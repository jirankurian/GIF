# POC/gif_poc_modules/data_utils.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from tqdm.auto import tqdm # For progress bars

# -----------------------------
# 1. Synthetic Data Generation Components
# -----------------------------

def transit_signal(times_indices, t0_index, duration_steps, depth, shape='box', ingress_duration_fraction=0.1):
    """
    Generates a transit signal (flux drop) centered at t0_index.

    Args:
        times_indices (np.ndarray): Array of time step indices (e.g., np.arange(seq_length)).
        t0_index (float): Center time index of the transit.
        duration_steps (float): Total duration of the transit in steps.
        depth (float): Maximum depth of the transit (positive value).
        shape (str): Shape of the transit ('box' or 'trapezoid').
        ingress_duration_fraction (float): Fraction of half-duration for ingress/egress in trapezoid shape.

    Returns:
        np.ndarray: Array representing the flux drop due to the transit.
    """
    if depth <= 0 or duration_steps <= 0:
        return np.zeros_like(times_indices, dtype=np.float32)

    half_duration = duration_steps / 2.0
    flux_drop = np.zeros_like(times_indices, dtype=np.float32)

    if shape == 'box':
        # Simple rectangular transit
        in_transit = (np.abs(times_indices - t0_index) <= half_duration)
        flux_drop[in_transit] = depth
    elif shape == 'trapezoid':
        # Trapezoidal transit with ingress/egress ramps
        ing_egr_steps = half_duration * ingress_duration_fraction
        flat_bottom_half_dur = half_duration - ing_egr_steps

        # Ensure non-negative flat bottom duration
        if flat_bottom_half_dur < 0:
            flat_bottom_half_dur = 0
            ing_egr_steps = half_duration # Ramps meet in the middle

        # Define time ranges for each phase
        t_start_ing = t0_index - half_duration
        t_end_ing = t0_index - flat_bottom_half_dur
        t_start_egr = t0_index + flat_bottom_half_dur
        t_end_egr = t0_index + half_duration

        # Calculate flux drop for each phase
        in_ingress = (times_indices >= t_start_ing) & (times_indices < t_end_ing)
        if ing_egr_steps > 1e-6: # Avoid division by zero
             flux_drop[in_ingress] = depth * (times_indices[in_ingress] - t_start_ing) / ing_egr_steps
        else: # Handle edge case of zero ingress duration (becomes box-like)
            flux_drop[in_ingress] = depth


        in_bottom = (np.abs(times_indices - t0_index) <= flat_bottom_half_dur)
        flux_drop[in_bottom] = depth

        in_egress = (times_indices > t_start_egr) & (times_indices <= t_end_egr)
        if ing_egr_steps > 1e-6: # Avoid division by zero
             flux_drop[in_egress] = depth * (t_end_egr - times_indices[in_egress]) / ing_egr_steps
        else: # Handle edge case of zero egress duration
             flux_drop[in_egress] = depth

        # Clip just in case of floating point issues at boundaries
        flux_drop = np.clip(flux_drop, 0, depth)

    else:
        raise ValueError(f"Unsupported transit shape: {shape}")

    return flux_drop

def generate_synthetic_light_curve(seq_length, transit_prob, noise_level,
                                   depth_range, duration_range_steps, shape):
    """
    Generates a single synthetic light curve with optional transit and noise.

    Args:
        seq_length (int): Length of the light curve sequence.
        transit_prob (float): Probability of the curve containing a transit.
        noise_level (float): Standard deviation of Gaussian noise.
        depth_range (tuple): Min and max possible transit depth.
        duration_range_steps (tuple): Min and max possible transit duration in steps.
        shape (str): Shape of the transit ('box' or 'trapezoid').

    Returns:
        tuple: (light_curve_array, label, actual_depth)
               light_curve_array: (seq_length, 1) numpy array
               label: 0 (no transit) or 1 (transit)
               actual_depth: Depth of the transit (0 if no transit)
    """
    times_indices = np.arange(seq_length)
    baseline_flux = np.ones(seq_length)
    label = 0
    actual_depth = 0.0

    if np.random.rand() < transit_prob:
        # Generate a transit
        label = 1
        min_dur_steps, max_dur_steps = duration_range_steps
        # Ensure duration is at least 1 step
        duration_steps = np.random.uniform(max(1, min_dur_steps), max_dur_steps)
        actual_depth = np.random.uniform(depth_range[0], depth_range[1])

        # Ensure t0 allows the full transit duration within bounds
        max_half_dur = duration_steps / 2.0
        # Add a small buffer (e.g., 1 step) to avoid edge effects
        buffer = max_half_dur + 1
        if seq_length > 2 * buffer: # Check if sequence is long enough for a transit
             t0_index = np.random.uniform(buffer, seq_length - buffer) # Center index
             transit = transit_signal(times_indices, t0_index, duration_steps, actual_depth, shape=shape)
             baseline_flux -= transit
        else:
             label = 0 # Cannot fit transit, revert label
             actual_depth = 0.0


    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, seq_length)
    light_curve = baseline_flux + noise

    # Clip flux to be non-negative (optional, depends on whether negative flux is plausible/desired)
    # light_curve = np.clip(light_curve, 0, None)

    # Reshape for consistency (SeqLength, Features=1)
    light_curve = light_curve.reshape(-1, 1)

    return light_curve.astype(np.float32), label, np.float32(actual_depth)

def generate_and_save_dataset(dataset_params, filename, seq_length):
    """
    Generates a dataset based on params and saves it as a .pt file.

    Args:
        dataset_params (dict): Dictionary containing generation parameters for the dataset.
        filename (str): Path to save the generated dataset.
        seq_length (int): The sequence length (passed explicitly).
    """
    if os.path.exists(filename):
        print(f"Dataset file {filename} already exists. Skipping generation.")
        return

    num_samples = dataset_params['num_samples']
    print(f"Generating {num_samples} synthetic samples for '{dataset_params['name']}'...")

    # Pre-allocate numpy arrays
    all_light_curves = np.zeros((num_samples, seq_length, 1), dtype=np.float32)
    all_labels = np.zeros(num_samples, dtype=np.int64) # Use int64 for labels often expected by PyTorch loss fns
    all_depths = np.zeros(num_samples, dtype=np.float32)

    # Use tqdm for progress bar
    for i in tqdm(range(num_samples), desc=f"Generating '{dataset_params['name']}'"):
        lc, label, depth = generate_synthetic_light_curve(
            seq_length=seq_length,
            transit_prob=dataset_params['transit_prob'],
            noise_level=dataset_params['noise_level'],
            depth_range=dataset_params['depth_range'],
            duration_range_steps=dataset_params['duration_range_steps'],
            shape=dataset_params['shape']
        )
        all_light_curves[i] = lc
        all_labels[i] = label
        all_depths[i] = depth

    # Convert to PyTorch tensors and save in a dictionary
    data_dict = {
        'light_curves': torch.from_numpy(all_light_curves),
        'labels': torch.from_numpy(all_labels),
        'depths': torch.from_numpy(all_depths)
    }

    try:
        # Ensure parent directory exists before saving
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(data_dict, filename)
        print(f"Dataset '{dataset_params['name']}' saved to {filename}")
    except Exception as e:
        print(f"Error saving dataset {filename}: {e}")

# -----------------------------
# 2. Data Preprocessing & Scaling
# -----------------------------

def preprocess_and_scale_depths(depths_numpy, scaler_path=None, fit_scaler=False):
    """
    Scales transit depths to the range [0, 1] using MinMaxScaler.
    Loads/saves the scaler based on the fit_scaler flag.

    Args:
        depths_numpy (np.ndarray): Numpy array of transit depths.
        scaler_path (str, optional): Path to load/save the scaler object.
        fit_scaler (bool): If True, fit a new scaler and save it.
                           If False, try to load an existing scaler.

    Returns:
        tuple: (scaled_depths_numpy, scaler_object)
               Returns (None, None) on failure.
    """
    scaler = None
    # Ensure input is a numpy array
    if not isinstance(depths_numpy, np.ndarray):
        try:
            depths_numpy = np.array(depths_numpy)
        except Exception as e:
            print(f"Error: Could not convert depths to numpy array: {e}")
            return None, None

    # Reshape for scaler: (n_samples,) -> (n_samples, 1)
    depths_reshaped = depths_numpy.reshape(-1, 1)

    # Try loading existing scaler if not fitting
    if not fit_scaler and scaler_path and os.path.exists(scaler_path):
        try:
            print(f"Loading depth scaler from {scaler_path}")
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Warning: Failed to load scaler from {scaler_path}: {e}. Fitting a new one.")
            fit_scaler = True # Force fitting if loading failed

    # Fit a new scaler if required or if loading failed
    if fit_scaler or scaler is None:
        print("Fitting new depth scaler...")
        scaler = MinMaxScaler(feature_range=(0, 1))
        try:
            scaler.fit(depths_reshaped)
            print("Scaler fitted.")
            # Save the newly fitted scaler if a path is provided
            if scaler_path:
                try:
                    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
                    joblib.dump(scaler, scaler_path)
                    print(f"Saving depth scaler to {scaler_path}")
                except Exception as e:
                    print(f"Error saving scaler to {scaler_path}: {e}")
            else:
                print("Warning: fit_scaler=True but scaler_path not provided. Scaler won't be saved.")
        except Exception as e:
             print(f"Error fitting scaler: {e}")
             return None, None # Return None on fitting error

    # Transform the data using the loaded or newly fitted scaler
    try:
        scaled_depths = scaler.transform(depths_reshaped)
        return scaled_depths.astype(np.float32), scaler
    except Exception as e:
        print(f"Error transforming depths with scaler: {e}")
        return None, None

# -----------------------------
# 3. PyTorch Dataset and DataLoader
# -----------------------------

class ExoplanetDatasetPT(Dataset):
    """
    Custom PyTorch Dataset that loads data from pre-generated tensors.
    Handles reshaping based on the encoder type (ANN_TYPE).
    """
    def __init__(self, light_curves_tensor, labels_tensor, depths_tensor, ann_type):
        """
        Args:
            light_curves_tensor (torch.Tensor): Shape (N, SeqLength, Features=1)
            labels_tensor (torch.Tensor): Shape (N,)
            depths_tensor (torch.Tensor): Shape (N, 1) - Scaled depths
            ann_type (str): 'CNN' or 'LSTM', determines light curve reshaping.
        """
        self.light_curves = light_curves_tensor
        # Ensure labels are float and have shape (N, 1) for BCEWithLogitsLoss
        self.labels = labels_tensor.float().unsqueeze(1)
        # Ensure depths have shape (N, 1)
        self.depths = depths_tensor.float()
        if self.depths.ndim == 1:
            self.depths = self.depths.unsqueeze(1)

        # Reshape light curves based on the expected input format of the encoder
        # Input shape from file is assumed: (N, SeqLength, Features=1)
        if ann_type == 'LSTM':
            # LSTM expects (N, SeqLength, Features) - current shape is fine
            pass
        elif ann_type == 'CNN':
            # Conv1d expects (N, Channels=Features, SeqLength)
            self.light_curves = self.light_curves.permute(0, 2, 1)
        else:
            raise ValueError(f"Unsupported ANN_TYPE: {ann_type} in ExoplanetDatasetPT")

        # Store original sequence length for potential checks
        self.seq_length = self.light_curves.shape[-1] if ann_type == 'CNN' else self.light_curves.shape[-2]


    def __len__(self):
        """Returns the total number of samples."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Returns a single sample: light curve, label, depth."""
        # Add error handling for index out of bounds
        if idx >= len(self.labels):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.labels)}")
        return self.light_curves[idx], self.labels[idx], self.depths[idx]


def get_dataloaders(dataset_params, dataset_filename, scaler_path, cfg,
                    train_split_ratio=None, fit_scaler=False):
    """
    Generates/loads dataset, scales depths, optionally splits into train/val,
    and creates PyTorch DataLoaders.

    Args:
        dataset_params (dict): Parameters for generating the dataset if it doesn't exist.
        dataset_filename (str): Path to the .pt dataset file.
        scaler_path (str): Path to load/save the depth scaler.
        cfg (module): The config module containing parameters like BATCH_SIZE, ANN_TYPE, SEQ_LENGTH.
        train_split_ratio (float, optional): Ratio for splitting into training and validation sets.
                                             If None, uses the entire dataset. Defaults to None.
        fit_scaler (bool): Whether to fit a new scaler (typically True for primary dataset A)
                           or load an existing one (typically False for dataset B). Defaults to False.

    Returns:
        tuple: Depending on train_split_ratio:
               - If split: (train_loader, val_loader, depth_scaler)
               - If no split: (data_loader, depth_scaler)
               Returns (None, None, None) or (None, None) on failure.
    """
    # Generate dataset if it doesn't exist
    generate_and_save_dataset(dataset_params, dataset_filename, cfg.SEQ_LENGTH)

    # Load dataset from file
    try:
        print(f"Loading dataset from {dataset_filename}...")
        data_dict = torch.load(dataset_filename)
        light_curves_np = data_dict['light_curves'].numpy()
        labels_np = data_dict['labels'].numpy()
        depths_np = data_dict['depths'].numpy()
        print(f"Dataset loaded. Shapes: LC={light_curves_np.shape}, Labels={labels_np.shape}, Depths={depths_np.shape}")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_filename} after attempting generation.")
        return (None, None, None) if train_split_ratio else (None, None)
    except Exception as e:
        print(f"Error loading dataset {dataset_filename}: {e}")
        return (None, None, None) if train_split_ratio else (None, None)

    # Preprocess and scale depths
    scaled_depths_np, depth_scaler = preprocess_and_scale_depths(depths_np, scaler_path, fit_scaler=fit_scaler)
    if depth_scaler is None or scaled_depths_np is None:
        print("Error: Depth scaling failed.")
        return (None, None, None) if train_split_ratio else (None, None)

    # Convert data back to tensors
    light_curves_tensor = torch.from_numpy(light_curves_np)
    labels_tensor = torch.from_numpy(labels_np)
    scaled_depths_tensor = torch.from_numpy(scaled_depths_np)

    # Split dataset or use the whole dataset
    if train_split_ratio is not None and 0 < train_split_ratio < 1:
        try:
            # Split data
            lc_train, lc_val, labels_train, labels_val, depths_train, depths_val = train_test_split(
                light_curves_tensor, labels_tensor, scaled_depths_tensor,
                train_size=train_split_ratio,
                random_state=42, # for reproducibility
                stratify=labels_tensor # ensure class proportion is similar in splits
            )
            print(f"Split dataset: Train={len(lc_train)}, Validation={len(lc_val)}")

            # Create Datasets
            train_dataset = ExoplanetDatasetPT(lc_train, labels_train, depths_train, cfg.ANN_TYPE)
            val_dataset = ExoplanetDatasetPT(lc_val, labels_val, depths_val, cfg.ANN_TYPE)

            # Create DataLoaders
            # drop_last=True recommended for SNNs to maintain consistent batch size fed to hidden states
            train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=min(4, os.cpu_count()), pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=min(4, os.cpu_count()), pin_memory=True)

            return train_loader, val_loader, depth_scaler

        except Exception as e:
            print(f"Error during dataset splitting or DataLoader creation: {e}")
            return None, None, None
    else:
        # Use the entire dataset
        print("Using the entire dataset (no train/validation split).")
        try:
            full_dataset = ExoplanetDatasetPT(light_curves_tensor, labels_tensor, scaled_depths_tensor, cfg.ANN_TYPE)
            data_loader = DataLoader(full_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=min(4, os.cpu_count()), pin_memory=True)
            return data_loader, depth_scaler
        except Exception as e:
            print(f"Error creating DataLoader for the full dataset: {e}")
            return None, None


# Example Standalone Usage (requires config.py in the same directory or sys.path manipulation)
# if __name__ == "__main__":
#     try:
#         import config as local_cfg # Assuming config.py is runnable
#         print("Running data_utils standalone example...")
#         # Example: Get loaders for Dataset A with splitting
#         train_loader_a, val_loader_a, scaler_a = get_dataloaders(
#             dataset_params=local_cfg.DATASET_PARAMS_A,
#             dataset_filename=local_cfg.DATASET_A_FILENAME,
#             scaler_path=local_cfg.SCALER_SAVE_PATH,
#             cfg=local_cfg,
#             train_split_ratio=local_cfg.TRAIN_SPLIT_RATIO,
#             fit_scaler=True # Fit scaler for Dataset A
#         )
#         if train_loader_a:
#             print(f"Dataset A: Train batches={len(train_loader_a)}, Val batches={len(val_loader_a)}")
#             # Get one batch to check shapes
#             dl_iter = iter(train_loader_a)
#             sample_lc, sample_lbl, sample_dpt = next(dl_iter)
#             print(f"Sample batch shapes: LC={sample_lc.shape}, Label={sample_lbl.shape}, Depth={sample_dpt.shape}")
#         else:
#             print("Failed to get loaders for Dataset A.")

#         # Example: Get loader for Dataset B without splitting (using scaler from A)
#         loader_b, scaler_b = get_dataloaders(
#             dataset_params=local_cfg.DATASET_PARAMS_B,
#             dataset_filename=local_cfg.DATASET_B_FILENAME,
#             scaler_path=local_cfg.SCALER_SAVE_PATH,
#             cfg=local_cfg,
#             train_split_ratio=None, # No split
#             fit_scaler=False # Use existing scaler
#         )
#         if loader_b:
#             print(f"Dataset B: Total batches={len(loader_b)}")
#         else:
#             print("Failed to get loader for Dataset B.")

#     except ImportError:
#         print("Could not import config.py. Standalone example requires it.")
#     except Exception as main_e:
#         print(f"An error occurred in the standalone example: {main_e}")
