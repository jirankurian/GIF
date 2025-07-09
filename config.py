# POC/config.py

import torch
import os

# --- Core Parameters ---
# These define the fundamental setup of the POC
SEQ_LENGTH = 100            # Input sequence length (number of time steps for synthetic data)
ANN_TYPE = 'CNN'            # Encoder type: 'CNN' or 'LSTM'. Determines the initial feature extractor.
BATCH_SIZE = 64             # Number of samples processed in each training/evaluation step.

# --- SNN Core (DU Proxy) Parameters ---
# Configuration for the Spiking Neural Network core module
SNN_HIDDEN_SIZE = 128       # Number of neurons in each SNN layer.
SNN_NUM_LAYERS = 1          # Number of layers in the SNN core.
SPIKE_BETA = 0.95           # Membrane potential decay rate (closer to 1 means slower decay).
SPIKE_THRESHOLD = 1.0       # Neuron firing threshold.
NUM_TIME_STEPS = 25         # SNN simulation steps per input sample. More steps allow for finer temporal processing.

# --- Output Decoder Parameters ---
# Configuration for the output layers
OUTPUT_CLS_SIZE = 1         # Output size for the binary classification head (e.g., transit vs. no transit).
OUTPUT_REG_SIZE = 1         # Output size for the regression head (e.g., predicting transit depth).

# --- Data Configuration ---
# Settings related to data generation, storage, and loading
DATA_DIR = 'data'           # Directory to store generated .pt dataset files.
# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- Synthetic Dataset A (Primary Training) ---
# Parameters for the initial dataset used for primary model training.
DATASET_A_FILENAME = os.path.join(DATA_DIR, 'synthetic_dataset_A.pt')
DATASET_PARAMS_A = {
    "name": "Baseline",             # Descriptive name
    "num_samples": 10000,           # Number of light curves to generate
    "seq_length": SEQ_LENGTH,       # Sequence length (links to global param)
    "transit_prob": 0.5,            # Probability of a sample containing a transit
    "noise_level": 0.01,            # Standard deviation of Gaussian noise added
    "depth_range": (0.005, 0.05),   # Range for random transit depth
    "duration_range_steps": (int(0.03 * SEQ_LENGTH), int(0.15 * SEQ_LENGTH)), # Range for transit duration (in steps)
    "shape": "box"                  # Shape of the transit signal ('box' or 'trapezoid')
}

# --- Synthetic Dataset B (Adaptation Target) ---
# Parameters for the second dataset, used for demonstrating fine-tuning/adaptation.
# Note the differences from Dataset A (e.g., noise level, shape).
DATASET_B_FILENAME = os.path.join(DATA_DIR, 'synthetic_dataset_B.pt')
DATASET_PARAMS_B = {
    "name": "Altered",              # Descriptive name
    "num_samples": 2000,            # Smaller size for fine-tuning/evaluation example
    "seq_length": SEQ_LENGTH,       # Sequence length (links to global param)
    "transit_prob": 0.5,            # Probability of a sample containing a transit
    "noise_level": 0.02,            # Increased noise level compared to Dataset A
    "depth_range": (0.005, 0.05),   # Range for random transit depth
    "duration_range_steps": (int(0.05 * SEQ_LENGTH), int(0.12 * SEQ_LENGTH)), # Different duration range
    "shape": "trapezoid"            # Different transit shape compared to Dataset A
}

# --- Data Splitting ---
TRAIN_SPLIT_RATIO = 0.8         # Ratio of Dataset A used for training (remainder for validation). Set to None to use all for training.

# --- Model Architecture Specifics ---
# Parameters specific to certain encoder types (only used if that ANN_TYPE is selected)
# CNN specific
CNN_OUT_CHANNELS = [16, 32]     # Number of output channels for each CNN layer
CNN_KERNEL_SIZES = [5, 5]       # Kernel size for each CNN layer

# LSTM specific
LSTM_HIDDEN_SIZE = 64           # Hidden size for LSTM layers (can differ from SNN_HIDDEN_SIZE)
LSTM_NUM_LAYERS = 1             # Number of LSTM layers

# --- Training Configuration ---
# Parameters controlling the primary training and fine-tuning processes
NUM_EPOCHS_PRIMARY = 50         # Number of epochs for initial training on Dataset A
LEARNING_RATE_PRIMARY = 1e-3    # Learning rate for the primary optimizer

NUM_EPOCHS_FINETUNE = 20        # Number of epochs for fine-tuning on Dataset B
LEARNING_RATE_FINETUNE_DECODER = 5e-4 # Learning rate for fine-tuning the decoder layers
FINETUNE_SNN_CORE = False       # <<< NEW: Set to True to also fine-tune the SNN core layers
LEARNING_RATE_FINETUNE_SNN = 1e-5 # <<< NEW: Separate (usually smaller) LR if fine-tuning SNN core

OPTIMIZER = 'Adam'              # Optimizer type (currently only 'Adam' supported in training script)

# Loss function weights and regularization
CLASSIFICATION_LOSS_WEIGHT = 0.6 # Weight for the classification loss component
REGRESSION_LOSS_WEIGHT = 0.4   # Weight for the regression loss component
PHYSICS_REG_LAMBDA = 0.01       # Strength of the physics penalty (for predicted depths outside [0,1]). Set to 0 to disable.

# --- Device Configuration ---
# Automatically selects the best available device (MPS > CUDA > CPU)
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# --- Paths Configuration ---
# Defines where generated data, results, models, logs, and plots are saved
SAVE_DIR = 'results'
os.makedirs(SAVE_DIR, exist_ok=True) # Ensure base results directory exists

MODEL_SAVE_DIR = os.path.join(SAVE_DIR, 'saved_models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # Ensure model save directory exists

PLOT_SAVE_DIR = os.path.join(SAVE_DIR, 'plots')
os.makedirs(PLOT_SAVE_DIR, exist_ok=True) # Ensure plot save directory exists

LOG_SAVE_DIR = os.path.join(SAVE_DIR, 'logs')
os.makedirs(LOG_SAVE_DIR, exist_ok=True) # Ensure log save directory exists

# Specific file paths
PRIMARY_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'gif_poc_primary_model.pth')
FINETUNED_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'gif_poc_finetuned_model.pth')
SCALER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'depth_scaler.pkl') # Scaler for regression target
PRIMARY_HISTORY_SAVE_PATH = os.path.join(LOG_SAVE_DIR, 'primary_training_history.json')
FINETUNE_HISTORY_SAVE_PATH = os.path.join(LOG_SAVE_DIR, 'finetune_training_history.json')
EVAL_RESULTS_SAVE_PATH = os.path.join(LOG_SAVE_DIR, 'evaluation_results.json')

# --- Initial Print Statement ---
print("-" * 30)
print("GIF-POC Configuration Loaded:")
print(f"  Device: {DEVICE}")
print(f"  Sequence Length: {SEQ_LENGTH}")
print(f"  Encoder Type (ANN_TYPE): {ANN_TYPE}")
print(f"  SNN Hidden Size: {SNN_HIDDEN_SIZE}")
print(f"  SNN Time Steps: {NUM_TIME_STEPS}")
print(f"  Fine-tune SNN Core: {FINETUNE_SNN_CORE}")
print(f"  Output Dirs: {DATA_DIR}/, {SAVE_DIR}/")
print("-" * 30)
