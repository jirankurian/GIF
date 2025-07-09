# POC/gif_poc_modules/training/train_evaluate.py

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm # Progress bars
from sklearn.metrics import (accuracy_score, mean_squared_error, roc_auc_score,
                             precision_recall_fscore_support, balanced_accuracy_score)
import os
import json
import time
import warnings

# Import model definition only needed if type checking or accessing specific layers directly
# from gif_poc_modules.architecture.gif_model import GIFModel

# Filter out user warnings from sklearn about zero_division
warnings.filterwarnings("ignore", category=UserWarning, message="Precision and F-score are ill-defined*")

# -----------------------------
# Helper Functions
# -----------------------------

def calculate_physics_penalty(predicted_depths_scaled, cfg):
    """
    Calculates a penalty for predicted scaled depths outside the plausible range [0, 1].
    The penalty encourages predictions to stay within the valid normalized range.

    Args:
        predicted_depths_scaled (torch.Tensor): Tensor of predicted depths (already scaled).
        cfg (module): The config module containing PHYSICS_REG_LAMBDA.

    Returns:
        torch.Tensor: The calculated penalty term (scalar).
    """
    # Return zero penalty if lambda is zero or negative
    if cfg.PHYSICS_REG_LAMBDA <= 0:
        return torch.tensor(0.0, device=predicted_depths_scaled.device)

    # Penalty for values below 0
    penalty_low = torch.relu(-predicted_depths_scaled)
    # Penalty for values above 1
    max_plausible_depth = 1.0
    penalty_high = torch.relu(predicted_depths_scaled - max_plausible_depth)

    # Mean penalty across the batch
    total_penalty = torch.mean(penalty_low + penalty_high)

    # Scale by lambda
    return total_penalty * cfg.PHYSICS_REG_LAMBDA

def calculate_sparsity(total_aggregated_spikes, num_samples, cfg):
    """
    Calculates the average firing rate per neuron per time step (proxy for sparsity).

    Args:
        total_aggregated_spikes (float): Sum of all spikes across batch, layers, time steps.
        num_samples (int): Number of samples processed.
        cfg (module): Config module with SNN parameters.

    Returns:
        float: Average firing rate. Returns 0.0 if denominator is zero.
    """
    # Extract SNN parameters from config
    num_snn_neurons = cfg.SNN_HIDDEN_SIZE
    num_snn_layers = cfg.SNN_NUM_LAYERS
    num_time_steps = cfg.NUM_TIME_STEPS

    # Basic validation
    if num_samples <= 0 or num_snn_neurons <= 0 or num_snn_layers <= 0 or num_time_steps <= 0:
        return 0.0

    # Calculate the total number of possible spike events
    total_possible_spikes = num_samples * num_snn_neurons * num_snn_layers * num_time_steps

    if total_possible_spikes == 0:
        return 0.0

    # Calculate average firing rate
    avg_firing_rate = total_aggregated_spikes / total_possible_spikes
    return avg_firing_rate

# -----------------------------
# Training Function (One Epoch)
# -----------------------------

def train_one_epoch(model, train_loader, optimizer, criterion_cls, criterion_reg, device, cfg, desc_prefix="Training"):
    """
    Trains the model for one epoch using the provided data loader and optimizer.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        criterion_cls (nn.Module): Loss function for classification.
        criterion_reg (nn.Module): Loss function for regression.
        device (str): Device to perform training on ('cpu', 'cuda', 'mps').
        cfg (module): The config module.
        desc_prefix (str): Prefix for the progress bar description.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()  # Set model to training mode
    total_loss = 0.0
    num_batches = len(train_loader)

    if num_batches == 0:
        print(f"Warning: Training loader for '{desc_prefix}' is empty. Skipping epoch.")
        return 0.0

    # Progress bar
    pbar = tqdm(train_loader, desc=f"{desc_prefix} Epoch Progress", leave=False, total=num_batches)

    for batch_idx, batch_data in enumerate(pbar):
        # Basic check for expected batch structure
        if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 3:
            print(f"Warning: Skipping invalid batch in {desc_prefix}. Expected 3 items, got {len(batch_data)}.")
            continue

        inputs, labels_cls, labels_reg = batch_data
        inputs, labels_cls, labels_reg = inputs.to(device), labels_cls.to(device), labels_reg.to(device)

        # --- Forward Pass ---
        optimizer.zero_grad(set_to_none=True) # More memory efficient than zero_grad()
        try:
            cls_logits, reg_preds, _ = model(inputs) # Get predictions and spike info
        except Exception as e:
            print(f"\nError during forward pass in {desc_prefix} (Batch {batch_idx}): {e}")
            print(f"Input shape: {inputs.shape}")
            # Optional: Add more debugging info here if needed
            continue # Skip this batch

        # --- Loss Calculation ---
        loss_cls = criterion_cls(cls_logits, labels_cls)
        loss_reg = criterion_reg(reg_preds, labels_reg)
        loss_phys = calculate_physics_penalty(reg_preds, cfg) # Calculate physics penalty

        # Combine losses using weights from config
        combined_loss = (cfg.CLASSIFICATION_LOSS_WEIGHT * loss_cls +
                         cfg.REGRESSION_LOSS_WEIGHT * loss_reg +
                         loss_phys) # Physics penalty added directly (already includes lambda)

        # --- Backward Pass & Optimization ---
        try:
            combined_loss.backward()
            optimizer.step()
        except Exception as e:
            print(f"\nError during backward pass or optimizer step in {desc_prefix} (Batch {batch_idx}): {e}")
            # Consider skipping optimizer step if backward failed critically
            # optimizer.zero_grad(set_to_none=True) # Reset gradients anyway
            continue # Skip this batch


        total_loss += combined_loss.item()

        # Update progress bar
        pbar.set_postfix(loss=f"{total_loss / (batch_idx + 1):.4f}", refresh=True)

    # Calculate average loss for the epoch
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    pbar.close()
    return avg_loss

# -----------------------------
# Evaluation Function
# -----------------------------

def evaluate(model, data_loader, criterion_cls, criterion_reg, device, cfg, desc_prefix="Evaluating"):
    """
    Evaluates the model performance on a given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation set.
        criterion_cls (nn.Module): Loss function for classification.
        criterion_reg (nn.Module): Loss function for regression.
        device (str): Device for evaluation ('cpu', 'cuda', 'mps').
        cfg (module): The config module.
        desc_prefix (str): Prefix for the progress bar description.

    Returns:
        dict: Dictionary containing various evaluation metrics and results.
    """
    model.eval()  # Set model to evaluation mode
    total_loss_cls = 0.0
    total_loss_reg = 0.0
    total_loss_phys = 0.0
    all_preds_cls_prob = []
    all_labels_cls = []
    all_preds_reg = []
    all_labels_reg = []
    total_spikes_agg = 0.0
    num_samples = 0
    num_batches = len(data_loader)

    # Default results dictionary for empty loader case
    default_results = {
        "loss": float('inf'), "loss_cls": float('inf'), "loss_reg": float('inf'), "loss_phys": float('inf'),
        "accuracy": 0.0, "balanced_accuracy": 0.0, "mse": float('inf'), "auc": 0.0,
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "firing_rate": 0.0,
        "predictions_reg": np.array([]), "actuals_reg": np.array([]),
        "predictions_cls_prob": np.array([]), "predictions_cls_binary": np.array([]),
        "actuals_cls": np.array([])
    }

    if num_batches == 0:
        print(f"Warning: Evaluation loader '{desc_prefix}' is empty.")
        return default_results

    pbar = tqdm(data_loader, desc=desc_prefix, leave=False, total=num_batches)

    with torch.no_grad(): # Disable gradient calculations for evaluation
        for batch_data in pbar:
            if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 3:
                print(f"Warning: Skipping invalid batch in {desc_prefix}. Expected 3 items, got {len(batch_data)}.")
                continue

            inputs, labels_cls, labels_reg = batch_data
            inputs, labels_cls, labels_reg = inputs.to(device), labels_cls.to(device), labels_reg.to(device)

            try:
                cls_logits, reg_preds, spikes_per_sample = model(inputs)
            except Exception as e:
                print(f"\nError during forward pass in {desc_prefix}: {e}")
                print(f"Input shape: {inputs.shape}")
                continue # Skip this batch

            # Calculate losses for this batch
            loss_cls = criterion_cls(cls_logits, labels_cls)
            loss_reg = criterion_reg(reg_preds, labels_reg)
            loss_phys = calculate_physics_penalty(reg_preds, cfg)

            batch_size = inputs.size(0)
            total_loss_cls += loss_cls.item() * batch_size # Accumulate total loss, not average
            total_loss_reg += loss_reg.item() * batch_size
            total_loss_phys += loss_phys.item() # Physics penalty is already averaged in helper

            # Store predictions and labels for metric calculation later
            preds_cls_prob = torch.sigmoid(cls_logits) # Convert logits to probabilities
            all_preds_cls_prob.append(preds_cls_prob.cpu().numpy())
            all_labels_cls.append(labels_cls.cpu().numpy())
            all_preds_reg.append(reg_preds.cpu().numpy())
            all_labels_reg.append(labels_reg.cpu().numpy())

            # Accumulate spikes
            if spikes_per_sample is not None:
                total_spikes_agg += spikes_per_sample.sum().item() # Sum spikes across batch

            num_samples += batch_size

    pbar.close()

    if num_samples == 0: # If all batches were skipped
        print(f"Warning: No valid samples processed in '{desc_prefix}'.")
        return default_results

    # Calculate average losses
    avg_loss_cls = total_loss_cls / num_samples
    avg_loss_reg = total_loss_reg / num_samples
    # Physics loss was already averaged per batch, so average over batches
    avg_loss_phys = total_loss_phys / num_batches if num_batches > 0 else 0
    avg_loss_combined = (cfg.CLASSIFICATION_LOSS_WEIGHT * avg_loss_cls +
                         cfg.REGRESSION_LOSS_WEIGHT * avg_loss_reg +
                         avg_loss_phys)

    # Concatenate results from all batches
    try:
        labels_cls_np = np.concatenate(all_labels_cls).flatten()
        preds_cls_prob_np = np.concatenate(all_preds_cls_prob).flatten()
        preds_cls_binary_np = (preds_cls_prob_np > 0.5).astype(int) # Threshold probabilities
        labels_reg_np = np.concatenate(all_labels_reg).flatten()
        preds_reg_np = np.concatenate(all_preds_reg).flatten()
    except ValueError: # Handle cases where lists might be empty if all batches failed
        print(f"Warning: Could not concatenate results for {desc_prefix}. Results might be empty.")
        return default_results


    # Calculate classification metrics
    accuracy = accuracy_score(labels_cls_np, preds_cls_binary_np)
    balanced_acc = balanced_accuracy_score(labels_cls_np, preds_cls_binary_np)
    # Use zero_division=0 to avoid warnings when a class has no predictions/labels
    precision, recall, f1, _ = precision_recall_fscore_support(labels_cls_np, preds_cls_binary_np, average='binary', zero_division=0)

    # Calculate AUC - requires at least one sample from each class
    auc = 0.0
    if len(np.unique(labels_cls_np)) > 1:
        try:
            auc = roc_auc_score(labels_cls_np, preds_cls_prob_np)
        except ValueError as e:
            print(f"Warning: AUC calculation failed for '{desc_prefix}': {e}. Often due to only one class present.")
            auc = 0.0 # Or np.nan, depending on desired handling
    else:
        print(f"Warning: Only one class present in labels for '{desc_prefix}'. AUC set to 0.0.")


    # Calculate regression metrics
    mse = mean_squared_error(labels_reg_np, preds_reg_np)

    # Calculate SNN sparsity
    avg_firing_rate = calculate_sparsity(total_aggregated_spikes=total_spikes_agg,
                                       num_samples=num_samples,
                                       cfg=cfg)

    # Compile results dictionary
    results = {
        "loss": avg_loss_combined,
        "loss_cls": avg_loss_cls,
        "loss_reg": avg_loss_reg,
        "loss_phys": avg_loss_phys,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "mse": mse,
        "firing_rate": avg_firing_rate,
        "predictions_reg": preds_reg_np,      # Return raw predictions for plotting
        "actuals_reg": labels_reg_np,
        "predictions_cls_prob": preds_cls_prob_np, # Return probabilities for AUC/analysis
        "predictions_cls_binary": preds_cls_binary_np, # Return binary predictions for Acc/F1 etc.
        "actuals_cls": labels_cls_np
    }

    return results

# ---------------------------------------
# Main Training Loop Orchestration (Primary)
# ---------------------------------------

def run_training_loop(model, train_loader, val_loader, optimizer, criterion_cls, criterion_reg, device, cfg):
    """
    Runs the primary training loop for a specified number of epochs,
    evaluates on the validation set, and saves the best model based on validation loss.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): Optimizer for primary training.
        criterion_cls (nn.Module): Loss function for classification.
        criterion_reg (nn.Module): Loss function for regression.
        device (str): Device to run on.
        cfg (module): The config module.

    Returns:
        dict: Training history dictionary.
    """
    best_val_loss = float('inf')
    history_keys = [
        'train_loss', 'val_loss', 'val_loss_cls', 'val_loss_reg', 'val_loss_phys',
        'val_accuracy', 'val_balanced_accuracy', 'val_precision', 'val_recall', 'val_f1',
        'val_auc', 'val_mse', 'val_firing_rate'
    ]
    history = {k: [] for k in history_keys}

    print(f"\n--- Starting Primary Training ---")
    print(f"  Device: {device}")
    print(f"  Epochs: {cfg.NUM_EPOCHS_PRIMARY}")
    print(f"  Optimizer: {type(optimizer).__name__}")
    print(f"  Learning Rate: {cfg.LEARNING_RATE_PRIMARY}")
    print(f"  Saving best model to: {cfg.PRIMARY_MODEL_SAVE_PATH}")
    print("-" * 30)

    training_start_time = time.time()

    for epoch in range(cfg.NUM_EPOCHS_PRIMARY):
        epoch_start_time = time.time()
        print(f"\n--- Primary Epoch {epoch + 1}/{cfg.NUM_EPOCHS_PRIMARY} ---")

        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_cls, criterion_reg, device, cfg, desc_prefix="Primary Train")

        # Evaluate on validation set
        val_results = evaluate(model, val_loader, criterion_cls, criterion_reg, device, cfg, desc_prefix="Primary Val")
        val_loss = val_results["loss"] # Use combined loss for saving best model

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} Summary | Time: {epoch_duration:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_results['auc']:.4f} | Val MSE: {val_results['mse']:.6f}")

        # Store history
        history['train_loss'].append(train_loss)
        for key in history.keys():
             # Map 'val_metric' key to 'metric' key in results dict
             results_key = key.replace('val_', '')
             if key != 'train_loss' and results_key in val_results:
                 history[key].append(val_results[results_key])
             elif key != 'train_loss':
                 history[key].append(None) # Append None if metric wasn't calculated

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                torch.save(model.state_dict(), cfg.PRIMARY_MODEL_SAVE_PATH)
                print(f"-------> Best primary model saved (Epoch {epoch + 1}, Val Loss: {best_val_loss:.4f}) <-------")
            except Exception as e:
                print(f"--- Error saving primary model: {e} ---")
        else:
             print(f"       (Val loss did not improve from {best_val_loss:.4f})")


    total_training_time = time.time() - training_start_time
    print(f"\n--- Primary Training finished ---")
    print(f"Total duration: {total_training_time:.2f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save training history log
    try:
        # Convert numpy types for JSON serialization
        serializable_history = {}
        for k, v_list in history.items():
            serializable_history[k] = [float(vi) if isinstance(vi, (np.float32, np.float64, np.number, np.bool_)) else vi for vi in v_list]

        with open(cfg.PRIMARY_HISTORY_SAVE_PATH, 'w') as f:
             json.dump(serializable_history, f, indent=4)
        print(f"Primary training history saved to {cfg.PRIMARY_HISTORY_SAVE_PATH}")
    except TypeError as e:
        print(f"--- Error serializing primary history to JSON: {e}. Check for non-serializable types. ---")
    except Exception as e:
        print(f"--- Error saving primary history: {e} ---")

    return history

# ---------------------------------------
# Fine-tuning Loop Orchestration
# ---------------------------------------

def run_finetuning_loop(model, finetune_loader, val_loader, criterion_cls, criterion_reg, device, cfg):
    """
    Runs the fine-tuning loop on a target dataset (e.g., Dataset B).
    Optionally freezes parts of the model based on config settings.

    Args:
        model (nn.Module): The model to fine-tune (usually loaded with pre-trained weights).
        finetune_loader (DataLoader): DataLoader for the fine-tuning dataset.
        val_loader (DataLoader): DataLoader for validation during fine-tuning (can be same as finetune_loader).
        criterion_cls (nn.Module): Loss function for classification.
        criterion_reg (nn.Module): Loss function for regression.
        device (str): Device to run on.
        cfg (module): The config module.

    Returns:
        dict: Fine-tuning history dictionary.
    """
    print(f"\n--- Starting Fine-tuning ---")
    print(f"  Device: {device}")
    print(f"  Epochs: {cfg.NUM_EPOCHS_FINETUNE}")
    print(f"  Fine-tuning SNN Core: {cfg.FINETUNE_SNN_CORE}")
    print(f"  Decoder LR: {cfg.LEARNING_RATE_FINETUNE_DECODER}")
    if cfg.FINETUNE_SNN_CORE:
        print(f"  SNN Core LR: {cfg.LEARNING_RATE_FINETUNE_SNN}")
    print(f"  Saving best model to: {cfg.FINETUNED_MODEL_SAVE_PATH}")
    print("-" * 30)

    start_time = time.time()

    # --- Layer Freezing Logic ---
    print("Configuring layers for fine-tuning...")
    # Freeze Encoder by default
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("  - Encoder: FROZEN")

    # Freeze or Unfreeze SNN Core based on config
    snn_params_to_tune = []
    if cfg.FINETUNE_SNN_CORE:
        for param in model.core_du.parameters():
            param.requires_grad = True
        snn_params_to_tune = list(model.core_du.parameters()) # Collect SNN params
        print("  - SNN Core (DU): UNFROZEN (Learning Rate applied)")
    else:
        for param in model.core_du.parameters():
            param.requires_grad = False
        print("  - SNN Core (DU): FROZEN")

    # Always fine-tune the projection layer and decoders
    for param in model.ann_to_snn_input.parameters():
         param.requires_grad = True
    print("  - ANN->SNN Projection: UNFROZEN")
    for param in model.decoder_cls.parameters():
        param.requires_grad = True
    print("  - Decoder CLS: UNFROZEN")
    for param in model.decoder_reg.parameters():
        param.requires_grad = True
    print("  - Decoder REG: UNFROZEN")

    # --- Optimizer Setup ---
    # Group parameters for different learning rates if tuning SNN core
    params_to_optimize = [
        {'params': model.ann_to_snn_input.parameters(), 'lr': cfg.LEARNING_RATE_FINETUNE_DECODER},
        {'params': model.decoder_cls.parameters(), 'lr': cfg.LEARNING_RATE_FINETUNE_DECODER},
        {'params': model.decoder_reg.parameters(), 'lr': cfg.LEARNING_RATE_FINETUNE_DECODER}
    ]
    if cfg.FINETUNE_SNN_CORE and snn_params_to_tune:
         params_to_optimize.append({'params': snn_params_to_tune, 'lr': cfg.LEARNING_RATE_FINETUNE_SNN})
         print(f"  Optimizer groups: Decoders/Projection (LR={cfg.LEARNING_RATE_FINETUNE_DECODER}), SNN Core (LR={cfg.LEARNING_RATE_FINETUNE_SNN})")
    else:
         print(f"  Optimizer groups: Decoders/Projection (LR={cfg.LEARNING_RATE_FINETUNE_DECODER})")


    if cfg.OPTIMIZER == 'Adam':
        optimizer_ft = torch.optim.Adam(params_to_optimize) # Pass param groups
    # Add AdamW or other optimizers if needed
    # elif cfg.OPTIMIZER == 'AdamW':
    #     optimizer_ft = torch.optim.AdamW(params_to_optimize)
    else:
        # Fallback if only one LR is needed (or if optimizer doesn't support groups well)
        print(f"Warning: Optimizer {cfg.OPTIMIZER} might not fully support parameter groups. Using default LR {cfg.LEARNING_RATE_FINETUNE_DECODER} for all trainable params.")
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        if cfg.OPTIMIZER == 'Adam':
             optimizer_ft = torch.optim.Adam(trainable_params, lr=cfg.LEARNING_RATE_FINETUNE_DECODER)
        else:
             raise NotImplementedError(f"Optimizer {cfg.OPTIMIZER} not implemented.")

    # --- Fine-tuning Epoch Loop ---
    best_val_loss_ft = float('inf')
    # Use same keys as primary training for consistency
    history_ft = { k: [] for k in [
        'train_loss', 'val_loss', 'val_loss_cls', 'val_loss_reg', 'val_loss_phys',
        'val_accuracy', 'val_balanced_accuracy', 'val_precision', 'val_recall', 'val_f1',
        'val_auc', 'val_mse', 'val_firing_rate'
    ] }

    for epoch in range(cfg.NUM_EPOCHS_FINETUNE):
        epoch_start_time = time.time()
        print(f"\n--- Fine-tuning Epoch {epoch + 1}/{cfg.NUM_EPOCHS_FINETUNE} ---")

        # Train one epoch on the fine-tuning data
        train_loss_ft = train_one_epoch(model, finetune_loader, optimizer_ft, criterion_cls, criterion_reg, device, cfg, desc_prefix="FineTune Train")

        # Evaluate on the validation set (often same as fine-tuning set for adaptation tasks)
        val_results_ft = evaluate(model, val_loader, criterion_cls, criterion_reg, device, cfg, desc_prefix="FineTune Val")
        val_loss_ft = val_results_ft["loss"] # Use combined loss

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} Summary | Time: {epoch_duration:.2f}s | Train Loss: {train_loss_ft:.4f} | Val Loss: {val_loss_ft:.4f} | Val AUC: {val_results_ft['auc']:.4f} | Val MSE: {val_results_ft['mse']:.6f}")

        # Store history
        history_ft['train_loss'].append(train_loss_ft)
        for key in history_ft.keys():
             results_key = key.replace('val_', '')
             if key != 'train_loss' and results_key in val_results_ft:
                 history_ft[key].append(val_results_ft[results_key])
             elif key != 'train_loss':
                 history_ft[key].append(None)

        # Save the best model based on validation loss
        if val_loss_ft < best_val_loss_ft:
            best_val_loss_ft = val_loss_ft
            try:
                torch.save(model.state_dict(), cfg.FINETUNED_MODEL_SAVE_PATH)
                print(f"-------> Best fine-tuned model saved (Epoch {epoch + 1}, Val Loss: {best_val_loss_ft:.4f}) <-------")
            except Exception as e:
                print(f"--- Error saving fine-tuned model: {e} ---")
        else:
             print(f"       (Val loss did not improve from {best_val_loss_ft:.4f})")


    total_time = time.time() - start_time
    print(f"\n--- Fine-tuning finished ---")
    print(f"Total duration: {total_time:.2f}s")
    print(f"Best validation loss during fine-tuning: {best_val_loss_ft:.4f}")


    # --- Unfreeze All Layers ---
    # Important to unfreeze if the same model instance might be used further
    print("Unfreezing all model layers...")
    for param in model.parameters():
        param.requires_grad = True
    print("Model layers unfrozen.")

    # Save fine-tuning history log
    try:
        # Convert numpy types for JSON serialization
        serializable_history_ft = {}
        for k, v_list in history_ft.items():
            serializable_history_ft[k] = [float(vi) if isinstance(vi, (np.float32, np.float64, np.number, np.bool_)) else vi for vi in v_list]

        with open(cfg.FINETUNE_HISTORY_SAVE_PATH, 'w') as f:
             json.dump(serializable_history_ft, f, indent=4)
        print(f"Fine-tuning history saved to {cfg.FINETUNE_HISTORY_SAVE_PATH}")
    except TypeError as e:
        print(f"--- Error serializing fine-tuning history to JSON: {e}. Check non-serializable types. ---")
    except Exception as e:
        print(f"--- Error saving fine-tuning history: {e} ---")

    return history_ft
