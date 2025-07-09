# POC/main.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import joblib # For loading scaler
import json
import time
import traceback # For detailed error printing

# Import project modules using the package structure
import config # Assumes config.py is in the same directory as main.py
from gif_poc_modules import data_utils
from gif_poc_modules.architecture import gif_model
from gif_poc_modules.training import train_evaluate
from gif_poc_modules.utils import plotting
from gif_poc_modules.utils import xai_utils

# Helper function to make dict JSON serializable (handles numpy types)
def make_serializable(data_dict):
    """Converts numpy types in a dictionary to standard Python types for JSON."""
    if not isinstance(data_dict, dict):
         # Return empty dict or raise error if input is not a dict
         print(f"Warning: make_serializable received non-dict input: {type(data_dict)}. Returning empty dict.")
         return {}
    serializable_dict = {}
    for k, v in data_dict.items():
         if isinstance(v, np.ndarray):
             serializable_dict[k] = v.tolist() # Convert numpy arrays to lists
         # CORRECTED: Check against base NumPy numeric and boolean types
         elif isinstance(v, (np.floating, np.integer)): # Checks for any numpy float or int type
             serializable_dict[k] = v.item() # Use .item() to convert numpy numbers to standard Python types
         elif isinstance(v, np.bool_):
             serializable_dict[k] = bool(v) # Convert numpy bool to standard Python bool
         elif isinstance(v, (float, int, str, bool, list, dict)) or v is None:
             serializable_dict[k] = v # Keep standard types
         else:
             # Fallback: Convert other unknown types to string
             # print(f"Warning: Converting unknown type {type(v)} to string for key '{k}'.")
             serializable_dict[k] = str(v)
    return serializable_dict

def main():
    """Main function to run the GIF-POC workflow."""
    print("--- Starting GIF-POC Workflow ---")
    print(f"Using Device: {config.DEVICE}")
    global_start_time = time.time()

    # --- 1. Data Preparation ---
    print("\n--- Step 1: Preparing Data ---")
    data_prep_start = time.time()
    depth_scaler = None # Initialize scaler

    # --- Dataset A (Primary Training) ---
    print("--> Loading/Generating Dataset A (for Primary Training)...")
    try:
        train_loader_A, val_loader_A, depth_scaler = data_utils.get_dataloaders(
            dataset_params=config.DATASET_PARAMS_A,
            dataset_filename=config.DATASET_A_FILENAME,
            scaler_path=config.SCALER_SAVE_PATH,
            cfg=config, # Pass config object
            train_split_ratio=config.TRAIN_SPLIT_RATIO,
            fit_scaler=True # Fit scaler on Dataset A
        )
        if not train_loader_A or not val_loader_A:
             raise RuntimeError("Failed to create data loaders for Dataset A.")
        print(f"Dataset A loaded: Train batches={len(train_loader_A)}, Val batches={len(val_loader_A)}")
        if depth_scaler:
             print(f"Depth scaler fitted/loaded from: {config.SCALER_SAVE_PATH}")
        else:
            # Should have been caught by the check above, but added for safety
            raise RuntimeError("Depth scaler failed to initialize for Dataset A.")
    except Exception as e:
        print(f"Fatal Error: Could not prepare Dataset A. Details: {e}")
        traceback.print_exc()
        return # Exit if primary data fails

    # --- Dataset B (Fine-tuning / Adaptation Target) ---
    print("\n--> Loading/Generating Dataset B (for Fine-tuning/Evaluation)...")
    try:
        # Use the *scaler* fitted on Dataset A, do not refit (fit_scaler=False)
        # Get the full dataset B as one loader for fine-tuning and evaluation in this POC
        loader_B, _ = data_utils.get_dataloaders( # _ ignores scaler return value here
            dataset_params=config.DATASET_PARAMS_B,
            dataset_filename=config.DATASET_B_FILENAME,
            scaler_path=config.SCALER_SAVE_PATH, # Use existing scaler path
            cfg=config, # Pass config object
            train_split_ratio=None, # Use whole dataset B for fine-tune/eval
            fit_scaler=False # DO NOT refit scaler
        )
        if not loader_B:
             raise RuntimeError("Failed to create data loader for Dataset B.")
        print(f"Dataset B loaded: Total batches={len(loader_B)}")
    except Exception as e:
        print(f"Fatal Error: Could not prepare Dataset B. Details: {e}")
        traceback.print_exc()
        return # Exit if adaptation data fails

    print(f"Data Preparation finished. Duration: {time.time() - data_prep_start:.2f}s")


    # --- 2. Model Initialization ---
    print("\n--- Step 2: Initializing Model ---")
    model_init_start = time.time()
    try:
        # Pass config explicitly to the model constructor
        model = gif_model.GIFModel(cfg=config).to(config.DEVICE)
        print(f"GIFModel initialized successfully on {config.DEVICE}")
        # Optional: Print model summary here if desired
        # print(model)
    except Exception as e:
        print(f"Fatal Error: Could not initialize GIFModel. Details: {e}")
        traceback.print_exc()
        return
    print(f"Model Initialization finished. Duration: {time.time() - model_init_start:.2f}s")


    # --- 3. Primary Training (Dataset A) ---
    print("\n--- Step 3: Starting Primary Training (Dataset A) ---")
    # Define loss functions
    # Using BCEWithLogitsLoss is numerically more stable than Sigmoid + BCELoss
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()

    # Define optimizer for primary training
    if config.OPTIMIZER.lower() == 'adam':
        optimizer_primary = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_PRIMARY)
    elif config.OPTIMIZER.lower() == 'adamw':
         optimizer_primary = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE_PRIMARY)
    else:
        print(f"Warning: Optimizer {config.OPTIMIZER} not explicitly supported. Defaulting to Adam.")
        optimizer_primary = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_PRIMARY)

    # Run primary training loop
    primary_history = train_evaluate.run_training_loop(
        model=model,
        train_loader=train_loader_A,
        val_loader=val_loader_A,
        optimizer=optimizer_primary,
        criterion_cls=criterion_cls,
        criterion_reg=criterion_reg,
        device=config.DEVICE,
        cfg=config # Pass config
    )

    # Plot primary training history
    try:
        plotting.plot_training_history(
            config.PRIMARY_HISTORY_SAVE_PATH,
            plot_filename="primary_training_history.png",
            plot_title="Primary Training History (Dataset A)",
            cfg=config # Pass config
        )
    except Exception as e:
        print(f"Warning: Could not plot primary training history: {e}")


    # --- 4. Evaluate Primary Model on Dataset B (Before Fine-tuning) ---
    print("\n--- Step 4: Evaluating Primary Model on Dataset B (Before Fine-tuning) ---")
    eval_pre_ft_start = time.time()
    results_pre_ft = {} # Initialize results dict
    try:
        # Load the *best* weights saved during primary training
        model.load_state_dict(torch.load(config.PRIMARY_MODEL_SAVE_PATH, map_location=config.DEVICE))
        print(f"Loaded best primary model weights from: {config.PRIMARY_MODEL_SAVE_PATH}")

        # Evaluate
        results_pre_ft = train_evaluate.evaluate(
            model=model,
            data_loader=loader_B, # Evaluate on the full Dataset B
            criterion_cls=criterion_cls,
            criterion_reg=criterion_reg,
            device=config.DEVICE,
            desc_prefix="Eval Primary on B",
            cfg=config # Pass config
        )
    except FileNotFoundError:
         print(f"Warning: Primary model file not found at {config.PRIMARY_MODEL_SAVE_PATH}. Skipping pre-fine-tune evaluation on B.")
    except Exception as e:
        print(f"Warning: Could not load or evaluate primary model on Dataset B: {e}. Skipping pre-fine-tune evaluation.")
        traceback.print_exc()

    # Print results only if evaluation was successful
    if results_pre_ft:
        print("\n--- Results on Dataset B (BEFORE Fine-tuning) ---")
        for key, value in results_pre_ft.items():
             if isinstance(value, (float, int, np.number)): # Check if value is numeric
                 print(f"  {key:<20}: {value:.4f}")
    else:
        print("\n--- Skipping Results on Dataset B (BEFORE Fine-tuning) due to previous error ---")

    print(f"Evaluation (Pre-FT) finished. Duration: {time.time() - eval_pre_ft_start:.2f}s")


    # --- 5. Fine-tuning (Dataset B) ---
    # Note: The model instance `model` still holds the weights loaded/trained previously.
    print("\n--- Step 5: Starting Fine-tuning (Dataset B) ---")

    # Run fine-tuning loop (handles layer freezing internally based on config)
    finetune_history = train_evaluate.run_finetuning_loop(
        model=model,
        finetune_loader=loader_B, # Fine-tune on the full Dataset B
        val_loader=loader_B,      # Validate on the full Dataset B
        criterion_cls=criterion_cls,
        criterion_reg=criterion_reg,
        device=config.DEVICE,
        cfg=config # Pass config
    )

    # Plot fine-tuning history
    try:
        plotting.plot_training_history(
            config.FINETUNE_HISTORY_SAVE_PATH,
            plot_filename="finetune_training_history.png",
            plot_title="Fine-tuning History (Dataset B)",
            cfg=config # Pass config
        )
    except Exception as e:
        print(f"Warning: Could not plot fine-tuning history: {e}")


    # --- 6. Final Evaluation on Dataset B (After Fine-tuning) ---
    print("\n--- Step 6: Evaluating Fine-tuned Model on Dataset B ---")
    eval_post_ft_start = time.time()
    results_post_ft = {} # Initialize results dict
    try:
        # Load the *best* weights saved during fine-tuning
        model.load_state_dict(torch.load(config.FINETUNED_MODEL_SAVE_PATH, map_location=config.DEVICE))
        print(f"Loaded best fine-tuned model weights from: {config.FINETUNED_MODEL_SAVE_PATH}")

        # Evaluate the fine-tuned model on Dataset B
        results_post_ft = train_evaluate.evaluate(
            model=model,
            data_loader=loader_B,
            criterion_cls=criterion_cls,
            criterion_reg=criterion_reg,
            device=config.DEVICE,
            desc_prefix="Eval FineTuned on B",
            cfg=config # Pass config
        )
    except FileNotFoundError:
        print(f"Warning: Fine-tuned model file not found at {config.FINETUNED_MODEL_SAVE_PATH}. Skipping post-fine-tune evaluation on B.")
    except Exception as e:
        print(f"Warning: Could not load or evaluate fine-tuned model on Dataset B: {e}. Skipping post-fine-tune evaluation.")
        traceback.print_exc()

    # Print results only if evaluation was successful
    if results_post_ft:
        print("\n--- Results on Dataset B (AFTER Fine-tuning) ---")
        for key, value in results_post_ft.items():
             if isinstance(value, (float, int, np.number)): # Check if numeric
                 print(f"  {key:<20}: {value:.4f}")
    else:
         print("\n--- Skipping Results on Dataset B (AFTER Fine-tuning) due to previous error ---")

    print(f"Evaluation (Post-FT) finished. Duration: {time.time() - eval_post_ft_start:.2f}s")


    # --- 7. Compare Pre- vs. Post-Fine-tuning Results ---
    # Only run comparison if both evaluations were successful
    if results_pre_ft and results_post_ft:
        print("\n--- Step 7: Adaptation Comparison (Performance on Dataset B) ---")
        print(f"{'Metric':<20} | {'Before Fine-tuning':<20} | {'After Fine-tuning':<20} | {'Change':<20}")
        print("-" * 85)
        metrics_to_compare = ['loss', 'loss_cls', 'loss_reg', 'loss_phys', 'accuracy', 'balanced_accuracy', 'auc', 'f1', 'precision', 'recall', 'mse', 'firing_rate']
        for metric in metrics_to_compare:
             before = results_pre_ft.get(metric, float('nan'))
             after = results_post_ft.get(metric, float('nan'))

             if isinstance(before, (int, float, np.number)) and isinstance(after, (int, float, np.number)) and not (np.isnan(before) or np.isnan(after)):
                 change = after - before
                 if 'loss' in metric or 'mse' in metric:
                     print(f"{metric:<20} | {before:<20.6f} | {after:<20.6f} | {change:<+20.6f}")
                 else:
                     print(f"{metric:<20} | {before:<20.4f} | {after:<20.4f} | {change:<+20.4f}")
             else:
                 print(f"{metric:<20} | {str(before):<20} | {str(after):<20} | {'N/A':<20}")
        print("-" * 85)
    else:
         print("\n--- Step 7: Skipping Adaptation Comparison due to missing evaluation results ---")


    # --- 8. Save Final Evaluation Summary ---
    print("\n--- Step 8: Saving Final Evaluation Logs ---")
    # Optionally, re-evaluate primary model on A's validation set
    results_primary_on_A_val = {}
    try:
         model.load_state_dict(torch.load(config.PRIMARY_MODEL_SAVE_PATH, map_location=config.DEVICE))
         results_primary_on_A_val = train_evaluate.evaluate(model, val_loader_A, criterion_cls, criterion_reg, config.DEVICE, desc_prefix="Final Eval Primary on A Val", cfg=config)
         print("-> Primary model evaluated on Dataset A validation set.")
    except Exception as e:
         print(f"Warning: Could not load/eval primary model on A val set: {e}")

    # Optionally, re-evaluate fine-tuned model on A's validation set (to check for catastrophic forgetting)
    results_finetuned_on_A_val = {}
    try:
         model.load_state_dict(torch.load(config.FINETUNED_MODEL_SAVE_PATH, map_location=config.DEVICE))
         results_finetuned_on_A_val = train_evaluate.evaluate(model, val_loader_A, criterion_cls, criterion_reg, config.DEVICE, desc_prefix="Final Eval FineTuned on A Val", cfg=config)
         print("-> Fine-tuned model evaluated on Dataset A validation set.")
    except Exception as e:
         print(f"Warning: Could not load/eval fine-tuned model on A val set: {e}")

    # Combine all evaluation results, ensuring keys exist even if evaluation failed
    all_eval_results = {
        "Primary_Model_on_Dataset_A_Validation": results_primary_on_A_val if results_primary_on_A_val else {},
        "Finetuned_Model_on_Dataset_A_Validation": results_finetuned_on_A_val if results_finetuned_on_A_val else {}, # Forgetting check
        "Primary_Model_on_Dataset_B": results_pre_ft if results_pre_ft else {},   # Before adaptation
        "Finetuned_Model_on_Dataset_B": results_post_ft if results_post_ft else {} # After adaptation
    }

    # --- Serialize results to JSON (using the helper function) ---
    try:
        serializable_results_to_save = {k: make_serializable(v) for k, v in all_eval_results.items()}
        with open(config.EVAL_RESULTS_SAVE_PATH, 'w') as f:
            json.dump(serializable_results_to_save, f, indent=4, sort_keys=True)
        print(f"Combined evaluation results saved to {config.EVAL_RESULTS_SAVE_PATH}")
    except Exception as e:
        # Catch errors during the make_serializable call or file writing
        print(f"Error saving final evaluation results: {e}")
        traceback.print_exc()


    # --- 9. Final Plots ---
    # Only plot if post-fine-tune evaluation was successful
    if results_post_ft:
        print("\n--- Step 9: Generating Final Evaluation Plots ---")
        try:
            plotting.plot_regression_results(
                 actuals=results_post_ft.get("actuals_reg"),
                 predictions=results_post_ft.get("predictions_reg"),
                 plot_title="Dataset B Regression (After Fine-tuning)",
                 filename="dataset_B_regression_post_finetune.png",
                 scaler=depth_scaler, # Pass the scaler to un-normalize depths
                 cfg=config # Pass config
            )
        except Exception as e:
            print(f"Warning: Could not plot post-finetune regression results: {e}")

        try:
            plotting.plot_classification_results(
                 actuals=results_post_ft.get("actuals_cls"),
                 predictions_binary=results_post_ft.get("predictions_cls_binary"),
                 plot_title="Dataset B Confusion Matrix (After Fine-tuning)",
                 filename="dataset_B_confusion_matrix_post_finetune.png",
                 cfg=config # Pass config
            )
        except Exception as e:
            print(f"Warning: Could not plot post-finetune classification results: {e}")
    else:
        print("\n--- Step 9: Skipping Final Evaluation Plots due to missing fine-tuning results ---")

    # --- 10. XAI Saliency Map ---
    # Only run if fine-tuned model exists and loader_B is valid
    if os.path.exists(config.FINETUNED_MODEL_SAVE_PATH) and loader_B:
        print("\n--- Step 10: Generating XAI Saliency Map (Fine-tuned Model) ---")
        xai_start_time = time.time()
        try:
             # Load the best fine-tuned model state again (safer)
             model.load_state_dict(torch.load(config.FINETUNED_MODEL_SAVE_PATH, map_location=config.DEVICE))
             print(f"Loaded fine-tuned model for XAI from: {config.FINETUNED_MODEL_SAVE_PATH}")

             # Get a sample from Dataset B
             dataset_B = loader_B.dataset # Access the underlying Dataset object
             if len(dataset_B) > 0:
                # Select a random sample index
                sample_idx = np.random.randint(0, len(dataset_B))
                # Get the sample (input tensor, label tensor, depth tensor)
                sample_input_tensor, sample_label, _ = dataset_B[sample_idx]

                # --- Prepare Sample for Model Input ---
                # Add batch dimension [1, C, L] or [1, L, F] and move to device
                sample_input_batch = sample_input_tensor.unsqueeze(0).to(config.DEVICE)

                print(f"Generating saliency map for Dataset B sample index: {sample_idx} (Label: {int(sample_label.item())})...")

                # Calculate saliency scores (input gradients)
                saliency_scores = xai_utils.calculate_input_gradients(
                    model=model,
                    input_data=sample_input_batch,
                    target_output='cls', # Or 'reg' if desired
                    cfg=config # Pass config
                )

                if saliency_scores is not None:
                    # Saliency scores shape should match input sample after batch removal
                    # For plotting, we usually want a 1D array representing importance over time steps.
                    if saliency_scores.ndim > 1:
                         # Average over channels/features if needed (simple approach)
                         saliency_scores_plot = saliency_scores.mean(axis=0).flatten()
                    else:
                         saliency_scores_plot = saliency_scores.flatten()

                    # Original input for plotting also needs to be 1D
                    original_input_plot = sample_input_tensor.squeeze().cpu().numpy()
                    if original_input_plot.ndim > 1:
                         original_input_plot = original_input_plot.mean(axis=0).flatten() # Average if needed

                    # Plot the saliency map
                    plotting.plot_saliency_map(
                        light_curve=original_input_plot,
                        saliency_scores=saliency_scores_plot,
                        filename=f"saliency_map_finetuned_sample_{sample_idx}.png",
                        title=f"XAI Saliency on Dataset B Sample {sample_idx} (Fine-tuned Model)",
                        cfg=config # Pass config
                    )
                else:
                    print("Saliency score calculation failed for the fine-tuned model.")
             else:
                print("Dataset B is empty, cannot generate saliency map.")
        except FileNotFoundError:
            print(f"Error: Fine-tuned model file not found at {config.FINETUNED_MODEL_SAVE_PATH}. Skipping XAI.")
        except Exception as e:
            print(f"An error occurred during XAI saliency map generation: {e}")
            traceback.print_exc()
        print(f"XAI generation finished. Duration: {time.time() - xai_start_time:.2f}s")
    else:
        print("\n--- Step 10: Skipping XAI due to missing fine-tuned model or data loader ---")


    # --- Workflow End ---
    global_end_time = time.time()
    print("\n" + "=" * 30)
    print("--- GIF-POC Workflow Finished ---")
    print(f"Total execution time: {global_end_time - global_start_time:.2f} seconds")
    print("Results, models, logs, and plots saved in the 'results/' directory.")
    print("=" * 30)

# Entry point for running the script directly
if __name__ == "__main__":
    # This ensures the script runs when executed as 'python main.py'
    main()
