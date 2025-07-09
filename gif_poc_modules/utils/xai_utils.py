# POC/gif_poc_modules/utils/xai_utils.py

import torch
import numpy as np
import traceback # For printing stack trace on error

def calculate_input_gradients(model, input_data, cfg, target_idx=0, target_output='cls'):
    """
    Calculates the gradient of a target output with respect to the input data.
    This is a basic form of saliency mapping.

    Args:
        model (torch.nn.Module): The trained model.
        input_data (torch.Tensor): A single input sample tensor with batch dimension (e.g., [1, C, L] or [1, L, F]).
        cfg (module): The config module (needed for device).
        target_idx (int): Index of the output neuron if the target output is multi-dimensional (default: 0).
        target_output (str): Which model output to compute gradients for ('cls' or 'reg').

    Returns:
        np.ndarray | None: Absolute values of the gradients w.r.t. input, squeezed to remove batch dim.
                           Returns None on error.
    """
    if cfg is None:
        print("Error: Config object (cfg) is required for calculate_input_gradients.")
        return None
    if not isinstance(model, torch.nn.Module):
        print("Error: Invalid model passed to calculate_input_gradients.")
        return None
    if not isinstance(input_data, torch.Tensor):
        print("Error: input_data must be a PyTorch Tensor.")
        return None
    # Ensure input has a batch dimension of 1
    if input_data.shape[0] != 1:
        print(f"Error: calculate_input_gradients expects batch size 1, but got shape {input_data.shape}.")
        return None

    model.eval() # Ensure model is in evaluation mode

    # --- Prepare Input ---
    # Clone, move to device, and enable gradient calculation for the input
    input_data_grad = input_data.clone().detach().to(cfg.DEVICE).requires_grad_(True)
    # Zero out any previous gradients attached to the input tensor
    if input_data_grad.grad is not None:
        input_data_grad.grad.zero_()

    # --- Forward Pass ---
    try:
        outputs = model(input_data_grad)
        # Expecting (cls_logits, reg_preds, total_spikes)
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            print("Error: Model output format unexpected in XAI forward pass. Expected tuple (cls, reg, ...).")
            return None
        cls_logits, reg_preds, _ = outputs # Unpack outputs
    except Exception as e:
        print(f"Error during XAI forward pass: {e}")
        traceback.print_exc() # Print stack trace for debugging
        return None

    # --- Select Target Output ---
    target = None
    if target_output == 'cls':
        if cls_logits is None:
            print("Error: Classification output (cls_logits) is None.")
            return None
        if cls_logits.numel() == 0 or target_idx >= cls_logits.shape[-1]:
             print(f"Error: CLS output invalid or target_idx out of bounds. Shape={cls_logits.shape}, target_idx={target_idx}")
             return None
        # Select the specific logit output (before sigmoid)
        target = cls_logits[:, target_idx] # Shape: [1] if output_size is 1
    elif target_output == 'reg':
        if reg_preds is None:
             print("Error: Regression output (reg_preds) is None.")
             return None
        if reg_preds.numel() == 0 or target_idx >= reg_preds.shape[-1]:
             print(f"Error: REG output invalid or target_idx out of bounds. Shape={reg_preds.shape}, target_idx={target_idx}")
             return None
        target = reg_preds[:, target_idx] # Shape: [1]
    else:
        print(f"Error: Invalid target_output specified: '{target_output}'. Use 'cls' or 'reg'.")
        return None

    # Ensure target is a scalar for backward pass
    if target is None:
         print(f"Error: Target for backward pass is None (target_output='{target_output}').")
         return None
    if target.numel() > 1:
        # If target is multi-dimensional (shouldn't be for cls/reg with size 1), sum it.
        print(f"Warning: Target output has more than one element ({target.numel()}). Summing for gradient calculation.")
        target = target.sum()
    elif target.numel() == 0:
        print("Error: Target output has zero elements.")
        return None

    # --- Backward Pass ---
    try:
        # Calculate gradients of the target scalar w.r.t. inputs requiring gradients
        target.backward()
    except RuntimeError as e:
         print(f"Error during XAI backward pass: {e}. Gradient might not be computable for the selected target.")
         traceback.print_exc()
         return None
    except Exception as e:
         print(f"Unexpected error during XAI backward pass: {e}")
         traceback.print_exc()
         return None


    # --- Extract and Process Gradients ---
    gradients = input_data_grad.grad
    if gradients is None:
        print("Error: Gradients w.r.t. input are None after backward pass.")
        return None

    # Take absolute value of gradients for saliency
    saliency_scores = torch.abs(gradients)

    # Remove the batch dimension (shape [1, C, L] -> [C, L] or [1, L, F] -> [L, F])
    saliency_scores = saliency_scores.squeeze(0)

    # Move to CPU and convert to numpy array
    saliency_scores_np = saliency_scores.detach().cpu().numpy()

    return saliency_scores_np
