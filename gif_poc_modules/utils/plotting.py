# POC/gif_poc_modules/utils/plotting.py

import matplotlib.pyplot as plt
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings

# Set a default Matplotlib style, trying a few common ones
# Filter specific UserWarning related to style not being found
# CORRECTED LINE: Escaped the parenthesis in the regex pattern
warnings.filterwarnings("ignore", category=UserWarning, message=r".*style sheets were found but not.*\)")
try:
    # Try a modern seaborn style first
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        # Fallback to ggplot if seaborn-v0_8 isn't available
        plt.style.use('ggplot')
    except OSError:
        # Fallback to default if ggplot also fails
        print("Warning: Using default Matplotlib style as 'seaborn-v0_8-darkgrid' and 'ggplot' were not found.")
        pass # Use default Matplotlib style


def _plot_metric(ax, epochs, data, label, style, title, y_label, y_lim=None, min_len=1, alpha=1.0):
    """Helper function to plot a single metric if data is valid."""
    if data and len(data) == len(epochs) and len(data) >= min_len:
        # Ensure data is numeric or None before plotting
        numeric_data = [d if isinstance(d, (int, float, np.number)) else np.nan for d in data]
        if not all(np.isnan(numeric_data)): # Check if there's at least one valid number
            ax.plot(epochs, numeric_data, style, label=label, markersize=4, alpha=alpha) # Smaller markers
            ax.set_title(title, fontsize=12) # Consistent font size
            ax.set_xlabel('Epochs', fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            if y_lim:
                ax.set_ylim(y_lim)
            ax.legend(fontsize=8) # Smaller legend font
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.tick_params(axis='both', which='major', labelsize=9) # Axis tick label size
            return True
    # If data is missing or not plottable
    ax.set_title(f'{title}\n(Data Missing/Incorrect Length)', fontsize=12)
    ax.text(0.5, 0.5, 'Data Missing', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='grey')
    ax.set_xlabel('Epochs', fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=9)
    return False

def plot_training_history(history_path, plot_filename="training_history.png", plot_title="Training History", cfg=None):
    """
    Plots training and validation history from a JSON log file.

    Args:
        history_path (str): Path to the JSON file containing training history.
        plot_filename (str): Name of the file to save the plot.
        plot_title (str): Title for the overall plot.
        cfg (module): The config module (needed for PLOT_SAVE_DIR).
    """
    if cfg is None:
        print("Error: Config object (cfg) required for plot_training_history.")
        return
    if not os.path.exists(cfg.PLOT_SAVE_DIR):
        try:
            os.makedirs(cfg.PLOT_SAVE_DIR)
        except OSError as e:
             print(f"Error creating plot directory {cfg.PLOT_SAVE_DIR}: {e}")
             return # Cannot save plot

    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        print(f"Loaded history from: {history_path}")
    except FileNotFoundError:
        print(f"Error: History file not found at {history_path}")
        return
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON from {history_path}. File might be corrupted or empty.")
         return
    except Exception as e:
        print(f"Error loading history file {history_path}: {e}")
        return

    # Determine number of epochs based on train_loss, assuming it's always present
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    if not epochs:
        print(f"Error: No 'train_loss' data found or data is empty in {history_path}.")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 14)) # Slightly smaller figure
    fig.suptitle(plot_title, fontsize=16, y=0.99) # Adjust title position slightly

    # --- Row 1 ---
    # Plot Combined Loss
    loss_axes = axes[0, 0]
    _plot_metric(loss_axes, epochs, history.get('train_loss'), 'Train Loss', 'bo-', 'Combined Loss vs. Epochs', 'Loss', y_lim=[0, None])
    _plot_metric(loss_axes, epochs, history.get('val_loss'), 'Val Loss', 'ro-', 'Combined Loss vs. Epochs', 'Loss', y_lim=[0, None])
    loss_axes.legend(fontsize=8) # Ensure legend appears even if only one line is plotted

    # Plot Classification Metrics (Accuracy & AUC)
    cls_axes = axes[0, 1]
    acc_plotted = _plot_metric(cls_axes, epochs, history.get('val_accuracy'), 'Val Accuracy', 'go-', 'Val Classification Metrics', 'Metric Value', y_lim=[0, 1.05])
    auc_plotted = _plot_metric(cls_axes, epochs, history.get('val_auc'), 'Val AUC', 'mo-', 'Val Classification Metrics', 'Metric Value', y_lim=[0, 1.05])
    if acc_plotted or auc_plotted:
        cls_axes.legend(fontsize=8)
    else: # If neither plotted, add missing text
         cls_axes.set_title('Val Classification Metrics\n(Data Missing/Incorrect Length)', fontsize=12)
         cls_axes.text(0.5, 0.5, 'Data Missing', ha='center', va='center', transform=cls_axes.transAxes, fontsize=10, color='grey')


    # --- Row 2 ---
    # Plot Regression Metric (MSE)
    _plot_metric(axes[1, 0], epochs, history.get('val_mse'), 'Val MSE', 'yo-', 'Val Regression MSE', 'MSE', y_lim=[0, None])

    # Plot SNN Firing Rate
    _plot_metric(axes[1, 1], epochs, history.get('val_firing_rate'), 'Val Firing Rate', 'co-', 'SNN Avg Firing Rate', 'Avg Rate', y_lim=[0, None])

    # --- Row 3 ---
    # Plot Component Losses
    ax = axes[2, 0]; plc = False
    if _plot_metric(ax, epochs, history.get('val_loss_cls'), 'Val Loss CLS', 'b.-', 'Val Loss Components', 'Loss', alpha=0.7): plc=True
    if _plot_metric(ax, epochs, history.get('val_loss_reg'), 'Val Loss REG', 'g.-', 'Val Loss Components', 'Loss', alpha=0.7): plc=True
    if _plot_metric(ax, epochs, history.get('val_loss_phys'), 'Val Loss PHYS', 'r.-', 'Val Loss Components', 'Loss', alpha=0.7): plc=True
    if plc:
        ax.legend(fontsize=8)
    else: # If none plotted, add missing text
        ax.set_title('Val Loss Components\n(Data Missing/Incorrect Length)', fontsize=12)
        ax.text(0.5, 0.5, 'Data Missing', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='grey')

    # Plot Precision/Recall/F1
    ax = axes[2, 1]; pm = False
    if _plot_metric(ax, epochs, history.get('val_precision'), 'Val Precision', 'b.-', 'Val Precision/Recall/F1', 'Score', y_lim=[0, 1.05], alpha=0.7): pm=True
    if _plot_metric(ax, epochs, history.get('val_recall'), 'Val Recall', 'g.-', 'Val Precision/Recall/F1', 'Score', y_lim=[0, 1.05], alpha=0.7): pm=True
    if _plot_metric(ax, epochs, history.get('val_f1'), 'Val F1-Score', 'r.-', 'Val Precision/Recall/F1', 'Score', y_lim=[0, 1.05], alpha=0.7): pm=True
    if pm:
         ax.legend(fontsize=8)
    else: # If none plotted, add missing text
         ax.set_title('Val P/R/F1\n(Data Missing/Incorrect Length)', fontsize=12)
         ax.text(0.5, 0.5, 'Data Missing', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='grey')

    # --- Save Plot ---
    plt.tight_layout(rect=[0, 0.02, 1, 0.97]) # Adjust layout to prevent title overlap
    save_path = os.path.join(cfg.PLOT_SAVE_DIR, plot_filename)
    try:
        plt.savefig(save_path, dpi=150) # Increase DPI slightly for better quality
        print(f"'{plot_title}' plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving plot {save_path}: {e}")
    finally:
        plt.close(fig) # Close the figure to free memory


def plot_regression_results(actuals, predictions, cfg, plot_title="Regression Results", filename="regression_results.png", scaler=None):
    """
    Creates a scatter plot comparing actual vs. predicted values for regression.
    Optionally inverse transforms values if a scaler is provided.

    Args:
        actuals (np.ndarray): Array of actual target values.
        predictions (np.ndarray): Array of predicted target values.
        cfg (module): Config module (for PLOT_SAVE_DIR).
        plot_title (str): Title for the plot.
        filename (str): Filename for saving the plot.
        scaler (sklearn.preprocessing.MinMaxScaler, optional): Scaler used for the target variable.
    """
    if cfg is None:
        print("Error: Config object (cfg) required for plot_regression_results.")
        return
    if actuals is None or predictions is None or not isinstance(actuals, np.ndarray) or not isinstance(predictions, np.ndarray) or actuals.size == 0 or predictions.size == 0 or len(actuals) != len(predictions):
        print(f"Warning: Invalid or empty data provided for regression plot '{filename}'. Skipping.")
        return

    actuals_plot = np.array(actuals).flatten()
    predictions_plot = np.array(predictions).flatten()
    xlabel = "Actual (Scaled)"
    ylabel = "Predicted (Scaled)"

    # Inverse transform if scaler is provided
    if scaler is not None:
        try:
            actuals_plot = scaler.inverse_transform(actuals_plot.reshape(-1, 1)).flatten()
            predictions_plot = scaler.inverse_transform(predictions_plot.reshape(-1, 1)).flatten()
            xlabel = "Actual Depth"
            ylabel = "Predicted Depth"
        except ValueError as e:
             print(f"Warning: Could not inverse transform depths for plotting: {e}. Plotting scaled values.")
        except Exception as e: # Catch other potential scaler errors
             print(f"Warning: Error during inverse transform: {e}. Plotting scaled values.")


    plt.figure(figsize=(8, 8))
    plt.scatter(actuals_plot, predictions_plot, alpha=0.4, label='Predictions', s=15, edgecolors='k', linewidths=0.5) # Add edges for visibility

    # Determine plot limits dynamically
    min_val = min(np.min(actuals_plot), np.min(predictions_plot))
    max_val = max(np.max(actuals_plot), np.max(predictions_plot))
    # Handle potential case where min_val == max_val
    range_val = max_val - min_val if max_val > min_val else 1.0
    # Add a small buffer to limits
    plot_min = min_val - 0.05 * range_val
    plot_max = max_val + 0.05 * range_val

    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', label='Ideal (y=x)', linewidth=2)
    plt.title(plot_title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)
    plt.gca().set_aspect('equal', adjustable='box') # Make axes equal
    plt.tick_params(axis='both', which='major', labelsize=9)

    # --- Save Plot ---
    if not os.path.exists(cfg.PLOT_SAVE_DIR): os.makedirs(cfg.PLOT_SAVE_DIR)
    save_path = os.path.join(cfg.PLOT_SAVE_DIR, filename)
    try:
        plt.savefig(save_path, dpi=150)
        print(f"Regression plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving regression plot {save_path}: {e}")
    finally:
        plt.close()


def plot_classification_results(actuals, predictions_binary, cfg, plot_title="Classification Results", filename="confusion_matrix.png"):
    """
    Plots a confusion matrix for binary classification results.

    Args:
        actuals (np.ndarray): Array of actual binary labels (0 or 1).
        predictions_binary (np.ndarray): Array of predicted binary labels (0 or 1).
        cfg (module): Config module (for PLOT_SAVE_DIR).
        plot_title (str): Title for the plot.
        filename (str): Filename for saving the plot.
    """
    if cfg is None:
        print("Error: Config object (cfg) required for plot_classification_results.")
        return
    if actuals is None or predictions_binary is None or not isinstance(actuals, np.ndarray) or not isinstance(predictions_binary, np.ndarray) or actuals.size == 0 or predictions_binary.size == 0 or len(actuals) != len(predictions_binary):
        print(f"Warning: Invalid or empty data provided for confusion matrix plot '{filename}'. Skipping.")
        return

    actuals_flat = np.array(actuals).flatten().astype(int)
    predictions_flat = np.array(predictions_binary).flatten().astype(int)

    # Get unique labels present in the data
    unique_labels = np.unique(np.concatenate((actuals_flat, predictions_flat)))
    display_labels = ['No Transit', 'Transit'] # Default labels

    # Adjust display labels if only one class is present (edge case)
    if len(unique_labels) == 1:
         if unique_labels[0] == 0:
             display_labels = ['No Transit']
         elif unique_labels[0] == 1:
             display_labels = ['Transit']
         else:
             display_labels = [str(unique_labels[0])] # Fallback

    try:
        cm = confusion_matrix(actuals_flat, predictions_flat, labels=unique_labels) # Use unique labels found
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

        fig, ax = plt.subplots(figsize=(7, 7))
        disp.plot(ax=ax, cmap='Blues', values_format='d') # Use 'd' for integer format
        ax.set_title(plot_title, fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)

        plt.tight_layout()

        # --- Save Plot ---
        if not os.path.exists(cfg.PLOT_SAVE_DIR): os.makedirs(cfg.PLOT_SAVE_DIR)
        save_path = os.path.join(cfg.PLOT_SAVE_DIR, filename)
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")

    except ValueError as e:
         # Handles case where labels might only contain one class after filtering bad batches etc.
         print(f"Error plotting confusion matrix for '{filename}': {e}. Check if labels contain both classes.")
    except Exception as e:
        print(f"An unexpected error occurred plotting confusion matrix '{filename}': {e}")
    finally:
        # Ensure plot is closed even if errors occurred
        if 'fig' in locals(): # Check if fig was created
             plt.close(fig)


def plot_saliency_map(light_curve, saliency_scores, cfg, filename="saliency_map.png", title="XAI Saliency Map"):
    """
    Plots the input light curve overlaid with its calculated saliency map (input gradients).

    Args:
        light_curve (np.ndarray): 1D numpy array of the input light curve.
        saliency_scores (np.ndarray): 1D numpy array of corresponding saliency scores (e.g., abs gradients).
        cfg (module): Config module (for PLOT_SAVE_DIR).
        filename (str): Filename for saving the plot.
        title (str): Title for the plot.
    """
    if cfg is None:
        print("Error: Config object (cfg) required for plot_saliency_map.")
        return
    if light_curve is None or saliency_scores is None:
        print("Error plotting saliency: Input light_curve or saliency_scores is None.")
        return

    lc = np.array(light_curve).squeeze()
    ss = np.array(saliency_scores).squeeze()

    if lc.ndim != 1 or ss.ndim != 1 or len(lc) != len(ss):
        print(f"Error plotting saliency: Invalid shapes/lengths. LC:{lc.shape}, SS:{ss.shape}")
        return
    if len(lc) == 0:
        print("Error plotting saliency: Input arrays are empty.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.suptitle(title, fontsize=14)
    time_steps = np.arange(len(lc))

    # Plot Light Curve on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Normalized Flux (Input)', color=color, fontsize=12)
    ax1.plot(time_steps, lc, color=color, label='Light Curve', alpha=0.8, lw=1.5)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(True, axis='x', linestyle='--', alpha=0.6)
    # Dynamic y-lim for light curve
    lc_min, lc_max = np.min(lc), np.max(lc)
    lc_range = lc_max - lc_min if lc_max > lc_min else 0.1
    ax1.set_ylim(lc_min - 0.05 * lc_range, lc_max + 0.05 * lc_range)


    # Plot Saliency Scores on secondary y-axis
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Saliency (|Input Gradient|)', color=color, fontsize=12)
    # Use fill_between for better visualization
    ax2.fill_between(time_steps, 0, ss, color=color, alpha=0.4, label='Saliency Score')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=10)
    ax2.set_ylim(bottom=0) # Saliency (abs gradient) should be non-negative

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    # --- Save Plot ---
    if not os.path.exists(cfg.PLOT_SAVE_DIR): os.makedirs(cfg.PLOT_SAVE_DIR)
    save_path = os.path.join(cfg.PLOT_SAVE_DIR, filename)
    try:
        plt.savefig(save_path, dpi=150)
        print(f"Saliency map plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving saliency plot {save_path}: {e}")
    finally:
        plt.close(fig)
