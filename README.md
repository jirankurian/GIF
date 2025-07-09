# GIF/DU Proof-of-Concept (POC)

This repository contains a Proof-of-Concept (POC) implementation demonstrating key ideas from the **General Intelligence Framework (GIF)** and **Deep Understanding (DU)** research.

## Purpose

The primary goal of this POC is *not* to achieve state-of-the-art performance on a specific task, but rather to illustrate two core principles of the proposed GIF/DU architecture:

1.  **Modularity:** Showcasing how different components (Encoder, SNN Core, Decoders) can be assembled into a flexible architecture.
2.  **Adaptability (Proxy via Fine-tuning):** Demonstrating the system's potential to adapt to changes in data characteristics. This is simulated by:
    * Training the model on a baseline synthetic dataset (Dataset A).
    * Fine-tuning parts of the trained model (primarily decoders, optionally the SNN core) on a second synthetic dataset (Dataset B) with different properties (e.g., noise level, signal shape).
    * Comparing performance on Dataset B before and after fine-tuning to show adaptation.

**Note:** This POC uses supervised fine-tuning as a proxy for the more complex Real-Time Learning (RTL) envisioned in the full GIF/DU framework. The synthetic data mimics simple time-series signals (like exoplanet transits) for demonstration purposes.

## Architecture Overview

The model implemented (`gif_model.py`) follows the GIF structure:

1.  **Encoder:** An initial feature extractor. Can be configured in `config.py` (`ANN_TYPE`) to use:
    * `CNNEncoder`: A 1D Convolutional Neural Network.
    * `LSTMEncoder`: A Long Short-Term Memory network.
2.  **Projection Layer:** A standard Linear layer (`ann_to_snn_input`) that maps the encoder's output features to the input size required by the SNN core.
3.  **SNN Core (DU Proxy):** (`core_du.py`) A Spiking Neural Network implemented using `snnTorch` Leaky Integrate-and-Fire (LIF) neurons. It processes the projected features over a configured number of time steps (`NUM_TIME_STEPS`). This module represents the Deep Understanding core.
4.  **Decoders:** Separate linear heads (`decoder.py`) take the aggregated output from the SNN core to produce predictions for:
    * **Classification:** Binary output (e.g., signal vs. no signal).
    * **Regression:** Continuous output (e.g., signal depth).

## Directory Structure
POC/
├── data/                     # Stores generated .pt datasets
├── results/                  # Stores all outputs (logs, plots, models)
│   ├── logs/
│   ├── plots/
│   └── saved_models/
├── gif_poc_modules/          # Source code package
│   ├── init.py
│   ├── architecture/         # Model components (encoder, core_du, etc.)
│   ├── data_utils.py         # Data generation and loading
│   ├── training/             # Training/evaluation loops
│   └── utils/                # Plotting, XAI helpers
├── config.py                 # Central configuration for parameters
├── main.py                   # Main script to run the workflow
├── requirements.txt          # Python dependencies
└── README.md                 # This file

## Setup

1.  **Create a Python Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration (`config.py`)

Key parameters can be adjusted in `config.py`:

* `ANN_TYPE`: Choose 'CNN' or 'LSTM' for the encoder.
* `SNN_HIDDEN_SIZE`, `NUM_TIME_STEPS`, `SPIKE_BETA`, etc.: Control SNN behavior.
* `DATASET_PARAMS_A`, `DATASET_PARAMS_B`: Modify synthetic data generation (noise, shapes, sizes).
* `NUM_EPOCHS_PRIMARY`, `NUM_EPOCHS_FINETUNE`: Set training durations.
* `FINETUNE_SNN_CORE`: Set to `True` to allow the SNN core weights to be updated during fine-tuning (with `LEARNING_RATE_FINETUNE_SNN`). Default (`False`) only fine-tunes the projection layer and decoders.
* `DEVICE`: Automatically detects 'mps', 'cuda', or 'cpu'.

## Running the POC

1.  Navigate to the `POC` directory in your terminal.
2.  Ensure your virtual environment is activated.
3.  Run the main script:
    ```bash
    python main.py
    ```

**Execution Flow:**

* Checks for/generates synthetic datasets (Dataset A & B) in `data/`.
* Initializes the GIF model based on `config.py`.
* Performs primary training on Dataset A, saving the best model (based on validation loss) to `results/saved_models/`. Logs and plots are saved to `results/logs/` and `results/plots/`.
* Evaluates the primary model on Dataset B (performance *before* adaptation).
* Performs fine-tuning using Dataset B (freezing layers based on `FINETUNE_SNN_CORE` setting), saving the best fine-tuned model.
* Evaluates the fine-tuned model on Dataset B (performance *after* adaptation).
* Prints a comparison table showing performance changes on Dataset B due to fine-tuning.
* Saves combined evaluation metrics to `results/logs/evaluation_results.json`.
* Generates final evaluation plots (regression, confusion matrix for Dataset B post-fine-tuning).
* Generates an example XAI saliency map for a random sample from Dataset B using the fine-tuned model.

## Interpreting Results

* **Adaptation Comparison Table:** This is key output in the console. Look for improvements in metrics (lower loss/MSE, higher Accuracy/AUC/F1) on Dataset B after fine-tuning compared to before. This demonstrates the adaptation.
* **Plots (`results/plots/`):**
    * `primary_training_history.png` / `finetune_training_history.png`: Show how loss and metrics evolved during training/fine-tuning. Check for convergence and reasonable SNN firing rates.
    * `dataset_B_*.png`: Visualize the final performance of the fine-tuned model on the target dataset (Dataset B).
    * `saliency_map_*.png`: Shows which parts of the input time series the fine-tuned model considered most important for its classification decision on a sample input.
* **Logs (`results/logs/`):** JSON files contain detailed metrics for each epoch and final evaluations.
* **Models (`results/saved_models/`):** Saved PyTorch state dictionaries (`.pth`) for the best primary and fine-tuned models, plus the data scaler (`.pkl`).
