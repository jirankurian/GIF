"""
Medical Diagnostics Potentiation Experiment - Master Script
===========================================================

This script implements the definitive System Potentiation experiment using the GIF framework.
It orchestrates three distinct experimental runs to rigorously test the hypothesis that
diverse prior experience improves the fundamental learning mechanism itself, not just
knowledge transfer.

Experimental Design:
===================

**Scientific Question**: Does diverse prior experience improve the learning mechanism
itself, or just provide transferable knowledge?

**Hypothesis**: System Potentiation - Prior experience in different domains improves
the fundamental learning capacity of neural systems.

**Method**: Weight-Reset Protocol - Load pre-trained model, immediately reset all
synaptic weights, then train on new task. Any improvement vs naive model proves
potentiation rather than knowledge transfer.

Three Experimental Arms:
=======================

1. **SOTA Baseline**: Standard CNN for performance comparison and validation
2. **Naive GIF-DU**: Fresh random weights (control group)
3. **Pre-Exposed GIF-DU**: Weight-reset protocol (experimental group)

Key Innovation:
==============

The weight-reset protocol is the critical scientific control that distinguishes this
from simple transfer learning. By wiping specific knowledge but preserving structural
improvements, we can isolate and measure true system potentiation effects.

Expected Outcome:
================

If system potentiation exists, the pre-exposed model should learn the medical task
faster than the naive model, despite having identical starting weights. This would
prove that prior experience improved the learning mechanism itself.

Scientific Impact:
=================

This experiment provides the first rigorous test of system potentiation in artificial
neural networks, with implications for AGI development and continual learning research.
"""

import os
import sys
import logging
import time
import json
import pickle
from typing import Dict, Any, List, Tuple, Optional
import warnings

# Scientific computing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# GIF Framework components
from gif_framework.orchestrator import GIF
from gif_framework.core.du_core import DU_Core_V1
from gif_framework.core.rtl_mechanisms import STDP_Rule
from gif_framework.core.memory_systems import EpisodicMemory
from gif_framework.training.trainer import Continual_Trainer

# Medical domain components
from applications.poc_medical.encoders.ecg_encoder import ECG_Encoder
from applications.poc_medical.decoders.arrhythmia_decoder import Arrhythmia_Decoder
from applications.poc_medical.config_med import (
    get_config, create_output_directories, get_device, set_reproducibility_seeds,
    SOTA_BASELINE_DIR, NAIVE_RUN_DIR, PRE_EXPOSED_RUN_DIR, ANALYSIS_DIR,
    PRE_TRAINED_EXO_MODEL_PATH
)

# Data generation and simulation
from data_generators.ecg_generator import RealisticECGGenerator
from simulators.neuromorphic_sim import NeuromorphicSimulator


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Set up comprehensive logging for the experiment."""

    # Create main log file
    log_file = os.path.join(ANALYSIS_DIR, "potentiation_experiment.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("MEDICAL DIAGNOSTICS POTENTIATION EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Experiment: {config['metadata']['experiment_name']}")
    logger.info(f"Hypothesis: {config['metadata']['hypothesis']}")
    logger.info(f"Critical Protocol: {config['metadata']['critical_protocol']}")

    return logger


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_ecg_datasets(config: Dict[str, Any], logger: logging.Logger) -> Tuple[List, List, List]:
    """
    Generate comprehensive ECG datasets for all experimental arms.

    Args:
        config: Experiment configuration
        logger: Logger instance

    Returns:
        Tuple of (training_data, validation_data, test_data) lists
    """
    logger.info("Generating synthetic ECG datasets for potentiation experiment...")

    data_config = config["data"]

    # Initialize ECG generator with clinical realism
    generator = RealisticECGGenerator(seed=data_config["random_seed"])

    # Generate training data with balanced arrhythmia classes
    logger.info(f"Generating {data_config['num_training_samples']} training samples...")
    training_data = []

    arrhythmia_classes = data_config["arrhythmia_classes"]
    samples_per_class = data_config["num_training_samples"] // len(arrhythmia_classes)

    for class_idx, arrhythmia_class in enumerate(arrhythmia_classes):
        for i in range(samples_per_class):
            # Generate ECG with specific arrhythmia characteristics
            heart_rate = np.random.uniform(*data_config["heart_rate_range"])

            ecg_data = generator.generate(
                duration_seconds=data_config["duration_seconds"],
                sampling_rate=data_config["sampling_rate"],
                heart_rate_bpm=heart_rate,
                noise_levels=data_config["noise_levels"],
                rhythm_class=arrhythmia_class
            )

            training_data.append((ecg_data, class_idx, arrhythmia_class))

        logger.info(f"  Generated {samples_per_class} samples for {arrhythmia_class}")

    # Generate validation data
    logger.info(f"Generating {data_config['num_validation_samples']} validation samples...")
    validation_data = []
    val_samples_per_class = data_config["num_validation_samples"] // len(arrhythmia_classes)

    for class_idx, arrhythmia_class in enumerate(arrhythmia_classes):
        for i in range(val_samples_per_class):
            heart_rate = np.random.uniform(*data_config["heart_rate_range"])
            ecg_data = generator.generate(
                duration_seconds=data_config["duration_seconds"],
                sampling_rate=data_config["sampling_rate"],
                heart_rate_bpm=heart_rate,
                noise_levels=data_config["noise_levels"],
                rhythm_class=arrhythmia_class
            )
            validation_data.append((ecg_data, class_idx, arrhythmia_class))

    # Generate test data
    logger.info(f"Generating {data_config['num_test_samples']} test samples...")
    test_data = []
    test_samples_per_class = data_config["num_test_samples"] // len(arrhythmia_classes)

    for class_idx, arrhythmia_class in enumerate(arrhythmia_classes):
        for i in range(test_samples_per_class):
            heart_rate = np.random.uniform(*data_config["heart_rate_range"])
            ecg_data = generator.generate(
                duration_seconds=data_config["duration_seconds"],
                sampling_rate=data_config["sampling_rate"],
                heart_rate_bpm=heart_rate,
                noise_levels=data_config["noise_levels"],
                rhythm_class=arrhythmia_class
            )
            test_data.append((ecg_data, class_idx, arrhythmia_class))

    # Shuffle datasets
    np.random.shuffle(training_data)
    np.random.shuffle(validation_data)
    np.random.shuffle(test_data)

    logger.info(f"Dataset generation complete:")
    logger.info(f"  Training: {len(training_data)} samples")
    logger.info(f"  Validation: {len(validation_data)} samples")
    logger.info(f"  Test: {len(test_data)} samples")
    logger.info(f"  Classes: {len(arrhythmia_classes)} arrhythmia types")

    return training_data, validation_data, test_data


# =============================================================================
# EXPERIMENTAL ARM 1: SOTA BASELINE
# =============================================================================

class ArrhythmiaCNN(nn.Module):
    """
    State-of-the-art CNN architecture for ECG arrhythmia classification.

    This model serves as the SOTA baseline for comparison with the GIF framework.
    It uses a standard convolutional architecture optimized for 1D time-series
    classification tasks like ECG analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        super(ArrhythmiaCNN, self).__init__()

        arch_config = config["architecture"]

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = arch_config["input_channels"]

        for conv_config in arch_config["conv_layers"]:
            self.conv_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=conv_config["out_channels"],
                kernel_size=conv_config["kernel_size"],
                stride=conv_config["stride"],
                padding=conv_config["padding"]
            ))
            in_channels = conv_config["out_channels"]

        # Pooling and activation
        self.pool = nn.MaxPool1d(arch_config["pool_size"])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(arch_config["dropout_rate"])

        # Calculate flattened size (approximate for ECG length)
        # Assuming ECG length of 5000 samples (10s at 500Hz)
        flattened_size = self._calculate_flattened_size(5000, arch_config)

        # Fully connected layers
        fc_layers = []
        prev_size = flattened_size

        for fc_size in arch_config["fc_layers"]:
            fc_layers.extend([
                nn.Linear(prev_size, fc_size),
                nn.ReLU(),
                nn.Dropout(arch_config["dropout_rate"])
            ])
            prev_size = fc_size

        # Final classification layer
        fc_layers.append(nn.Linear(prev_size, arch_config["num_classes"]))

        self.classifier = nn.Sequential(*fc_layers)

    def _calculate_flattened_size(self, input_length: int, arch_config: Dict) -> int:
        """Calculate the flattened size after conv and pooling layers."""
        length = input_length

        for conv_config in arch_config["conv_layers"]:
            # After convolution
            length = (length + 2 * conv_config["padding"] - conv_config["kernel_size"]) // conv_config["stride"] + 1
            # After pooling
            length = length // arch_config["pool_size"]

        # Multiply by final number of channels
        final_channels = arch_config["conv_layers"][-1]["out_channels"]
        return length * final_channels

    def forward(self, x):
        # Convolutional feature extraction
        for conv_layer in self.conv_layers:
            x = self.relu(conv_layer(x))
            x = self.pool(x)
            x = self.dropout(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def run_sota_baseline(config: Dict[str, Any], training_data: List, validation_data: List,
                     test_data: List, logger: logging.Logger) -> Dict[str, Any]:
    """
    Run SOTA baseline experiment using standard CNN architecture.

    This establishes the performance benchmark using conventional deep learning
    approaches for ECG arrhythmia classification.

    Args:
        config: Experiment configuration
        training_data: Training dataset
        validation_data: Validation dataset
        test_data: Test dataset
        logger: Logger instance

    Returns:
        Dict containing experimental results and metrics
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENTAL ARM 1: SOTA BASELINE (CNN)")
    logger.info("=" * 60)

    baseline_config = config["sota_baseline_config"]
    device = get_device()

    # Initialize CNN model
    logger.info("Initializing SOTA CNN baseline model...")
    model = ArrhythmiaCNN(baseline_config)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ CNN model initialized with {total_params:,} parameters")

    # Prepare data for CNN (convert ECG to tensor format)
    logger.info("Preparing data for CNN training...")

    def prepare_cnn_data(data_list):
        """Convert ECG data to CNN-compatible format."""
        X, y = [], []
        for ecg_data, label, class_name in data_list:
            # Extract voltage signal and normalize
            voltage = ecg_data['voltage'].to_numpy()
            voltage = (voltage - voltage.mean()) / (voltage.std() + 1e-8)
            X.append(voltage)
            y.append(label)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    train_X, train_y = prepare_cnn_data(training_data)
    val_X, val_y = prepare_cnn_data(validation_data)
    test_X, test_y = prepare_cnn_data(test_data)

    # Add channel dimension for CNN
    train_X = train_X.unsqueeze(1)  # [batch, 1, length]
    val_X = val_X.unsqueeze(1)
    test_X = test_X.unsqueeze(1)

    logger.info(f"✓ Data prepared: Train={train_X.shape}, Val={val_X.shape}, Test={test_X.shape}")

    # Initialize training components
    train_config = baseline_config["training"]
    optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"],
                          weight_decay=train_config["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, **train_config["scheduler_params"])

    # Create data loaders
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True)

    val_dataset = TensorDataset(val_X, val_y)
    val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=False)

    logger.info("✓ Training components initialized")

    # Training loop
    logger.info("Starting CNN baseline training...")
    start_time = time.time()

    learning_curve = []
    best_val_accuracy = 0.0

    for epoch in range(train_config["num_epochs"]):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)

        # Calculate metrics
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        epoch_time = time.time() - epoch_start

        # Update learning rate
        scheduler.step()

        # Log progress
        logger.info(f"Epoch {epoch+1}/{train_config['num_epochs']}: "
                   f"Train Loss={train_loss/len(train_loader):.4f}, "
                   f"Train Acc={train_accuracy:.4f}, "
                   f"Val Acc={val_accuracy:.4f}, "
                   f"Time={epoch_time:.2f}s")

        # Save learning curve
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "train_accuracy": train_accuracy,
            "val_loss": val_loss / len(val_loader),
            "val_accuracy": val_accuracy,
            "learning_rate": scheduler.get_last_lr()[0],
            "timestamp": time.time() - start_time
        }
        learning_curve.append(log_entry)

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_path = os.path.join(SOTA_BASELINE_DIR, "models", "best_model.pth")
            torch.save(model.state_dict(), model_path)

    total_training_time = time.time() - start_time

    # Final evaluation on test set
    logger.info("Evaluating CNN baseline on test set...")
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        test_dataset = TensorDataset(test_X, test_y)
        test_loader = DataLoader(test_dataset, batch_size=train_config["batch_size"], shuffle=False)

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
            test_total += target.size(0)

    final_accuracy = test_correct / test_total

    # Compile results
    results = {
        "experiment_type": "sota_baseline",
        "model_type": baseline_config["model_type"],
        "final_accuracy": final_accuracy,
        "best_val_accuracy": best_val_accuracy,
        "training_time": total_training_time,
        "total_parameters": total_params,
        "learning_curve": learning_curve,
        "model_path": os.path.join(SOTA_BASELINE_DIR, "models", "best_model.pth"),
        "config": baseline_config.copy()
    }

    # Save results
    save_experimental_results(results, SOTA_BASELINE_DIR, "sota_baseline")

    logger.info(f"SOTA CNN baseline completed")
    logger.info(f"Final test accuracy: {final_accuracy:.4f}")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    logger.info(f"Total training time: {total_training_time:.2f}s")

    return results


# =============================================================================
# EXPERIMENTAL ARM 2: NAIVE GIF-DU (Control Group)
# =============================================================================

def run_naive_gif_du(config: Dict[str, Any], training_data: List, validation_data: List,
                    test_data: List, logger: logging.Logger) -> Dict[str, Any]:
    """
    Run naive GIF-DU experiment with fresh random weights (control group).

    This establishes the baseline learning performance of the GIF architecture
    on the medical task without any prior experience.

    Args:
        config: Experiment configuration
        training_data: Training dataset
        validation_data: Validation dataset
        test_data: Test dataset
        logger: Logger instance

    Returns:
        Dict containing experimental results and detailed learning curves
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENTAL ARM 2: NAIVE GIF-DU (Control Group)")
    logger.info("=" * 60)
    logger.info("Testing baseline learning performance with fresh random weights")

    gif_config = config["gif_config"]
    device = get_device()

    # Initialize components with fresh weights
    logger.info("Initializing GIF components with fresh random weights...")

    # 1. Create ECG encoder
    encoder = ECG_Encoder(**gif_config["encoder_config"])
    logger.info(f"✓ ECG_Encoder initialized: {encoder.get_config()['encoder_type']}")

    # 2. Create arrhythmia decoder
    decoder = Arrhythmia_Decoder(**gif_config["decoder_config"])
    logger.info(f"✓ Arrhythmia_Decoder initialized: {len(decoder.class_names)} classes")

    # 3. Create fresh DU_Core with random weights
    du_core = DU_Core_V1(**gif_config["snn_architecture"])
    du_core.to(device)
    logger.info(f"✓ DU_Core_V1 initialized with fresh random weights")
    logger.info(f"  Architecture: {du_core}")

    # 4. Assemble GIF model
    gif_model = GIF(du_core)
    gif_model.attach_encoder(encoder)
    gif_model.attach_decoder(decoder)
    logger.info("✓ GIF model assembled with naive components")

    # 5. Initialize training components
    train_config = gif_config["training"]
    optimizer = optim.Adam(du_core.parameters(), lr=train_config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    trainer = Continual_Trainer(
        gif_model=gif_model,
        optimizer=optimizer,
        criterion=criterion,
        memory_batch_size=train_config["memory_batch_size"]
    )

    # 6. Initialize neuromorphic simulator
    simulator = NeuromorphicSimulator(
        snn_model=du_core,
        energy_per_synop=config["simulation"]["energy_per_synop"]
    )

    logger.info("✓ Training components initialized")

    # 7. Execute training with detailed logging
    logger.info("Starting naive GIF-DU training...")
    start_time = time.time()

    learning_curve = []
    log_interval = config["logging"]["log_every_n_samples"]

    # Training loop with detailed logging
    for epoch in range(train_config["num_epochs"]):
        epoch_start = time.time()
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (ecg_data, label, class_name) in enumerate(training_data):
            # Process through GIF pipeline
            spike_train = encoder.encode(ecg_data)
            output_spikes = du_core.process(spike_train.unsqueeze(0))  # Add batch dimension

            # Convert to logits for training
            decoder.train()
            logits = decoder.forward(output_spikes)
            target = torch.tensor([label], dtype=torch.long, device=device)

            # Training step
            loss = trainer.train_step(
                raw_data=ecg_data,
                target=target,
                task_id=train_config["task_id"]
            )

            epoch_loss += loss

            # Calculate accuracy
            predicted = torch.argmax(logits, dim=1)
            correct_predictions += (predicted == target).sum().item()
            total_samples += 1

            # Log at specified intervals
            if (batch_idx + 1) % log_interval == 0:
                current_accuracy = correct_predictions / total_samples
                energy_stats = simulator.get_energy_stats()

                log_entry = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "sample": epoch * len(training_data) + batch_idx,
                    "loss": loss,
                    "accuracy": current_accuracy,
                    "energy_per_sample": energy_stats.get("avg_energy_per_sample", 0),
                    "timestamp": time.time() - start_time
                }
                learning_curve.append(log_entry)

        # Epoch summary
        epoch_time = time.time() - epoch_start
        epoch_accuracy = correct_predictions / total_samples

        logger.info(f"Epoch {epoch+1}/{train_config['num_epochs']}: "
                   f"Loss={epoch_loss/len(training_data):.4f}, "
                   f"Accuracy={epoch_accuracy:.4f}, "
                   f"Time={epoch_time:.2f}s")

    total_training_time = time.time() - start_time

    # Final evaluation
    logger.info("Evaluating naive GIF-DU model...")
    final_accuracy = evaluate_gif_model(gif_model, test_data, device, logger)

    # Compile results
    results = {
        "experiment_type": "naive_gif_du",
        "model_type": "GIF_DU_V1",
        "final_accuracy": final_accuracy,
        "training_time": total_training_time,
        "learning_curve": learning_curve,
        "model_path": os.path.join(NAIVE_RUN_DIR, "models", "final_model.pth"),
        "config": gif_config.copy()
    }

    # Save model and results
    save_experimental_results(results, NAIVE_RUN_DIR, "naive_gif_du")

    logger.info(f"Naive GIF-DU experiment completed")
    logger.info(f"Final accuracy: {final_accuracy:.4f}")
    logger.info(f"Total training time: {total_training_time:.2f}s")
    logger.info(f"Learning curve points: {len(learning_curve)}")

    return results


# =============================================================================
# EXPERIMENTAL ARM 3: PRE-EXPOSED GIF-DU (Experimental Group)
# =============================================================================

def run_pre_exposed_gif_du(config: Dict[str, Any], training_data: List, validation_data: List,
                          test_data: List, logger: logging.Logger) -> Dict[str, Any]:
    """
    Run pre-exposed GIF-DU experiment with weight-reset protocol (experimental group).

    This is the critical experimental arm that tests the system potentiation hypothesis.
    The protocol:
    1. Load pre-trained DU_Core from exoplanet task
    2. IMMEDIATELY reset all synaptic weights
    3. Train on medical task with identical protocol as naive model
    4. Any improvement vs naive proves potentiation, not knowledge transfer

    Args:
        config: Experiment configuration
        training_data: Training dataset
        validation_data: Validation dataset
        test_data: Test dataset
        logger: Logger instance

    Returns:
        Dict containing experimental results and detailed learning curves
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENTAL ARM 3: PRE-EXPOSED GIF-DU (Experimental Group)")
    logger.info("=" * 60)
    logger.info("Testing system potentiation hypothesis with weight-reset protocol")
    logger.info("CRITICAL PROTOCOL: Load pre-trained model → Reset weights → Train on new task")

    gif_config = config["gif_config"]
    device = get_device()

    # Initialize components (same as naive, but DU_Core will be pre-exposed)
    logger.info("Initializing GIF components for potentiation experiment...")

    # 1. Create ECG encoder (identical to naive)
    encoder = ECG_Encoder(**gif_config["encoder_config"])
    logger.info(f"✓ ECG_Encoder initialized: {encoder.get_config()['encoder_type']}")

    # 2. Create arrhythmia decoder (identical to naive)
    decoder = Arrhythmia_Decoder(**gif_config["decoder_config"])
    logger.info(f"✓ Arrhythmia_Decoder initialized: {len(decoder.class_names)} classes")

    # 3. Create DU_Core and load pre-trained weights
    logger.info("=" * 40)
    logger.info("EXECUTING WEIGHT-RESET PROTOCOL")
    logger.info("=" * 40)

    # Create DU_Core with same architecture as naive
    du_core = DU_Core_V1(**gif_config["snn_architecture"])
    du_core.to(device)
    logger.info(f"✓ DU_Core_V1 created with target architecture")

    # Load pre-trained exoplanet model state
    if os.path.exists(PRE_TRAINED_EXO_MODEL_PATH):
        logger.info(f"Loading pre-trained exoplanet model from: {PRE_TRAINED_EXO_MODEL_PATH}")
        try:
            # Load the state dict
            checkpoint = torch.load(PRE_TRAINED_EXO_MODEL_PATH, map_location=device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Load the pre-trained weights
            du_core.load_state_dict(state_dict, strict=False)  # strict=False for architecture differences
            logger.info("✓ Pre-trained exoplanet weights loaded successfully")

            # Get model info before reset
            pre_reset_config = du_core.get_config()
            logger.info(f"  Pre-trained model parameters: {pre_reset_config['total_parameters']:,}")

        except Exception as e:
            logger.warning(f"Failed to load pre-trained model: {e}")
            logger.warning("Proceeding with random initialization (equivalent to naive run)")
    else:
        logger.warning(f"Pre-trained model not found at: {PRE_TRAINED_EXO_MODEL_PATH}")
        logger.warning("Proceeding with random initialization (equivalent to naive run)")

    # 4. CRITICAL STEP: Reset all synaptic weights
    logger.info("EXECUTING WEIGHT RESET - This is the key to proving potentiation!")
    logger.info("Wiping all specific knowledge while preserving structural experience...")

    du_core.reset_weights()  # This is the critical method we added

    logger.info("✓ Weight reset protocol completed")
    logger.info("✓ DU_Core now has 'experienced structure' but 'naive weights'")
    logger.info("✓ Any learning improvement vs naive model proves system potentiation")

    # 5. Assemble GIF model (identical to naive from this point)
    gif_model = GIF(du_core)
    gif_model.attach_encoder(encoder)
    gif_model.attach_decoder(decoder)
    logger.info("✓ GIF model assembled with pre-exposed (weight-reset) DU_Core")

    # 6. Initialize training components (identical to naive)
    train_config = gif_config["training"]
    optimizer = optim.Adam(du_core.parameters(), lr=train_config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    trainer = Continual_Trainer(
        gif_model=gif_model,
        optimizer=optimizer,
        criterion=criterion,
        memory_batch_size=train_config["memory_batch_size"]
    )

    # 7. Initialize neuromorphic simulator
    simulator = NeuromorphicSimulator(
        snn_model=du_core,
        energy_per_synop=config["simulation"]["energy_per_synop"]
    )

    logger.info("✓ Training components initialized (identical to naive)")

    # 8. Execute training with detailed logging (IDENTICAL to naive)
    logger.info("Starting pre-exposed GIF-DU training...")
    logger.info("Training protocol is IDENTICAL to naive - only difference is weight-reset history")
    start_time = time.time()

    learning_curve = []
    log_interval = config["logging"]["log_every_n_samples"]

    # Training loop (identical to naive)
    for epoch in range(train_config["num_epochs"]):
        epoch_start = time.time()
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (ecg_data, label, class_name) in enumerate(training_data):
            # Process through GIF pipeline
            spike_train = encoder.encode(ecg_data)
            output_spikes = du_core.process(spike_train.unsqueeze(0))

            # Convert to logits for training
            decoder.train()
            logits = decoder.forward(output_spikes)
            target = torch.tensor([label], dtype=torch.long, device=device)

            # Training step
            loss = trainer.train_step(
                raw_data=ecg_data,
                target=target,
                task_id=train_config["task_id"]
            )

            epoch_loss += loss

            # Calculate accuracy
            predicted = torch.argmax(logits, dim=1)
            correct_predictions += (predicted == target).sum().item()
            total_samples += 1

            # Log at specified intervals
            if (batch_idx + 1) % log_interval == 0:
                current_accuracy = correct_predictions / total_samples
                energy_stats = simulator.get_energy_stats()

                log_entry = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "sample": epoch * len(training_data) + batch_idx,
                    "loss": loss,
                    "accuracy": current_accuracy,
                    "energy_per_sample": energy_stats.get("avg_energy_per_sample", 0),
                    "timestamp": time.time() - start_time
                }
                learning_curve.append(log_entry)

        # Epoch summary
        epoch_time = time.time() - epoch_start
        epoch_accuracy = correct_predictions / total_samples

        logger.info(f"Epoch {epoch+1}/{train_config['num_epochs']}: "
                   f"Loss={epoch_loss/len(training_data):.4f}, "
                   f"Accuracy={epoch_accuracy:.4f}, "
                   f"Time={epoch_time:.2f}s")

    total_training_time = time.time() - start_time

    # Final evaluation
    logger.info("Evaluating pre-exposed GIF-DU model...")
    final_accuracy = evaluate_gif_model(gif_model, test_data, device, logger)

    # Compile results
    results = {
        "experiment_type": "pre_exposed_gif_du",
        "model_type": "GIF_DU_V1_PreExposed",
        "final_accuracy": final_accuracy,
        "training_time": total_training_time,
        "learning_curve": learning_curve,
        "model_path": os.path.join(PRE_EXPOSED_RUN_DIR, "models", "final_model.pth"),
        "config": gif_config.copy(),
        "weight_reset_protocol": True,
        "pre_trained_source": PRE_TRAINED_EXO_MODEL_PATH
    }

    # Save model and results
    save_experimental_results(results, PRE_EXPOSED_RUN_DIR, "pre_exposed_gif_du")

    logger.info("=" * 40)
    logger.info("PRE-EXPOSED EXPERIMENT COMPLETED")
    logger.info("=" * 40)
    logger.info(f"Final accuracy: {final_accuracy:.4f}")
    logger.info(f"Total training time: {total_training_time:.2f}s")
    logger.info(f"Learning curve points: {len(learning_curve)}")
    logger.info("Ready for potentiation analysis vs naive model")

    return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def evaluate_gif_model(gif_model: GIF, test_data: List, device: torch.device,
                      logger: logging.Logger) -> float:
    """
    Evaluate GIF model on test dataset.

    Args:
        gif_model: Assembled GIF model
        test_data: Test dataset
        device: Computation device
        logger: Logger instance

    Returns:
        Test accuracy as float
    """
    gif_model._du_core.eval()
    gif_model._decoder.eval()

    correct_predictions = 0
    total_samples = len(test_data)

    with torch.no_grad():
        for ecg_data, label, class_name in test_data:
            # Process through GIF pipeline
            spike_train = gif_model._encoder.encode(ecg_data)
            output_spikes = gif_model._du_core.process(spike_train.unsqueeze(0))

            # Get prediction
            prediction = gif_model._decoder.decode(output_spikes)
            predicted_class = gif_model._decoder.class_names.index(prediction)

            if predicted_class == label:
                correct_predictions += 1

    accuracy = correct_predictions / total_samples
    logger.info(f"Test evaluation: {correct_predictions}/{total_samples} correct ({accuracy:.4f})")

    return accuracy


def save_experimental_results(results: Dict[str, Any], output_dir: str,
                            experiment_name: str) -> None:
    """
    Save experimental results to multiple formats.

    Args:
        results: Results dictionary
        output_dir: Output directory path
        experiment_name: Name of experiment for file naming
    """
    # Save learning curve as CSV
    if "learning_curve" in results and results["learning_curve"]:
        learning_curve_df = pd.DataFrame(results["learning_curve"])
        csv_path = os.path.join(output_dir, "logs", f"{experiment_name}_learning_curve.csv")
        learning_curve_df.to_csv(csv_path, index=False)
        print(f"Learning curve saved to: {csv_path}")

    # Save full results as JSON
    json_results = results.copy()
    # Remove non-serializable items
    if "learning_curve" in json_results:
        json_results["learning_curve_length"] = len(json_results["learning_curve"])
        del json_results["learning_curve"]  # Already saved as CSV

    json_path = os.path.join(output_dir, "logs", f"{experiment_name}_results.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Save full results as pickle for Python analysis
    pkl_path = os.path.join(output_dir, "logs", f"{experiment_name}_results.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Full results saved to: {pkl_path}")


# =============================================================================
# MAIN EXPERIMENT ORCHESTRATOR
# =============================================================================

def main():
    """
    Master experiment orchestrator for the System Potentiation experiment.

    This function coordinates all three experimental arms and generates the
    data needed to rigorously test the system potentiation hypothesis.
    """
    print("=" * 80)
    print("MEDICAL DIAGNOSTICS SYSTEM POTENTIATION EXPERIMENT")
    print("=" * 80)
    print("Testing the hypothesis that diverse prior experience improves")
    print("the fundamental learning mechanism itself, not just knowledge transfer")
    print("=" * 80)

    try:
        # Load configuration and set up environment
        config = get_config()
        set_reproducibility_seeds(config)
        create_output_directories(config)
        logger = setup_logging(config)

        logger.info("Experiment configuration loaded and environment prepared")
        logger.info(f"Device: {get_device()}")
        logger.info(f"Reproducibility seeds set: {config['reproducibility']}")

        # Generate shared datasets for all experimental arms
        logger.info("Generating shared ECG datasets for all experimental arms...")
        training_data, validation_data, test_data = generate_ecg_datasets(config, logger)

        logger.info("Starting three experimental arms...")

        # Experimental Arm 1: SOTA Baseline
        logger.info("\n" + "="*80)
        logger.info("STARTING EXPERIMENTAL ARM 1: SOTA BASELINE")
        logger.info("="*80)
        sota_results = run_sota_baseline(config, training_data, validation_data, test_data, logger)

        # Experimental Arm 2: Naive GIF-DU (Control Group)
        logger.info("\n" + "="*80)
        logger.info("STARTING EXPERIMENTAL ARM 2: NAIVE GIF-DU (Control)")
        logger.info("="*80)
        naive_results = run_naive_gif_du(config, training_data, validation_data, test_data, logger)

        # Experimental Arm 3: Pre-Exposed GIF-DU (Experimental Group)
        logger.info("\n" + "="*80)
        logger.info("STARTING EXPERIMENTAL ARM 3: PRE-EXPOSED GIF-DU (Experimental)")
        logger.info("="*80)
        pre_exposed_results = run_pre_exposed_gif_du(config, training_data, validation_data, test_data, logger)

        # Compile comparative results
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT COMPLETED - COMPILING RESULTS")
        logger.info("="*80)

        comparative_results = {
            "experiment_metadata": config["metadata"],
            "sota_baseline": sota_results,
            "naive_gif_du": naive_results,
            "pre_exposed_gif_du": pre_exposed_results,
            "potentiation_analysis": {
                "naive_final_accuracy": naive_results["final_accuracy"],
                "pre_exposed_final_accuracy": pre_exposed_results["final_accuracy"],
                "accuracy_improvement": pre_exposed_results["final_accuracy"] - naive_results["final_accuracy"],
                "relative_improvement": (pre_exposed_results["final_accuracy"] - naive_results["final_accuracy"]) / naive_results["final_accuracy"],
                "naive_training_time": naive_results["training_time"],
                "pre_exposed_training_time": pre_exposed_results["training_time"]
            }
        }

        # Save comparative results
        comparative_path = os.path.join(ANALYSIS_DIR, "comparative_results.json")
        with open(comparative_path, 'w') as f:
            json.dump(comparative_results, f, indent=2)

        # Print final summary
        logger.info("FINAL EXPERIMENTAL SUMMARY:")
        logger.info(f"SOTA Baseline Accuracy: {sota_results['final_accuracy']:.4f}")
        logger.info(f"Naive GIF-DU Accuracy: {naive_results['final_accuracy']:.4f}")
        logger.info(f"Pre-Exposed GIF-DU Accuracy: {pre_exposed_results['final_accuracy']:.4f}")

        improvement = comparative_results["potentiation_analysis"]["accuracy_improvement"]
        relative_improvement = comparative_results["potentiation_analysis"]["relative_improvement"]

        logger.info(f"Accuracy Improvement: {improvement:+.4f} ({relative_improvement:+.2%})")

        if improvement > 0:
            logger.info("🎉 POSITIVE RESULT: Pre-exposed model outperformed naive model!")
            logger.info("This suggests evidence for SYSTEM POTENTIATION")
        else:
            logger.info("📊 NEGATIVE RESULT: No improvement detected")
            logger.info("Further analysis needed to understand results")

        logger.info(f"Comparative results saved to: {comparative_path}")
        logger.info("Ready for statistical analysis in Task 5.3")

        logger.info("=" * 80)
        logger.info("SYSTEM POTENTIATION EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return comparative_results

    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        logger.error("Full traceback:", exc_info=True)
        raise


if __name__ == "__main__":
    # Execute the complete potentiation experiment
    results = main()
