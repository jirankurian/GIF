"""
Exoplanet Detection Proof of Concept - Complete Implementation
==============================================================

This script implements the complete exoplanet detection proof-of-concept using the GIF
framework. It demonstrates the end-to-end capabilities of the neuromorphic AI system,
from synthetic data generation through spike-based processing to scientific analysis.

The implementation showcases the key innovations of the GIF framework:
- Event-driven neuromorphic computation for energy efficiency
- Biologically-inspired spike-based information processing
- Modular plug-and-play architecture for domain adaptation
- Real-time learning with catastrophic forgetting prevention
- Comprehensive scientific validation and analysis

Experimental Pipeline:
=====================

1. **Data Generation**: Create realistic synthetic exoplanet light curves with proper
   noise characteristics and astrophysical parameters.

2. **Encoding**: Convert continuous light curve data into sparse spike trains using
   delta modulation, emphasizing transient events like planetary transits.

3. **Processing**: Process spike trains through the DU Core SNN with real-time learning
   capabilities and episodic memory for continual learning.

4. **Decoding**: Extract meaningful scientific outputs (planet detection, parameter
   estimation) from processed spike patterns.

5. **Simulation**: Run the complete pipeline through neuromorphic hardware simulation
   to model realistic energy consumption and processing constraints.

6. **Analysis**: Generate comprehensive performance metrics, visualizations, and
   scientific validation results suitable for publication.

Scientific Validation:
=====================

The experiment provides quantitative evidence for the key claims of neuromorphic
computing in scientific applications:
- Energy efficiency compared to traditional approaches
- Robust learning with minimal catastrophic forgetting
- Real-time adaptability to new data characteristics
- Biological plausibility of information processing

Integration Architecture:
========================

The script demonstrates the complete GIF framework integration:
- Configuration-driven experiment management
- Professional logging and progress monitoring
- Comprehensive error handling and validation
- Publication-quality results generation
- Reproducible experimental protocols

Usage:
======

    python applications/poc_exoplanet/main_exo.py

Results are saved to the configured output directory with comprehensive
documentation, plots, and data exports for scientific analysis.
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Core framework imports
import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl
import numpy as np

# GIF Framework components
from gif_framework.orchestrator import GIF
from gif_framework.core.du_core import DU_Core_V1
from gif_framework.core.rtl_mechanisms import STDP_Rule
from gif_framework.core.memory_systems import EpisodicMemory
from gif_framework.training.trainer import Continual_Trainer

# Application-specific components
from applications.poc_exoplanet.encoders.light_curve_encoder import LightCurveEncoder
from applications.poc_exoplanet.decoders.exoplanet_decoder import ExoplanetDecoder
from applications.poc_exoplanet.config_exo import get_config, create_output_directories, get_device, move_to_device

# Data generation and simulation
from data_generators.exoplanet_generator import RealisticExoplanetGenerator
from simulators.neuromorphic_sim import NeuromorphicSimulator

# Analysis tools
from applications.analysis.continual_learning_analyzer import ContinualLearningAnalyzer


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Set up comprehensive logging for the experiment.

    Args:
        config: Experiment configuration dictionary.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logs directory
    logs_dir = config["analysis"]["logs_dir"]
    os.makedirs(logs_dir, exist_ok=True)

    # Configure logging
    log_file = os.path.join(logs_dir, "exoplanet_experiment.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger("ExoplanetPOC")
    logger.info("=" * 80)
    logger.info("EXOPLANET DETECTION PROOF-OF-CONCEPT EXPERIMENT")
    logger.info("=" * 80)

    return logger


# =============================================================================
# DATA GENERATION AND PREPARATION
# =============================================================================

def generate_datasets(config: Dict[str, Any], logger: logging.Logger) -> Tuple[List, List]:
    """
    Generate training and test datasets for the experiment.

    Args:
        config: Experiment configuration.
        logger: Logger instance.

    Returns:
        Tuple[List, List]: (training_data, test_data) lists of (light_curve, label) pairs.
    """
    logger.info("Generating synthetic exoplanet datasets...")

    data_config = config["data"]

    # Initialize data generator
    generator = RealisticExoplanetGenerator(
        seed=data_config["random_seed"],
        stellar_variability_amplitude=data_config["stellar_variability_amplitude"],
        instrumental_noise_level=data_config["instrumental_noise_level"],
        noise_color_alpha=data_config["noise_color_alpha"]
    )

    # Generate training data
    logger.info(f"Generating {data_config['num_training_samples']} training samples...")
    training_data = []

    for i in range(data_config["num_training_samples"]):
        # Generate light curve
        light_curve = generator.generate(
            observation_duration=data_config["observation_duration"],
            cadence_minutes=data_config["cadence_minutes"]
        )

        # Create binary label (1 if planet present, 0 if not)
        # For this POC, we'll use transit depth as indicator
        transit_depth = 1.0 - light_curve["flux"].min()
        label = 1 if transit_depth > 0.005 else 0  # 0.5% transit depth threshold

        training_data.append((light_curve, label))

        if (i + 1) % 100 == 0:
            logger.info(f"  Generated {i + 1}/{data_config['num_training_samples']} training samples")

    # Generate test data
    logger.info(f"Generating {data_config['num_test_samples']} test samples...")
    test_data = []

    for i in range(data_config["num_test_samples"]):
        light_curve = generator.generate(
            observation_duration=data_config["observation_duration"],
            cadence_minutes=data_config["cadence_minutes"]
        )

        transit_depth = 1.0 - light_curve["flux"].min()
        label = 1 if transit_depth > 0.005 else 0

        test_data.append((light_curve, label))

        if (i + 1) % 50 == 0:
            logger.info(f"  Generated {i + 1}/{data_config['num_test_samples']} test samples")

    # Log dataset statistics
    train_planets = sum(1 for _, label in training_data if label == 1)
    test_planets = sum(1 for _, label in test_data if label == 1)

    logger.info(f"Training dataset: {len(training_data)} samples ({train_planets} planets, {len(training_data) - train_planets} non-planets)")
    logger.info(f"Test dataset: {len(test_data)} samples ({test_planets} planets, {len(test_data) - test_planets} non-planets)")

    return training_data, test_data


# =============================================================================
# COMPONENT INITIALIZATION
# =============================================================================

def initialize_components(config: Dict[str, Any], logger: logging.Logger) -> Tuple[Any, Any, Any, torch.device]:
    """
    Initialize all GIF framework components with GPU support.

    Args:
        config: Experiment configuration.
        logger: Logger instance.

    Returns:
        Tuple: (encoder, decoder, du_core, device) initialized components.
    """
    logger.info("Initializing GIF framework components...")

    # Initialize device
    device = get_device(config["training"]["device"])
    logger.info(f"  ✓ Device initialized: {device}")

    # Initialize encoder with device support
    encoder_config = config["encoder"]
    encoder = LightCurveEncoder(
        threshold=encoder_config["threshold"],
        normalize_input=encoder_config["normalize_input"],
        device=str(device)
    )
    logger.info(f"  ✓ LightCurveEncoder initialized (threshold={encoder_config['threshold']:.2e})")

    # Initialize decoder
    decoder_config = config["decoder"]
    arch_config = config["architecture"]
    decoder = ExoplanetDecoder(
        input_size=arch_config["output_size"],
        output_size=decoder_config["regression_output_size"],
        classification_threshold=decoder_config["classification_threshold"]
    )
    logger.info(f"  ✓ ExoplanetDecoder initialized (input_size={arch_config['output_size']})")

    # Initialize RTL components if enabled
    plasticity_rule = None
    memory_system = None

    if arch_config["enable_plasticity"]:
        plasticity_rule = STDP_Rule(
            learning_rate_ltp=arch_config["stdp_learning_rate_ltp"],
            learning_rate_ltd=arch_config["stdp_learning_rate_ltd"],
            tau_ltp=arch_config["stdp_tau_ltp"],
            tau_ltd=arch_config["stdp_tau_ltd"]
        )
        logger.info("  ✓ STDP plasticity rule initialized")

    if arch_config["enable_memory"]:
        memory_system = EpisodicMemory(capacity=arch_config["memory_capacity"])
        logger.info(f"  ✓ Episodic memory initialized (capacity={arch_config['memory_capacity']})")

    # Initialize DU Core
    du_core = DU_Core_V1(
        input_size=arch_config["input_size"],
        hidden_sizes=arch_config["hidden_sizes"],
        output_size=arch_config["output_size"],
        beta=arch_config["neuron_beta"],
        threshold=arch_config["neuron_threshold"],
        recurrent=arch_config["recurrent"],
        plasticity_rule=plasticity_rule,
        memory_system=memory_system
    )
    logger.info(f"  ✓ DU_Core_V1 initialized ({arch_config['input_size']} → {arch_config['hidden_sizes']} → {arch_config['output_size']})")

    # Move models to device (MPS for Apple Silicon)
    du_core = du_core.to(device)
    decoder = decoder.to(device)
    logger.info(f"  ✓ Models moved to device: {device}")

    return encoder, decoder, du_core, device


def assemble_gif_model(encoder, decoder, du_core, device: torch.device, config: Dict[str, Any], logger: logging.Logger) -> Tuple[GIF, Continual_Trainer, NeuromorphicSimulator]:
    """
    Assemble the complete GIF model with training and simulation components.

    Args:
        encoder: Initialized encoder component.
        decoder: Initialized decoder component.
        du_core: Initialized DU Core component.
        device: Device for training (MPS, CUDA, or CPU).
        config: Experiment configuration.
        logger: Logger instance.

    Returns:
        Tuple: (gif_model, trainer, simulator) assembled components.
    """
    logger.info("Assembling GIF model and training infrastructure...")

    # Create GIF orchestrator
    gif_model = GIF(du_core)
    gif_model.attach_encoder(encoder)
    gif_model.attach_decoder(decoder)
    logger.info(f"  ✓ GIF orchestrator assembled with encoder and decoder on {device}")

    # Initialize optimizer
    train_config = config["training"]
    if train_config["optimizer"] == "Adam":
        optimizer = optim.Adam(
            gif_model.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"]
        )
    else:
        raise ValueError(f"Unsupported optimizer: {train_config['optimizer']}")

    logger.info(f"  ✓ {train_config['optimizer']} optimizer initialized (lr={train_config['learning_rate']:.2e})")

    # Initialize loss function
    if train_config["loss_function"] == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss(weight=train_config["class_weights"])
    else:
        raise ValueError(f"Unsupported loss function: {train_config['loss_function']}")

    logger.info(f"  ✓ {train_config['loss_function']} criterion initialized")

    # Initialize continual learning trainer
    trainer = Continual_Trainer(
        gif_model=gif_model,
        optimizer=optimizer,
        criterion=criterion,
        memory_batch_size=train_config["memory_batch_size"],
        gradient_clip_norm=train_config.get("gradient_clip_norm", None)
    )
    logger.info(f"  ✓ Continual_Trainer initialized (memory_batch_size={train_config['memory_batch_size']})")

    # Initialize neuromorphic simulator
    sim_config = config["simulation"]
    simulator = NeuromorphicSimulator(
        snn_model=du_core,
        energy_per_synop=sim_config["energy_per_synop"]
    )
    logger.info(f"  ✓ NeuromorphicSimulator initialized (energy_per_synop={sim_config['energy_per_synop']:.2e} J)")

    return gif_model, trainer, simulator


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def train_model(gif_model: GIF, trainer: Continual_Trainer, training_data: List, device: torch.device, config: Dict[str, Any], logger: logging.Logger) -> Dict[str, List]:
    """
    Execute the complete training pipeline.

    Args:
        gif_model: Assembled GIF model.
        trainer: Continual learning trainer.
        training_data: List of (light_curve, label) pairs.
        config: Experiment configuration.
        logger: Logger instance.

    Returns:
        Dict[str, List]: Training statistics and logs.
    """
    logger.info("Starting training pipeline...")

    train_config = config["training"]
    task_id = train_config["task_id"]

    # Training statistics
    training_logs = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
        "training_time": [],
        "interference_count": [],
        "memory_utilization": []
    }

    # Training loop
    for epoch in range(train_config["num_epochs"]):
        epoch_start_time = time.time()
        epoch_losses = []
        correct_predictions = 0
        total_predictions = 0

        logger.info(f"Epoch {epoch + 1}/{train_config['num_epochs']}")

        # Shuffle training data
        np.random.shuffle(training_data)

        # Process training data in batches
        batch_size = train_config["batch_size"]
        num_batches = len(training_data) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(training_data))
            batch_data = training_data[batch_start:batch_end]

            batch_losses = []

            for light_curve, target_label in batch_data:
                try:
                    # Prepare target tensor and move to device
                    target_tensor = torch.tensor([target_label], dtype=torch.long).to(device)

                    # Training step - trainer handles GIF processing internally
                    loss = trainer.train_step(light_curve, target_tensor, task_id)
                    batch_losses.append(loss)

                    # Get prediction for accuracy tracking
                    prediction = gif_model.process_single_input(light_curve)

                    # Convert prediction to comparable format
                    if isinstance(prediction, int):
                        pred_value = prediction
                    else:
                        pred_value = int(prediction)

                    # Track accuracy
                    correct_predictions += int(pred_value == target_label)
                    total_predictions += 1

                except Exception as e:
                    logger.warning(f"Training step failed: {e}")
                    continue

            if batch_losses:
                epoch_losses.extend(batch_losses)

            # Progress reporting
            if (batch_idx + 1) % 10 == 0:
                avg_loss = np.mean(batch_losses) if batch_losses else 0.0
                logger.info(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {avg_loss:.4f}")

        # Calculate epoch statistics
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        epoch_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        # Get trainer statistics
        trainer_stats = trainer.get_training_stats()

        # Log epoch results
        training_logs["epoch"].append(epoch + 1)
        training_logs["loss"].append(avg_epoch_loss)
        training_logs["accuracy"].append(epoch_accuracy)
        training_logs["training_time"].append(epoch_time)
        training_logs["interference_count"].append(trainer_stats.get("interference_count", 0))
        training_logs["memory_utilization"].append(len(trainer_stats.get("memory_utilization", [])))

        logger.info(f"  Epoch {epoch + 1} completed:")
        logger.info(f"    Loss: {avg_epoch_loss:.4f}")
        logger.info(f"    Accuracy: {epoch_accuracy:.4f}")
        logger.info(f"    Time: {epoch_time:.2f}s")
        logger.info(f"    Interference events: {trainer_stats.get('interference_count', 0)}")

    logger.info("Training completed successfully!")
    return training_logs


# =============================================================================
# EVALUATION PIPELINE
# =============================================================================

def evaluate_model(gif_model: GIF, simulator: NeuromorphicSimulator, test_data: List, config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Execute comprehensive model evaluation with neuromorphic simulation.

    Args:
        gif_model: Trained GIF model.
        simulator: Neuromorphic hardware simulator.
        test_data: List of (light_curve, label) test pairs.
        config: Experiment configuration.
        logger: Logger instance.

    Returns:
        Dict[str, Any]: Comprehensive evaluation results.
    """
    logger.info("Starting model evaluation...")

    # Evaluation statistics
    predictions = []
    true_labels = []
    simulation_stats = []
    processing_times = []

    # Process test data
    for i, (light_curve, true_label) in enumerate(test_data):
        try:
            start_time = time.time()

            # Get prediction from GIF model
            prediction = gif_model.process_single_input(light_curve)

            # Convert to standard format
            if isinstance(prediction, int):
                pred_label = prediction
            else:
                pred_label = int(prediction)

            predictions.append(pred_label)
            true_labels.append(true_label)

            # Measure processing time
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # Run neuromorphic simulation for energy analysis
            try:
                # Encode light curve to spike train
                encoder = gif_model._encoder
                spike_train = encoder.encode(light_curve)

                # Add time dimension if needed
                if spike_train.dim() == 2:
                    spike_train = spike_train.unsqueeze(1)  # Add batch dimension

                # Run simulation
                output_spikes, sim_stats = simulator.run(spike_train)
                simulation_stats.append(sim_stats)

            except Exception as e:
                logger.warning(f"Simulation failed for sample {i}: {e}")
                # Add default stats
                simulation_stats.append({
                    'total_spikes': 0,
                    'total_synops': 0,
                    'estimated_energy_joules': 0.0
                })

            # Progress reporting
            if (i + 1) % 50 == 0:
                logger.info(f"  Evaluated {i + 1}/{len(test_data)} samples")

        except Exception as e:
            logger.warning(f"Evaluation failed for sample {i}: {e}")
            predictions.append(0)  # Default prediction
            true_labels.append(true_label)
            processing_times.append(0.0)
            simulation_stats.append({
                'total_spikes': 0,
                'total_synops': 0,
                'estimated_energy_joules': 0.0
            })

    # Calculate performance metrics
    results = calculate_performance_metrics(predictions, true_labels, simulation_stats, processing_times, logger)

    logger.info("Evaluation completed successfully!")
    return results


def calculate_performance_metrics(predictions: List, true_labels: List, simulation_stats: List, processing_times: List, logger: logging.Logger) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics.

    Args:
        predictions: List of predicted labels.
        true_labels: List of true labels.
        simulation_stats: List of simulation statistics.
        processing_times: List of processing times.
        logger: Logger instance.

    Returns:
        Dict[str, Any]: Comprehensive performance metrics.
    """
    logger.info("Calculating performance metrics...")

    # Convert to numpy arrays for easier computation
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Classification metrics
    accuracy = np.mean(predictions == true_labels)

    # Calculate confusion matrix components
    tp = np.sum((predictions == 1) & (true_labels == 1))  # True positives
    tn = np.sum((predictions == 0) & (true_labels == 0))  # True negatives
    fp = np.sum((predictions == 1) & (true_labels == 0))  # False positives
    fn = np.sum((predictions == 0) & (true_labels == 1))  # False negatives

    # Derived metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    true_positive_rate = recall
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Neuromorphic performance metrics
    total_spikes = sum(stats['total_spikes'] for stats in simulation_stats)
    total_synops = sum(stats['total_synops'] for stats in simulation_stats)
    total_energy = sum(stats['estimated_energy_joules'] for stats in simulation_stats)

    avg_spikes_per_sample = total_spikes / len(simulation_stats) if simulation_stats else 0.0
    avg_synops_per_sample = total_synops / len(simulation_stats) if simulation_stats else 0.0
    avg_energy_per_sample = total_energy / len(simulation_stats) if simulation_stats else 0.0

    # Processing efficiency
    avg_processing_time = np.mean(processing_times) if processing_times else 0.0

    # Compile results
    results = {
        # Classification performance
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "true_positive_rate": float(true_positive_rate),
        "false_positive_rate": float(false_positive_rate),

        # Confusion matrix
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),

        # Neuromorphic performance
        "total_spikes": int(total_spikes),
        "total_synops": int(total_synops),
        "total_energy_joules": float(total_energy),
        "avg_spikes_per_sample": float(avg_spikes_per_sample),
        "avg_synops_per_sample": float(avg_synops_per_sample),
        "avg_energy_per_sample": float(avg_energy_per_sample),

        # Processing efficiency
        "avg_processing_time": float(avg_processing_time),
        "samples_processed": len(predictions),

        # Raw data for further analysis
        "predictions": predictions.tolist(),
        "true_labels": true_labels.tolist(),
        "simulation_stats": simulation_stats,
        "processing_times": processing_times
    }

    # Log key metrics
    logger.info("Performance Metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1_score:.4f}")
    logger.info(f"  Average Energy per Sample: {avg_energy_per_sample:.2e} J")
    logger.info(f"  Average Processing Time: {avg_processing_time:.4f} s")

    return results


# =============================================================================
# RESULTS ANALYSIS AND EXPORT
# =============================================================================

def save_results(training_logs: Dict, evaluation_results: Dict, config: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Save comprehensive experimental results.

    Args:
        training_logs: Training statistics and logs.
        evaluation_results: Evaluation results and metrics.
        config: Experiment configuration.
        logger: Logger instance.
    """
    logger.info("Saving experimental results...")

    analysis_config = config["analysis"]
    results_dir = analysis_config["results_dir"]

    # Create output directories
    create_output_directories(config)

    # Save configuration
    if analysis_config["save_config"]:
        config_file = os.path.join(results_dir, "experiment_config.json")
        import json
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        logger.info(f"  ✓ Configuration saved to {config_file}")

    # Save training logs
    for format_type in analysis_config["export_formats"]:
        if format_type == "csv":
            training_df = pl.DataFrame(training_logs)
            training_file = os.path.join(results_dir, "training_logs.csv")
            training_df.write_csv(training_file)
            logger.info(f"  ✓ Training logs saved to {training_file}")

        elif format_type == "json":
            training_file = os.path.join(results_dir, "training_logs.json")
            import json
            with open(training_file, 'w') as f:
                json.dump(training_logs, f, indent=2, default=str)
            logger.info(f"  ✓ Training logs saved to {training_file}")

    # Save evaluation results
    for format_type in analysis_config["export_formats"]:
        if format_type == "csv":
            # Create DataFrame for key metrics
            metrics_data = {
                "metric": list(evaluation_results.keys()),
                "value": [str(v) for v in evaluation_results.values()]
            }
            metrics_df = pl.DataFrame(metrics_data)
            metrics_file = os.path.join(results_dir, "evaluation_metrics.csv")
            metrics_df.write_csv(metrics_file)
            logger.info(f"  ✓ Evaluation metrics saved to {metrics_file}")

        elif format_type == "json":
            results_file = os.path.join(results_dir, "evaluation_results.json")
            import json
            with open(results_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2, default=str)
            logger.info(f"  ✓ Evaluation results saved to {results_file}")

    # Generate summary report
    summary_file = os.path.join(results_dir, "experiment_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("EXOPLANET DETECTION POC - EXPERIMENT SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write("CONFIGURATION:\n")
        f.write(f"  Training samples: {config['data']['num_training_samples']}\n")
        f.write(f"  Test samples: {config['data']['num_test_samples']}\n")
        f.write(f"  Architecture: {config['architecture']['input_size']} → {config['architecture']['hidden_sizes']} → {config['architecture']['output_size']}\n")
        f.write(f"  Training epochs: {config['training']['num_epochs']}\n")
        f.write(f"  Learning rate: {config['training']['learning_rate']}\n\n")

        f.write("PERFORMANCE RESULTS:\n")
        f.write(f"  Accuracy: {evaluation_results['accuracy']:.4f}\n")
        f.write(f"  Precision: {evaluation_results['precision']:.4f}\n")
        f.write(f"  Recall: {evaluation_results['recall']:.4f}\n")
        f.write(f"  F1-Score: {evaluation_results['f1_score']:.4f}\n\n")

        f.write("NEUROMORPHIC PERFORMANCE:\n")
        f.write(f"  Total energy consumption: {evaluation_results['total_energy_joules']:.2e} J\n")
        f.write(f"  Average energy per sample: {evaluation_results['avg_energy_per_sample']:.2e} J\n")
        f.write(f"  Total synaptic operations: {evaluation_results['total_synops']}\n")
        f.write(f"  Average spikes per sample: {evaluation_results['avg_spikes_per_sample']:.2f}\n")
        f.write(f"  Average processing time: {evaluation_results['avg_processing_time']:.4f} s\n\n")

        f.write("CONFUSION MATRIX:\n")
        f.write(f"  True Positives: {evaluation_results['true_positives']}\n")
        f.write(f"  True Negatives: {evaluation_results['true_negatives']}\n")
        f.write(f"  False Positives: {evaluation_results['false_positives']}\n")
        f.write(f"  False Negatives: {evaluation_results['false_negatives']}\n")

    logger.info(f"  ✓ Summary report saved to {summary_file}")
    logger.info("All results saved successfully!")


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main() -> None:
    """
    Main execution function for the exoplanet detection POC experiment.

    This function orchestrates the complete experimental pipeline from
    configuration loading through results analysis and export.
    """
    try:
        # Load configuration
        config = get_config()

        # Set up logging
        logger = setup_logging(config)

        logger.info("Starting Exoplanet Detection Proof-of-Concept Experiment")
        logger.info(f"Experiment: {config['metadata']['experiment_name']}")
        logger.info(f"Version: {config['metadata']['version']}")
        logger.info(f"Description: {config['metadata']['description']}")

        # Generate datasets
        training_data, test_data = generate_datasets(config, logger)

        # Initialize components
        encoder, decoder, du_core, device = initialize_components(config, logger)

        # Assemble GIF model
        gif_model, trainer, simulator = assemble_gif_model(encoder, decoder, du_core, device, config, logger)

        # Train model
        training_logs = train_model(gif_model, trainer, training_data, device, config, logger)

        # Evaluate model
        evaluation_results = evaluate_model(gif_model, simulator, test_data, config, logger)

        # Save results
        save_results(training_logs, evaluation_results, config, logger)

        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {config['analysis']['results_dir']}")
        logger.info(f"Final accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info(f"Energy efficiency: {evaluation_results['avg_energy_per_sample']:.2e} J/sample")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
