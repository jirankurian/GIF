"""
Medical Diagnostics Potentiation Experiment Configuration
=========================================================

This module contains the complete configuration for the medical diagnostics potentiation
experiment using the GIF framework. This experiment is the definitive test of the
system potentiation hypothesis - that diverse prior experience improves the fundamental
learning mechanism itself, not just knowledge transfer.

The configuration supports three experimental arms:
1. SOTA Baseline: Standard CNN for performance comparison
2. Naive GIF-DU: Fresh random weights (control group)
3. Pre-Exposed GIF-DU: Weight-reset protocol (experimental group)

Scientific Method:
- Identical datasets and training protocols across all runs
- Weight-reset protocol ensures fair comparison
- Detailed logging for statistical analysis
- Reproducible experimental conditions

Key Innovation:
The weight-reset protocol is the critical scientific control that distinguishes
this from simple transfer learning. By wiping specific knowledge but preserving
structural improvements, we isolate true system potentiation effects.
"""

import os
from typing import Dict, Any, List
import torch

# =============================================================================
# EXPERIMENT PATHS AND MODEL REFERENCES
# =============================================================================

# Path to pre-trained exoplanet model from Phase 4
PRE_TRAINED_EXO_MODEL_PATH = "results/poc_exoplanet/models/best_model.pth"

# Base results directory for medical POC
RESULTS_BASE_DIR = "results/poc_medical"

# Specific output directories for each experimental arm
SOTA_BASELINE_DIR = os.path.join(RESULTS_BASE_DIR, "sota_baseline")
NAIVE_RUN_DIR = os.path.join(RESULTS_BASE_DIR, "naive_run") 
PRE_EXPOSED_RUN_DIR = os.path.join(RESULTS_BASE_DIR, "pre_exposed_run")
ANALYSIS_DIR = os.path.join(RESULTS_BASE_DIR, "analysis")

# =============================================================================
# SHARED EXPERIMENTAL CONFIGURATION
# =============================================================================

def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration for the medical potentiation experiment.
    
    Returns:
        Dict containing all experimental parameters organized by category.
    """
    
    config = {
        
        # =====================================================================
        # EXPERIMENTAL METADATA
        # =====================================================================
        "metadata": {
            "experiment_name": "medical_potentiation_experiment",
            "description": "System potentiation experiment comparing naive vs pre-exposed GIF-DU models",
            "version": "1.0.0",
            "author": "GIF Development Team",
            "date_created": "2025-07-09",
            
            # Scientific context
            "domain": "medical_diagnostics",
            "task_type": "multi_class_classification",
            "data_type": "ecg_time_series",
            "hypothesis": "system_potentiation",
            
            # Experimental design
            "control_group": "naive_gif_du",
            "experimental_group": "pre_exposed_gif_du",
            "baseline_comparison": "sota_cnn",
            "critical_protocol": "weight_reset"
        },
        
        # =====================================================================
        # DATASET CONFIGURATION
        # =====================================================================
        "data": {
            # ECG generation parameters
            "duration_seconds": 10.0,
            "sampling_rate": 500,  # Hz
            "num_training_samples": 1000,
            "num_test_samples": 200,
            "num_validation_samples": 100,
            
            # Heart rate and arrhythmia parameters
            "heart_rate_range": [60, 100],  # BPM
            "arrhythmia_classes": [
                "Normal Sinus Rhythm",
                "Atrial Fibrillation", 
                "Atrial Flutter",
                "Supraventricular Tachycardia",
                "Ventricular Tachycardia",
                "Ventricular Fibrillation",
                "Premature Ventricular Contraction",
                "Premature Atrial Contraction"
            ],
            
            # Noise levels for realistic ECG
            "noise_levels": {
                "baseline": 0.08,      # Baseline wander
                "muscle": 0.015,       # Muscle artifact
                "powerline": 0.005,    # Powerline interference
                "powerline_freq": 60   # Hz (North American standard)
            },
            
            # Reproducibility
            "random_seed": 42,
            "train_test_split_seed": 123
        },
        
        # =====================================================================
        # GIF FRAMEWORK CONFIGURATION (Shared by Naive and Pre-Exposed)
        # =====================================================================
        "gif_config": {
            # SNN Architecture
            "snn_architecture": {
                "input_size": 3,        # ECG_Encoder output channels
                "hidden_sizes": [128, 64, 32],  # Multi-layer SNN
                "output_size": 8,       # Number of arrhythmia classes
                "beta": 0.95,          # LIF neuron leak parameter
                "threshold": 1.0       # Firing threshold
            },
            
            # Training parameters
            "training": {
                "learning_rate": 1e-4,
                "num_epochs": 50,
                "batch_size": 16,
                "optimizer": "Adam",
                "weight_decay": 1e-5,
                "
                
                # Loss function
                "loss_function": "CrossEntropyLoss",
                "class_weights": None,  # Will be calculated from data
                
                # Continual learning
                "memory_batch_size": 32,
                "task_id": "medical_arrhythmia_detection"
            },
            
            # Encoder configuration
            "encoder_config": {
                "sampling_rate": 500,
                "num_channels": 3,
                "r_peak_threshold": 0.6,
                "heart_rate_window_ms": 200.0,
                "qrs_search_window_ms": 80.0
            },
            
            # Decoder configuration  
            "decoder_config": {
                "input_size": 8,        # Must match SNN output_size
                "num_classes": 8,       # Number of arrhythmia classes
                "dropout_rate": 0.1
            }
        },
        
        # =====================================================================
        # SOTA BASELINE CONFIGURATION
        # =====================================================================
        "sota_baseline_config": {
            "model_type": "ArrhythmiaCNN",
            "architecture": {
                "input_channels": 1,    # Single ECG channel
                "conv_layers": [
                    {"out_channels": 32, "kernel_size": 7, "stride": 1, "padding": 3},
                    {"out_channels": 64, "kernel_size": 5, "stride": 1, "padding": 2},
                    {"out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1}
                ],
                "pool_size": 2,
                "fc_layers": [256, 128],
                "num_classes": 8,
                "dropout_rate": 0.3
            },
            
            "training": {
                "learning_rate": 1e-3,
                "num_epochs": 50,
                "batch_size": 32,
                "optimizer": "Adam",
                "weight_decay": 1e-4,
                "scheduler": "StepLR",
                "scheduler_params": {"step_size": 15, "gamma": 0.5}
            }
        },
        
        # =====================================================================
        # NEUROMORPHIC SIMULATION PARAMETERS
        # =====================================================================
        "simulation": {
            "energy_per_synop": 2.5e-11,  # Joules per synaptic operation
            "hardware_target": "neuromorphic",
            "enable_energy_tracking": True,
            "enable_spike_monitoring": True
        },
        
        # =====================================================================
        # LOGGING AND ANALYSIS CONFIGURATION
        # =====================================================================
        "logging": {
            # Learning curve logging intervals
            "log_every_n_samples": 50,     # Log every 50 training samples
            "log_every_n_epochs": 1,       # Log every epoch
            
            # Metrics to track
            "metrics": [
                "loss", "accuracy", "precision", "recall", "f1_score",
                "training_time", "memory_utilization", "energy_consumption",
                "interference_count", "spike_rate"
            ],
            
            # Output file formats
            "save_learning_curves": True,
            "save_model_checkpoints": True,
            "save_detailed_logs": True,
            "export_formats": ["csv", "json", "pkl"]
        },
        
        # =====================================================================
        # STATISTICAL ANALYSIS PARAMETERS
        # =====================================================================
        "analysis": {
            "significance_level": 0.05,
            "confidence_interval": 0.95,
            "bootstrap_samples": 1000,
            "learning_efficiency_threshold": 0.9,  # 90% of peak accuracy
            
            # Comparison metrics
            "potentiation_metrics": [
                "samples_to_threshold",     # Key potentiation metric
                "final_accuracy",
                "learning_rate_slope",
                "convergence_speed",
                "energy_efficiency"
            ]
        },
        
        # =====================================================================
        # REPRODUCIBILITY SETTINGS
        # =====================================================================
        "reproducibility": {
            "torch_seed": 42,
            "numpy_seed": 42,
            "python_seed": 42,
            "deterministic_algorithms": True,
            "benchmark_mode": False
        }
    }
    
    return config


def create_output_directories(config: Dict[str, Any]) -> None:
    """
    Create all necessary output directories for the experiment.
    
    Args:
        config: Experiment configuration dictionary
    """
    directories = [
        RESULTS_BASE_DIR,
        SOTA_BASELINE_DIR,
        NAIVE_RUN_DIR,
        PRE_EXPOSED_RUN_DIR,
        ANALYSIS_DIR,
        
        # Subdirectories for each experimental arm
        os.path.join(SOTA_BASELINE_DIR, "models"),
        os.path.join(SOTA_BASELINE_DIR, "logs"),
        os.path.join(NAIVE_RUN_DIR, "models"),
        os.path.join(NAIVE_RUN_DIR, "logs"),
        os.path.join(PRE_EXPOSED_RUN_DIR, "models"),
        os.path.join(PRE_EXPOSED_RUN_DIR, "logs"),
        os.path.join(ANALYSIS_DIR, "figures"),
        os.path.join(ANALYSIS_DIR, "tables"),
        os.path.join(ANALYSIS_DIR, "statistics")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"Created output directories for medical potentiation experiment")
    print(f"Base directory: {RESULTS_BASE_DIR}")


def get_device() -> torch.device:
    """Get the appropriate device for training (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU for training")
    
    return device


def set_reproducibility_seeds(config: Dict[str, Any]) -> None:
    """Set all random seeds for reproducible experiments."""
    repro_config = config["reproducibility"]
    
    torch.manual_seed(repro_config["torch_seed"])
    torch.cuda.manual_seed_all(repro_config["torch_seed"])
    
    import numpy as np
    np.random.seed(repro_config["numpy_seed"])
    
    import random
    random.seed(repro_config["python_seed"])
    
    if repro_config["deterministic_algorithms"]:
        torch.use_deterministic_algorithms(True)
    
    torch.backends.cudnn.benchmark = repro_config["benchmark_mode"]
    
    print("Reproducibility seeds set for deterministic experiments")
