"""
Exoplanet Detection Experiment Configuration
===========================================

This module contains the complete configuration for the exoplanet detection proof-of-concept
experiment using the GIF framework. It provides a centralized, professional approach to
experiment parameter management, ensuring reproducibility and easy modification of
experimental conditions.

The configuration is organized into logical sections covering all aspects of the experiment:
- Data generation parameters for realistic synthetic light curves
- Neural network architecture specifications for the SNN
- Training parameters for continual learning
- Simulation parameters for neuromorphic hardware modeling
- Analysis and output parameters for scientific validation

Key Design Principles:
=====================

**Reproducibility**: All parameters are explicitly defined and documented, enabling
exact reproduction of experimental results.

**Flexibility**: Parameters can be easily modified to explore different experimental
conditions without changing the main code.

**Scientific Rigor**: Default values are based on realistic astronomical and
neuromorphic hardware specifications.

**Professional Standards**: Configuration follows software engineering best practices
for complex scientific experiments.

Usage Example:
=============

    from applications.poc_exoplanet.config_exo import EXP_CONFIG
    
    # Access configuration parameters
    num_samples = EXP_CONFIG["data"]["num_training_samples"]
    learning_rate = EXP_CONFIG["training"]["learning_rate"]
    
    # Use in component initialization
    du_core = DU_Core_V1(
        input_size=EXP_CONFIG["architecture"]["input_size"],
        hidden_sizes=EXP_CONFIG["architecture"]["hidden_sizes"],
        output_size=EXP_CONFIG["architecture"]["output_size"]
    )

This configuration enables systematic exploration of the GIF framework's capabilities
while maintaining scientific rigor and experimental reproducibility.
"""

from typing import List, Dict, Any
import os

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

EXP_CONFIG: Dict[str, Any] = {
    
    # =========================================================================
    # DATA GENERATION PARAMETERS
    # =========================================================================
    "data": {
        # Training and evaluation dataset sizes
        "num_training_samples": 1000,      # Number of training light curves
        "num_test_samples": 200,           # Number of test light curves
        "validation_split": 0.2,           # Fraction of training data for validation
        
        # Astronomical observation parameters (based on TESS mission)
        "observation_duration": 27.4,      # Days (one TESS sector)
        "cadence_minutes": 30.0,           # Time between observations (minutes)
        
        # Realism parameters for synthetic data generation
        "stellar_variability_amplitude": 0.001,    # 0.1% stellar variability
        "instrumental_noise_level": 0.0001,        # 100 ppm instrumental noise
        "noise_color_alpha": 1.0,                  # Pink noise (1/f)
        
        # Data preprocessing
        "normalize_flux": True,             # Normalize flux to zero mean, unit variance
        "random_seed": 42,                 # For reproducible data generation
    },
    
    # =========================================================================
    # ENCODER CONFIGURATION
    # =========================================================================
    "encoder": {
        "type": "LightCurveEncoder",
        "threshold": 1e-4,                 # Delta modulation sensitivity threshold
        "normalize_input": True,           # Enable input normalization
        "auto_calibrate": True,            # Auto-calibrate threshold on sample data
    },
    
    # =========================================================================
    # SNN ARCHITECTURE PARAMETERS
    # =========================================================================
    "architecture": {
        # Network topology
        "input_size": 2,                   # Two channels from delta modulation
        "hidden_sizes": [64, 32],          # Hidden layer sizes
        "output_size": 2,                  # Binary classification (Planet/No Planet)
        
        # LIF neuron parameters
        "neuron_beta": 0.95,               # Membrane potential decay (0 < beta ≤ 1)
        "neuron_threshold": 1.0,           # Firing threshold
        "recurrent": False,                # No recurrent connections (for now)
        
        # RTL (Real-Time Learning) parameters
        "enable_plasticity": True,         # Enable synaptic plasticity
        "plasticity_rule": "STDP",         # Spike-Timing Dependent Plasticity
        "stdp_learning_rate_ltp": 0.01,    # Long-term potentiation rate
        "stdp_learning_rate_ltd": 0.005,   # Long-term depression rate
        "stdp_tau_ltp": 20.0,              # LTP time constant (ms)
        "stdp_tau_ltd": 20.0,              # LTD time constant (ms)
        
        # Episodic memory parameters
        "enable_memory": True,             # Enable episodic memory
        "memory_capacity": 10000,          # Maximum stored experiences
    },
    
    # =========================================================================
    # DECODER CONFIGURATION
    # =========================================================================
    "decoder": {
        "type": "ExoplanetDecoder",
        "classification_threshold": 0.5,   # Decision threshold for binary classification
        "regression_output_size": 1,       # Number of regression parameters
        "default_mode": "classification",  # Default decoding mode
    },
    
    # =========================================================================
    # TRAINING PARAMETERS
    # =========================================================================
    "training": {
        # Optimization parameters
        "learning_rate": 1e-3,             # Adam optimizer learning rate
        "weight_decay": 1e-5,              # L2 regularization
        "optimizer": "Adam",               # Optimizer type
        
        # Training schedule
        "num_epochs": 20,                  # Number of training epochs
        "batch_size": 32,                  # Training batch size
        "eval_frequency": 5,               # Evaluate every N epochs
        
        # Continual learning parameters
        "memory_batch_size": 32,           # GEM memory batch size
        "task_id": "exoplanet_detection",  # Task identifier
        
        # Loss function
        "loss_function": "CrossEntropyLoss", # For classification
        "class_weights": None,             # Balanced classes (None = equal weights)
    },
    
    # =========================================================================
    # NEUROMORPHIC SIMULATION PARAMETERS
    # =========================================================================
    "simulation": {
        # Hardware simulation parameters
        "energy_per_synop": 2.5e-11,      # Joules per synaptic operation (Loihi 2)
        "simulation_time_steps": 100,     # Number of time steps per sample
        
        # Event-driven processing
        "enable_event_driven": True,      # Use event-driven simulation
        "spike_threshold": 0.1,           # Minimum spike value to process
    },
    
    # =========================================================================
    # ANALYSIS AND OUTPUT PARAMETERS
    # =========================================================================
    "analysis": {
        # Performance metrics
        "metrics": [
            "accuracy", "precision", "recall", "f1_score",
            "true_positive_rate", "false_positive_rate",
            "energy_consumption", "spike_efficiency"
        ],
        
        # Visualization parameters
        "generate_plots": True,            # Generate performance plots
        "plot_format": "png",              # Plot file format
        "plot_dpi": 300,                   # Plot resolution for publication
        
        # Output directories
        "results_dir": "results/poc_exoplanet",
        "plots_dir": "results/poc_exoplanet/plots",
        "logs_dir": "results/poc_exoplanet/logs",
        
        # Export formats
        "export_formats": ["csv", "json"], # Data export formats
        "save_model": True,                # Save trained model
        "save_config": True,               # Save configuration with results
    },
    
    # =========================================================================
    # EXPERIMENTAL METADATA
    # =========================================================================
    "metadata": {
        "experiment_name": "exoplanet_detection_poc",
        "description": "Proof-of-concept for neuromorphic exoplanet detection using GIF framework",
        "version": "1.0.0",
        "author": "GIF Development Team",
        "date_created": "2025-07-09",
        
        # Scientific context
        "domain": "astronomy",
        "task_type": "binary_classification",
        "data_type": "time_series",
        "hardware_target": "neuromorphic",
        
        # Framework information
        "gif_version": "1.0.0",
        "framework_components": [
            "RealisticExoplanetGenerator",
            "LightCurveEncoder", 
            "DU_Core_V1",
            "ExoplanetDecoder",
            "NeuromorphicSimulator",
            "Continual_Trainer"
        ]
    }
}


# =============================================================================
# CONFIGURATION VALIDATION AND UTILITIES
# =============================================================================

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate the experiment configuration for completeness and consistency.
    
    Args:
        config: Configuration dictionary to validate.
        
    Returns:
        bool: True if configuration is valid.
        
    Raises:
        ValueError: If configuration is invalid.
    """
    required_sections = ["data", "architecture", "training", "simulation", "analysis"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate data parameters
    data_config = config["data"]
    if data_config["num_training_samples"] <= 0:
        raise ValueError("num_training_samples must be positive")
    
    if data_config["num_test_samples"] <= 0:
        raise ValueError("num_test_samples must be positive")
    
    # Validate architecture parameters
    arch_config = config["architecture"]
    if arch_config["input_size"] != 2:
        raise ValueError("input_size must be 2 for delta modulation encoding")
    
    if arch_config["output_size"] != 2:
        raise ValueError("output_size must be 2 for binary classification")
    
    # Validate training parameters
    train_config = config["training"]
    if train_config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    
    if train_config["num_epochs"] <= 0:
        raise ValueError("num_epochs must be positive")
    
    return True


def get_config() -> Dict[str, Any]:
    """
    Get the validated experiment configuration.
    
    Returns:
        Dict[str, Any]: Validated configuration dictionary.
    """
    validate_config(EXP_CONFIG)
    return EXP_CONFIG


def create_output_directories(config: Dict[str, Any]) -> None:
    """
    Create output directories specified in the configuration.
    
    Args:
        config: Configuration dictionary containing directory paths.
    """
    analysis_config = config["analysis"]
    
    directories = [
        analysis_config["results_dir"],
        analysis_config["plots_dir"],
        analysis_config["logs_dir"]
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# Validate configuration on import
validate_config(EXP_CONFIG)
