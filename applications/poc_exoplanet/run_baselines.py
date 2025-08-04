"""
Baseline Experiments Runner for Exoplanet Detection
===================================================

This script runs the CNN and Transformer baseline experiments on the same synthetic
exoplanet dataset used by the GIF-DU model, ensuring fair comparison for scientific
analysis and Table IV generation.

The script generates identical synthetic data, trains both baseline models, and
collects comprehensive performance metrics for comparison with the neuromorphic
GIF-DU approach.

Key Features:
============

**Fair Comparison**: Uses identical synthetic dataset as GIF-DU experiment
**Comprehensive Metrics**: Collects classification and efficiency metrics
**Structured Logging**: Saves results in format compatible with analysis notebook
**Energy Modeling**: Estimates GPU energy consumption for traditional approaches

Integration:
===========

Results are saved in structured format for automatic ingestion by the analysis
notebook, enabling direct comparison with GIF-DU results and Table IV generation.
"""

import sys
import os
import time
import json
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import polars as pl
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import data generator and baseline models
from data_generators.exoplanet_generator import RealisticExoplanetGenerator
from applications.poc_exoplanet.baselines.cnn_baseline import CNNBaseline
from applications.poc_exoplanet.baselines.transformer_baseline import TransformerBaseline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BaselineExperiments')

def generate_synthetic_data(config: Dict[str, Any]) -> Tuple[List, List]:
    """
    Generate synthetic exoplanet data identical to GIF-DU experiment.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Tuple of (training_data, test_data) lists
    """
    logger.info("Generating synthetic exoplanet datasets...")
    
    # Initialize generator with same parameters as main experiment
    generator = RealisticExoplanetGenerator(
        seed=config['data']['random_seed'],
        stellar_variability_amplitude=config['data']['stellar_variability_amplitude'],
        instrumental_noise_level=config['data']['instrumental_noise_level'],
        noise_color_alpha=config['data']['noise_color_alpha']
    )
    
    # Generate training data
    logger.info(f"Generating {config['data']['num_training_samples']} training samples...")
    training_data = []
    for i in range(config['data']['num_training_samples']):
        # Generate light curve
        light_curve = generator.generate(
            observation_duration=config['data']['observation_duration'],
            cadence_minutes=config['data']['cadence_minutes']
        )

        # Create binary label (1 if planet present, 0 if not)
        # Use transit depth as indicator
        transit_depth = 1.0 - light_curve["flux"].min()
        label = 1 if transit_depth > 0.005 else 0  # 0.5% transit depth threshold

        training_data.append((light_curve, label))

        if (i + 1) % 100 == 0:
            logger.info(f"  Generated {i + 1}/{config['data']['num_training_samples']} training samples")

    # Generate test data
    logger.info(f"Generating {config['data']['num_test_samples']} test samples...")
    test_data = []
    for i in range(config['data']['num_test_samples']):
        # Generate light curve
        light_curve = generator.generate(
            observation_duration=config['data']['observation_duration'],
            cadence_minutes=config['data']['cadence_minutes']
        )

        # Create binary label (1 if planet present, 0 if not)
        transit_depth = 1.0 - light_curve["flux"].min()
        label = 1 if transit_depth > 0.005 else 0  # 0.5% transit depth threshold

        test_data.append((light_curve, label))

        if (i + 1) % 50 == 0:
            logger.info(f"  Generated {i + 1}/{config['data']['num_test_samples']} test samples")
    
    # Log dataset statistics
    train_planets = sum(1 for _, label in training_data if label == 1)
    test_planets = sum(1 for _, label in test_data if label == 1)
    
    logger.info(f"Training dataset: {len(training_data)} samples ({train_planets} planets, {len(training_data) - train_planets} non-planets)")
    logger.info(f"Test dataset: {len(test_data)} samples ({test_planets} planets, {len(test_data) - test_planets} non-planets)")
    
    return training_data, test_data

def run_cnn_baseline(training_data: List, test_data: List, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run CNN baseline experiment.
    
    Args:
        training_data: Training dataset
        test_data: Test dataset
        config: Experiment configuration
        
    Returns:
        Dictionary with CNN results
    """
    logger.info("=" * 80)
    logger.info("RUNNING CNN BASELINE EXPERIMENT")
    logger.info("=" * 80)
    
    # Calculate input length from generated data
    sample_light_curve = training_data[0][0]  # First training sample
    input_length = len(sample_light_curve)

    # Initialize CNN model
    cnn_model = CNNBaseline(
        input_length=input_length,
        num_classes=2
    )

    logger.info(f"CNN Model initialized:")
    logger.info(f"  Input length: {input_length}")
    logger.info(f"  Conv layers: {cnn_model.conv_layers}")
    logger.info(f"  FC layers: {cnn_model.fc_layers}")
    logger.info(f"  Model size: {cnn_model.model_size_mb:.1f} MB")
    
    # Train model
    start_time = time.time()
    training_logs = cnn_model.train(
        training_data=training_data,
        epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate']
    )
    training_time = time.time() - start_time
    
    logger.info(f"CNN training completed in {training_time:.1f} seconds")
    
    # Evaluate model
    evaluation_results = cnn_model.evaluate(test_data)
    
    # Combine results
    results = {
        'model_type': 'CNN',
        'training_time': training_time,
        'training_logs': training_logs,
        'evaluation_results': evaluation_results,
        'model_info': cnn_model.get_model_info()
    }
    
    logger.info(f"CNN Results:")
    logger.info(f"  Accuracy: {evaluation_results['accuracy']:.3f}")
    logger.info(f"  F1-Score: {evaluation_results['f1_score']:.3f}")
    logger.info(f"  Energy per sample: {evaluation_results['energy_per_sample']:.2e} J")
    
    return results

def run_transformer_baseline(training_data: List, test_data: List, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Transformer baseline experiment.
    
    Args:
        training_data: Training dataset
        test_data: Test dataset
        config: Experiment configuration
        
    Returns:
        Dictionary with Transformer results
    """
    logger.info("=" * 80)
    logger.info("RUNNING TRANSFORMER BASELINE EXPERIMENT")
    logger.info("=" * 80)
    
    # Calculate input length from generated data
    sample_light_curve = training_data[0][0]  # First training sample
    input_length = len(sample_light_curve)

    # Initialize Transformer model
    transformer_model = TransformerBaseline(
        sequence_length=input_length,
        num_classes=2,
        d_model=config['transformer']['d_model'],
        num_heads=config['transformer']['num_heads'],
        num_layers=config['transformer']['num_layers']
    )

    logger.info(f"Transformer Model initialized:")
    logger.info(f"  Input length: {input_length}")
    logger.info(f"  d_model: {config['transformer']['d_model']}")
    logger.info(f"  Num heads: {config['transformer']['num_heads']}")
    logger.info(f"  Num layers: {config['transformer']['num_layers']}")
    logger.info(f"  Model size: {transformer_model.model_size_mb:.1f} MB")
    
    # Train model
    start_time = time.time()
    training_logs = transformer_model.train(
        training_data=training_data,
        epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate']
    )
    training_time = time.time() - start_time
    
    logger.info(f"Transformer training completed in {training_time:.1f} seconds")
    
    # Evaluate model
    evaluation_results = transformer_model.evaluate(test_data)
    
    # Combine results
    results = {
        'model_type': 'Transformer',
        'training_time': training_time,
        'training_logs': training_logs,
        'evaluation_results': evaluation_results,
        'model_info': transformer_model.get_model_info()
    }
    
    logger.info(f"Transformer Results:")
    logger.info(f"  Accuracy: {evaluation_results['accuracy']:.3f}")
    logger.info(f"  F1-Score: {evaluation_results['f1_score']:.3f}")
    logger.info(f"  Energy per sample: {evaluation_results['energy_per_sample']:.2e} J")
    
    return results

def save_results(cnn_results: Dict[str, Any], transformer_results: Dict[str, Any], output_dir: Path) -> None:
    """
    Save baseline results to structured files for analysis.
    
    Args:
        cnn_results: CNN experiment results
        transformer_results: Transformer experiment results
        output_dir: Output directory for results
    """
    output_dir.mkdir(exist_ok=True)
    
    # Save CNN results
    cnn_file = output_dir / 'cnn_baseline_results.json'
    with open(cnn_file, 'w') as f:
        json.dump(cnn_results, f, indent=2)
    logger.info(f"CNN results saved to {cnn_file}")
    
    # Save Transformer results
    transformer_file = output_dir / 'transformer_baseline_results.json'
    with open(transformer_file, 'w') as f:
        json.dump(transformer_results, f, indent=2)
    logger.info(f"Transformer results saved to {transformer_file}")
    
    # Save combined summary
    summary = {
        'experiment_timestamp': time.strftime('%Y-%m-%d_%H-%M-%S'),
        'cnn_summary': {
            'accuracy': cnn_results['evaluation_results']['accuracy'],
            'f1_score': cnn_results['evaluation_results']['f1_score'],
            'energy_per_sample': cnn_results['evaluation_results']['energy_per_sample'],
            'model_size_mb': cnn_results['model_info']['model_size_mb'],
            'training_time': cnn_results['training_time']
        },
        'transformer_summary': {
            'accuracy': transformer_results['evaluation_results']['accuracy'],
            'f1_score': transformer_results['evaluation_results']['f1_score'],
            'energy_per_sample': transformer_results['evaluation_results']['energy_per_sample'],
            'model_size_mb': transformer_results['model_info']['model_size_mb'],
            'training_time': transformer_results['training_time']
        }
    }
    
    summary_file = output_dir / 'baseline_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_file}")

def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("BASELINE EXPERIMENTS FOR EXOPLANET DETECTION")
    logger.info("=" * 80)
    
    # Configuration matching main experiment
    config = {
        'data': {
            'num_training_samples': 1000,
            'num_test_samples': 200,
            'observation_duration': 27.4,
            'cadence_minutes': 30.0,
            'random_seed': 42,
            'stellar_variability_amplitude': 0.001,
            'instrumental_noise_level': 0.0001,
            'noise_color_alpha': 1.0
        },
        'training': {
            'epochs': 20,
            'learning_rate': 0.001
        },
        'cnn': {
            'conv_layers': [32, 64, 128],
            'fc_layers': [256, 128]
        },
        'transformer': {
            'd_model': 128,
            'num_heads': 8,
            'num_layers': 4
        }
    }
    
    # Generate synthetic data
    training_data, test_data = generate_synthetic_data(config)
    
    # Run baseline experiments
    cnn_results = run_cnn_baseline(training_data, test_data, config)
    transformer_results = run_transformer_baseline(training_data, test_data, config)
    
    # Save results
    output_dir = Path('applications/poc_exoplanet/data/baseline_results')
    save_results(cnn_results, transformer_results, output_dir)
    
    logger.info("=" * 80)
    logger.info("BASELINE EXPERIMENTS COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
