#!/usr/bin/env python3
"""
Analysis script for exoplanet detection POC results.

This script analyzes the training results, creates visualizations,
and generates comprehensive performance reports.

Usage:
    python analyze_results.py [results_dir]

Author: GIF Framework Team
Date: 2025-07-11
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from applications.poc_exoplanet.analysis.performance_analyzer import analyze_experiment_results


def setup_logging() -> logging.Logger:
    """Set up logging for the analysis script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def main(results_dir: Optional[str] = None):
    """
    Main analysis function.
    
    Args:
        results_dir: Directory containing experimental results
    """
    logger = setup_logging()
    
    # Default results directory
    if results_dir is None:
        results_dir = "results/poc_exoplanet"
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_path}")
        return
    
    logger.info(f"Analyzing results from: {results_path}")
    
    try:
        # Perform comprehensive analysis
        analysis = analyze_experiment_results(str(results_path), logger)
        
        # Print summary to console
        print("\n" + "="*80)
        print("EXOPLANET DETECTION POC - ANALYSIS SUMMARY")
        print("="*80)
        
        summary = analysis['summary']
        
        # Training summary
        if 'training' in summary:
            training = summary['training']
            print("\n📊 TRAINING PERFORMANCE:")
            print(f"  Final Loss: {training.get('final_loss', 'N/A'):.4f}" if training.get('final_loss') else "  Final Loss: N/A")
            print(f"  Final Accuracy: {training.get('final_accuracy', 'N/A'):.2%}" if training.get('final_accuracy') else "  Final Accuracy: N/A")
            print(f"  Best Accuracy: {training.get('best_accuracy', 'N/A'):.2%}" if training.get('best_accuracy') else "  Best Accuracy: N/A")
            print(f"  Total Training Time: {training.get('total_training_time', 'N/A'):.2f}s" if training.get('total_training_time') else "  Total Training Time: N/A")
            print(f"  Convergence Stability: {training.get('convergence_stability', 'N/A'):.6f}" if training.get('convergence_stability') else "  Convergence Stability: N/A")
        
        # Evaluation summary
        if 'evaluation' in summary:
            evaluation = summary['evaluation']
            print("\n🎯 EVALUATION PERFORMANCE:")
            print(f"  Test Accuracy: {evaluation.get('test_accuracy', 'N/A'):.2%}" if evaluation.get('test_accuracy') else "  Test Accuracy: N/A")
            print(f"  Test Precision: {evaluation.get('test_precision', 'N/A'):.2%}" if evaluation.get('test_precision') else "  Test Precision: N/A")
            print(f"  Test Recall: {evaluation.get('test_recall', 'N/A'):.2%}" if evaluation.get('test_recall') else "  Test Recall: N/A")
            print(f"  Test F1-Score: {evaluation.get('test_f1_score', 'N/A'):.2%}" if evaluation.get('test_f1_score') else "  Test F1-Score: N/A")
        
        # Neuromorphic performance
        if 'evaluation' in summary:
            evaluation = summary['evaluation']
            print("\n⚡ NEUROMORPHIC PERFORMANCE:")
            print(f"  Energy Efficiency: {evaluation.get('energy_efficiency', 'N/A'):.2e} J/sample" if evaluation.get('energy_efficiency') else "  Energy Efficiency: N/A")
            print(f"  Processing Speed: {evaluation.get('processing_speed', 'N/A'):.4f} s/sample" if evaluation.get('processing_speed') else "  Processing Speed: N/A")
        
        # Configuration
        if 'configuration' in summary:
            config = summary['configuration']
            print("\n⚙️  CONFIGURATION:")
            arch = config.get('architecture', {})
            training_params = config.get('training_params', {})
            print(f"  Architecture: {arch.get('input_size', 'N/A')} → {arch.get('hidden_sizes', 'N/A')} → {arch.get('output_size', 'N/A')}")
            print(f"  Learning Rate: {training_params.get('learning_rate', 'N/A')}")
            print(f"  Device: {training_params.get('device', 'N/A')}")
        
        # Files generated
        print("\n📁 GENERATED FILES:")
        generated_files = []
        
        if (results_path / "training_metrics.png").exists():
            generated_files.append("training_metrics.png")
        if (results_path / "training_times.png").exists():
            generated_files.append("training_times.png")
        if (results_path / "performance_report.md").exists():
            generated_files.append("performance_report.md")
        
        for file in generated_files:
            print(f"  ✓ {file}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    # Parse command line arguments
    results_dir = None
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    
    main(results_dir)
