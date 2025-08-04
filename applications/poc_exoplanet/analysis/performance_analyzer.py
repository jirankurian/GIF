"""
Performance analysis and visualization tools for the exoplanet detection POC.

This module provides tools for analyzing training performance, creating
visualizations, and generating comprehensive reports.

Author: GIF Framework Team
Date: 2025-07-11
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis and visualization tool.
    
    This class provides methods for analyzing training logs, creating
    visualizations, and generating performance reports.
    """
    
    def __init__(self, results_dir: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the performance analyzer.
        
        Args:
            results_dir: Directory containing experimental results
            logger: Optional logger instance
        """
        self.results_dir = Path(results_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_results(self) -> Dict[str, Any]:
        """
        Load experimental results from the results directory.
        
        Returns:
            Dictionary containing all experimental data
        """
        results = {}
        
        # Load training logs
        training_logs_path = self.results_dir / "training_logs.json"
        if training_logs_path.exists():
            with open(training_logs_path, 'r') as f:
                results['training_logs'] = json.load(f)
        
        # Load evaluation results
        eval_results_path = self.results_dir / "evaluation_results.json"
        if eval_results_path.exists():
            with open(eval_results_path, 'r') as f:
                results['evaluation_results'] = json.load(f)
        
        # Load configuration
        config_path = self.results_dir / "experiment_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                results['config'] = json.load(f)
        
        return results
    
    def create_training_plots(self, results: Dict[str, Any], save_plots: bool = True) -> Dict[str, plt.Figure]:
        """
        Create comprehensive training performance plots.
        
        Args:
            results: Experimental results dictionary
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        training_logs = results.get('training_logs', {})
        
        if not training_logs:
            self.logger.warning("No training logs found for plotting")
            return figures
        
        # Extract training data
        epochs = training_logs.get('epoch', [])
        losses = training_logs.get('loss', [])
        accuracies = training_logs.get('accuracy', [])
        training_times = training_logs.get('training_time', [])
        
        # Plot 1: Loss and Accuracy over Epochs
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss plot
        ax1.plot(epochs, losses, 'b-', marker='o', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, accuracies, 'g-', marker='s', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        figures['training_metrics'] = fig1
        
        if save_plots:
            fig1.savefig(self.results_dir / "training_metrics.png", dpi=300, bbox_inches='tight')
        
        # Plot 2: Training Time Analysis
        if training_times:
            fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            ax.bar(epochs, training_times, alpha=0.7, color='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Training Time (seconds)')
            ax.set_title('Training Time per Epoch')
            ax.grid(True, alpha=0.3)
            
            # Add average line
            avg_time = np.mean(training_times)
            ax.axhline(y=avg_time, color='red', linestyle='--', 
                      label=f'Average: {avg_time:.2f}s')
            ax.legend()
            
            plt.tight_layout()
            figures['training_times'] = fig2
            
            if save_plots:
                fig2.savefig(self.results_dir / "training_times.png", dpi=300, bbox_inches='tight')
        
        return figures
    
    def create_performance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive performance summary.
        
        Args:
            results: Experimental results dictionary
            
        Returns:
            Performance summary dictionary
        """
        summary = {}
        
        # Training performance
        training_logs = results.get('training_logs', {})
        if training_logs:
            losses = training_logs.get('loss', [])
            accuracies = training_logs.get('accuracy', [])
            training_times = training_logs.get('training_time', [])
            
            summary['training'] = {
                'final_loss': losses[-1] if losses else None,
                'final_accuracy': accuracies[-1] if accuracies else None,
                'best_accuracy': max(accuracies) if accuracies else None,
                'worst_accuracy': min(accuracies) if accuracies else None,
                'total_training_time': sum(training_times) if training_times else None,
                'avg_epoch_time': np.mean(training_times) if training_times else None,
                'convergence_stability': self._calculate_convergence_stability(losses)
            }
        
        # Evaluation performance
        eval_results = results.get('evaluation_results', {})
        if eval_results:
            summary['evaluation'] = {
                'test_accuracy': eval_results.get('accuracy'),
                'test_precision': eval_results.get('precision'),
                'test_recall': eval_results.get('recall'),
                'test_f1_score': eval_results.get('f1_score'),
                'energy_efficiency': eval_results.get('avg_energy_per_sample'),
                'processing_speed': eval_results.get('avg_processing_time')
            }
        
        # Configuration summary
        config = results.get('config', {})
        if config:
            summary['configuration'] = {
                'architecture': config.get('architecture', {}),
                'training_params': config.get('training', {}),
                'data_params': config.get('data', {})
            }
        
        return summary
    
    def _calculate_convergence_stability(self, losses: List[float]) -> float:
        """
        Calculate convergence stability metric.
        
        Args:
            losses: List of loss values over epochs
            
        Returns:
            Stability score (lower is more stable)
        """
        if len(losses) < 3:
            return float('inf')
        
        # Calculate variance in the last half of training
        second_half = losses[len(losses)//2:]
        return float(np.var(second_half))
    
    def create_comparison_plot(self, multiple_results: Dict[str, Dict[str, Any]], 
                             metric: str = 'loss', save_plot: bool = True) -> plt.Figure:
        """
        Create comparison plots for multiple experiments.
        
        Args:
            multiple_results: Dictionary of experiment_name -> results
            metric: Metric to compare ('loss', 'accuracy', etc.)
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for exp_name, results in multiple_results.items():
            training_logs = results.get('training_logs', {})
            epochs = training_logs.get('epoch', [])
            values = training_logs.get(metric, [])
            
            if epochs and values:
                ax.plot(epochs, values, marker='o', linewidth=2, 
                       label=exp_name, markersize=4)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison Across Experiments')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            fig.savefig(self.results_dir / f"{metric}_comparison.png", 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_report(self, results: Dict[str, Any], output_file: str = "performance_report.md") -> str:
        """
        Generate a comprehensive markdown performance report.
        
        Args:
            results: Experimental results dictionary
            output_file: Output filename for the report
            
        Returns:
            Report content as string
        """
        summary = self.create_performance_summary(results)
        
        report_lines = [
            "# Exoplanet Detection POC - Performance Report",
            "",
            f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
        ]
        
        # Training summary
        if 'training' in summary:
            training = summary['training']
            report_lines.extend([
                "### Training Performance",
                "",
                f"- **Final Loss:** {training.get('final_loss', 'N/A'):.4f}" if training.get('final_loss') else "- **Final Loss:** N/A",
                f"- **Final Accuracy:** {training.get('final_accuracy', 'N/A'):.2%}" if training.get('final_accuracy') else "- **Final Accuracy:** N/A",
                f"- **Best Accuracy:** {training.get('best_accuracy', 'N/A'):.2%}" if training.get('best_accuracy') else "- **Best Accuracy:** N/A",
                f"- **Total Training Time:** {training.get('total_training_time', 'N/A'):.2f}s" if training.get('total_training_time') else "- **Total Training Time:** N/A",
                f"- **Convergence Stability:** {training.get('convergence_stability', 'N/A'):.6f}" if training.get('convergence_stability') else "- **Convergence Stability:** N/A",
                "",
            ])
        
        # Evaluation summary
        if 'evaluation' in summary:
            evaluation = summary['evaluation']
            report_lines.extend([
                "### Evaluation Performance",
                "",
                f"- **Test Accuracy:** {evaluation.get('test_accuracy', 'N/A'):.2%}" if evaluation.get('test_accuracy') else "- **Test Accuracy:** N/A",
                f"- **Test Precision:** {evaluation.get('test_precision', 'N/A'):.2%}" if evaluation.get('test_precision') else "- **Test Precision:** N/A",
                f"- **Test Recall:** {evaluation.get('test_recall', 'N/A'):.2%}" if evaluation.get('test_recall') else "- **Test Recall:** N/A",
                f"- **Test F1-Score:** {evaluation.get('test_f1_score', 'N/A'):.2%}" if evaluation.get('test_f1_score') else "- **Test F1-Score:** N/A",
                "",
            ])
        
        # Neuromorphic performance
        if 'evaluation' in summary:
            evaluation = summary['evaluation']
            report_lines.extend([
                "### Neuromorphic Performance",
                "",
                f"- **Energy Efficiency:** {evaluation.get('energy_efficiency', 'N/A'):.2e} J/sample" if evaluation.get('energy_efficiency') else "- **Energy Efficiency:** N/A",
                f"- **Processing Speed:** {evaluation.get('processing_speed', 'N/A'):.4f} s/sample" if evaluation.get('processing_speed') else "- **Processing Speed:** N/A",
                "",
            ])
        
        # Configuration details
        if 'configuration' in summary:
            config = summary['configuration']
            report_lines.extend([
                "### Configuration",
                "",
                "#### Architecture",
                f"- Input Size: {config.get('architecture', {}).get('input_size', 'N/A')}",
                f"- Hidden Sizes: {config.get('architecture', {}).get('hidden_sizes', 'N/A')}",
                f"- Output Size: {config.get('architecture', {}).get('output_size', 'N/A')}",
                "",
                "#### Training Parameters",
                f"- Learning Rate: {config.get('training_params', {}).get('learning_rate', 'N/A')}",
                f"- Epochs: {config.get('training_params', {}).get('num_epochs', 'N/A')}",
                f"- Device: {config.get('training_params', {}).get('device', 'N/A')}",
                "",
            ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_path = self.results_dir / output_file
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Performance report saved to {report_path}")
        
        return report_content


def analyze_experiment_results(results_dir: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze experiment results.
    
    Args:
        results_dir: Directory containing experimental results
        logger: Optional logger instance
        
    Returns:
        Analysis results dictionary
    """
    analyzer = PerformanceAnalyzer(results_dir, logger)
    results = analyzer.load_results()
    
    # Create visualizations
    figures = analyzer.create_training_plots(results)
    
    # Generate summary
    summary = analyzer.create_performance_summary(results)
    
    # Generate report
    report = analyzer.generate_report(results)
    
    return {
        'results': results,
        'summary': summary,
        'figures': figures,
        'report': report
    }
