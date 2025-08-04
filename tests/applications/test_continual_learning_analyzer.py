"""
Comprehensive Test Suite for Continual Learning Analyzer
========================================================

This module provides exhaustive testing for the ContinualLearningAnalyzer class,
ensuring correct implementation of all continual learning metrics and proper
integration with the GIF framework's continual learning infrastructure.

Test Coverage:
- Analyzer initialization and data validation
- Average accuracy calculation
- Forgetting measure computation
- Forward transfer measurement
- Summary plot generation
- Comprehensive reporting
- Integration with trainer statistics
- Edge cases and error handling

The tests use realistic continual learning scenarios to validate that the
analyzer correctly implements the mathematical formulas from the continual
learning literature and provides accurate scientific measurements.
"""

import pytest
import numpy as np
import polars as pl
from pathlib import Path
import tempfile
import os
from typing import Dict, List, Any

# Import the analyzer class
from applications.analysis.continual_learning_analyzer import ContinualLearningAnalyzer

# Try to import matplotlib, skip visualization tests if not available
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TestContinualLearningAnalyzerInitialization:
    """Test suite for analyzer initialization and data validation."""
    
    def test_valid_initialization(self):
        """Test successful initialization with valid experimental data."""
        # Create valid experimental data
        data = {
            'training_task_id': ['task_A', 'task_A', 'task_B', 'task_B', 'task_B'],
            'evaluation_task_id': ['task_A', 'task_A', 'task_A', 'task_B', 'task_B'],
            'epoch': [1, 10, 5, 1, 10],
            'accuracy': [0.6, 0.9, 0.85, 0.4, 0.8]
        }
        df = pl.DataFrame(data)
        
        # Initialize analyzer
        analyzer = ContinualLearningAnalyzer(df)
        
        # Verify initialization
        assert analyzer.experiment_logs.height == 5
        assert set(analyzer.task_ids) == {'task_A', 'task_B'}
        assert analyzer.training_sequence == ['task_A', 'task_B']
    
    def test_initialization_with_invalid_type(self):
        """Test that initialization fails with non-DataFrame input."""
        with pytest.raises(TypeError, match="experiment_logs must be a Polars DataFrame"):
            ContinualLearningAnalyzer("not_a_dataframe")
    
    def test_initialization_missing_columns(self):
        """Test that initialization fails with missing required columns."""
        # Missing 'accuracy' column
        data = {
            'training_task_id': ['task_A'],
            'evaluation_task_id': ['task_A'],
            'epoch': [1]
        }
        df = pl.DataFrame(data)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            ContinualLearningAnalyzer(df)
    
    def test_initialization_invalid_accuracy_range(self):
        """Test that initialization fails with accuracy values outside [0,1] range."""
        # Accuracy > 1.0
        data = {
            'training_task_id': ['task_A'],
            'evaluation_task_id': ['task_A'],
            'epoch': [1],
            'accuracy': [1.5]  # Invalid accuracy
        }
        df = pl.DataFrame(data)
        
        with pytest.raises(ValueError, match="accuracy values must be in range"):
            ContinualLearningAnalyzer(df)


class TestContinualLearningAnalyzerMetrics:
    """Test suite for continual learning metric calculations."""
    
    @pytest.fixture
    def sample_analyzer(self):
        """Create a sample analyzer with realistic continual learning data."""
        # Simulate a 3-task sequential learning experiment following proper protocol
        data = {
            'training_task_id': [
                # Training on task_A (only evaluate task_A)
                'task_A', 'task_A', 'task_A',
                # Training on task_B (evaluate on A and B)
                'task_B', 'task_B', 'task_B', 'task_B',
                # Training on task_C (evaluate on A, B, and C)
                'task_C', 'task_C', 'task_C', 'task_C', 'task_C', 'task_C'
            ],
            'evaluation_task_id': [
                # Training on task_A (only evaluate A)
                'task_A', 'task_A', 'task_A',
                # Training on task_B (evaluate A and B)
                'task_A', 'task_A', 'task_B', 'task_B',
                # Training on task_C (evaluate A, B, and C)
                'task_A', 'task_A', 'task_B', 'task_B', 'task_C', 'task_C'
            ],
            'epoch': [
                # Training on task_A
                1, 5, 10,
                # Training on task_B
                1, 10, 1, 10,
                # Training on task_C
                1, 10, 1, 10, 1, 10
            ],
            'accuracy': [
                # Training on task_A: learns from 0.5 to 0.9
                0.5, 0.7, 0.9,
                # Training on task_B: A drops slightly, B learns
                0.85, 0.8, 0.3, 0.85,
                # Training on task_C: A drops more, B stable, C learns
                0.75, 0.7, 0.8, 0.82, 0.2, 0.88
            ]
        }
        df = pl.DataFrame(data)
        return ContinualLearningAnalyzer(df)
    
    def test_calculate_average_accuracy(self, sample_analyzer):
        """Test average accuracy calculation."""
        avg_accuracy = sample_analyzer.calculate_average_accuracy()
        
        # Final accuracies: task_A=0.7, task_B=0.82, task_C=0.88
        expected_avg = (0.7 + 0.82 + 0.88) / 3
        assert abs(avg_accuracy - expected_avg) < 0.001
    
    def test_calculate_forgetting_measure_task_a(self, sample_analyzer):
        """Test forgetting measure calculation for task A."""
        forgetting = sample_analyzer.calculate_forgetting_measure('task_A')
        
        # Task A: max accuracy = 0.9, final accuracy = 0.7
        expected_forgetting = 0.9 - 0.7
        assert abs(forgetting - expected_forgetting) < 0.001
    
    def test_calculate_forgetting_measure_task_b(self, sample_analyzer):
        """Test forgetting measure calculation for task B."""
        forgetting = sample_analyzer.calculate_forgetting_measure('task_B')
        
        # Task B: max accuracy = 0.85, final accuracy = 0.82
        expected_forgetting = 0.85 - 0.82
        assert abs(forgetting - expected_forgetting) < 0.001
    
    def test_calculate_forward_transfer_first_task(self, sample_analyzer):
        """Test forward transfer for first task (should be 0)."""
        transfer = sample_analyzer.calculate_forward_transfer('task_A')
        
        # First task has no prior experience to compare against
        assert transfer == 0.0
    
    def test_calculate_forward_transfer_second_task(self, sample_analyzer):
        """Test forward transfer calculation for second task."""
        transfer = sample_analyzer.calculate_forward_transfer('task_B')
        
        # Task B first epoch: 0.3, Task A first epoch (naive): 0.5
        # Note: This is a simplified baseline comparison
        expected_transfer = 0.3 - 0.5  # Negative transfer in this case
        assert abs(transfer - expected_transfer) < 0.001
    
    def test_forgetting_measure_invalid_task(self, sample_analyzer):
        """Test forgetting measure with invalid task ID."""
        with pytest.raises(ValueError, match="Task 'invalid_task' not found"):
            sample_analyzer.calculate_forgetting_measure('invalid_task')
    
    def test_forward_transfer_invalid_task(self, sample_analyzer):
        """Test forward transfer with invalid task ID."""
        with pytest.raises(ValueError, match="Task 'invalid_task' not found"):
            sample_analyzer.calculate_forward_transfer('invalid_task')


class TestContinualLearningAnalyzerVisualization:
    """Test suite for visualization and reporting functionality."""
    
    @pytest.fixture
    def sample_analyzer(self):
        """Create a sample analyzer for visualization tests."""
        data = {
            'training_task_id': ['task_A', 'task_A', 'task_B', 'task_B', 'task_B'],
            'evaluation_task_id': ['task_A', 'task_A', 'task_A', 'task_B', 'task_B'],
            'epoch': [1, 10, 5, 1, 10],
            'accuracy': [0.6, 0.9, 0.85, 0.4, 0.8]
        }
        df = pl.DataFrame(data)
        return ContinualLearningAnalyzer(df)
    
    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
    def test_generate_summary_plot_no_save(self, sample_analyzer):
        """Test summary plot generation without saving."""
        # This should not raise an exception
        sample_analyzer.generate_summary_plot()

        # Close the plot to prevent display issues in testing
        plt.close('all')

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
    def test_generate_summary_plot_with_save(self, sample_analyzer):
        """Test summary plot generation with file saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_plot.png")

            # Generate and save plot
            sample_analyzer.generate_summary_plot(save_path=save_path)

            # Verify file was created
            assert os.path.exists(save_path)

            # Close the plot
            plt.close('all')
    
    def test_generate_comprehensive_report(self, sample_analyzer):
        """Test comprehensive report generation."""
        report = sample_analyzer.generate_comprehensive_report()
        
        # Verify report structure
        assert 'overall_metrics' in report
        assert 'task_specific_metrics' in report
        assert 'experimental_info' in report
        assert 'recommendations' in report
        
        # Verify overall metrics
        overall = report['overall_metrics']
        assert 'average_accuracy' in overall
        assert 'mean_forgetting' in overall
        assert 'total_tasks' in overall
        
        # Verify task-specific metrics
        task_metrics = report['task_specific_metrics']
        assert 'task_A' in task_metrics
        assert 'task_B' in task_metrics
        
        # Verify each task has forgetting and transfer measures
        for task_id, metrics in task_metrics.items():
            assert 'forgetting_measure' in metrics
            assert 'forward_transfer' in metrics


class TestContinualLearningAnalyzerIntegration:
    """Test suite for integration with GIF framework components."""
    
    def test_from_trainer_stats_basic(self):
        """Test creation from trainer statistics."""
        # Mock trainer statistics
        trainer_stats = {
            'task_losses': {
                'task_A': [0.8, 0.6, 0.4, 0.2],
                'task_B': [0.9, 0.7, 0.5, 0.3]
            }
        }
        
        # Create analyzer from trainer stats
        analyzer = ContinualLearningAnalyzer.from_trainer_stats(trainer_stats)
        
        # Verify analyzer was created successfully
        assert len(analyzer.task_ids) == 2
        assert 'task_A' in analyzer.task_ids
        assert 'task_B' in analyzer.task_ids
    
    def test_from_trainer_stats_invalid_format(self):
        """Test error handling for invalid trainer statistics format."""
        # Missing required key
        invalid_stats = {'other_data': [1, 2, 3]}
        
        with pytest.raises(ValueError, match="trainer_stats must contain 'task_losses'"):
            ContinualLearningAnalyzer.from_trainer_stats(invalid_stats)


class TestContinualLearningAnalyzerEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_single_task_experiment(self):
        """Test analyzer with single-task experiment."""
        data = {
            'training_task_id': ['task_A', 'task_A'],
            'evaluation_task_id': ['task_A', 'task_A'],
            'epoch': [1, 10],
            'accuracy': [0.5, 0.9]
        }
        df = pl.DataFrame(data)
        analyzer = ContinualLearningAnalyzer(df)
        
        # Should work without errors
        avg_accuracy = analyzer.calculate_average_accuracy()
        assert avg_accuracy == 0.9
        
        # Forgetting should be 0 for single task
        forgetting = analyzer.calculate_forgetting_measure('task_A')
        assert forgetting == 0.0
        
        # Forward transfer should be 0 for first task
        transfer = analyzer.calculate_forward_transfer('task_A')
        assert transfer == 0.0
    
    def test_perfect_learning_no_forgetting(self):
        """Test analyzer with perfect learning scenario (no forgetting)."""
        data = {
            'training_task_id': ['task_A', 'task_A', 'task_B', 'task_B', 'task_B'],
            'evaluation_task_id': ['task_A', 'task_A', 'task_A', 'task_B', 'task_B'],
            'epoch': [1, 10, 5, 1, 10],
            'accuracy': [0.5, 1.0, 1.0, 0.6, 1.0]  # No forgetting on task_A
        }
        df = pl.DataFrame(data)
        analyzer = ContinualLearningAnalyzer(df)
        
        # Forgetting should be 0
        forgetting = analyzer.calculate_forgetting_measure('task_A')
        assert forgetting == 0.0
        
        # Average accuracy should be 1.0
        avg_accuracy = analyzer.calculate_average_accuracy()
        assert avg_accuracy == 1.0
    
    def test_empty_dataframe(self):
        """Test analyzer with empty DataFrame."""
        # Create empty DataFrame with correct columns
        df = pl.DataFrame({
            'training_task_id': [],
            'evaluation_task_id': [],
            'epoch': [],
            'accuracy': []
        })
        
        # Should raise error during validation
        with pytest.raises(ValueError):
            ContinualLearningAnalyzer(df)


# Additional tests specified in Prompt 3.5
def test_forgetting_measure_calculation():
    """Validates that the forgetting measure is calculated correctly."""
    # Create a mock log DataFrame showing performance on Task_A
    # dropping after Task_B is learned.
    mock_logs_data = {
        "training_task_id": ["task_A", "task_A", "task_B", "task_B", "task_B", "task_B"],
        "evaluation_task_id": ["task_A", "task_A", "task_A", "task_A", "task_B", "task_B"],
        "epoch": [1, 10, 1, 10, 1, 10],
        "accuracy": [0.5, 0.95, 0.85, 0.75, 0.6, 0.9]  # Accuracy on A was 95%, then dropped to 75%
    }
    mock_logs_df = pl.DataFrame(mock_logs_data)

    analyzer = ContinualLearningAnalyzer(mock_logs_df)

    # The forgetting should be max_accuracy - final_accuracy
    forgetting = analyzer.calculate_forgetting_measure(target_task_id="task_A")

    assert pytest.approx(forgetting) == 0.20  # 0.95 - 0.75
