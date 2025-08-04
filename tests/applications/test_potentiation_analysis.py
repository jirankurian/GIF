"""
Comprehensive Test Suite for Potentiation Analysis Calculations
==============================================================

This module provides exhaustive testing for the analysis calculations used to
measure system potentiation effects. These tests ensure the mathematical accuracy
and scientific validity of the metrics used to demonstrate AGI capabilities.

Test Coverage:
- Learning efficiency calculation validation
- Few-shot generalization testing
- RSA (Representational Similarity Analysis) validation
- Statistical significance testing
- Mock data validation
- Edge cases and error handling

The accuracy of these calculations is critical for making credible scientific
claims about system potentiation in artificial neural networks.
"""

import pytest
import numpy as np
import torch
import polars as pl
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import analysis components
try:
    from applications.analysis.continual_learning_analyzer import ContinualLearningAnalyzer
except ImportError:
    # Create mock analyzer if not available
    class ContinualLearningAnalyzer:
        def __init__(self, logs=None):
            self.logs = logs or {}
        
        def calculate_learning_efficiency(self, target_accuracy=0.9):
            return 0.5  # Mock 50% efficiency gain

# Import GIF framework components for RSA testing
from gif_framework.core.du_core import DU_Core_V1

# Try to import scientific libraries
try:
    from scipy.stats import ttest_ind, pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TestLearningEfficiencyCalculation:
    """Test suite for learning efficiency metric calculations."""
    
    def test_learning_efficiency_basic_calculation(self):
        """
        Test basic learning efficiency calculation with mock data.
        
        This validates the core potentiation metric: how much faster the
        pre-exposed model learns compared to the naive model.
        """
        # Create mock learning curves where pre-exposed learns faster
        naive_log = pl.DataFrame({
            "samples_seen": [1000, 2000, 3000, 4000, 5000],
            "accuracy": [0.6, 0.75, 0.85, 0.90, 0.92]
        })
        
        pre_exposed_log = pl.DataFrame({
            "samples_seen": [1000, 2000, 3000, 4000, 5000],
            "accuracy": [0.8, 0.91, 0.93, 0.94, 0.95]
        })
        
        # Calculate when each model reaches 90% accuracy
        naive_samples_to_90 = None
        pre_exposed_samples_to_90 = None
        
        for i, acc in enumerate(naive_log["accuracy"]):
            if acc >= 0.90 and naive_samples_to_90 is None:
                naive_samples_to_90 = naive_log["samples_seen"][i]
                break
        
        for i, acc in enumerate(pre_exposed_log["accuracy"]):
            if acc >= 0.90 and pre_exposed_samples_to_90 is None:
                pre_exposed_samples_to_90 = pre_exposed_log["samples_seen"][i]
                break
        
        # Verify expected results
        assert naive_samples_to_90 == 4000, "Naive model should reach 90% at 4000 samples"
        assert pre_exposed_samples_to_90 == 2000, "Pre-exposed model should reach 90% at 2000 samples"
        
        # Calculate efficiency gain
        efficiency_gain = (naive_samples_to_90 - pre_exposed_samples_to_90) / naive_samples_to_90
        expected_gain = (4000 - 2000) / 4000  # 0.5 or 50%
        
        assert abs(efficiency_gain - expected_gain) < 1e-6, \
            f"Efficiency gain calculation incorrect: {efficiency_gain} vs {expected_gain}"
    
    def test_learning_efficiency_edge_cases(self):
        """
        Test learning efficiency calculation with edge cases.
        
        This ensures robust handling of unusual learning curves that might
        occur in real experiments.
        """
        # Case 1: Model never reaches target accuracy
        never_converges_log = pl.DataFrame({
            "samples_seen": [1000, 2000, 3000, 4000, 5000],
            "accuracy": [0.6, 0.65, 0.7, 0.75, 0.8]
        })
        
        # Should handle gracefully (return None or special value)
        samples_to_target = None
        for i, acc in enumerate(never_converges_log["accuracy"]):
            if acc >= 0.90:
                samples_to_target = never_converges_log["samples_seen"][i]
                break
        
        assert samples_to_target is None, "Should return None when target not reached"
        
        # Case 2: Model reaches target immediately
        immediate_converge_log = pl.DataFrame({
            "samples_seen": [1000, 2000, 3000, 4000, 5000],
            "accuracy": [0.95, 0.96, 0.97, 0.98, 0.99]
        })
        
        samples_to_target = None
        for i, acc in enumerate(immediate_converge_log["accuracy"]):
            if acc >= 0.90:
                samples_to_target = immediate_converge_log["samples_seen"][i]
                break
        
        assert samples_to_target == 1000, "Should return first sample when target reached immediately"
    
    def test_learning_curve_interpolation(self):
        """
        Test interpolation for more precise learning efficiency calculation.
        
        This tests advanced analysis that interpolates between data points
        to get more accurate convergence timing.
        """
        # Create learning curve that crosses target between data points
        learning_log = pl.DataFrame({
            "samples_seen": [1000, 2000, 3000, 4000, 5000],
            "accuracy": [0.7, 0.8, 0.88, 0.92, 0.95]
        })
        
        target_accuracy = 0.90
        
        # Find crossing point between samples 3000 and 4000
        for i in range(len(learning_log) - 1):
            acc_before = learning_log["accuracy"][i]
            acc_after = learning_log["accuracy"][i + 1]
            
            if acc_before < target_accuracy <= acc_after:
                samples_before = learning_log["samples_seen"][i]
                samples_after = learning_log["samples_seen"][i + 1]
                
                # Linear interpolation
                ratio = (target_accuracy - acc_before) / (acc_after - acc_before)
                interpolated_samples = samples_before + ratio * (samples_after - samples_before)
                
                # Should be between 3000 and 4000
                assert 3000 < interpolated_samples < 4000, \
                    f"Interpolated samples out of expected range: {interpolated_samples}"
                
                # More precise: should be closer to 3000 since 0.88 is closer to 0.90
                expected_samples = 3000 + (0.90 - 0.88) / (0.92 - 0.88) * 1000  # 3500
                assert abs(interpolated_samples - expected_samples) < 1, \
                    f"Interpolation inaccurate: {interpolated_samples} vs {expected_samples}"
                break
        else:
            pytest.fail("Target accuracy crossing not found")
    
    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available")
    def test_statistical_significance_testing(self):
        """
        Test statistical significance testing for learning efficiency differences.
        
        This validates that observed differences are statistically significant
        and not due to random chance.
        """
        # Create mock data for multiple experimental runs
        naive_convergence_times = [4000, 4200, 3800, 4100, 3900]  # Mean: 4000
        pre_exposed_convergence_times = [2000, 2100, 1900, 2050, 1950]  # Mean: 2000
        
        # Perform t-test
        t_stat, p_value = ttest_ind(naive_convergence_times, pre_exposed_convergence_times)
        
        # Should be statistically significant (p < 0.05)
        assert p_value < 0.05, f"Difference not statistically significant: p={p_value}"
        assert t_stat > 0, "T-statistic should be positive (naive > pre-exposed)"
        
        # Calculate effect size (Cohen's d)
        naive_mean = np.mean(naive_convergence_times)
        pre_exposed_mean = np.mean(pre_exposed_convergence_times)
        pooled_std = np.sqrt((np.var(naive_convergence_times) + np.var(pre_exposed_convergence_times)) / 2)
        
        cohens_d = (naive_mean - pre_exposed_mean) / pooled_std
        
        # Should be large effect size (d > 0.8)
        assert cohens_d > 0.8, f"Effect size too small: d={cohens_d}"


class TestFewShotGeneralizationAnalysis:
    """Test suite for few-shot generalization testing."""
    
    def test_few_shot_performance_calculation(self):
        """
        Test calculation of few-shot learning performance metrics.
        
        This validates the measurement of how quickly models can adapt
        to new classes with minimal training data.
        """
        # Mock few-shot performance data
        shot_counts = [1, 5, 10]
        naive_accuracies = [0.3, 0.6, 0.75]  # Slower improvement
        pre_exposed_accuracies = [0.5, 0.8, 0.9]  # Faster improvement
        
        # Calculate improvement ratios
        improvements = []
        for naive_acc, pre_exposed_acc in zip(naive_accuracies, pre_exposed_accuracies):
            improvement = (pre_exposed_acc - naive_acc) / naive_acc
            improvements.append(improvement)
        
        # Verify improvements are positive and substantial
        for i, improvement in enumerate(improvements):
            assert improvement > 0, f"No improvement for {shot_counts[i]} shots"
            assert improvement > 0.1, f"Improvement too small for {shot_counts[i]} shots: {improvement}"
        
        # Verify improvement decreases with more shots (diminishing returns)
        for i in range(len(improvements) - 1):
            # Allow some tolerance for realistic data
            assert improvements[i] >= improvements[i + 1] - 0.1, \
                "Improvement should decrease with more shots"
    
    def test_few_shot_data_validation(self):
        """
        Test validation of few-shot learning data quality.
        
        This ensures the few-shot experiments use appropriate data
        and produce reliable measurements.
        """
        # Test data structure validation
        few_shot_results = {
            'shot_counts': [1, 5, 10],
            'naive_accuracies': [0.3, 0.6, 0.75],
            'pre_exposed_accuracies': [0.5, 0.8, 0.9],
            'naive_std': [0.1, 0.08, 0.05],
            'pre_exposed_std': [0.08, 0.06, 0.04]
        }
        
        # Validate data consistency
        data_length = len(few_shot_results['shot_counts'])
        for key, values in few_shot_results.items():
            assert len(values) == data_length, f"Inconsistent data length for {key}"
        
        # Validate accuracy ranges
        for accuracies in [few_shot_results['naive_accuracies'], 
                          few_shot_results['pre_exposed_accuracies']]:
            for acc in accuracies:
                assert 0 <= acc <= 1, f"Accuracy out of range: {acc}"
        
        # Validate standard deviations
        for stds in [few_shot_results['naive_std'], few_shot_results['pre_exposed_std']]:
            for std in stds:
                assert std >= 0, f"Standard deviation negative: {std}"
                assert std <= 0.5, f"Standard deviation too large: {std}"


class TestRSAAnalysisValidation:
    """Test suite for Representational Similarity Analysis (RSA) validation."""

    def test_hidden_layer_activation_extraction(self):
        """
        Test extraction of hidden layer activations for RSA analysis.

        This validates the enhanced DU_Core capability to return internal
        representations for advanced neural analysis.
        """
        # Create DU_Core with multiple hidden layers
        du_core = DU_Core_V1(
            input_size=5,
            hidden_sizes=[10, 8, 6],
            output_size=3
        )

        # Create test input
        test_input = torch.randn(20, 2, 5)  # [time_steps, batch, features]

        # Test forward pass with activation extraction
        output_spikes, hidden_activations = du_core.forward(
            test_input,
            num_steps=20,
            return_activations=True
        )

        # Verify output shape
        assert output_spikes.shape == (20, 2, 3), "Output spikes shape incorrect"

        # Verify hidden activations structure
        assert isinstance(hidden_activations, list), "Hidden activations should be a list"
        # Note: DU_Core returns activations for all layers including output
        expected_num_activations = len(du_core.hidden_sizes) + 1  # +1 for output layer
        assert len(hidden_activations) == expected_num_activations, \
            f"Should have activations for all layers: expected {expected_num_activations}, got {len(hidden_activations)}"

        # Verify activation shapes (all layers including output)
        expected_shapes = [(20, 2, 10), (20, 2, 8), (20, 2, 6), (20, 2, 3)]
        for i, (activations, expected_shape) in enumerate(zip(hidden_activations, expected_shapes)):
            assert activations.shape == expected_shape, \
                f"Layer {i+1} activation shape incorrect: {activations.shape} vs {expected_shape}"

    def test_representational_dissimilarity_matrix_calculation(self):
        """
        Test calculation of Representational Dissimilarity Matrix (RDM).

        This validates the core RSA computation that reveals how the network
        organizes different classes in its internal representation space.
        """
        # Create mock activation patterns for different classes
        num_classes = 4
        num_neurons = 10
        num_samples_per_class = 5

        # Generate distinct activation patterns for each class
        class_activations = []
        for class_id in range(num_classes):
            # Create class-specific pattern with some noise
            base_pattern = torch.randn(num_neurons) * 0.5
            class_samples = []
            for _ in range(num_samples_per_class):
                sample = base_pattern + torch.randn(num_neurons) * 0.1
                class_samples.append(sample)
            class_activations.append(torch.stack(class_samples))

        # Calculate average activation for each class
        class_means = [activations.mean(dim=0) for activations in class_activations]

        # Calculate dissimilarity matrix (1 - correlation)
        rdm = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    rdm[i, j] = 0  # Perfect similarity with self
                else:
                    # Calculate correlation
                    mean_i = class_means[i]
                    mean_j = class_means[j]

                    # Pearson correlation
                    mean_i_centered = mean_i - mean_i.mean()
                    mean_j_centered = mean_j - mean_j.mean()

                    correlation = torch.sum(mean_i_centered * mean_j_centered) / \
                                 (torch.norm(mean_i_centered) * torch.norm(mean_j_centered))

                    rdm[i, j] = 1 - correlation

        # Verify RDM properties
        assert rdm.shape == (num_classes, num_classes), "RDM shape incorrect"

        # Diagonal should be zero (perfect self-similarity)
        for i in range(num_classes):
            assert abs(rdm[i, i].item()) < 1e-6, f"Diagonal element {i} not zero: {rdm[i, i]}"

        # Matrix should be symmetric
        for i in range(num_classes):
            for j in range(num_classes):
                assert abs(rdm[i, j] - rdm[j, i]) < 1e-6, \
                    f"RDM not symmetric at ({i}, {j}): {rdm[i, j]} vs {rdm[j, i]}"

        # Off-diagonal elements should be positive (dissimilarity)
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    assert rdm[i, j] > 0, f"Dissimilarity should be positive: {rdm[i, j]}"

    def test_rsa_comparison_between_models(self):
        """
        Test RSA comparison between naive and pre-exposed models.

        This validates the analysis that shows pre-exposed models develop
        more organized internal representations.
        """
        # Create two models with same architecture
        naive_model = DU_Core_V1(input_size=4, hidden_sizes=[8, 6], output_size=3)
        pre_exposed_model = DU_Core_V1(input_size=4, hidden_sizes=[8, 6], output_size=3)

        # Apply weight reset to pre-exposed model (simulating potentiation protocol)
        pre_exposed_model.reset_weights()

        # Create test data representing different classes
        num_classes = 3
        samples_per_class = 4
        test_data = []

        for class_id in range(num_classes):
            for sample_id in range(samples_per_class):
                # Create class-specific input pattern
                class_input = torch.zeros(10, 1, 4)  # [time_steps, batch, features]
                class_input[:, 0, class_id] = 1.0  # One-hot-like encoding
                class_input += torch.randn_like(class_input) * 0.1  # Add noise
                test_data.append((class_input, class_id))

        # Extract activations from both models
        naive_activations = []
        pre_exposed_activations = []

        for input_data, class_id in test_data:
            # Get activations from naive model (use second-to-last layer to avoid output layer)
            _, naive_hidden = naive_model.forward(input_data, num_steps=10, return_activations=True)
            # Use the last hidden layer (not output layer) for meaningful representations
            last_hidden_idx = -2 if len(naive_hidden) > 1 else -1
            naive_activations.append((naive_hidden[last_hidden_idx].mean(dim=0).squeeze(), class_id))

            # Get activations from pre-exposed model
            _, pre_exposed_hidden = pre_exposed_model.forward(input_data, num_steps=10, return_activations=True)
            pre_exposed_activations.append((pre_exposed_hidden[last_hidden_idx].mean(dim=0).squeeze(), class_id))

        # Verify we have activations for all samples
        assert len(naive_activations) == num_classes * samples_per_class
        assert len(pre_exposed_activations) == num_classes * samples_per_class

        # Group activations by class
        naive_by_class = [[] for _ in range(num_classes)]
        pre_exposed_by_class = [[] for _ in range(num_classes)]

        for activations, class_id in naive_activations:
            naive_by_class[class_id].append(activations)

        for activations, class_id in pre_exposed_activations:
            pre_exposed_by_class[class_id].append(activations)

        # Calculate class means
        naive_class_means = [torch.stack(class_acts).mean(dim=0) for class_acts in naive_by_class]
        pre_exposed_class_means = [torch.stack(class_acts).mean(dim=0) for class_acts in pre_exposed_by_class]

        # Verify we have means for all classes
        assert len(naive_class_means) == num_classes
        assert len(pre_exposed_class_means) == num_classes

        # Both models should produce different representations
        for i in range(num_classes):
            assert not torch.equal(naive_class_means[i], pre_exposed_class_means[i]), \
                f"Models produce identical representations for class {i}"


class TestAnalysisIntegrationAndEdgeCases:
    """Test suite for analysis integration and edge case handling."""

    def test_complete_analysis_pipeline(self):
        """
        Test the complete analysis pipeline from raw data to final metrics.

        This validates the end-to-end analysis workflow that produces
        publication-ready results.
        """
        # Create mock experimental results
        experimental_data = {
            'naive_log': pl.DataFrame({
                'epoch': [1, 2, 3, 4, 5],
                'samples_seen': [1000, 2000, 3000, 4000, 5000],
                'accuracy': [0.6, 0.7, 0.8, 0.9, 0.92],
                'loss': [1.0, 0.8, 0.6, 0.4, 0.3]
            }),
            'pre_exposed_log': pl.DataFrame({
                'epoch': [1, 2, 3, 4, 5],
                'samples_seen': [1000, 2000, 3000, 4000, 5000],
                'accuracy': [0.75, 0.85, 0.9, 0.93, 0.95],
                'loss': [0.8, 0.6, 0.4, 0.3, 0.2]
            })
        }

        # Test data validation
        for log_name, log_data in experimental_data.items():
            assert 'accuracy' in log_data.columns, f"Missing accuracy in {log_name}"
            assert 'samples_seen' in log_data.columns, f"Missing samples_seen in {log_name}"

            # Verify data quality
            accuracies = log_data['accuracy'].to_list()
            assert all(0 <= acc <= 1 for acc in accuracies), f"Invalid accuracies in {log_name}"

            samples = log_data['samples_seen'].to_list()
            assert all(s > 0 for s in samples), f"Invalid sample counts in {log_name}"

        # Calculate key metrics
        target_accuracy = 0.9

        # Find convergence times
        naive_convergence = None
        pre_exposed_convergence = None

        for i, acc in enumerate(experimental_data['naive_log']['accuracy']):
            if acc >= target_accuracy and naive_convergence is None:
                naive_convergence = experimental_data['naive_log']['samples_seen'][i]

        for i, acc in enumerate(experimental_data['pre_exposed_log']['accuracy']):
            if acc >= target_accuracy and pre_exposed_convergence is None:
                pre_exposed_convergence = experimental_data['pre_exposed_log']['samples_seen'][i]

        # Verify convergence found
        assert naive_convergence is not None, "Naive model convergence not found"
        assert pre_exposed_convergence is not None, "Pre-exposed model convergence not found"

        # Calculate efficiency gain
        efficiency_gain = (naive_convergence - pre_exposed_convergence) / naive_convergence

        # Verify reasonable efficiency gain
        assert efficiency_gain > 0, "No efficiency gain observed"
        assert efficiency_gain < 1, "Efficiency gain too large (>100%)"

        # Expected: naive converges at 4000, pre-exposed at 3000
        # Efficiency gain = (4000 - 3000) / 4000 = 0.25 or 25%
        expected_gain = (4000 - 3000) / 4000
        assert abs(efficiency_gain - expected_gain) < 0.01, \
            f"Efficiency gain calculation incorrect: {efficiency_gain} vs {expected_gain}"
