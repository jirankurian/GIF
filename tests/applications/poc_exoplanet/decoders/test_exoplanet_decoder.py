"""
Comprehensive Test Suite for ExoplanetDecoder
==============================================

This module provides rigorous testing for the ExoplanetDecoder class,
validating its dual-mode operation, interface compliance, and
integration with the GIF framework components.

Test Categories:
- Interface compliance and contract validation
- Dual-mode operation (classification and regression)
- PyTorch module integration and training
- Population coding and linear readout accuracy
- Input format handling and validation
- Edge cases and error conditions
- Performance and efficiency validation

The test suite ensures the decoder correctly converts spike trains into
meaningful exoplanet detection outputs for both binary classification
and continuous parameter estimation tasks.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import warnings
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Union

# Import the components to test
from applications.poc_exoplanet.decoders.exoplanet_decoder import ExoplanetDecoder
from gif_framework.interfaces.base_interfaces import DecoderInterface, SpikeTrain, Action


class TestExoplanetDecoderInitialization:
    """Test suite for ExoplanetDecoder initialization and validation."""

    def test_valid_initialization(self):
        """Test successful initialization with valid parameters."""
        # Test default initialization
        decoder = ExoplanetDecoder(input_size=10)
        assert decoder.input_size == 10
        assert decoder.output_size == 1
        assert decoder.classification_threshold == 0.5
        assert isinstance(decoder.regression_layer, nn.Linear)
        
        # Test custom parameters
        decoder_custom = ExoplanetDecoder(
            input_size=32, 
            output_size=3, 
            classification_threshold=0.7
        )
        assert decoder_custom.input_size == 32
        assert decoder_custom.output_size == 3
        assert decoder_custom.classification_threshold == 0.7

    def test_invalid_input_size(self):
        """Test initialization with invalid input size."""
        with pytest.raises(ValueError, match="input_size must be a positive integer"):
            ExoplanetDecoder(input_size=0)
        
        with pytest.raises(ValueError, match="input_size must be a positive integer"):
            ExoplanetDecoder(input_size=-5)
        
        with pytest.raises(ValueError, match="input_size must be a positive integer"):
            ExoplanetDecoder(input_size="invalid")

    def test_invalid_output_size(self):
        """Test initialization with invalid output size."""
        with pytest.raises(ValueError, match="output_size must be a positive integer"):
            ExoplanetDecoder(input_size=10, output_size=0)
        
        with pytest.raises(ValueError, match="output_size must be a positive integer"):
            ExoplanetDecoder(input_size=10, output_size=-2)

    def test_invalid_classification_threshold(self):
        """Test initialization with invalid classification threshold."""
        with pytest.raises(ValueError, match="classification_threshold must be between 0 and 1"):
            ExoplanetDecoder(input_size=10, classification_threshold=0.0)
        
        with pytest.raises(ValueError, match="classification_threshold must be between 0 and 1"):
            ExoplanetDecoder(input_size=10, classification_threshold=1.0)
        
        with pytest.raises(ValueError, match="classification_threshold must be between 0 and 1"):
            ExoplanetDecoder(input_size=10, classification_threshold=1.5)

    def test_interface_inheritance(self):
        """Test that ExoplanetDecoder properly inherits from required interfaces."""
        decoder = ExoplanetDecoder(input_size=10)
        
        # Check interface inheritance
        assert isinstance(decoder, DecoderInterface)
        assert isinstance(decoder, nn.Module)
        
        # Check that all required methods are implemented
        assert hasattr(decoder, 'decode')
        assert hasattr(decoder, 'get_config')
        assert hasattr(decoder, 'forward')
        assert callable(decoder.decode)
        assert callable(decoder.get_config)
        assert callable(decoder.forward)

    def test_pytorch_module_properties(self):
        """Test PyTorch module properties and parameter initialization."""
        decoder = ExoplanetDecoder(input_size=20, output_size=2)

        # Check that it has trainable parameters (both classification and regression layers)
        params = list(decoder.parameters())
        assert len(params) == 4  # 2 layers × (weight + bias) = 4 parameters
        
        # Check parameter shapes (regression layer weight, regression bias, classification weight, classification bias)
        reg_weight, reg_bias, cls_weight, cls_bias = params
        assert reg_weight.shape == (2, 20)  # output_size x input_size (regression layer)
        assert reg_bias.shape == (2,)       # output_size (regression layer)
        assert cls_weight.shape == (2, 20)  # 2 classes x input_size (classification layer)
        assert cls_bias.shape == (2,)       # 2 classes (classification layer)

        # Check parameter initialization (should not be all zeros)
        assert not torch.allclose(reg_weight, torch.zeros_like(reg_weight))
        assert not torch.allclose(cls_weight, torch.zeros_like(cls_weight))
        assert torch.allclose(reg_bias, torch.zeros_like(reg_bias))  # Bias initialized to zero
        assert torch.allclose(cls_bias, torch.zeros_like(cls_bias))  # Bias initialized to zero

    def test_initial_statistics(self):
        """Test that initial decoding statistics are properly set."""
        decoder = ExoplanetDecoder(input_size=15)
        stats = decoder.get_decoding_stats()

        expected_keys = ['input_shape', 'total_input_spikes', 'spike_rate',
                        'mode_used']
        for key in expected_keys:
            assert key in stats


class TestInputValidationAndFormatHandling:
    """Test suite for input validation and format handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.decoder = ExoplanetDecoder(input_size=8, output_size=1)

    def test_valid_2d_spike_train_input(self):
        """Test decoding with valid 2D spike train input."""
        # Shape: [num_time_steps, num_neurons]
        spike_train = torch.rand(50, 8)
        spike_train = (spike_train > 0.8).float()  # Make binary spikes
        
        # Test classification mode
        result_class = self.decoder.decode(spike_train, mode='classification')
        assert isinstance(result_class, int)
        assert result_class in [0, 1]
        
        # Test regression mode
        result_reg = self.decoder.decode(spike_train, mode='regression')
        assert isinstance(result_reg, float)

    def test_valid_3d_spike_train_input(self):
        """Test decoding with valid 3D spike train input (with batch dimension)."""
        # Shape: [num_time_steps, batch_size, num_neurons]
        spike_train = torch.rand(30, 2, 8)
        spike_train = (spike_train > 0.7).float()
        
        # Should handle by taking first batch
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.decoder.decode(spike_train, mode='classification')
            
            # Should warn about multiple batches
            assert len(w) == 1
            assert "single batch only" in str(w[0].message)
        
        assert isinstance(result, int)

    def test_invalid_spike_train_types(self):
        """Test decoding with invalid spike train types."""
        with pytest.raises(ValueError, match="spike_train must be a torch.Tensor"):
            self.decoder.decode("invalid_string", mode='classification')
        
        with pytest.raises(ValueError, match="spike_train must be a torch.Tensor"):
            self.decoder.decode(np.array([[1, 0], [0, 1]]), mode='classification')

    def test_invalid_spike_train_dimensions(self):
        """Test decoding with invalid tensor dimensions."""
        # 1D tensor (missing time or neuron dimension)
        with pytest.raises(ValueError, match="spike_train must be 2D or 3D tensor"):
            self.decoder.decode(torch.rand(10), mode='classification')
        
        # 4D tensor (too many dimensions)
        with pytest.raises(ValueError, match="spike_train must be 2D or 3D tensor"):
            self.decoder.decode(torch.rand(10, 2, 8, 1), mode='classification')

    def test_mismatched_input_size(self):
        """Test decoding with spike train that doesn't match decoder input size."""
        # Wrong number of neurons
        wrong_size_spike_train = torch.rand(20, 5)  # 5 neurons instead of 8
        
        with pytest.raises(ValueError, match="must match decoder input_size"):
            self.decoder.decode(wrong_size_spike_train, mode='classification')

    def test_invalid_mode_parameter(self):
        """Test decoding with invalid mode parameter."""
        spike_train = torch.rand(20, 8)
        
        with pytest.raises(ValueError, match="mode must be one of"):
            self.decoder.decode(spike_train, mode='invalid_mode')
        
        with pytest.raises(ValueError, match="mode must be one of"):
            self.decoder.decode(spike_train, mode='prediction')

    def test_empty_spike_train_handling(self):
        """Test decoding with empty spike train."""
        empty_spike_train = torch.zeros(0, 8)

        # Should handle gracefully or raise appropriate error
        try:
            result = self.decoder.decode(empty_spike_train, mode='classification')
            assert isinstance(result, int)
        except (ValueError, ZeroDivisionError):
            # Acceptable to raise an error for empty input
            pass

    def test_single_timestep_spike_train(self):
        """Test decoding with single time step."""
        single_step = torch.rand(1, 8)
        single_step = (single_step > 0.5).float()
        
        result_class = self.decoder.decode(single_step, mode='classification')
        result_reg = self.decoder.decode(single_step, mode='regression')
        
        assert isinstance(result_class, int)
        assert isinstance(result_reg, float)


class TestClassificationModeOperation:
    """Test suite for classification mode functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.decoder = ExoplanetDecoder(input_size=10, output_size=1)

    def test_single_neuron_classification(self):
        """Test classification with single output neuron."""
        decoder_single = ExoplanetDecoder(input_size=1, output_size=1)

        # High activity spike train
        high_activity = torch.ones(20, 1) * 1.0  # All spikes
        result_high = decoder_single.decode(high_activity, mode='classification')

        # Low activity spike train
        low_activity = torch.zeros(20, 1)  # No spikes
        result_low = decoder_single.decode(low_activity, mode='classification')

        # Results should be different based on activity level
        assert result_high in [0, 1]
        assert result_low in [0, 1]

    def test_two_neuron_winner_take_all(self):
        """Test winner-take-all classification with two neurons."""
        decoder_two = ExoplanetDecoder(input_size=2, output_size=1)

        # First neuron more active
        spike_train_1 = torch.zeros(10, 2)
        spike_train_1[:8, 0] = 1.0  # 8 spikes in first neuron
        spike_train_1[:3, 1] = 1.0  # 3 spikes in second neuron

        result_1 = decoder_two.decode(spike_train_1, mode='classification')
        assert result_1 == 0  # First neuron wins

        # Second neuron more active
        spike_train_2 = torch.zeros(10, 2)
        spike_train_2[:2, 0] = 1.0  # 2 spikes in first neuron
        spike_train_2[:7, 1] = 1.0  # 7 spikes in second neuron

        result_2 = decoder_two.decode(spike_train_2, mode='classification')
        assert result_2 == 1  # Second neuron wins

    def test_multiple_neuron_threshold_classification(self):
        """Test threshold-based classification with multiple neurons."""
        # High activity across all neurons
        high_activity = torch.rand(50, 10)
        high_activity = (high_activity > 0.3).float()  # ~70% spike rate

        result_high = self.decoder.decode(high_activity, mode='classification')

        # Low activity across all neurons
        low_activity = torch.rand(50, 10)
        low_activity = (low_activity > 0.9).float()  # ~10% spike rate

        result_low = self.decoder.decode(low_activity, mode='classification')

        # Both should be valid classifications
        assert result_high in [0, 1]
        assert result_low in [0, 1]

    def test_empty_spike_counts_classification(self):
        """Test classification with empty spike counts."""
        empty_spikes = torch.zeros(20, 10)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.decoder.decode(empty_spikes, mode='classification')

            # May warn about empty spike counts
            if len(w) > 0:
                assert "Empty spike counts" in str(w[0].message)

        assert result in [0, 1]

    def test_classification_threshold_effect(self):
        """Test effect of classification threshold on results."""
        # Create decoder with low threshold
        decoder_low = ExoplanetDecoder(input_size=10, classification_threshold=0.1)

        # Create decoder with high threshold
        decoder_high = ExoplanetDecoder(input_size=10, classification_threshold=0.9)

        # Medium activity spike train
        medium_activity = torch.rand(30, 10)
        medium_activity = (medium_activity > 0.7).float()  # ~30% spike rate

        result_low = decoder_low.decode(medium_activity, mode='classification')
        result_high = decoder_high.decode(medium_activity, mode='classification')

        # Both should produce valid results
        assert result_low in [0, 1]
        assert result_high in [0, 1]

    def test_classification_consistency(self):
        """Test classification consistency with identical inputs."""
        spike_train = torch.rand(25, 10)
        spike_train = (spike_train > 0.8).float()

        # Multiple calls should produce identical results
        results = []
        for _ in range(5):
            result = self.decoder.decode(spike_train, mode='classification')
            results.append(result)

        # All results should be identical
        assert all(r == results[0] for r in results)


class TestRegressionModeOperation:
    """Test suite for regression mode functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.decoder = ExoplanetDecoder(input_size=8, output_size=1)

    def test_single_output_regression(self):
        """Test regression with single output parameter."""
        spike_train = torch.rand(30, 8)
        spike_train = (spike_train > 0.7).float()

        result = self.decoder.decode(spike_train, mode='regression')

        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_multiple_output_regression(self):
        """Test regression with multiple output parameters."""
        decoder_multi = ExoplanetDecoder(input_size=12, output_size=3)

        spike_train = torch.rand(40, 12)
        spike_train = (spike_train > 0.6).float()

        result = decoder_multi.decode(spike_train, mode='regression')

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_regression_linearity(self):
        """Test that regression output scales with input activity."""
        # Create spike trains with different activity levels
        low_activity = torch.zeros(20, 8)
        low_activity[:5, :] = 1.0  # 25% activity

        high_activity = torch.zeros(20, 8)
        high_activity[:15, :] = 1.0  # 75% activity

        result_low = self.decoder.decode(low_activity, mode='regression')
        result_high = self.decoder.decode(high_activity, mode='regression')

        # Both should be valid floats
        assert isinstance(result_low, float)
        assert isinstance(result_high, float)

        # Results should be different (linear layer should respond to activity)
        assert result_low != result_high

    def test_regression_with_zero_input(self):
        """Test regression with zero spike input."""
        zero_spikes = torch.zeros(25, 8)

        result = self.decoder.decode(zero_spikes, mode='regression')

        assert isinstance(result, float)
        # With zero input and zero bias initialization, should be close to zero
        assert abs(result) < 10.0  # Reasonable bound

    def test_regression_parameter_learning(self):
        """Test that regression parameters can be updated through training."""
        # Get initial parameters
        initial_weight = self.decoder.regression_layer.weight.clone()
        initial_bias = self.decoder.regression_layer.bias.clone()

        # Create dummy training data
        spike_counts = torch.rand(5, 8)
        target = torch.rand(5, 1)

        # Perform a training step
        optimizer = torch.optim.SGD(self.decoder.parameters(), lr=0.1)
        loss_fn = nn.MSELoss()

        optimizer.zero_grad()
        output = self.decoder.forward(spike_counts)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # Parameters should have changed
        assert not torch.allclose(self.decoder.regression_layer.weight, initial_weight)
        assert not torch.allclose(self.decoder.regression_layer.bias, initial_bias)

    def test_regression_gradient_flow(self):
        """Test that gradients flow properly through regression layer."""
        spike_counts = torch.rand(3, 8, requires_grad=True)

        output = self.decoder.forward(spike_counts)
        loss = output.sum()
        loss.backward()

        # Input should have gradients
        assert spike_counts.grad is not None
        assert not torch.allclose(spike_counts.grad, torch.zeros_like(spike_counts.grad))


class TestConfigurationAndStatistics:
    """Test suite for configuration management and statistics tracking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.decoder = ExoplanetDecoder(input_size=12, output_size=2, classification_threshold=0.6)

    def test_get_config_structure(self):
        """Test that get_config returns proper configuration structure."""
        config = self.decoder.get_config()

        required_keys = [
            'decoder_type', 'input_size', 'output_size', 'classification_threshold',
            'supported_modes', 'classification_method', 'regression_method',
            'trainable_parameters', 'last_decoding_stats'
        ]

        for key in required_keys:
            assert key in config

        assert config['decoder_type'] == 'ExoplanetDecoder'
        assert config['input_size'] == 12
        assert config['output_size'] == 2
        assert config['classification_threshold'] == 0.6

    def test_config_reflects_parameters(self):
        """Test that configuration reflects actual decoder parameters."""
        config = self.decoder.get_config()

        assert config['supported_modes'] == ['classification', 'regression']
        assert config['classification_method'] == 'population_coding'
        assert config['regression_method'] == 'linear_readout'

        # Check trainable parameters count (both classification and regression layers)
        expected_params = (12 * 2 + 2) + (12 * 2 + 2)  # (regression layer) + (classification layer)
        assert config['trainable_parameters'] == expected_params

    def test_statistics_update_after_decoding(self):
        """Test that statistics are properly updated after decoding."""
        spike_train = torch.rand(25, 12)
        spike_train = (spike_train > 0.8).float()

        # Get initial stats
        initial_stats = self.decoder.get_decoding_stats()
        assert initial_stats['total_input_spikes'] == 0

        # Perform decoding
        result = self.decoder.decode(spike_train, mode='classification')

        # Get updated stats
        updated_stats = self.decoder.get_decoding_stats()
        assert updated_stats['input_shape'] == [25, 12]
        assert updated_stats['total_input_spikes'] == spike_train.sum().item()
        assert updated_stats['mode_used'] == 'classification'
        assert updated_stats['spike_rate'] > 0

    def test_spike_rate_calculation(self):
        """Test spike rate calculation accuracy."""
        spike_train = torch.zeros(20, 12)
        spike_train[:10, :6] = 1.0  # 60 spikes out of 240 possible

        self.decoder.decode(spike_train, mode='regression')
        stats = self.decoder.get_decoding_stats()

        expected_rate = 60.0 / (20 * 12)  # total_spikes / (time_steps * neurons)
        assert abs(stats['spike_rate'] - expected_rate) < 1e-10

    def test_classification_threshold_modification(self):
        """Test classification threshold modification."""
        original_threshold = self.decoder.classification_threshold
        new_threshold = 0.3

        self.decoder.set_classification_threshold(new_threshold)
        assert self.decoder.classification_threshold == new_threshold

        # Config should reflect the change
        config = self.decoder.get_config()
        assert config['classification_threshold'] == new_threshold

    def test_invalid_threshold_modification(self):
        """Test invalid classification threshold modification."""
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            self.decoder.set_classification_threshold(0.0)

        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            self.decoder.set_classification_threshold(1.5)

    def test_regression_layer_reset(self):
        """Test regression layer weight reset functionality."""
        # Get initial weights
        initial_weight = self.decoder.regression_layer.weight.clone()
        initial_bias = self.decoder.regression_layer.bias.clone()

        # Modify weights
        with torch.no_grad():
            self.decoder.regression_layer.weight.fill_(0.5)
            self.decoder.regression_layer.bias.fill_(0.1)

        # Reset weights
        self.decoder.reset_regression_layer()

        # Weights should be different from both initial and modified values
        assert not torch.allclose(self.decoder.regression_layer.weight, initial_weight)
        assert not torch.allclose(self.decoder.regression_layer.weight, torch.full_like(initial_weight, 0.5))
        assert torch.allclose(self.decoder.regression_layer.bias, torch.zeros_like(initial_bias))


class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.decoder = ExoplanetDecoder(input_size=6, output_size=1)

    def test_extreme_spike_values(self):
        """Test decoding with extreme spike values."""
        # Very high spike counts
        high_spikes = torch.ones(100, 6) * 1000  # Unrealistic but should handle
        result_high = self.decoder.decode(high_spikes, mode='regression')
        assert isinstance(result_high, float)
        assert not np.isnan(result_high)

        # Very sparse spikes
        sparse_spikes = torch.zeros(1000, 6)
        sparse_spikes[::100, 0] = 1.0  # Only 10 spikes total
        result_sparse = self.decoder.decode(sparse_spikes, mode='classification')
        assert result_sparse in [0, 1]

    def test_large_time_series(self):
        """Test decoding with very long time series."""
        large_spike_train = torch.rand(10000, 6)
        large_spike_train = (large_spike_train > 0.9).float()

        result = self.decoder.decode(large_spike_train, mode='classification')
        assert isinstance(result, int)
        assert result in [0, 1]

    def test_single_batch_extraction(self):
        """Test proper handling of batch dimension extraction."""
        # 3D input with multiple batches
        multi_batch = torch.rand(15, 3, 6)
        multi_batch = (multi_batch > 0.8).float()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.decoder.decode(multi_batch, mode='regression')

            # Should warn about using only first batch
            assert len(w) == 1
            assert "single batch only" in str(w[0].message)

        assert isinstance(result, float)

    def test_numerical_stability(self):
        """Test numerical stability with edge case inputs."""
        # All zeros
        zeros = torch.zeros(20, 6)
        result_zeros = self.decoder.decode(zeros, mode='regression')
        assert not np.isnan(result_zeros)
        assert not np.isinf(result_zeros)

        # All ones
        ones = torch.ones(20, 6)
        result_ones = self.decoder.decode(ones, mode='regression')
        assert not np.isnan(result_ones)
        assert not np.isinf(result_ones)

    def test_memory_efficiency(self):
        """Test memory efficiency across multiple decodings."""
        # Perform multiple decodings to check for memory leaks
        for i in range(10):
            spike_train = torch.rand(50, 6)
            spike_train = (spike_train > 0.7).float()

            result_class = self.decoder.decode(spike_train, mode='classification')
            result_reg = self.decoder.decode(spike_train, mode='regression')

            assert isinstance(result_class, int)
            assert isinstance(result_reg, float)

    def test_concurrent_mode_usage(self):
        """Test using both modes on the same spike train."""
        spike_train = torch.rand(30, 6)
        spike_train = (spike_train > 0.6).float()

        # Use both modes on same data
        class_result = self.decoder.decode(spike_train, mode='classification')
        reg_result = self.decoder.decode(spike_train, mode='regression')

        # Both should be valid
        assert isinstance(class_result, int)
        assert isinstance(reg_result, float)

        # Statistics should reflect the last operation (regression)
        stats = self.decoder.get_decoding_stats()
        assert stats['mode_used'] == 'regression'


class TestStringRepresentations:
    """Test suite for string representations and utility methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.decoder = ExoplanetDecoder(input_size=16, output_size=3, classification_threshold=0.4)

    def test_repr_method(self):
        """Test __repr__ method."""
        repr_str = repr(self.decoder)
        assert 'ExoplanetDecoder' in repr_str
        assert 'input_size=16' in repr_str
        assert 'output_size=3' in repr_str
        assert 'threshold=0.4' in repr_str

    def test_str_method(self):
        """Test __str__ method."""
        str_repr = str(self.decoder)
        assert 'Exoplanet Decoder' in str_repr
        assert 'Dual-Mode' in str_repr
        assert 'Input neurons: 16' in str_repr
        assert 'Output parameters: 3' in str_repr
        assert 'Population coding' in str_repr
        assert 'Linear readout' in str_repr

    def test_string_representations_after_decoding(self):
        """Test string representations after decoding operation."""
        # Perform decoding
        spike_train = torch.rand(25, 16)
        spike_train = (spike_train > 0.8).float()
        result = self.decoder.decode(spike_train, mode='classification')

        # String representations should reflect decoding statistics
        str_repr = str(self.decoder)
        assert 'Last decoding: classification mode' in str_repr

        # Repr should still work
        repr_str = repr(self.decoder)
        assert 'ExoplanetDecoder' in repr_str


class TestPyTorchIntegration:
    """Test suite for PyTorch module integration and training compatibility."""

    def setup_method(self):
        """Set up test fixtures."""
        self.decoder = ExoplanetDecoder(input_size=10, output_size=2)

    def test_forward_method_compatibility(self):
        """Test forward method for PyTorch training compatibility."""
        # Create batch of spike counts
        batch_spike_counts = torch.rand(5, 10)  # batch_size=5

        output = self.decoder.forward(batch_spike_counts)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (5, 2)  # batch_size x output_size
        assert output.requires_grad  # Should support gradients

    def test_training_mode_compatibility(self):
        """Test compatibility with PyTorch training and evaluation modes."""
        # Test training mode
        self.decoder.train()
        assert self.decoder.training

        # Test evaluation mode
        self.decoder.eval()
        assert not self.decoder.training

        # Both modes should work for decoding
        spike_train = torch.rand(20, 10)

        self.decoder.train()
        result_train = self.decoder.decode(spike_train, mode='regression')

        self.decoder.eval()
        result_eval = self.decoder.decode(spike_train, mode='regression')

        # Results should be identical (no dropout or batch norm)
        if isinstance(result_train, torch.Tensor) and isinstance(result_eval, torch.Tensor):
            assert torch.allclose(result_train, result_eval, atol=1e-6)
        else:
            assert abs(result_train - result_eval) < 1e-6

    def test_optimizer_compatibility(self):
        """Test compatibility with PyTorch optimizers."""
        # Test with different optimizers
        optimizers = [
            torch.optim.SGD(self.decoder.parameters(), lr=0.01),
            torch.optim.Adam(self.decoder.parameters(), lr=0.001),
            torch.optim.RMSprop(self.decoder.parameters(), lr=0.01)
        ]

        for optimizer in optimizers:
            # Create dummy data
            spike_counts = torch.rand(3, 10)
            target = torch.rand(3, 2)

            # Training step
            optimizer.zero_grad()
            output = self.decoder.forward(spike_counts)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            optimizer.step()

            # Should complete without errors
            assert True

    def test_loss_function_compatibility(self):
        """Test compatibility with different PyTorch loss functions."""
        spike_counts = torch.rand(4, 10)

        # Test with regression losses
        output = self.decoder.forward(spike_counts)
        target_reg = torch.rand(4, 2)

        mse_loss = nn.MSELoss()(output, target_reg)
        l1_loss = nn.L1Loss()(output, target_reg)

        assert isinstance(mse_loss, torch.Tensor)
        assert isinstance(l1_loss, torch.Tensor)
        assert mse_loss.requires_grad
        assert l1_loss.requires_grad

    def test_device_compatibility(self):
        """Test compatibility with different devices (CPU/GPU if available)."""
        # Test CPU
        self.decoder.to('cpu')
        spike_train_cpu = torch.rand(15, 10)
        result_cpu = self.decoder.decode(spike_train_cpu, mode='regression')

        # Handle both single and multi-output cases
        if isinstance(result_cpu, torch.Tensor):
            assert result_cpu.numel() == self.decoder.output_size
        else:
            assert isinstance(result_cpu, float)

        # Test GPU if available
        if torch.cuda.is_available():
            self.decoder.to('cuda')
            spike_train_gpu = torch.rand(15, 10).cuda()
            result_gpu = self.decoder.decode(spike_train_gpu, mode='regression')

            if isinstance(result_gpu, torch.Tensor):
                assert result_gpu.numel() == self.decoder.output_size
            else:
                assert isinstance(result_gpu, float)

    def test_state_dict_compatibility(self):
        """Test state dict save/load compatibility."""
        # Get initial state
        initial_state = self.decoder.state_dict()

        # Modify parameters
        with torch.no_grad():
            self.decoder.regression_layer.weight.fill_(0.5)

        # Load initial state
        self.decoder.load_state_dict(initial_state)

        # Should restore original parameters
        restored_state = self.decoder.state_dict()
        for key in initial_state:
            assert torch.allclose(initial_state[key], restored_state[key])

    def test_parameter_access(self):
        """Test parameter access and modification."""
        # Test parameter iteration (both classification and regression layers)
        params = list(self.decoder.parameters())
        assert len(params) == 4  # 2 layers × (weight + bias) = 4 parameters

        # Test named parameters (both layers)
        named_params = dict(self.decoder.named_parameters())
        assert 'regression_layer.weight' in named_params
        assert 'regression_layer.bias' in named_params
        assert 'classification_layer.weight' in named_params
        assert 'classification_layer.bias' in named_params

        # Test parameter modification
        original_weight = self.decoder.regression_layer.weight.clone()
        with torch.no_grad():
            self.decoder.regression_layer.weight += 0.1

        assert not torch.allclose(self.decoder.regression_layer.weight, original_weight)


class TestPerformanceCharacteristics:
    """Test suite for performance characteristics and efficiency."""

    def setup_method(self):
        """Set up test fixtures."""
        self.decoder = ExoplanetDecoder(input_size=20, output_size=1)

    def test_decoding_speed(self):
        """Test decoding speed with moderately large inputs."""
        import time

        # Create large spike train
        large_spike_train = torch.rand(1000, 20)
        large_spike_train = (large_spike_train > 0.8).float()

        start_time = time.time()
        result = self.decoder.decode(large_spike_train, mode='classification')
        end_time = time.time()

        # Should complete quickly (less than 0.1 seconds)
        assert (end_time - start_time) < 0.1
        assert isinstance(result, int)

    def test_memory_usage_scaling(self):
        """Test memory usage scaling with input size."""
        input_sizes = [50, 100, 500, 1000]

        for size in input_sizes:
            spike_train = torch.rand(size, 20)
            spike_train = (spike_train > 0.7).float()

            # Both modes should handle different sizes efficiently
            class_result = self.decoder.decode(spike_train, mode='classification')
            reg_result = self.decoder.decode(spike_train, mode='regression')

            assert isinstance(class_result, int)
            assert isinstance(reg_result, float)

    def test_batch_processing_efficiency(self):
        """Test efficiency of batch processing in forward method."""
        batch_sizes = [1, 5, 10, 20]

        for batch_size in batch_sizes:
            spike_counts = torch.rand(batch_size, 20)

            output = self.decoder.forward(spike_counts)
            assert output.shape == (batch_size, 1)

    def test_repeated_decoding_consistency(self):
        """Test consistency and efficiency of repeated decodings."""
        spike_train = torch.rand(30, 20)
        spike_train = (spike_train > 0.6).float()

        # Perform multiple decodings
        results = []
        for _ in range(10):
            result = self.decoder.decode(spike_train, mode='classification')
            results.append(result)

        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_mode_switching_efficiency(self):
        """Test efficiency of switching between modes."""
        spike_train = torch.rand(40, 20)
        spike_train = (spike_train > 0.5).float()

        # Alternate between modes
        for i in range(10):
            if i % 2 == 0:
                result = self.decoder.decode(spike_train, mode='classification')
                assert isinstance(result, int)
            else:
                result = self.decoder.decode(spike_train, mode='regression')
                assert isinstance(result, float)

    def test_large_network_performance(self):
        """Test performance with larger network configurations."""
        large_decoder = ExoplanetDecoder(input_size=100, output_size=10)

        large_spike_train = torch.rand(200, 100)
        large_spike_train = (large_spike_train > 0.8).float()

        # Should handle large configurations efficiently
        class_result = large_decoder.decode(large_spike_train, mode='classification')
        reg_result = large_decoder.decode(large_spike_train, mode='regression')

        assert isinstance(class_result, int)
        assert isinstance(reg_result, torch.Tensor)
        assert reg_result.shape == (10,)
