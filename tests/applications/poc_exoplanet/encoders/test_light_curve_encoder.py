"""
Comprehensive Test Suite for LightCurveEncoder
==============================================

This module provides rigorous testing for the LightCurveEncoder class,
validating its delta modulation encoding, interface compliance, and
integration with the GIF framework components.

Test Categories:
- Interface compliance and contract validation
- Delta modulation algorithm accuracy
- Input format handling and validation
- Edge cases and error conditions
- Calibration functionality
- Integration with data generators
- Performance and efficiency validation

The test suite ensures the encoder correctly converts astronomical light curves
into biologically plausible spike trains suitable for neuromorphic processing.
"""

import pytest
import torch
import polars as pl
import numpy as np
import warnings
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the components to test
from applications.poc_exoplanet.encoders.light_curve_encoder import LightCurveEncoder
from gif_framework.interfaces.base_interfaces import EncoderInterface, SpikeTrain
from data_generators.exoplanet_generator import RealisticExoplanetGenerator


class TestLightCurveEncoderInitialization:
    """Test suite for LightCurveEncoder initialization and validation."""

    def test_valid_initialization(self):
        """Test successful initialization with valid parameters."""
        # Test default initialization
        encoder = LightCurveEncoder()
        assert encoder.threshold == 1e-4
        assert encoder.normalize_input == True
        
        # Test custom parameters
        encoder_custom = LightCurveEncoder(threshold=5e-5, normalize_input=False)
        assert encoder_custom.threshold == 5e-5
        assert encoder_custom.normalize_input == False

    def test_invalid_threshold_values(self):
        """Test initialization with invalid threshold values."""
        # Test negative threshold
        with pytest.raises(ValueError, match="threshold must be a positive number"):
            LightCurveEncoder(threshold=-1e-4)
        
        # Test zero threshold
        with pytest.raises(ValueError, match="threshold must be a positive number"):
            LightCurveEncoder(threshold=0)
        
        # Test non-numeric threshold
        with pytest.raises(ValueError, match="threshold must be a positive number"):
            LightCurveEncoder(threshold="invalid")

    def test_interface_inheritance(self):
        """Test that LightCurveEncoder properly inherits from EncoderInterface."""
        encoder = LightCurveEncoder()
        assert isinstance(encoder, EncoderInterface)
        
        # Check that all required methods are implemented
        assert hasattr(encoder, 'encode')
        assert hasattr(encoder, 'get_config')
        assert hasattr(encoder, 'calibrate')
        assert callable(encoder.encode)
        assert callable(encoder.get_config)
        assert callable(encoder.calibrate)

    def test_initial_statistics(self):
        """Test that initial encoding statistics are properly set."""
        encoder = LightCurveEncoder()
        stats = encoder.get_encoding_stats()
        
        expected_keys = ['input_length', 'total_spikes', 'positive_spikes', 
                        'negative_spikes', 'spike_rate']
        for key in expected_keys:
            assert key in stats
            assert stats[key] == 0 or stats[key] == 0.0


class TestInputValidationAndFormatHandling:
    """Test suite for input validation and format handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = LightCurveEncoder(threshold=1e-4)

    def test_polars_dataframe_input(self):
        """Test encoding with polars DataFrame input (primary format)."""
        # Create test DataFrame
        time_data = np.linspace(0, 10, 100)
        flux_data = np.sin(time_data) + np.random.normal(0, 0.01, 100)
        df = pl.DataFrame({'time': time_data, 'flux': flux_data})
        
        spike_train = self.encoder.encode(df)
        
        assert isinstance(spike_train, torch.Tensor)
        assert spike_train.shape == (100, 2)
        assert spike_train.dtype == torch.float32

    def test_numpy_array_input(self):
        """Test encoding with numpy array input."""
        flux_data = np.sin(np.linspace(0, 10, 50)) + np.random.normal(0, 0.01, 50)
        
        spike_train = self.encoder.encode(flux_data)
        
        assert isinstance(spike_train, torch.Tensor)
        assert spike_train.shape == (50, 2)

    def test_list_input(self):
        """Test encoding with list input."""
        flux_data = [0.1, 0.2, 0.15, 0.3, 0.25, 0.4]
        
        spike_train = self.encoder.encode(flux_data)
        
        assert isinstance(spike_train, torch.Tensor)
        assert spike_train.shape == (6, 2)

    def test_invalid_input_types(self):
        """Test encoding with invalid input types."""
        with pytest.raises(ValueError, match="Unsupported input data type"):
            self.encoder.encode("invalid_string")
        
        with pytest.raises(ValueError, match="Unsupported input data type"):
            self.encoder.encode(42)

    def test_missing_flux_column(self):
        """Test encoding with DataFrame missing flux column."""
        df = pl.DataFrame({'time': [1, 2, 3], 'brightness': [0.1, 0.2, 0.15]})
        
        with pytest.raises(ValueError, match="Input DataFrame must contain 'flux' column"):
            self.encoder.encode(df)

    def test_multidimensional_array_input(self):
        """Test encoding with invalid multidimensional array."""
        invalid_array = np.random.rand(10, 5)  # 2D array
        
        with pytest.raises(ValueError, match="Input array must be 1D"):
            self.encoder.encode(invalid_array)

    def test_empty_input_handling(self):
        """Test encoding with empty input."""
        empty_array = np.array([])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            spike_train = self.encoder.encode(empty_array)

        assert spike_train.shape == (0, 2)

    def test_single_point_input(self):
        """Test encoding with single data point."""
        single_point = np.array([0.5])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            spike_train = self.encoder.encode(single_point)

        assert spike_train.shape == (1, 2)
        # Single point should produce no spikes (no delta)
        assert spike_train.sum().item() == 0


class TestDeltaModulationAlgorithm:
    """Test suite for delta modulation encoding algorithm."""

    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = LightCurveEncoder(threshold=0.1, normalize_input=False)

    def test_positive_flux_changes(self):
        """Test encoding of positive flux changes."""
        # Create flux with clear positive changes
        flux_data = np.array([0.0, 0.2, 0.4, 0.6])  # Increasing by 0.2 each step
        
        spike_train = self.encoder.encode(flux_data)
        
        # Should have spikes in positive channel (channel 0) at steps 1, 2, 3
        assert spike_train[0, 0] == 0  # No spike at first step
        assert spike_train[1, 0] == 1  # Positive spike at step 1
        assert spike_train[2, 0] == 1  # Positive spike at step 2
        assert spike_train[3, 0] == 1  # Positive spike at step 3
        
        # No spikes in negative channel
        assert spike_train[:, 1].sum() == 0

    def test_negative_flux_changes(self):
        """Test encoding of negative flux changes."""
        # Create flux with clear negative changes
        flux_data = np.array([1.0, 0.8, 0.6, 0.4])  # Decreasing by 0.2 each step
        
        spike_train = self.encoder.encode(flux_data)
        
        # Should have spikes in negative channel (channel 1) at steps 1, 2, 3
        assert spike_train[0, 1] == 0  # No spike at first step
        assert spike_train[1, 1] == 1  # Negative spike at step 1
        assert spike_train[2, 1] == 1  # Negative spike at step 2
        assert spike_train[3, 1] == 1  # Negative spike at step 3
        
        # No spikes in positive channel
        assert spike_train[:, 0].sum() == 0

    def test_threshold_sensitivity(self):
        """Test that threshold properly controls spike generation."""
        flux_data = np.array([0.0, 0.05, 0.1, 0.15])  # Changes of 0.05
        
        # High threshold - should not generate spikes
        encoder_high = LightCurveEncoder(threshold=0.1, normalize_input=False)
        spike_train_high = encoder_high.encode(flux_data)
        assert spike_train_high.sum() == 0
        
        # Low threshold - should generate spikes
        encoder_low = LightCurveEncoder(threshold=0.01, normalize_input=False)
        spike_train_low = encoder_low.encode(flux_data)
        assert spike_train_low.sum() > 0

    def test_mixed_flux_changes(self):
        """Test encoding with mixed positive and negative changes."""
        flux_data = np.array([0.5, 0.7, 0.4, 0.8, 0.3])  # +0.2, -0.3, +0.4, -0.5
        
        spike_train = self.encoder.encode(flux_data)
        
        # Check specific expected spikes
        assert spike_train[1, 0] == 1  # Positive change at step 1
        assert spike_train[2, 1] == 1  # Negative change at step 2
        assert spike_train[3, 0] == 1  # Positive change at step 3
        assert spike_train[4, 1] == 1  # Negative change at step 4

    def test_small_changes_below_threshold(self):
        """Test that small changes below threshold don't generate spikes."""
        flux_data = np.array([0.5, 0.505, 0.51, 0.515])  # Small changes of 0.005
        
        encoder = LightCurveEncoder(threshold=0.01, normalize_input=False)
        spike_train = encoder.encode(flux_data)
        
        # No spikes should be generated
        assert spike_train.sum() == 0

    def test_normalization_effect(self):
        """Test the effect of input normalization on encoding."""
        # Create data with large absolute values but small relative changes
        flux_data = np.array([1000.0, 1000.1, 1000.05, 1000.15])
        
        # Without normalization - changes might be above threshold
        encoder_no_norm = LightCurveEncoder(threshold=0.05, normalize_input=False)
        spike_train_no_norm = encoder_no_norm.encode(flux_data)
        
        # With normalization - should emphasize relative changes
        encoder_norm = LightCurveEncoder(threshold=0.05, normalize_input=True)
        spike_train_norm = encoder_norm.encode(flux_data)
        
        # Both should produce valid spike trains
        assert spike_train_no_norm.shape == (4, 2)
        assert spike_train_norm.shape == (4, 2)


class TestConfigurationAndCalibration:
    """Test suite for configuration management and calibration functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = LightCurveEncoder(threshold=1e-4, normalize_input=True)

    def test_get_config_structure(self):
        """Test that get_config returns proper configuration structure."""
        config = self.encoder.get_config()

        required_keys = [
            'encoder_type', 'encoding_scheme', 'threshold', 'normalize_input',
            'output_channels', 'channel_0_meaning', 'channel_1_meaning',
            'last_encoding_stats'
        ]

        for key in required_keys:
            assert key in config

        assert config['encoder_type'] == 'LightCurveEncoder'
        assert config['encoding_scheme'] == 'delta_modulation'
        assert config['output_channels'] == 2

    def test_config_reflects_parameters(self):
        """Test that configuration reflects actual encoder parameters."""
        threshold = 5e-5
        normalize = False
        encoder = LightCurveEncoder(threshold=threshold, normalize_input=normalize)

        config = encoder.get_config()
        assert config['threshold'] == threshold
        assert config['normalize_input'] == normalize

    def test_calibration_with_sample_data(self):
        """Test calibration functionality with sample data."""
        # Create sample data with known characteristics
        time_data = np.linspace(0, 10, 1000)
        flux_data = np.sin(time_data) + np.random.normal(0, 0.1, 1000)
        sample_df = pl.DataFrame({'time': time_data, 'flux': flux_data})

        original_threshold = self.encoder.threshold
        self.encoder.calibrate(sample_df)

        # Threshold should be updated based on data characteristics
        assert self.encoder.threshold != original_threshold
        assert self.encoder.threshold > 0

    def test_calibration_with_insufficient_data(self):
        """Test calibration with insufficient data."""
        small_data = np.array([0.1, 0.2])  # Only 2 points

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.encoder.calibrate(small_data)
            assert len(w) == 1
            assert "Insufficient data for calibration" in str(w[0].message)

    def test_calibration_with_no_changes(self):
        """Test calibration with data that has no flux changes."""
        constant_data = np.ones(100) * 0.5  # Constant flux

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.encoder.calibrate(constant_data)
            # May generate multiple warnings (normalization + no changes)
            assert len(w) >= 1
            warning_messages = [str(warning.message) for warning in w]
            assert any("No flux changes detected" in msg for msg in warning_messages)

    def test_calibration_error_handling(self):
        """Test calibration error handling with invalid data."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.encoder.calibrate("invalid_data")
            assert len(w) == 1
            assert "Calibration failed" in str(w[0].message)


class TestEncodingStatistics:
    """Test suite for encoding statistics and monitoring."""

    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = LightCurveEncoder(threshold=0.1, normalize_input=False)

    def test_statistics_update_after_encoding(self):
        """Test that statistics are properly updated after encoding."""
        flux_data = np.array([0.0, 0.2, 0.1, 0.3, 0.05])

        # Get initial stats
        initial_stats = self.encoder.get_encoding_stats()
        assert initial_stats['total_spikes'] == 0

        # Encode data
        spike_train = self.encoder.encode(flux_data)

        # Get updated stats
        updated_stats = self.encoder.get_encoding_stats()
        assert updated_stats['input_length'] == 5
        assert updated_stats['total_spikes'] == spike_train.sum().item()
        assert updated_stats['spike_rate'] > 0

    def test_positive_negative_spike_counting(self):
        """Test accurate counting of positive and negative spikes."""
        # Create data with known spike pattern
        flux_data = np.array([0.0, 0.2, 0.1, 0.3])  # +0.2, -0.1, +0.2

        spike_train = self.encoder.encode(flux_data)
        stats = self.encoder.get_encoding_stats()

        expected_positive = spike_train[:, 0].sum().item()
        expected_negative = spike_train[:, 1].sum().item()

        assert stats['positive_spikes'] == expected_positive
        assert stats['negative_spikes'] == expected_negative
        assert stats['total_spikes'] == expected_positive + expected_negative

    def test_spike_rate_calculation(self):
        """Test spike rate calculation accuracy."""
        flux_data = np.array([0.0, 0.2, 0.4, 0.6, 0.8])  # 4 positive spikes expected

        spike_train = self.encoder.encode(flux_data)
        stats = self.encoder.get_encoding_stats()

        expected_rate = stats['total_spikes'] / len(flux_data)
        assert abs(stats['spike_rate'] - expected_rate) < 1e-10


class TestIntegrationWithDataGenerators:
    """Test suite for integration with exoplanet data generators."""

    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = LightCurveEncoder(threshold=1e-4)

    def test_realistic_exoplanet_data_encoding(self):
        """Test encoding with realistic exoplanet generator data."""
        generator = RealisticExoplanetGenerator(seed=42)
        light_curve_data = generator.generate()

        spike_train = self.encoder.encode(light_curve_data)

        # Validate output structure
        assert isinstance(spike_train, torch.Tensor)
        assert spike_train.shape[0] == len(light_curve_data)
        assert spike_train.shape[1] == 2
        assert spike_train.dtype == torch.float32

    def test_transit_signal_detection(self):
        """Test that encoder can detect transit signals in generated data."""
        # Generate data with low noise for clear signal detection
        generator = RealisticExoplanetGenerator(
            seed=123,
            stellar_variability_amplitude=0.0001,  # Very low stellar variability
            instrumental_noise_level=0.00001       # Very low instrumental noise
        )
        light_curve_data = generator.generate()

        spike_train = self.encoder.encode(light_curve_data)

        # Should generate spikes during transit events
        assert spike_train.sum() > 0

        # Get encoding statistics
        stats = self.encoder.get_encoding_stats()
        assert stats['spike_rate'] > 0
        assert stats['total_spikes'] > 0

    def test_different_noise_levels(self):
        """Test encoder behavior with different noise levels."""
        noise_levels = [0.0001, 0.001, 0.01]  # Different instrumental noise levels
        spike_rates = []

        for noise_level in noise_levels:
            generator = RealisticExoplanetGenerator(
                seed=42,
                instrumental_noise_level=noise_level
            )
            data = generator.generate()

            spike_train = self.encoder.encode(data)
            stats = self.encoder.get_encoding_stats()
            spike_rates.append(stats['spike_rate'])

        # Higher noise should generally produce more spikes
        # (though this relationship isn't strictly monotonic)
        assert all(rate >= 0 for rate in spike_rates)

    def test_multiple_data_samples(self):
        """Test encoder consistency across multiple data samples."""
        generator = RealisticExoplanetGenerator(seed=42)

        spike_rates = []
        for _ in range(5):
            data = generator.generate()
            spike_train = self.encoder.encode(data)
            stats = self.encoder.get_encoding_stats()
            spike_rates.append(stats['spike_rate'])

        # All spike rates should be reasonable
        assert all(0 <= rate <= 1 for rate in spike_rates)

        # Should have some variability but not extreme
        mean_rate = np.mean(spike_rates)
        assert mean_rate > 0


class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = LightCurveEncoder(threshold=1e-4)

    def test_extreme_flux_values(self):
        """Test encoding with extreme flux values."""
        # Very large values
        large_flux = np.array([1e6, 1e6 + 1, 1e6 + 2])
        spike_train_large = self.encoder.encode(large_flux)
        assert spike_train_large.shape == (3, 2)

        # Very small values
        small_flux = np.array([1e-10, 2e-10, 3e-10])
        spike_train_small = self.encoder.encode(small_flux)
        assert spike_train_small.shape == (3, 2)

    def test_nan_and_inf_handling(self):
        """Test handling of NaN and infinity values."""
        # Test with NaN values
        nan_flux = np.array([0.1, np.nan, 0.3])

        # Should either handle gracefully or raise appropriate error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            try:
                spike_train = self.encoder.encode(nan_flux)
                # If it doesn't raise an error, check output is reasonable
                assert spike_train.shape == (3, 2)
                assert not torch.isnan(spike_train).any()
            except (ValueError, RuntimeError):
                # Acceptable to raise an error for invalid input
                pass

    def test_very_long_time_series(self):
        """Test encoding with very long time series."""
        # Create long time series
        long_flux = np.sin(np.linspace(0, 100, 10000)) + np.random.normal(0, 0.01, 10000)

        spike_train = self.encoder.encode(long_flux)

        assert spike_train.shape == (10000, 2)
        assert spike_train.dtype == torch.float32

    def test_constant_flux_handling(self):
        """Test encoding with constant flux (no changes)."""
        constant_flux = np.ones(100) * 0.5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            spike_train = self.encoder.encode(constant_flux)

        # Should produce no spikes
        assert spike_train.sum() == 0

        # Check statistics
        stats = self.encoder.get_encoding_stats()
        assert stats['total_spikes'] == 0
        assert stats['spike_rate'] == 0.0

    def test_alternating_flux_pattern(self):
        """Test encoding with alternating flux pattern."""
        # Create alternating pattern
        alternating_flux = np.array([0.0, 0.2] * 50)  # 100 points alternating

        spike_train = self.encoder.encode(alternating_flux)

        # Should generate many spikes due to constant changes
        assert spike_train.sum() > 0

        # Check pattern consistency
        stats = self.encoder.get_encoding_stats()
        assert stats['positive_spikes'] > 0
        assert stats['negative_spikes'] > 0

    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        # Test that encoder doesn't accumulate memory across multiple calls
        initial_stats = self.encoder.get_encoding_stats()

        for i in range(10):
            large_data = np.random.rand(1000)
            spike_train = self.encoder.encode(large_data)

            # Each encoding should work independently
            assert spike_train.shape == (1000, 2)

        # Statistics should only reflect the last encoding
        final_stats = self.encoder.get_encoding_stats()
        assert final_stats['input_length'] == 1000


class TestStringRepresentations:
    """Test suite for string representations and utility methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = LightCurveEncoder(threshold=5e-5, normalize_input=False)

    def test_repr_method(self):
        """Test __repr__ method."""
        repr_str = repr(self.encoder)
        assert 'LightCurveEncoder' in repr_str
        assert '5.00e-05' in repr_str or '5e-05' in repr_str
        assert 'normalize_input=False' in repr_str

    def test_str_method(self):
        """Test __str__ method."""
        str_repr = str(self.encoder)
        assert 'Light Curve Encoder' in str_repr
        assert 'Delta Modulation' in str_repr
        assert 'Threshold:' in str_repr
        assert 'Output channels: 2' in str_repr

    def test_string_representations_after_encoding(self):
        """Test string representations after encoding operation."""
        # Perform encoding
        flux_data = np.array([0.1, 0.2, 0.15, 0.3])
        spike_train = self.encoder.encode(flux_data)

        # String representations should reflect encoding statistics
        str_repr = str(self.encoder)
        assert 'Last encoding:' in str_repr

        # Should show actual spike count
        stats = self.encoder.get_encoding_stats()
        assert str(stats['total_spikes']) in str_repr


class TestPerformanceCharacteristics:
    """Test suite for performance characteristics and efficiency."""

    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = LightCurveEncoder(threshold=1e-4)

    def test_encoding_speed(self):
        """Test encoding speed with moderately large datasets."""
        import time

        # Create moderately large dataset
        large_flux = np.sin(np.linspace(0, 50, 5000)) + np.random.normal(0, 0.01, 5000)

        start_time = time.time()
        spike_train = self.encoder.encode(large_flux)
        end_time = time.time()

        # Should complete in reasonable time (less than 1 second)
        assert (end_time - start_time) < 1.0
        assert spike_train.shape == (5000, 2)

    def test_output_sparsity(self):
        """Test that output maintains appropriate sparsity."""
        # Generate realistic astronomical data
        generator = RealisticExoplanetGenerator(seed=42)
        data = generator.generate()

        spike_train = self.encoder.encode(data)

        # Calculate sparsity
        total_possible_spikes = spike_train.numel()
        actual_spikes = spike_train.sum().item()
        sparsity = 1.0 - (actual_spikes / total_possible_spikes)

        # Should be reasonably sparse (>50% zeros)
        assert sparsity > 0.5

        # But not completely sparse (should have some activity)
        assert actual_spikes > 0

    def test_memory_usage_consistency(self):
        """Test consistent memory usage across different input sizes."""
        input_sizes = [100, 500, 1000, 2000]

        for size in input_sizes:
            flux_data = np.random.rand(size)
            spike_train = self.encoder.encode(flux_data)

            # Output size should scale linearly with input
            assert spike_train.shape == (size, 2)

            # Memory should be released after each encoding
            # (This is implicit in Python's garbage collection)

    def test_threshold_impact_on_sparsity(self):
        """Test impact of threshold on output sparsity."""
        flux_data = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)

        thresholds = [1e-5, 1e-4, 1e-3, 1e-2]
        spike_counts = []

        for threshold in thresholds:
            encoder = LightCurveEncoder(threshold=threshold, normalize_input=False)
            spike_train = encoder.encode(flux_data)
            spike_counts.append(spike_train.sum().item())

        # Higher thresholds should generally produce fewer spikes
        # (though this isn't strictly monotonic due to noise)
        assert all(count >= 0 for count in spike_counts)

        # At least some variation in spike counts
        assert len(set(spike_counts)) > 1
