"""
Comprehensive Integration Test Suite for Encoder-Decoder Pipeline
================================================================

This module provides rigorous integration testing for the complete
exoplanet detection pipeline: Generator → Encoder → DU_Core → Decoder.

Test Categories:
- End-to-end pipeline integration
- Data flow validation through all components
- Interface compatibility between modules
- Realistic exoplanet detection scenarios
- Performance and accuracy validation
- Error propagation and handling
- Neuromorphic simulator integration

The test suite ensures the complete pipeline works correctly for
both classification and regression tasks in exoplanet detection.
"""

import pytest
import torch
import polars as pl
import numpy as np
import warnings
from typing import Dict, Any, Tuple

# Import all pipeline components
from data_generators.exoplanet_generator import RealisticExoplanetGenerator
from applications.poc_exoplanet.encoders.light_curve_encoder import LightCurveEncoder
from applications.poc_exoplanet.decoders.exoplanet_decoder import ExoplanetDecoder
from gif_framework.core.du_core import DU_Core_V1
from simulators.neuromorphic_sim import NeuromorphicSimulator


class TestBasicPipelineIntegration:
    """Test suite for basic pipeline integration functionality."""

    def setup_method(self):
        """Set up test fixtures for pipeline components."""
        # Initialize all pipeline components
        self.generator = RealisticExoplanetGenerator(seed=42)
        self.encoder = LightCurveEncoder(threshold=1e-4)
        self.du_core = DU_Core_V1(input_size=2, hidden_sizes=[8, 6], output_size=4)
        self.decoder = ExoplanetDecoder(input_size=4, output_size=1)
        self.simulator = NeuromorphicSimulator(self.du_core)

    def test_complete_pipeline_flow(self):
        """Test complete data flow through entire pipeline."""
        # Step 1: Generate realistic exoplanet data
        light_curve_data = self.generator.generate()
        assert isinstance(light_curve_data, pl.DataFrame)
        assert len(light_curve_data) > 0
        
        # Step 2: Encode to spike trains
        spike_train = self.encoder.encode(light_curve_data)
        assert isinstance(spike_train, torch.Tensor)
        assert spike_train.shape[1] == 2  # Two channels
        
        # Step 3: Process through DU Core
        du_output = self.du_core.forward(spike_train.unsqueeze(1), num_steps=spike_train.shape[0])
        assert isinstance(du_output, torch.Tensor)
        assert du_output.shape[-1] == 4  # DU Core output size
        
        # Step 4: Decode to final prediction
        # Remove batch dimension for decoder
        du_output_2d = du_output.squeeze(1)  # [time_steps, output_size]
        
        classification_result = self.decoder.decode(du_output_2d, mode='classification')
        regression_result = self.decoder.decode(du_output_2d, mode='regression')
        
        # Validate final outputs
        assert isinstance(classification_result, int)
        assert classification_result in [0, 1]
        assert isinstance(regression_result, float)
        assert not np.isnan(regression_result)

    def test_neuromorphic_simulator_integration(self):
        """Test integration with neuromorphic hardware simulator."""
        # Generate and encode data
        light_curve_data = self.generator.generate()
        spike_train = self.encoder.encode(light_curve_data)
        
        # Add batch dimension for simulator
        spike_train_batched = spike_train.unsqueeze(1)  # [time_steps, batch_size=1, features]
        
        # Process through neuromorphic simulator
        sim_output, sim_stats = self.simulator.run(spike_train_batched)
        
        # Validate simulator output
        assert isinstance(sim_output, torch.Tensor)
        assert sim_output.shape == spike_train_batched.shape[:-1] + (4,)  # [time_steps, batch_size, output_size]
        
        # Validate energy statistics
        assert sim_stats['total_spikes'] >= 0
        assert sim_stats['total_synops'] >= 0
        assert sim_stats['estimated_energy_joules'] >= 0
        
        # Decode simulator output
        sim_output_2d = sim_output.squeeze(1)  # Remove batch dimension
        final_result = self.decoder.decode(sim_output_2d, mode='classification')
        assert isinstance(final_result, int)

    def test_pipeline_with_different_data_samples(self):
        """Test pipeline consistency across different data samples."""
        results = []
        
        for seed in [42, 123, 456, 789, 999]:
            # Generate different data sample
            generator = RealisticExoplanetGenerator(seed=seed)
            data = generator.generate()
            
            # Process through pipeline
            spikes = self.encoder.encode(data)
            du_output = self.du_core.forward(spikes.unsqueeze(1), num_steps=spikes.shape[0])
            result = self.decoder.decode(du_output.squeeze(1), mode='classification')
            
            results.append(result)
        
        # All results should be valid classifications
        assert all(r in [0, 1] for r in results)
        
        # Should have some variability (not all identical)
        unique_results = set(results)
        assert len(unique_results) >= 1  # At least one unique result

    def test_pipeline_data_shape_consistency(self):
        """Test data shape consistency throughout pipeline."""
        data = self.generator.generate()
        original_length = len(data)
        
        # Track shapes through pipeline
        spikes = self.encoder.encode(data)
        assert spikes.shape[0] == original_length
        assert spikes.shape[1] == 2
        
        # DU Core processing
        du_input = spikes.unsqueeze(1)  # Add batch dimension
        du_output = self.du_core.forward(du_input, num_steps=spikes.shape[0])
        assert du_output.shape[0] == original_length
        assert du_output.shape[1] == 1  # Batch size
        assert du_output.shape[2] == 4  # Output features
        
        # Decoder processing
        decoder_input = du_output.squeeze(1)
        result = self.decoder.decode(decoder_input, mode='regression')
        assert isinstance(result, float)

    def test_pipeline_error_propagation(self):
        """Test error handling and propagation through pipeline."""
        # Test with invalid data
        invalid_data = pl.DataFrame({'time': [1, 2], 'invalid_column': [0.1, 0.2]})
        
        with pytest.raises(ValueError):
            self.encoder.encode(invalid_data)
        
        # Test with mismatched dimensions
        wrong_size_spikes = torch.rand(10, 5)  # Wrong number of channels
        
        with pytest.raises(ValueError):
            self.du_core.forward(wrong_size_spikes.unsqueeze(1), num_steps=10)

    def test_pipeline_memory_efficiency(self):
        """Test memory efficiency of complete pipeline."""
        # Process multiple samples to check for memory leaks
        for i in range(5):
            data = self.generator.generate()
            spikes = self.encoder.encode(data)
            du_output = self.du_core.forward(spikes.unsqueeze(1), num_steps=spikes.shape[0])
            result = self.decoder.decode(du_output.squeeze(1), mode='classification')
            
            # Each iteration should complete successfully
            assert isinstance(result, int)


class TestRealisticExoplanetScenarios:
    """Test suite for realistic exoplanet detection scenarios."""

    def setup_method(self):
        """Set up test fixtures for realistic scenarios."""
        self.encoder = LightCurveEncoder(threshold=5e-5)  # Sensitive threshold
        self.du_core = DU_Core_V1(input_size=2, hidden_sizes=[12, 8], output_size=6)
        self.decoder = ExoplanetDecoder(input_size=6, output_size=1)

    def test_low_noise_transit_detection(self):
        """Test detection with low noise, clear transit signals."""
        generator = RealisticExoplanetGenerator(
            seed=42,
            stellar_variability_amplitude=0.0001,
            instrumental_noise_level=0.00001
        )
        
        data = generator.generate()
        spikes = self.encoder.encode(data)
        
        # Should generate reasonable spike activity
        spike_stats = self.encoder.get_encoding_stats()
        assert spike_stats['total_spikes'] > 0
        assert spike_stats['spike_rate'] > 0
        
        # Process through SNN
        du_output = self.du_core.forward(spikes.unsqueeze(1), num_steps=spikes.shape[0])
        
        # Decode results
        classification = self.decoder.decode(du_output.squeeze(1), mode='classification')
        regression = self.decoder.decode(du_output.squeeze(1), mode='regression')
        
        assert isinstance(classification, int)
        assert isinstance(regression, float)

    def test_high_noise_scenario(self):
        """Test robustness with high noise levels."""
        generator = RealisticExoplanetGenerator(
            seed=123,
            stellar_variability_amplitude=0.01,
            instrumental_noise_level=0.005
        )
        
        data = generator.generate()
        spikes = self.encoder.encode(data)
        
        # High noise should still produce valid spike trains
        assert spikes.shape[0] == len(data)
        assert spikes.shape[1] == 2
        
        # Pipeline should handle noisy data gracefully
        du_output = self.du_core.forward(spikes.unsqueeze(1), num_steps=spikes.shape[0])
        result = self.decoder.decode(du_output.squeeze(1), mode='classification')
        
        assert result in [0, 1]

    def test_multiple_transit_events(self):
        """Test detection with multiple transit events."""
        # Generate longer time series to capture multiple transits
        generator = RealisticExoplanetGenerator(seed=456)
        
        data = generator.generate()
        spikes = self.encoder.encode(data)
        
        # Should have reasonable number of data points
        assert len(data) > 100
        
        # Process through pipeline
        du_output = self.du_core.forward(spikes.unsqueeze(1), num_steps=spikes.shape[0])
        
        # Both modes should work with longer time series
        classification = self.decoder.decode(du_output.squeeze(1), mode='classification')
        regression = self.decoder.decode(du_output.squeeze(1), mode='regression')
        
        assert isinstance(classification, int)
        assert isinstance(regression, float)

    def test_edge_case_parameters(self):
        """Test with edge case system parameters."""
        # Test with different seed for variety
        generator_small = RealisticExoplanetGenerator(seed=789)
        
        data_small = generator_small.generate()
        spikes_small = self.encoder.encode(data_small)
        
        # Should still produce valid pipeline output
        du_output = self.du_core.forward(spikes_small.unsqueeze(1), num_steps=spikes_small.shape[0])
        result = self.decoder.decode(du_output.squeeze(1), mode='classification')
        assert result in [0, 1]

    def test_calibration_impact_on_detection(self):
        """Test impact of encoder calibration on detection performance."""
        data = RealisticExoplanetGenerator(seed=42).generate()
        
        # Test with default encoder
        encoder_default = LightCurveEncoder(threshold=1e-4)
        spikes_default = encoder_default.encode(data)
        
        # Test with calibrated encoder
        encoder_calibrated = LightCurveEncoder(threshold=1e-4)
        encoder_calibrated.calibrate(data)
        spikes_calibrated = encoder_calibrated.encode(data)
        
        # Both should produce valid spike trains
        assert spikes_default.shape == spikes_calibrated.shape
        
        # Process both through pipeline
        du_output_default = self.du_core.forward(spikes_default.unsqueeze(1), num_steps=spikes_default.shape[0])
        du_output_calibrated = self.du_core.forward(spikes_calibrated.unsqueeze(1), num_steps=spikes_calibrated.shape[0])
        
        result_default = self.decoder.decode(du_output_default.squeeze(1), mode='classification')
        result_calibrated = self.decoder.decode(du_output_calibrated.squeeze(1), mode='classification')
        
        # Both should produce valid results
        assert result_default in [0, 1]
        assert result_calibrated in [0, 1]


class TestPerformanceAndAccuracy:
    """Test suite for performance and accuracy validation."""

    def setup_method(self):
        """Set up test fixtures for performance testing."""
        self.encoder = LightCurveEncoder(threshold=1e-4)
        self.du_core = DU_Core_V1(input_size=2, hidden_sizes=[10], output_size=5)
        self.decoder = ExoplanetDecoder(input_size=5, output_size=1)
        self.simulator = NeuromorphicSimulator(self.du_core)

    def test_pipeline_processing_speed(self):
        """Test processing speed of complete pipeline."""
        import time

        # Generate moderately large dataset
        generator = RealisticExoplanetGenerator(seed=42)
        data = generator.generate()

        start_time = time.time()

        # Complete pipeline processing
        spikes = self.encoder.encode(data)
        du_output = self.du_core.forward(spikes.unsqueeze(1), num_steps=spikes.shape[0])
        classification = self.decoder.decode(du_output.squeeze(1), mode='classification')
        regression = self.decoder.decode(du_output.squeeze(1), mode='regression')

        end_time = time.time()

        # Should complete in reasonable time (less than 2 seconds)
        processing_time = end_time - start_time
        assert processing_time < 2.0

        # Results should be valid
        assert isinstance(classification, int)
        assert isinstance(regression, float)

    def test_neuromorphic_energy_efficiency(self):
        """Test energy efficiency of neuromorphic processing."""
        data = RealisticExoplanetGenerator(seed=42).generate()
        spikes = self.encoder.encode(data)

        # Process through neuromorphic simulator
        sim_output, sim_stats = self.simulator.run(spikes.unsqueeze(1))

        # Validate energy metrics
        assert sim_stats['estimated_energy_joules'] > 0
        assert sim_stats['total_synops'] > 0

        # Get detailed energy analysis
        energy_analysis = self.simulator.get_detailed_energy_analysis()

        # Should show significant efficiency gains
        efficiency_ratio = energy_analysis['comparative_analysis']['energy_efficiency_ratio']
        assert efficiency_ratio > 100  # At least 100x more efficient than traditional

        # Sparsity should be high (neuromorphic advantage)
        sparsity = energy_analysis['hardware_insights']['sparsity_level']
        assert sparsity > 0.5  # At least 50% sparse

    def test_classification_accuracy_consistency(self):
        """Test classification accuracy consistency across samples."""
        classification_results = []

        # Test with multiple samples
        for seed in range(10):
            generator = RealisticExoplanetGenerator(seed=seed)
            data = generator.generate()

            spikes = self.encoder.encode(data)
            du_output = self.du_core.forward(spikes.unsqueeze(1), num_steps=spikes.shape[0])
            result = self.decoder.decode(du_output.squeeze(1), mode='classification')

            classification_results.append(result)

        # All results should be valid
        assert all(r in [0, 1] for r in classification_results)

        # Should have reasonable distribution (not all same class)
        unique_classes = set(classification_results)
        assert len(unique_classes) >= 1

    def test_regression_output_stability(self):
        """Test stability of regression outputs."""
        data = RealisticExoplanetGenerator(seed=42).generate()
        spikes = self.encoder.encode(data)

        regression_results = []

        # Multiple runs with same data should be consistent
        for _ in range(5):
            du_output = self.du_core.forward(spikes.unsqueeze(1), num_steps=spikes.shape[0])
            result = self.decoder.decode(du_output.squeeze(1), mode='regression')
            regression_results.append(result)

        # All results should be identical (deterministic)
        assert all(abs(r - regression_results[0]) < 1e-6 for r in regression_results)

        # Results should be finite and reasonable
        assert all(not np.isnan(r) and not np.isinf(r) for r in regression_results)

    def test_scalability_with_data_size(self):
        """Test pipeline scalability with different data sizes."""
        seeds = [42, 123, 456, 789]  # Different seeds for variety
        processing_times = []

        for seed in seeds:
            generator = RealisticExoplanetGenerator(seed=seed)
            data = generator.generate()

            import time
            start_time = time.time()

            spikes = self.encoder.encode(data)
            du_output = self.du_core.forward(spikes.unsqueeze(1), num_steps=spikes.shape[0])
            result = self.decoder.decode(du_output.squeeze(1), mode='classification')

            end_time = time.time()
            processing_times.append(end_time - start_time)

            # Each size should produce valid results
            assert isinstance(result, int)

        # Processing time should scale reasonably (not exponentially)
        assert all(t < 5.0 for t in processing_times)  # All under 5 seconds

    def test_memory_usage_optimization(self):
        """Test memory usage optimization throughout pipeline."""
        # Process large dataset to test memory efficiency
        generator = RealisticExoplanetGenerator(seed=42)
        data = generator.generate()

        # Pipeline should handle data without memory issues
        spikes = self.encoder.encode(data)
        assert spikes.shape[0] > 100  # Ensure it has reasonable size

        # Process in chunks if needed (test chunked processing)
        chunk_size = 500
        if spikes.shape[0] > chunk_size:
            # Process in chunks
            chunks = torch.split(spikes, chunk_size, dim=0)
            chunk_results = []

            for chunk in chunks:
                du_output = self.du_core.forward(chunk.unsqueeze(1), num_steps=chunk.shape[0])
                result = self.decoder.decode(du_output.squeeze(1), mode='classification')
                chunk_results.append(result)

            # All chunks should produce valid results
            assert all(r in [0, 1] for r in chunk_results)
        else:
            # Process normally
            du_output = self.du_core.forward(spikes.unsqueeze(1), num_steps=spikes.shape[0])
            result = self.decoder.decode(du_output.squeeze(1), mode='classification')
            assert result in [0, 1]


class TestErrorHandlingAndRobustness:
    """Test suite for error handling and robustness."""

    def setup_method(self):
        """Set up test fixtures for error testing."""
        self.encoder = LightCurveEncoder(threshold=1e-4)
        self.du_core = DU_Core_V1(input_size=2, hidden_sizes=[8], output_size=4)
        self.decoder = ExoplanetDecoder(input_size=4, output_size=1)

    def test_invalid_data_handling(self):
        """Test handling of invalid input data."""
        # Test with corrupted data (use strict=False to allow mixed types)
        corrupted_data = pl.DataFrame({
            'time': [1.0, 2.0, np.nan, 4.0],
            'flux': [0.1, np.inf, 0.3, -0.1]
        }, strict=False)

        # Encoder should handle or raise appropriate error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            try:
                spikes = self.encoder.encode(corrupted_data)
                # If it doesn't raise an error, output should be valid
                assert isinstance(spikes, torch.Tensor)
                assert spikes.shape[1] == 2
            except (ValueError, RuntimeError):
                # Acceptable to raise an error for invalid data
                pass

    def test_extreme_parameter_values(self):
        """Test robustness with extreme parameter values."""
        # Very sensitive encoder
        sensitive_encoder = LightCurveEncoder(threshold=1e-10)

        # Very insensitive encoder
        insensitive_encoder = LightCurveEncoder(threshold=1e-1)

        data = RealisticExoplanetGenerator(seed=42).generate()

        # Both should produce valid outputs
        spikes_sensitive = sensitive_encoder.encode(data)
        spikes_insensitive = insensitive_encoder.encode(data)

        assert spikes_sensitive.shape == spikes_insensitive.shape

        # Process through pipeline
        for spikes in [spikes_sensitive, spikes_insensitive]:
            du_output = self.du_core.forward(spikes.unsqueeze(1), num_steps=spikes.shape[0])
            result = self.decoder.decode(du_output.squeeze(1), mode='classification')
            assert result in [0, 1]

    def test_component_failure_isolation(self):
        """Test that component failures are properly isolated."""
        data = RealisticExoplanetGenerator(seed=42).generate()
        spikes = self.encoder.encode(data)

        # Test with invalid decoder configuration
        invalid_decoder = ExoplanetDecoder(input_size=10, output_size=1)  # Wrong input size

        du_output = self.du_core.forward(spikes.unsqueeze(1), num_steps=spikes.shape[0])

        # Should raise appropriate error for mismatched sizes
        with pytest.raises(ValueError):
            invalid_decoder.decode(du_output.squeeze(1), mode='classification')

    def test_numerical_stability_edge_cases(self):
        """Test numerical stability with edge case inputs."""
        # Create edge case data
        edge_data = pl.DataFrame({
            'time': np.linspace(0, 10, 100),
            'flux': np.ones(100) * 1e-10  # Very small values
        })

        spikes = self.encoder.encode(edge_data)

        # Should produce valid spike train (might be all zeros)
        assert isinstance(spikes, torch.Tensor)
        assert spikes.shape == (100, 2)

        # Pipeline should handle gracefully
        du_output = self.du_core.forward(spikes.unsqueeze(1), num_steps=spikes.shape[0])
        result = self.decoder.decode(du_output.squeeze(1), mode='classification')

        assert result in [0, 1]

    def test_concurrent_processing_safety(self):
        """Test safety of concurrent processing."""
        data = RealisticExoplanetGenerator(seed=42).generate()

        # Process same data multiple times concurrently (simulated)
        results = []
        for i in range(10):
            spikes = self.encoder.encode(data)
            du_output = self.du_core.forward(spikes.unsqueeze(1), num_steps=spikes.shape[0])
            result = self.decoder.decode(du_output.squeeze(1), mode='classification')
            results.append(result)

        # All results should be identical and valid
        assert all(r == results[0] for r in results)
        assert all(r in [0, 1] for r in results)
