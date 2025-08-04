"""
Medical Domain Integration Tests
===============================

This module provides comprehensive integration testing for the medical domain
proof-of-concept, validating the complete pipeline from ECG data to arrhythmia
classification through the GIF framework.

Test Categories:
- End-to-end pipeline integration
- ECG encoder + decoder compatibility
- GIF framework integration
- Neuromorphic simulation compatibility
- Performance and scalability testing
- Cross-domain comparison with exoplanet POC
"""

import pytest
import torch
import numpy as np
import polars as pl
from unittest.mock import patch, MagicMock

from applications.poc_medical.encoders.ecg_encoder import ECG_Encoder
from applications.poc_medical.decoders.arrhythmia_decoder import Arrhythmia_Decoder
from data_generators.ecg_generator import RealisticECGGenerator
from gif_framework.orchestrator import GIF
from gif_framework.core.du_core import DU_Core_V1
from simulators.neuromorphic_sim import NeuromorphicSimulator


class TestMedicalPipelineIntegration:
    """Test complete medical domain pipeline integration."""
    
    @pytest.fixture
    def ecg_generator(self):
        """Create ECG data generator."""
        return RealisticECGGenerator(seed=42)
    
    @pytest.fixture
    def encoder(self):
        """Create ECG encoder."""
        return ECG_Encoder(sampling_rate=500, num_channels=3)
    
    @pytest.fixture
    def decoder(self):
        """Create arrhythmia decoder."""
        return Arrhythmia_Decoder(input_size=50, num_classes=8)
    
    @pytest.fixture
    def du_core(self):
        """Create DU Core for testing."""
        return DU_Core_V1(
            input_size=3,
            hidden_sizes=[50],
            output_size=50,
            beta=0.95,
            threshold=1.0
        )
    
    def test_encoder_decoder_compatibility(self, encoder, decoder, ecg_generator):
        """Test that encoder output is compatible with decoder input."""
        # Generate ECG data
        ecg_data = ecg_generator.generate(
            duration_seconds=5.0,
            sampling_rate=500,
            heart_rate_bpm=75.0,
            noise_levels={'baseline': 0.05, 'muscle': 0.01, 'powerline': 0.005}
        )
        
        # Encode to spike train
        spike_train = encoder.encode(ecg_data)
        assert spike_train.shape[1] == 3  # 3 channels from encoder
        
        # Create mock DU Core output with correct size for decoder
        mock_du_output = torch.rand(spike_train.shape[0], 50)  # 50 neurons for decoder
        
        # Decode to prediction
        prediction = decoder.decode(mock_du_output)
        assert isinstance(prediction, str)
        assert prediction in decoder.class_names
    
    def test_end_to_end_pipeline(self, encoder, decoder, du_core, ecg_generator):
        """Test complete end-to-end pipeline."""
        # Generate ECG data
        ecg_data = ecg_generator.generate(
            duration_seconds=3.0,
            sampling_rate=500,
            heart_rate_bpm=80.0,
            noise_levels={'baseline': 0.05, 'muscle': 0.01, 'powerline': 0.005}
        )
        
        # Encode ECG to spike train
        input_spikes = encoder.encode(ecg_data)
        assert input_spikes.shape == (1500, 3)  # 3 seconds * 500 Hz, 3 channels
        
        # Process through DU Core
        du_core.eval()
        with torch.no_grad():
            output_spikes = du_core.process(input_spikes.unsqueeze(0))  # Add batch dimension
            output_spikes = output_spikes.squeeze(0)  # Remove batch dimension
        
        # Decode to arrhythmia prediction
        prediction = decoder.decode(output_spikes)
        
        assert isinstance(prediction, str)
        assert prediction in decoder.class_names
        print(f"Pipeline prediction: {prediction}")
    
    def test_batch_processing(self, encoder, decoder, du_core, ecg_generator):
        """Test batch processing capabilities."""
        batch_size = 3
        ecg_batch = []
        
        # Generate batch of ECG data
        for i in range(batch_size):
            ecg_data = ecg_generator.generate(
                duration_seconds=2.0,
                sampling_rate=500,
                heart_rate_bpm=70 + i * 10,  # Vary heart rate
                noise_levels={'baseline': 0.05, 'muscle': 0.01, 'powerline': 0.005}
            )
            ecg_batch.append(ecg_data)
        
        # Process batch through encoder
        spike_batch = []
        for ecg_data in ecg_batch:
            spikes = encoder.encode(ecg_data)
            spike_batch.append(spikes)
        
        # Stack into batch tensor
        spike_batch_tensor = torch.stack(spike_batch)
        assert spike_batch_tensor.shape == (batch_size, 1000, 3)
        
        # Process through DU Core
        du_core.eval()
        with torch.no_grad():
            output_batch = du_core.process(spike_batch_tensor)
        
        assert output_batch.shape == (batch_size, 1000, 50)
        
        # Decode batch predictions
        predictions = []
        for i in range(batch_size):
            prediction = decoder.decode(output_batch[i])
            predictions.append(prediction)
            assert prediction in decoder.class_names
        
        assert len(predictions) == batch_size
    
    def test_different_ecg_parameters(self, encoder, decoder, ecg_generator):
        """Test pipeline with different ECG parameters."""
        test_cases = [
            {'heart_rate_bpm': 60, 'duration': 4.0},   # Bradycardia
            {'heart_rate_bpm': 100, 'duration': 3.0},  # Tachycardia
            {'heart_rate_bpm': 75, 'duration': 5.0},   # Normal
        ]
        
        for case in test_cases:
            ecg_data = ecg_generator.generate(
                duration_seconds=case['duration'],
                sampling_rate=500,
                heart_rate_bpm=case['heart_rate_bpm'],
                noise_levels={'baseline': 0.05, 'muscle': 0.01, 'powerline': 0.005}
            )
            
            # Encode
            spike_train = encoder.encode(ecg_data)
            expected_length = int(case['duration'] * 500)
            assert spike_train.shape == (expected_length, 3)
            
            # Mock DU Core processing
            mock_output = torch.rand(expected_length, 50)
            
            # Decode
            prediction = decoder.decode(mock_output)
            assert prediction in decoder.class_names
    
    def test_noise_robustness(self, encoder, decoder, ecg_generator):
        """Test pipeline robustness to different noise levels."""
        noise_levels_list = [
            {'baseline': 0.01, 'muscle': 0.005, 'powerline': 0.002},  # Low noise
            {'baseline': 0.1, 'muscle': 0.02, 'powerline': 0.01},    # Medium noise
            {'baseline': 0.2, 'muscle': 0.05, 'powerline': 0.02},    # High noise
        ]
        
        for noise_levels in noise_levels_list:
            ecg_data = ecg_generator.generate(
                duration_seconds=3.0,
                sampling_rate=500,
                heart_rate_bpm=75.0,
                noise_levels=noise_levels
            )
            
            # Pipeline should handle noisy data gracefully
            spike_train = encoder.encode(ecg_data)
            assert spike_train.shape == (1500, 3)
            
            # Check that encoding produces reasonable spike counts
            total_spikes = torch.sum(spike_train).item()
            assert total_spikes > 0  # Should have some spikes
            assert total_spikes < spike_train.numel() * 0.5  # But not too many


class TestGIFFrameworkIntegration:
    """Test integration with GIF framework orchestrator."""
    
    @pytest.fixture
    def gif_system(self):
        """Create complete GIF system with medical components."""
        du_core = DU_Core_V1(
            input_size=3,
            hidden_sizes=[30],
            output_size=30,
            beta=0.95
        )
        
        encoder = ECG_Encoder(sampling_rate=500)
        decoder = Arrhythmia_Decoder(input_size=30, num_classes=4)
        
        gif = GIF(du_core)
        gif.attach_encoder(encoder)
        gif.attach_decoder(decoder)
        
        return gif
    
    def test_gif_attachment(self, gif_system):
        """Test that medical components attach properly to GIF."""
        assert gif_system._encoder is not None
        assert gif_system._decoder is not None
        assert isinstance(gif_system._encoder, ECG_Encoder)
        assert isinstance(gif_system._decoder, Arrhythmia_Decoder)
    
    def test_gif_processing(self, gif_system):
        """Test processing through complete GIF system."""
        generator = RealisticECGGenerator(seed=42)
        ecg_data = generator.generate(
            duration_seconds=2.0,
            sampling_rate=500,
            heart_rate_bpm=75.0,
            noise_levels={'baseline': 0.05, 'muscle': 0.01, 'powerline': 0.005}
        )
        
        # Process through GIF system
        result = gif_system.process_single_input(ecg_data)
        
        assert isinstance(result, str)
        assert result in gif_system._decoder.class_names
    
    def test_gif_configuration(self, gif_system):
        """Test GIF system configuration retrieval."""
        config = gif_system.get_system_config()
        
        assert 'encoder_config' in config
        assert 'decoder_config' in config
        assert 'du_core_config' in config
        
        assert config['encoder_config']['encoder_type'] == 'ECG_Encoder'
        assert config['decoder_config']['decoder_type'] == 'Arrhythmia_Decoder'


class TestNeuromorphicSimulationIntegration:
    """Test integration with neuromorphic simulation."""
    
    @pytest.fixture
    def neuromorphic_setup(self):
        """Create neuromorphic simulation setup."""
        du_core = DU_Core_V1(
            input_size=3,
            hidden_sizes=[25],
            output_size=25,
            beta=0.95
        )
        
        simulator = NeuromorphicSimulator(
            snn_model=du_core,
            energy_per_synop=2.5e-11
        )
        
        encoder = ECG_Encoder(sampling_rate=500)
        decoder = Arrhythmia_Decoder(input_size=25, num_classes=3)
        
        return {
            'du_core': du_core,
            'simulator': simulator,
            'encoder': encoder,
            'decoder': decoder
        }
    
    def test_neuromorphic_simulation(self, neuromorphic_setup):
        """Test medical pipeline with neuromorphic simulation."""
        components = neuromorphic_setup
        
        # Generate ECG data
        generator = RealisticECGGenerator(seed=42)
        ecg_data = generator.generate(
            duration_seconds=2.0,
            sampling_rate=500,
            heart_rate_bpm=75.0,
            noise_levels={'baseline': 0.05, 'muscle': 0.01, 'powerline': 0.005}
        )
        
        # Encode to spikes
        input_spikes = components['encoder'].encode(ecg_data)
        
        # Simulate through neuromorphic processor
        simulation_results = components['simulator'].run(
            input_spikes.unsqueeze(0)  # Add batch dimension
        )
        
        output_spikes, stats = simulation_results

        # Decode results
        output_spikes = output_spikes.squeeze(0)
        prediction = components['decoder'].decode(output_spikes)

        assert isinstance(prediction, str)
        assert prediction in components['decoder'].class_names

        # Verify energy consumption is calculated
        assert 'estimated_energy_joules' in stats
        assert 'total_synops' in stats
        assert stats['estimated_energy_joules'] > 0
        assert stats['total_synops'] > 0
    
    def test_energy_efficiency_comparison(self, neuromorphic_setup):
        """Test energy efficiency metrics for medical processing."""
        components = neuromorphic_setup
        
        # Generate test data
        generator = RealisticECGGenerator(seed=42)
        ecg_data = generator.generate(
            duration_seconds=1.0,
            sampling_rate=500,
            heart_rate_bpm=75.0,
            noise_levels={'baseline': 0.05, 'muscle': 0.01, 'powerline': 0.005}
        )
        
        input_spikes = components['encoder'].encode(ecg_data)
        
        # Run simulation
        output_spikes, stats = components['simulator'].run(input_spikes.unsqueeze(0))

        # Calculate energy per classification
        energy_per_classification = stats['estimated_energy_joules']
        
        # Should be in reasonable range for neuromorphic processing
        assert energy_per_classification < 1e-6  # Less than 1 microjoule
        assert energy_per_classification > 1e-12  # More than 1 picojoule
        
        print(f"Energy per ECG classification: {energy_per_classification:.2e} J")


class TestCrossDomainComparison:
    """Test cross-domain capabilities by comparing with exoplanet POC."""
    
    def test_interface_consistency(self):
        """Test that medical and exoplanet modules use consistent interfaces."""
        from applications.poc_exoplanet.encoders.light_curve_encoder import LightCurveEncoder
        from applications.poc_exoplanet.decoders.exoplanet_decoder import ExoplanetDecoder
        
        # Create both domain modules
        ecg_encoder = ECG_Encoder(sampling_rate=500)
        arrhythmia_decoder = Arrhythmia_Decoder(input_size=10, num_classes=3)
        
        light_curve_encoder = LightCurveEncoder(threshold=1e-4)
        exoplanet_decoder = ExoplanetDecoder(input_size=10, output_size=1)
        
        # Both should implement the same interfaces
        from gif_framework.interfaces.base_interfaces import EncoderInterface, DecoderInterface
        
        assert isinstance(ecg_encoder, EncoderInterface)
        assert isinstance(arrhythmia_decoder, DecoderInterface)
        assert isinstance(light_curve_encoder, EncoderInterface)
        assert isinstance(exoplanet_decoder, DecoderInterface)
        
        # Both should have the same interface methods
        encoder_methods = ['encode', 'get_config', 'calibrate']
        decoder_methods = ['decode', 'get_config']
        
        for method in encoder_methods:
            assert hasattr(ecg_encoder, method)
            assert hasattr(light_curve_encoder, method)
        
        for method in decoder_methods:
            assert hasattr(arrhythmia_decoder, method)
            assert hasattr(exoplanet_decoder, method)
    
    def test_du_core_compatibility(self):
        """Test that both domains can use the same DU Core."""
        du_core = DU_Core_V1(
            input_size=3,  # Works for both 2-channel exoplanet and 3-channel ECG
            hidden_sizes=[20],
            output_size=20,
            beta=0.95
        )
        
        # Test with ECG data
        ecg_encoder = ECG_Encoder(sampling_rate=500)
        generator = RealisticECGGenerator(seed=42)
        ecg_data = generator.generate(
            duration_seconds=1.0,
            sampling_rate=500,
            heart_rate_bpm=75.0,
            noise_levels={'baseline': 0.05, 'muscle': 0.01, 'powerline': 0.005}
        )
        ecg_spikes = ecg_encoder.encode(ecg_data)
        
        # Process ECG through DU Core
        du_core.eval()
        with torch.no_grad():
            ecg_output = du_core.process(ecg_spikes.unsqueeze(0))
        
        assert ecg_output.shape == (1, 500, 20)  # batch, time, neurons
        
        # Test with exoplanet data (simulated)
        # Create mock light curve spikes (2 channels)
        mock_light_curve_spikes = torch.rand(500, 2)
        # Pad to 3 channels for compatibility
        padded_spikes = torch.cat([mock_light_curve_spikes, torch.zeros(500, 1)], dim=1)
        
        with torch.no_grad():
            exoplanet_output = du_core.process(padded_spikes.unsqueeze(0))
        
        assert exoplanet_output.shape == (1, 500, 20)
        
        # Both outputs should have similar structure
        assert ecg_output.shape == exoplanet_output.shape


class TestPerformanceAndScalability:
    """Test performance and scalability of medical domain pipeline."""
    
    def test_processing_speed(self):
        """Test processing speed for medical pipeline."""
        import time
        
        encoder = ECG_Encoder(sampling_rate=500)
        decoder = Arrhythmia_Decoder(input_size=50, num_classes=8)
        generator = RealisticECGGenerator(seed=42)
        
        # Generate longer ECG data
        ecg_data = generator.generate(
            duration_seconds=10.0,
            sampling_rate=500,
            heart_rate_bpm=75.0,
            noise_levels={'baseline': 0.05, 'muscle': 0.01, 'powerline': 0.005}
        )
        
        # Time encoding
        start_time = time.time()
        spike_train = encoder.encode(ecg_data)
        encoding_time = time.time() - start_time
        
        # Time decoding (with mock DU output)
        mock_output = torch.rand(spike_train.shape[0], 50)
        start_time = time.time()
        prediction = decoder.decode(mock_output)
        decoding_time = time.time() - start_time
        
        # Should process reasonably quickly
        assert encoding_time < 5.0  # Less than 5 seconds for 10s of ECG
        assert decoding_time < 1.0  # Less than 1 second for decoding
        
        print(f"Encoding time: {encoding_time:.3f}s, Decoding time: {decoding_time:.3f}s")
    
    def test_memory_usage(self):
        """Test memory usage of medical pipeline components."""
        encoder = ECG_Encoder(sampling_rate=500)
        decoder = Arrhythmia_Decoder(input_size=100, num_classes=8)
        
        # Test with various input sizes
        for duration in [1.0, 5.0, 10.0]:
            generator = RealisticECGGenerator(seed=42)
            ecg_data = generator.generate(
                duration_seconds=duration,
                sampling_rate=500,
                heart_rate_bpm=75.0,
                noise_levels={'baseline': 0.05, 'muscle': 0.01, 'powerline': 0.005}
            )
            
            spike_train = encoder.encode(ecg_data)
            expected_shape = (int(duration * 500), 3)
            assert spike_train.shape == expected_shape
            
            # Memory usage should scale linearly with input size
            memory_mb = spike_train.numel() * 4 / (1024 * 1024)  # 4 bytes per float32
            assert memory_mb < 100  # Should be reasonable for typical use
