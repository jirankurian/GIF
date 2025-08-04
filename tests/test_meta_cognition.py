"""
Comprehensive Test Suite for Meta-Cognitive Routing
==================================================

This module provides rigorous testing for the meta-cognitive routing capabilities
of the GIF framework, including intelligent encoder selection, task description
analysis, and module library integration.

Test Categories:
- Meta-cognitive encoder selection based on task descriptions
- Specialized encoder functionality (Fourier and Wavelet)
- Module library integration and routing
- Fallback mechanisms and error handling
- Integration with GIF orchestrator

The test suite ensures that the meta-cognitive routing system can intelligently
select appropriate encoders based on natural language task descriptions,
representing a significant step toward AGI capabilities.
"""

import pytest
import torch
import numpy as np
import polars as pl
from unittest.mock import Mock, MagicMock
from typing import Any, Dict

from gif_framework.orchestrator import GIF, MetaController
from gif_framework.module_library import ModuleLibrary, ModuleMetadata
from gif_framework.interfaces.base_interfaces import EncoderInterface, DecoderInterface, SpikeTrain, Action
from applications.poc_exoplanet.encoders.fourier_encoder import FourierEncoder
from applications.poc_exoplanet.encoders.wavelet_encoder import WaveletEncoder
from applications.poc_exoplanet.encoders.light_curve_encoder import LightCurveEncoder


class MockDUCore:
    """Mock DU Core for testing."""

    def __init__(self):
        self.processed_count = 0

    def process(self, spike_train: SpikeTrain) -> SpikeTrain:
        """Mock processing that returns the input unchanged."""
        self.processed_count += 1
        return spike_train

    def get_config(self) -> Dict[str, Any]:
        return {'type': 'MockDUCore', 'processed_count': self.processed_count}

    def parameters(self):
        return iter([])

    def named_parameters(self):
        return iter([])

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def train(self, mode=True):
        return self

    def eval(self):
        return self


class MockDecoder(DecoderInterface):
    """Mock decoder for testing."""

    def decode(self, spike_train: SpikeTrain) -> Action:
        """Mock decoding that returns spike count."""
        return torch.sum(spike_train).item()

    def get_config(self) -> Dict[str, Any]:
        return {'type': 'MockDecoder', 'output_format': 'regression'}


class TestMetaCognitiveRouting:
    """Test meta-cognitive routing capabilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.du_core = MockDUCore()
        self.decoder = MockDecoder()

        # Create sample light curve data
        time = np.linspace(0, 10, 1000)
        flux = 1.0 + 0.01 * np.sin(2 * np.pi * time) + 0.005 * np.random.randn(1000)
        self.light_curve_data = pl.DataFrame({
            'time': time,
            'flux': flux
        })

    def test_meta_controller_selects_fourier_encoder_for_periodic_signals(self):
        """Test that meta-controller selects FourierEncoder for periodic signal tasks."""
        # Create module library with specialized encoders
        library = ModuleLibrary()

        # Register FourierEncoder
        fourier_encoder = FourierEncoder()
        fourier_metadata = ModuleMetadata(
            module_type='encoder',
            data_modality='timeseries',
            signal_type='periodic',
            domain='astronomy',
            input_format='dataframe',
            description='Best for detecting periodic signals'
        )
        library.register_module(fourier_encoder, fourier_metadata)

        # Register WaveletEncoder
        wavelet_encoder = WaveletEncoder()
        wavelet_metadata = ModuleMetadata(
            module_type='encoder',
            data_modality='timeseries',
            signal_type='transient',
            domain='astronomy',
            input_format='dataframe',
            description='Best for detecting transient events'
        )
        library.register_module(wavelet_encoder, wavelet_metadata)

        # Create GIF with meta-cognitive routing
        gif = GIF(self.du_core, module_library=library)
        gif.attach_decoder(self.decoder)

        # Test periodic signal task description
        task_description = "Analyze this signal to find repeating patterns and periodic behavior"
        selected_encoder = gif._meta_control_select_encoder(task_description)

        assert selected_encoder is not None
        assert isinstance(selected_encoder, FourierEncoder)


    def test_meta_controller_selects_wavelet_encoder_for_transient_signals(self):
        """Test that meta-controller selects WaveletEncoder for transient signal tasks."""
        # Create module library with specialized encoders
        library = ModuleLibrary()

        # Register encoders
        fourier_encoder = FourierEncoder()
        fourier_metadata = ModuleMetadata(
            module_type='encoder',
            data_modality='timeseries',
            signal_type='periodic',
            domain='astronomy',
            input_format='dataframe'
        )
        library.register_module(fourier_encoder, fourier_metadata)

        wavelet_encoder = WaveletEncoder()
        wavelet_metadata = ModuleMetadata(
            module_type='encoder',
            data_modality='timeseries',
            signal_type='transient',
            domain='astronomy',
            input_format='dataframe'
        )
        library.register_module(wavelet_encoder, wavelet_metadata)

        # Create GIF with meta-cognitive routing
        gif = GIF(self.du_core, module_library=library)
        gif.attach_decoder(self.decoder)

        # Test transient signal task description
        task_description = "Find sudden bursts and flares in this astronomical data"
        selected_encoder = gif._meta_control_select_encoder(task_description)

        assert selected_encoder is not None
        assert isinstance(selected_encoder, WaveletEncoder)
    def test_meta_controller_fallback_to_general_encoder(self):
        """Test fallback to general encoder when no specific match found."""
        # Create module library with general encoder
        library = ModuleLibrary()

        # Register general-purpose encoder
        light_curve_encoder = LightCurveEncoder()
        general_metadata = ModuleMetadata(
            module_type='encoder',
            data_modality='timeseries',
            signal_type='general',
            domain='astronomy',
            input_format='dataframe'
        )
        library.register_module(light_curve_encoder, general_metadata)

        # Create GIF with meta-cognitive routing
        gif = GIF(self.du_core, module_library=library)
        gif.attach_decoder(self.decoder)

        # Test ambiguous task description
        task_description = "Process this data for analysis"
        selected_encoder = gif._meta_control_select_encoder(task_description)

        assert selected_encoder is not None
        assert isinstance(selected_encoder, LightCurveEncoder)

    def test_process_single_input_with_task_description(self):
        """Test complete processing workflow with task description-based routing."""
        # Create module library with specialized encoders
        library = ModuleLibrary()

        # Register FourierEncoder for periodic signals
        fourier_encoder = FourierEncoder()
        fourier_metadata = ModuleMetadata(
            module_type='encoder',
            data_modality='timeseries',
            signal_type='periodic',
            domain='astronomy',
            input_format='dataframe'
        )
        library.register_module(fourier_encoder, fourier_metadata)

        # Create GIF with meta-cognitive routing
        gif = GIF(self.du_core, module_library=library)
        gif.attach_decoder(self.decoder)

        # Process with task description
        task_description = "Find periodic patterns in this light curve data"
        result = gif.process_single_input(self.light_curve_data, task_description)

        # Verify processing completed successfully
        assert result is not None
        assert isinstance(result, (int, float))
        assert self.du_core.processed_count == 1
    def test_meta_cognitive_routing_without_module_library(self):
        """Test that system falls back gracefully when no module library is available."""
        # Create GIF without module library
        gif = GIF(self.du_core)

        # Attach encoder and decoder manually
        encoder = LightCurveEncoder()
        gif.attach_encoder(encoder)
        gif.attach_decoder(self.decoder)

        # Process with task description (should use attached encoder)
        task_description = "Find periodic patterns in this data"
        result = gif.process_single_input(self.light_curve_data, task_description)

        # Verify processing completed successfully
        assert result is not None
        assert isinstance(result, (int, float))


class TestSpecializedEncoders:
    """Test specialized encoder functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample light curve data with periodic component
        time = np.linspace(0, 10, 1000)
        periodic_signal = 1.0 + 0.05 * np.sin(2 * np.pi * 0.5 * time)  # 0.5 Hz periodic
        noise = 0.01 * np.random.randn(1000)
        flux = periodic_signal + noise

        self.periodic_data = pl.DataFrame({
            'time': time,
            'flux': flux
        })

        # Create transient signal
        transient_signal = np.zeros(1000)
        transient_signal[500:520] = 0.1  # Burst at middle
        transient_flux = 1.0 + transient_signal + 0.005 * np.random.randn(1000)

        self.transient_data = pl.DataFrame({
            'time': time,
            'flux': transient_flux
        })
    def test_fourier_encoder_detects_periodic_signals(self):
        """Test that FourierEncoder effectively detects periodic components."""
        encoder = FourierEncoder(n_frequency_channels=4)

        # Encode periodic signal
        spike_train = encoder.encode(self.periodic_data)

        # Verify output shape and content
        assert spike_train.shape[0] == len(self.periodic_data)
        assert spike_train.shape[1] == 4  # n_frequency_channels
        assert torch.sum(spike_train) > 0  # Should generate spikes

        # Check encoding statistics
        stats = encoder.get_encoding_stats()
        assert stats['dominant_frequency'] > 0  # Should detect frequency
        assert stats['total_spikes'] > 0

        # Verify configuration
        config = encoder.get_config()
        assert config['signal_type'] == 'periodic'
        assert config['encoder_type'] == 'FourierEncoder'

    def test_wavelet_encoder_detects_transient_events(self):
        """Test that WaveletEncoder effectively detects transient events."""
        encoder = WaveletEncoder(n_scales=8, n_time_channels=4)

        # Encode transient signal
        spike_train = encoder.encode(self.transient_data)

        # Verify output shape and content
        assert spike_train.shape[0] == len(self.transient_data)
        assert spike_train.shape[1] == 4  # n_time_channels
        assert torch.sum(spike_train) > 0  # Should generate spikes

        # Check encoding statistics
        stats = encoder.get_encoding_stats()
        assert stats['transient_events'] > 0  # Should detect transient events
        assert stats['max_energy'] > 0

        # Verify configuration
        config = encoder.get_config()
        assert config['signal_type'] == 'transient'
        assert config['encoder_type'] == 'WaveletEncoder'
    def test_encoder_configuration_metadata(self):
        """Test that encoders provide correct metadata for meta-cognitive routing."""
        fourier_encoder = FourierEncoder()
        wavelet_encoder = WaveletEncoder()

        # Test FourierEncoder metadata
        fourier_config = fourier_encoder.get_config()
        assert fourier_config['signal_type'] == 'periodic'
        assert 'periodic' in fourier_config['description'].lower()
        assert 'optimal_for' in fourier_config

        # Test WaveletEncoder metadata
        wavelet_config = wavelet_encoder.get_config()
        assert wavelet_config['signal_type'] == 'transient'
        assert 'transient' in wavelet_config['description'].lower()
        assert 'optimal_for' in wavelet_config


class TestModuleLibraryIntegration:
    """Test module library integration with meta-cognitive routing."""

    def test_module_library_encoder_selection_by_signal_type(self):
        """Test module library can find encoders by signal type."""
        library = ModuleLibrary()

        # Register different encoder types
        fourier_encoder = FourierEncoder()
        fourier_metadata = ModuleMetadata(
            module_type='encoder',
            signal_type='periodic',
            data_modality='timeseries',
            domain='astronomy',
            input_format='dataframe'
        )
        library.register_module(fourier_encoder, fourier_metadata)

        wavelet_encoder = WaveletEncoder()
        wavelet_metadata = ModuleMetadata(
            module_type='encoder',
            signal_type='transient',
            data_modality='timeseries',
            domain='astronomy',
            input_format='dataframe'
        )
        library.register_module(wavelet_encoder, wavelet_metadata)

        # Test signal type selection
        periodic_encoders = library.get_encoders_by_signal_type('periodic')
        assert len(periodic_encoders) == 1
        assert isinstance(periodic_encoders[0], FourierEncoder)

        transient_encoders = library.get_encoders_by_signal_type('transient')
        assert len(transient_encoders) == 1
        assert isinstance(transient_encoders[0], WaveletEncoder)

        # Test non-existent signal type
        unknown_encoders = library.get_encoders_by_signal_type('unknown')
        assert len(unknown_encoders) == 0
    def test_module_library_statistics_include_signal_types(self):
        """Test that library statistics include signal type information."""
        library = ModuleLibrary()

        # Register encoders with different signal types
        fourier_encoder = FourierEncoder()
        fourier_metadata = ModuleMetadata(
            module_type='encoder',
            signal_type='periodic',
            data_modality='timeseries',
            domain='astronomy',
            input_format='dataframe'
        )
        library.register_module(fourier_encoder, fourier_metadata)

        wavelet_encoder = WaveletEncoder()
        wavelet_metadata = ModuleMetadata(
            module_type='encoder',
            signal_type='transient',
            data_modality='timeseries',
            domain='astronomy',
            input_format='dataframe'
        )
        library.register_module(wavelet_encoder, wavelet_metadata)

        # Check statistics
        stats = library.get_library_stats()
        assert 'encoder_signal_types' in stats
        assert stats['encoder_signal_types']['periodic'] == 1
        assert stats['encoder_signal_types']['transient'] == 1
        assert stats['encoder_count'] == 2


class TestErrorHandlingAndFallbacks:
    """Test error handling and fallback mechanisms."""

    def test_meta_controller_handles_empty_task_description(self):
        """Test meta-controller handles empty or None task descriptions."""
        library = ModuleLibrary()
        du_core = MockDUCore()
        gif = GIF(du_core, module_library=library)

        # Test with None task description
        result = gif._meta_control_select_encoder(None)
        assert result is None  # Should return None when no encoder attached

        # Test with empty string
        result = gif._meta_control_select_encoder("")
        assert result is None
    def test_meta_controller_handles_no_matching_encoders(self):
        """Test meta-controller handles case when no encoders match task description."""
        library = ModuleLibrary()  # Empty library
        du_core = MockDUCore()
        gif = GIF(du_core, module_library=library)

        # Test with task description but no registered encoders
        task_description = "Find periodic patterns in this data"
        result = gif._meta_control_select_encoder(task_description)
        assert result is None

    def test_process_single_input_error_handling_with_meta_routing(self):
        """Test error handling in process_single_input with meta-cognitive routing."""
        library = ModuleLibrary()
        du_core = MockDUCore()
        gif = GIF(du_core, module_library=library)

        # Try to process without any encoders or decoders
        with pytest.raises(RuntimeError, match="No encoder available"):
            gif.process_single_input([], "test task")

        # Attach decoder but no encoder
        gif.attach_decoder(MockDecoder())
        with pytest.raises(RuntimeError, match="No encoder available"):
            gif.process_single_input([], "test task")






