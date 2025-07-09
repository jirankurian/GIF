"""
Unit Tests for GIF Orchestrator and Integration Testing
========================================================

This module contains comprehensive tests for the GIF orchestrator class
and the critical integration test that validates the entire framework
pipeline from end to end.

Test Categories:
- GIF initialization and dependency injection
- Module attachment and validation
- Error handling and edge cases
- System configuration and monitoring
- Complete integration pipeline testing

The integration test is the most important test in the entire framework,
as it validates that all components work together correctly through the
complete cognitive cycle: encode → process → decode.
"""

import pytest
import torch
from typing import Any

from gif_framework.orchestrator import GIF
from gif_framework.interfaces.base_interfaces import (
    EncoderInterface, 
    DecoderInterface,
    SpikeTrain,
    Action
)
from tests.mocks import MockEncoder, MockDecoder, InvalidMock, create_test_raw_data

# Import DU Core for integration testing
try:
    from gif_framework.core.du_core import DU_Core_V1
    DU_CORE_AVAILABLE = True
except ImportError:
    DU_CORE_AVAILABLE = False


class TestGIFInitialization:
    """Test GIF orchestrator initialization and dependency injection."""
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_valid_initialization(self):
        """Test valid GIF initialization with DU Core dependency injection."""
        # Create a DU Core instance
        du_core = DU_Core_V1(input_size=20, hidden_sizes=[15], output_size=10)
        
        # Initialize GIF with dependency injection
        gif = GIF(du_core)
        
        # Check that DU core is properly stored
        assert gif._du_core is du_core
        
        # Check that encoder and decoder are initially None
        assert gif._encoder is None
        assert gif._decoder is None
    
    def test_initialization_with_none_du_core(self):
        """Test that GIF initialization fails with None DU core."""
        with pytest.raises(TypeError, match="DU Core cannot be None"):
            GIF(None)
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_initialization_stores_du_core_reference(self):
        """Test that GIF stores reference to DU core, not a copy."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[], output_size=5)
        gif = GIF(du_core)
        
        # Should be the same object (reference, not copy)
        assert gif._du_core is du_core


class TestGIFModuleAttachment:
    """Test GIF module attachment and validation."""
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_attach_valid_encoder(self):
        """Test attaching a valid encoder to GIF."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)
        gif = GIF(du_core)
        
        # Create and attach mock encoder
        mock_encoder = MockEncoder()
        gif.attach_encoder(mock_encoder)
        
        # Check that encoder is properly attached
        assert gif._encoder is mock_encoder
        assert isinstance(gif._encoder, EncoderInterface)
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_attach_valid_decoder(self):
        """Test attaching a valid decoder to GIF."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)
        gif = GIF(du_core)
        
        # Create and attach mock decoder
        mock_decoder = MockDecoder()
        gif.attach_decoder(mock_decoder)
        
        # Check that decoder is properly attached
        assert gif._decoder is mock_decoder
        assert isinstance(gif._decoder, DecoderInterface)
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_attach_none_encoder(self):
        """Test that attaching None encoder raises TypeError."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)
        gif = GIF(du_core)
        
        with pytest.raises(TypeError, match="Encoder cannot be None"):
            gif.attach_encoder(None)
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_attach_none_decoder(self):
        """Test that attaching None decoder raises TypeError."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)
        gif = GIF(du_core)
        
        with pytest.raises(TypeError, match="Decoder cannot be None"):
            gif.attach_decoder(None)
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_attach_invalid_encoder(self):
        """Test that attaching invalid encoder raises TypeError."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)
        gif = GIF(du_core)
        
        # Try to attach invalid object
        invalid_mock = InvalidMock()
        
        with pytest.raises(TypeError, match="Encoder must implement EncoderInterface"):
            gif.attach_encoder(invalid_mock)
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_attach_invalid_decoder(self):
        """Test that attaching invalid decoder raises TypeError."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)
        gif = GIF(du_core)
        
        # Try to attach invalid object
        invalid_mock = InvalidMock()
        
        with pytest.raises(TypeError, match="Decoder must implement DecoderInterface"):
            gif.attach_decoder(invalid_mock)
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_attach_string_as_encoder(self):
        """Test that attaching string as encoder raises TypeError."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)
        gif = GIF(du_core)
        
        with pytest.raises(TypeError, match="Encoder must implement EncoderInterface"):
            gif.attach_encoder("not_an_encoder")
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_encoder_replacement(self):
        """Test that attaching new encoder replaces the old one."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)
        gif = GIF(du_core)
        
        # Attach first encoder
        encoder1 = MockEncoder(output_shape=(10, 1, 15))
        gif.attach_encoder(encoder1)
        assert gif._encoder is encoder1
        
        # Attach second encoder (should replace first)
        encoder2 = MockEncoder(output_shape=(10, 1, 20))
        gif.attach_encoder(encoder2)
        assert gif._encoder is encoder2
        assert gif._encoder is not encoder1


class TestGIFProcessingErrors:
    """Test GIF processing error handling."""
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_process_without_encoder_fails(self):
        """Test that processing without encoder raises RuntimeError."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)
        gif = GIF(du_core)
        
        # Attach only decoder, not encoder
        gif.attach_decoder(MockDecoder())
        
        with pytest.raises(RuntimeError, match="No encoder attached"):
            gif.process_single_input("test_data")


class TestGIFSystemConfiguration:
    """Test GIF system configuration and monitoring."""

    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_get_system_config_complete_system(self):
        """Test get_system_config with complete system setup."""
        du_core = DU_Core_V1(input_size=20, hidden_sizes=[15], output_size=10)
        gif = GIF(du_core)

        # Attach modules
        encoder = MockEncoder()
        decoder = MockDecoder()
        gif.attach_encoder(encoder)
        gif.attach_decoder(decoder)

        # Get system configuration
        config = gif.get_system_config()

        # Check configuration structure
        assert "du_core_config" in config
        assert "encoder_config" in config
        assert "decoder_config" in config
        assert "system_status" in config

        # Check system status
        status = config["system_status"]
        assert status["ready"] == True
        assert status["encoder_attached"] == True
        assert status["decoder_attached"] == True
        assert status["encoder_type"] == "MockEncoder"
        assert status["decoder_type"] == "MockDecoder"
        assert status["du_core_type"] == "DU_Core_V1"

    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_get_system_config_incomplete_system(self):
        """Test get_system_config with incomplete system setup."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[], output_size=5)
        gif = GIF(du_core)

        # Don't attach any modules
        config = gif.get_system_config()

        # Check system status
        status = config["system_status"]
        assert status["ready"] == False
        assert status["encoder_attached"] == False
        assert status["decoder_attached"] == False
        assert status["encoder_type"] is None
        assert status["decoder_type"] is None
        assert status["du_core_type"] == "DU_Core_V1"


class TestGIFIntegration:
    """
    CRITICAL INTEGRATION TESTS

    These tests validate the complete GIF framework pipeline from end to end.
    The integration test is the most important test in the entire framework.
    """

    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_full_integration_cycle(self):
        """
        CRITICAL TEST: Complete integration of the entire GIF framework.

        This test validates the complete cognitive cycle:
        Raw Data → Encoder → SpikeTrain → DU Core → Processed SpikeTrain → Decoder → Action

        This is the most important test in the framework as it proves that
        all architectural components work together correctly.
        """
        # Step 1: Create the DU Core (the "brain")
        du_core = DU_Core_V1(
            input_size=20,  # Must match MockEncoder output
            hidden_sizes=[15, 10],
            output_size=8
        )

        # Step 2: Create GIF orchestrator with dependency injection
        gif = GIF(du_core)

        # Step 3: Create and attach mock components
        mock_encoder = MockEncoder(output_shape=(25, 1, 20))  # 25 steps, batch 1, 20 features
        mock_decoder = MockDecoder(mock_action="mock_action_success")

        gif.attach_encoder(mock_encoder)
        gif.attach_decoder(mock_decoder)

        # Step 4: Create test raw data
        raw_data = create_test_raw_data()

        # Step 5: Execute the complete cognitive cycle
        result = gif.process_single_input(raw_data)

        # Step 6: Validate the result
        assert result == "mock_action_success"

        # Step 7: Validate that all components were called
        assert mock_encoder.calibration_calls == 0  # Calibration not called in normal processing
        assert mock_decoder.decode_calls == 1  # Decoder should have been called once

        # Step 8: Validate system is ready
        config = gif.get_system_config()
        assert config["system_status"]["ready"] == True

    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_integration_with_different_configurations(self):
        """Test integration with different DU Core configurations."""
        # Test with minimal configuration
        du_core_minimal = DU_Core_V1(input_size=10, hidden_sizes=[], output_size=5)
        gif_minimal = GIF(du_core_minimal)

        encoder = MockEncoder(output_shape=(15, 1, 10))
        decoder = MockDecoder(mock_action="minimal_success")

        gif_minimal.attach_encoder(encoder)
        gif_minimal.attach_decoder(decoder)

        result = gif_minimal.process_single_input("test")
        assert result == "minimal_success"

        # Test with complex configuration
        du_core_complex = DU_Core_V1(
            input_size=50,
            hidden_sizes=[40, 30, 20, 10],
            output_size=5,
            beta=0.9,
            threshold=0.8
        )
        gif_complex = GIF(du_core_complex)

        encoder_complex = MockEncoder(output_shape=(20, 2, 50))  # Batch size 2
        decoder_complex = MockDecoder(mock_action="complex_success")

        gif_complex.attach_encoder(encoder_complex)
        gif_complex.attach_decoder(decoder_complex)

        result_complex = gif_complex.process_single_input("complex_test")
        assert result_complex == "complex_success"

    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_integration_error_propagation(self):
        """Test that errors in the pipeline are properly propagated."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)
        gif = GIF(du_core)

        # Create encoder that outputs wrong size (should cause error in DU Core)
        class BadEncoder(MockEncoder):
            def encode(self, raw_data):
                # Return wrong size spike train (15 instead of 10)
                return torch.rand(10, 1, 15)

        bad_encoder = BadEncoder()
        decoder = MockDecoder()

        gif.attach_encoder(bad_encoder)
        gif.attach_decoder(decoder)

        # Should raise an error due to size mismatch
        with pytest.raises(RuntimeError, match="SNN processing failed"):
            gif.process_single_input("test_data")

    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_integration_multiple_cycles(self):
        """Test multiple processing cycles with the same GIF instance."""
        du_core = DU_Core_V1(input_size=15, hidden_sizes=[10], output_size=5)
        gif = GIF(du_core)

        encoder = MockEncoder(output_shape=(20, 1, 15))
        decoder = MockDecoder(mock_action="cycle_success")

        gif.attach_encoder(encoder)
        gif.attach_decoder(decoder)

        # Run multiple cycles
        for i in range(5):
            result = gif.process_single_input(f"test_data_{i}")
            assert result == "cycle_success"

        # Check that decoder was called for each cycle
        assert decoder.decode_calls == 5
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_process_without_decoder_fails(self):
        """Test that processing without decoder raises RuntimeError."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)
        gif = GIF(du_core)
        
        # Attach only encoder, not decoder
        gif.attach_encoder(MockEncoder())
        
        with pytest.raises(RuntimeError, match="No decoder attached"):
            gif.process_single_input("test_data")
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_process_without_any_modules_fails(self):
        """Test that processing without any modules raises RuntimeError."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)
        gif = GIF(du_core)
        
        # Don't attach any modules
        
        with pytest.raises(RuntimeError, match="No encoder attached"):
            gif.process_single_input("test_data")
