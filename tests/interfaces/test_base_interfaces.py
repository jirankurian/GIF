"""
Unit Tests for Base Interfaces
===============================

This module contains tests for the base interface contracts of the GIF framework.
These tests ensure that the interface definitions are correct and that the
contract enforcement mechanisms work properly.

Test Categories:
- EncoderInterface contract validation
- DecoderInterface contract validation
- Type alias verification
- Mock implementation validation
- Interface inheritance testing

The tests verify that the interfaces properly define the contracts that
all encoders and decoders must follow.
"""

import pytest
import torch
from abc import ABC
from typing import Any, Dict

from gif_framework.interfaces.base_interfaces import (
    EncoderInterface,
    DecoderInterface,
    SpikeTrain,
    Action
)
from tests.mocks import (
    MockEncoder,
    MockDecoder,
    InvalidMock,
    FailingMockEncoder,
    FailingMockDecoder,
    PerformanceMockEncoder
)


class TestTypeAliases:
    """Test the type aliases defined in base_interfaces."""
    
    def test_spike_train_alias(self):
        """Test that SpikeTrain is correctly aliased to torch.Tensor."""
        # Create a tensor and verify it's a valid SpikeTrain
        tensor = torch.rand(10, 1, 20)
        assert isinstance(tensor, torch.Tensor)
        
        # SpikeTrain should be the same as torch.Tensor
        assert SpikeTrain == torch.Tensor
    
    def test_action_alias(self):
        """Test that Action alias accepts various types."""
        # Action should accept any type (it's aliased to Any)
        string_action = "classification_result"
        int_action = 42
        dict_action = {"prediction": 0.95, "confidence": 0.87}
        list_action = [1, 2, 3]
        
        # All of these should be valid Action types
        # (We can't directly test isinstance with Any, but we can verify the alias exists)
        assert Action == Any


class TestEncoderInterface:
    """Test EncoderInterface contract and implementation."""
    
    def test_encoder_interface_is_abstract(self):
        """Test that EncoderInterface cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            EncoderInterface()
    
    def test_encoder_interface_inheritance(self):
        """Test that EncoderInterface properly inherits from ABC."""
        assert issubclass(EncoderInterface, ABC)
        
        # Check that required methods are abstract
        abstract_methods = EncoderInterface.__abstractmethods__
        assert "encode" in abstract_methods
        assert "get_config" in abstract_methods
        assert "calibrate" in abstract_methods
    
    def test_mock_encoder_implements_interface(self):
        """Test that MockEncoder properly implements EncoderInterface."""
        mock_encoder = MockEncoder()
        
        # Should be instance of the interface
        assert isinstance(mock_encoder, EncoderInterface)
        
        # Should have all required methods
        assert hasattr(mock_encoder, "encode")
        assert hasattr(mock_encoder, "get_config")
        assert hasattr(mock_encoder, "calibrate")
        
        # Methods should be callable
        assert callable(mock_encoder.encode)
        assert callable(mock_encoder.get_config)
        assert callable(mock_encoder.calibrate)
    
    def test_mock_encoder_encode_method(self):
        """Test MockEncoder encode method returns valid SpikeTrain."""
        mock_encoder = MockEncoder(output_shape=(15, 2, 25))
        
        # Test encode method
        result = mock_encoder.encode("test_data")
        
        # Should return a torch.Tensor (SpikeTrain)
        assert isinstance(result, torch.Tensor)
        assert isinstance(result, SpikeTrain)  # Same thing, but explicit
        assert result.shape == (15, 2, 25)
    
    def test_mock_encoder_get_config_method(self):
        """Test MockEncoder get_config method returns valid configuration."""
        mock_encoder = MockEncoder()
        
        config = mock_encoder.get_config()
        
        # Should return a dictionary
        assert isinstance(config, dict)
        assert "encoder_type" in config
        assert "encoding_method" in config
        assert config["encoder_type"] == "MockEncoder"
    
    def test_mock_encoder_calibrate_method(self):
        """Test MockEncoder calibrate method works correctly."""
        mock_encoder = MockEncoder()
        
        # Initial calibration calls should be 0
        assert mock_encoder.calibration_calls == 0
        
        # Call calibrate
        mock_encoder.calibrate("sample_data")
        
        # Should increment counter
        assert mock_encoder.calibration_calls == 1
        
        # Call again
        mock_encoder.calibrate("more_data")
        assert mock_encoder.calibration_calls == 2


class TestDecoderInterface:
    """Test DecoderInterface contract and implementation."""
    
    def test_decoder_interface_is_abstract(self):
        """Test that DecoderInterface cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            DecoderInterface()
    
    def test_decoder_interface_inheritance(self):
        """Test that DecoderInterface properly inherits from ABC."""
        assert issubclass(DecoderInterface, ABC)
        
        # Check that required methods are abstract
        abstract_methods = DecoderInterface.__abstractmethods__
        assert "decode" in abstract_methods
        assert "get_config" in abstract_methods
    
    def test_mock_decoder_implements_interface(self):
        """Test that MockDecoder properly implements DecoderInterface."""
        mock_decoder = MockDecoder()
        
        # Should be instance of the interface
        assert isinstance(mock_decoder, DecoderInterface)
        
        # Should have all required methods
        assert hasattr(mock_decoder, "decode")
        assert hasattr(mock_decoder, "get_config")
        
        # Methods should be callable
        assert callable(mock_decoder.decode)
        assert callable(mock_decoder.get_config)
    
    def test_mock_decoder_decode_method(self):
        """Test MockDecoder decode method returns valid Action."""
        mock_decoder = MockDecoder(mock_action="test_action")
        
        # Create test spike train
        spike_train = torch.rand(10, 1, 5)
        
        # Test decode method
        result = mock_decoder.decode(spike_train)
        
        # Should return the configured action
        assert result == "test_action"
        assert mock_decoder.decode_calls == 1
        
        # Call again to test counter
        result2 = mock_decoder.decode(spike_train)
        assert result2 == "test_action"
        assert mock_decoder.decode_calls == 2
    
    def test_mock_decoder_get_config_method(self):
        """Test MockDecoder get_config method returns valid configuration."""
        mock_decoder = MockDecoder(mock_action="custom_action")
        
        config = mock_decoder.get_config()
        
        # Should return a dictionary
        assert isinstance(config, dict)
        assert "decoder_type" in config
        assert "decoding_method" in config
        assert "mock_action" in config
        assert config["decoder_type"] == "MockDecoder"
        assert config["mock_action"] == "custom_action"


class TestInvalidMock:
    """Test that invalid objects are properly rejected."""
    
    def test_invalid_mock_not_encoder(self):
        """Test that InvalidMock is not recognized as an EncoderInterface."""
        invalid_mock = InvalidMock()
        
        # Should not be instance of EncoderInterface
        assert not isinstance(invalid_mock, EncoderInterface)
    
    def test_invalid_mock_not_decoder(self):
        """Test that InvalidMock is not recognized as a DecoderInterface."""
        invalid_mock = InvalidMock()
        
        # Should not be instance of DecoderInterface
        assert not isinstance(invalid_mock, DecoderInterface)
    
    def test_invalid_mock_missing_methods(self):
        """Test that InvalidMock doesn't have required interface methods."""
        invalid_mock = InvalidMock()
        
        # Should not have encoder methods
        assert not hasattr(invalid_mock, "encode")
        assert not hasattr(invalid_mock, "calibrate")
        
        # Should not have decoder methods
        assert not hasattr(invalid_mock, "decode")
        
        # May have get_config but that alone doesn't make it valid
        assert hasattr(invalid_mock, "get_config")


class TestInterfaceContractEnforcement:
    """Test that interface contracts are properly enforced."""
    
    def test_incomplete_encoder_implementation(self):
        """Test that incomplete encoder implementations cannot be instantiated."""
        
        # Define an incomplete encoder (missing calibrate method)
        class IncompleteEncoder(EncoderInterface):
            def encode(self, raw_data: Any) -> SpikeTrain:
                return torch.zeros(10, 1, 5)
            
            def get_config(self) -> Dict[str, Any]:
                return {}
            
            # Missing calibrate method
        
        # Should not be able to instantiate
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteEncoder()
    
    def test_incomplete_decoder_implementation(self):
        """Test that incomplete decoder implementations cannot be instantiated."""
        
        # Define an incomplete decoder (missing decode method)
        class IncompleteDecoder(DecoderInterface):
            def get_config(self) -> Dict[str, Any]:
                return {}
            
            # Missing decode method
        
        # Should not be able to instantiate
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteDecoder()
    
    def test_complete_custom_implementations(self):
        """Test that complete custom implementations work correctly."""
        
        # Define complete custom implementations
        class CustomEncoder(EncoderInterface):
            def encode(self, raw_data: Any) -> SpikeTrain:
                return torch.ones(5, 1, 10)
            
            def get_config(self) -> Dict[str, Any]:
                return {"type": "custom"}
            
            def calibrate(self, sample_data: Any) -> None:
                pass
        
        class CustomDecoder(DecoderInterface):
            def decode(self, spike_train: SpikeTrain) -> Action:
                return "custom_result"
            
            def get_config(self) -> Dict[str, Any]:
                return {"type": "custom"}
        
        # Should be able to instantiate
        encoder = CustomEncoder()
        decoder = CustomDecoder()
        
        # Should be instances of the interfaces
        assert isinstance(encoder, EncoderInterface)
        assert isinstance(decoder, DecoderInterface)
        
        # Should work correctly
        spike_train = encoder.encode("test")
        assert isinstance(spike_train, torch.Tensor)
        
        action = decoder.decode(spike_train)
        assert action == "custom_result"


class TestAdvancedContractEnforcement:
    """Test advanced interface contract enforcement and validation."""

    def test_encoder_method_signature_validation(self):
        """Test that encoder methods have correct signatures."""
        encoder = MockEncoder()

        # Test encode method signature
        import inspect
        encode_sig = inspect.signature(encoder.encode)
        assert len(encode_sig.parameters) == 1  # Only raw_data parameter
        assert 'raw_data' in encode_sig.parameters

        # Test get_config method signature
        config_sig = inspect.signature(encoder.get_config)
        assert len(config_sig.parameters) == 0  # No parameters

        # Test calibrate method signature
        calibrate_sig = inspect.signature(encoder.calibrate)
        assert len(calibrate_sig.parameters) == 1  # Only sample_data parameter
        assert 'sample_data' in calibrate_sig.parameters

    def test_decoder_method_signature_validation(self):
        """Test that decoder methods have correct signatures."""
        decoder = MockDecoder()

        # Test decode method signature
        import inspect
        decode_sig = inspect.signature(decoder.decode)
        assert len(decode_sig.parameters) == 1  # Only spike_train parameter
        assert 'spike_train' in decode_sig.parameters

        # Test get_config method signature
        config_sig = inspect.signature(decoder.get_config)
        assert len(config_sig.parameters) == 0  # No parameters

    def test_interface_inheritance_hierarchy(self):
        """Test that interfaces properly inherit from ABC."""
        # EncoderInterface should be abstract
        assert issubclass(EncoderInterface, ABC)
        assert issubclass(DecoderInterface, ABC)

        # Should not be able to instantiate abstract interfaces directly
        with pytest.raises(TypeError):
            EncoderInterface()

        with pytest.raises(TypeError):
            DecoderInterface()

    def test_contract_violation_detection(self):
        """Test that contract violations are properly detected."""
        # Test incomplete encoder implementation
        class IncompleteEncoder(EncoderInterface):
            def encode(self, raw_data: Any) -> SpikeTrain:
                return torch.zeros(5, 1, 10)
            # Missing get_config and calibrate methods

        # Should not be able to instantiate incomplete implementation
        with pytest.raises(TypeError):
            IncompleteEncoder()

        # Test incomplete decoder implementation
        class IncompleteDecoder(DecoderInterface):
            def decode(self, spike_train: SpikeTrain) -> Action:
                return "result"
            # Missing get_config method

        # Should not be able to instantiate incomplete implementation
        with pytest.raises(TypeError):
            IncompleteDecoder()

    def test_runtime_contract_enforcement(self):
        """Test runtime enforcement of interface contracts."""
        encoder = MockEncoder()
        decoder = MockDecoder()

        # Test that methods return expected types
        spike_train = encoder.encode("test_data")
        assert isinstance(spike_train, torch.Tensor)

        config = encoder.get_config()
        assert isinstance(config, dict)

        action = decoder.decode(spike_train)
        # Action can be any type, so just check it's not None
        assert action is not None

        decoder_config = decoder.get_config()
        assert isinstance(decoder_config, dict)

    def test_error_handling_in_implementations(self):
        """Test error handling in interface implementations."""
        # Test failing encoder
        failing_encoder = FailingMockEncoder(failure_mode="encode")
        with pytest.raises(RuntimeError, match="Mock encoder encode failure"):
            failing_encoder.encode("test")

        failing_encoder_config = FailingMockEncoder(failure_mode="config")
        with pytest.raises(RuntimeError, match="Mock encoder config failure"):
            failing_encoder_config.get_config()

        failing_encoder_calibrate = FailingMockEncoder(failure_mode="calibrate")
        with pytest.raises(RuntimeError, match="Mock encoder calibrate failure"):
            failing_encoder_calibrate.calibrate("test")

        # Test failing decoder
        failing_decoder = FailingMockDecoder(failure_mode="decode")
        with pytest.raises(RuntimeError, match="Mock decoder decode failure"):
            failing_decoder.decode(torch.zeros(5, 1, 10))

        failing_decoder_config = FailingMockDecoder(failure_mode="config")
        with pytest.raises(RuntimeError, match="Mock decoder config failure"):
            failing_decoder_config.get_config()

    def test_performance_monitoring_interface(self):
        """Test performance monitoring capabilities in interfaces."""
        perf_encoder = PerformanceMockEncoder(latency_ms=1.0)

        # Perform multiple encodings
        for i in range(5):
            spike_train = perf_encoder.encode(f"test_data_{i}")
            assert isinstance(spike_train, torch.Tensor)

        # Check performance metrics
        metrics = perf_encoder.get_performance_metrics()
        assert metrics["total_calls"] == 5
        assert len(metrics["call_times"]) == 5
        assert metrics["avg_time"] > 0
        assert metrics["min_time"] >= 0
        assert metrics["max_time"] >= metrics["min_time"]

        # Check config includes performance data
        config = perf_encoder.get_config()
        assert "total_calls" in config
        assert "avg_encode_time" in config
