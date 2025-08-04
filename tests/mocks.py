"""
Mock Components for GIF Framework Testing
==========================================

This module provides mock implementations of the GIF framework interfaces
for use in unit testing. These mocks allow for isolated testing of components
without requiring complex real implementations.

The mock objects implement the required interface contracts but with simple,
predictable behavior that makes testing deterministic and fast.

Key Components:
- MockEncoder: Simple encoder that returns predictable spike trains
- MockDecoder: Simple decoder that returns predictable actions
- InvalidMock: Object that doesn't implement interfaces (for negative testing)
"""

from typing import Any, Dict
import torch
from gif_framework.interfaces.base_interfaces import (
    EncoderInterface, 
    DecoderInterface, 
    SpikeTrain, 
    Action
)


class MockEncoder(EncoderInterface):
    """
    Mock encoder implementation for testing purposes.
    
    This encoder provides predictable, deterministic behavior for testing
    the GIF orchestrator and other components without requiring complex
    real encoding logic.
    
    The encoder always returns the same spike train pattern regardless
    of input, making test assertions reliable and repeatable.
    """
    
    def __init__(self, output_shape: tuple = (10, 1, 20)) -> None:
        """
        Initialize the mock encoder.
        
        Args:
            output_shape (tuple): Shape of spike train to return (num_steps, batch_size, features).
                                 Defaults to (10, 1, 20) for testing.
        """
        self.output_shape = output_shape
        self.encoding_method = "mock_rate_coding"
        self.calibration_calls = 0  # Track calibration calls for testing
        self.encoding_calls = 0  # Track encoding calls for testing
    
    def encode(self, raw_data: Any) -> SpikeTrain:
        """
        Convert raw data to a predictable spike train.
        
        This mock implementation ignores the actual raw_data and always
        returns the same spike train pattern for deterministic testing.
        
        Args:
            raw_data (Any): Input data (ignored in mock).
        
        Returns:
            SpikeTrain: Predictable spike train tensor with shape self.output_shape.
        """
        # Increment call counter
        self.encoding_calls += 1

        # Create a predictable spike pattern for testing
        # Use a simple pattern: alternating 0.0 and 0.8 values
        spike_train = torch.zeros(self.output_shape)
        spike_train[::2] = 0.8  # Every other time step has spikes

        return spike_train
    
    def get_config(self) -> Dict[str, Any]:
        """
        Return mock encoder configuration.
        
        Returns:
            Dict[str, Any]: Configuration dictionary for testing.
        """
        return {
            "encoder_type": "MockEncoder",
            "encoding_method": self.encoding_method,
            "output_shape": self.output_shape,
            "calibration_calls": self.calibration_calls,
            "version": "test_v1.0"
        }
    
    def calibrate(self, sample_data: Any) -> None:
        """
        Mock calibration method.
        
        This implementation just tracks how many times calibration
        was called, which is useful for testing the orchestrator's
        calibration workflow.
        
        Args:
            sample_data (Any): Sample data for calibration (ignored in mock).
        """
        self.calibration_calls += 1


class MockDecoder(DecoderInterface):
    """
    Mock decoder implementation for testing purposes.
    
    This decoder provides predictable, deterministic behavior for testing
    the GIF orchestrator and integration workflows without requiring complex
    real decoding logic.
    
    The decoder always returns the same action regardless of input spike train,
    making test assertions reliable and repeatable.
    """
    
    def __init__(self, mock_action: Action = "mock_action_success") -> None:
        """
        Initialize the mock decoder.
        
        Args:
            mock_action (Action): The action to return from decode().
                                 Defaults to "mock_action_success".
        """
        self.mock_action = mock_action
        self.decoding_method = "mock_spike_count"
        self.decode_calls = 0  # Track decode calls for testing
    
    def decode(self, spike_train: SpikeTrain) -> Action:
        """
        Convert spike train to a predictable action.

        This mock implementation ignores the actual spike_train and always
        returns the same action for deterministic testing.

        Args:
            spike_train (SpikeTrain): Input spike train (ignored in mock).

        Returns:
            Action: Predictable action for testing.
        """
        self.decode_calls += 1
        return self.mock_action

    def forward_classification(self, spike_counts: torch.Tensor) -> torch.Tensor:
        """
        Mock forward classification method for testing.

        Args:
            spike_counts (torch.Tensor): Spike count tensor.

        Returns:
            torch.Tensor: Mock classification output.
        """
        # Return a simple mock output for classification
        batch_size = spike_counts.shape[0] if len(spike_counts.shape) > 0 else 1
        return torch.zeros(batch_size, 2)  # Mock 2-class output
    
    def get_config(self) -> Dict[str, Any]:
        """
        Return mock decoder configuration.
        
        Returns:
            Dict[str, Any]: Configuration dictionary for testing.
        """
        return {
            "decoder_type": "MockDecoder",
            "decoding_method": self.decoding_method,
            "mock_action": self.mock_action,
            "decode_calls": self.decode_calls,
            "version": "test_v1.0"
        }


class InvalidMock:
    """
    Invalid mock object that does NOT implement any GIF interfaces.
    
    This class is used for negative testing to ensure that the GIF
    orchestrator properly rejects objects that don't implement the
    required interface contracts.
    
    This class intentionally does not inherit from EncoderInterface
    or DecoderInterface, making it invalid for attachment to the
    GIF orchestrator.
    """
    
    def __init__(self) -> None:
        """Initialize the invalid mock object."""
        self.invalid_type = "InvalidMock"
    
    def some_method(self) -> str:
        """A method that doesn't match any interface contract."""
        return "This object doesn't implement GIF interfaces"
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration (but this doesn't make it a valid interface)."""
        return {
            "type": "InvalidMock",
            "implements_encoder_interface": False,
            "implements_decoder_interface": False
        }


# Utility functions for creating test data

def create_test_spike_train(num_steps: int = 10, batch_size: int = 1, features: int = 20) -> SpikeTrain:
    """
    Create a test spike train with predictable pattern.
    
    Args:
        num_steps (int): Number of time steps.
        batch_size (int): Batch size.
        features (int): Number of features.
    
    Returns:
        SpikeTrain: Test spike train tensor.
    """
    spike_train = torch.zeros(num_steps, batch_size, features)
    # Create a simple pattern: spikes at even time steps
    spike_train[::2] = 0.5
    return spike_train


def create_test_raw_data(data_type: str = "simple") -> Any:
    """
    Create test raw data for encoder input.

    Args:
        data_type (str): Type of test data to create.
                        Options: "simple", "complex", "edge_case", "large", "empty"

    Returns:
        Any: Test raw data appropriate for the specified type.
    """
    if data_type == "simple":
        return {
            "data_type": "test_data",
            "values": [1, 2, 3, 4, 5],
            "metadata": {"source": "test_suite", "version": "1.0"}
        }
    elif data_type == "complex":
        return {
            "sensor_data": torch.randn(100, 10),
            "metadata": {"timestamp": "2024-01-01", "source": "test"},
            "labels": ["A", "B", "C"],
            "nested": {"deep": {"data": torch.randn(50, 5)}}
        }
    elif data_type == "edge_case":
        return None
    elif data_type == "large":
        return torch.randn(10000, 100)  # Large tensor for stress testing
    elif data_type == "empty":
        return []
    else:
        return "default_test_data"


class FailingMockEncoder(EncoderInterface):
    """
    Mock encoder that can be configured to fail in various ways.

    This is used for testing error handling and robustness of the
    GIF framework when components fail during operation.
    """

    def __init__(self, failure_mode: str = "none"):
        """
        Initialize the failing mock encoder.

        Args:
            failure_mode (str): How this encoder should fail.
                              Options: "none", "encode", "config", "calibrate", "exception"
        """
        self.failure_mode = failure_mode
        self.call_count = 0

    def encode(self, raw_data: Any) -> SpikeTrain:
        """Encode method that can fail based on failure_mode."""
        self.call_count += 1

        if self.failure_mode == "encode":
            raise RuntimeError("Mock encoder encode failure")
        elif self.failure_mode == "exception":
            raise ValueError("Mock encoder unexpected exception")

        return torch.zeros(5, 1, 10)  # Minimal valid output

    def get_config(self) -> Dict[str, Any]:
        """Get config method that can fail based on failure_mode."""
        if self.failure_mode == "config":
            raise RuntimeError("Mock encoder config failure")

        return {"name": "FailingMockEncoder", "failure_mode": self.failure_mode}

    def calibrate(self, sample_data: Any) -> None:
        """Calibrate method that can fail based on failure_mode."""
        if self.failure_mode == "calibrate":
            raise RuntimeError("Mock encoder calibrate failure")


class FailingMockDecoder(DecoderInterface):
    """
    Mock decoder that can be configured to fail in various ways.

    This is used for testing error handling and robustness of the
    GIF framework when components fail during operation.
    """

    def __init__(self, failure_mode: str = "none"):
        """
        Initialize the failing mock decoder.

        Args:
            failure_mode (str): How this decoder should fail.
                              Options: "none", "decode", "config", "exception"
        """
        self.failure_mode = failure_mode
        self.call_count = 0

    def decode(self, spike_train: SpikeTrain) -> Action:
        """Decode method that can fail based on failure_mode."""
        self.call_count += 1

        if self.failure_mode == "decode":
            raise RuntimeError("Mock decoder decode failure")
        elif self.failure_mode == "exception":
            raise ValueError("Mock decoder unexpected exception")

        return "failing_mock_action"

    def get_config(self) -> Dict[str, Any]:
        """Get config method that can fail based on failure_mode."""
        if self.failure_mode == "config":
            raise RuntimeError("Mock decoder config failure")

        return {"name": "FailingMockDecoder", "failure_mode": self.failure_mode}


class PerformanceMockEncoder(EncoderInterface):
    """
    Mock encoder that tracks performance metrics for testing.

    This encoder measures encoding time, memory usage, and call patterns
    to help validate performance characteristics of the framework.
    """

    def __init__(self, output_shape: tuple = (10, 1, 20), latency_ms: float = 0.0):
        """
        Initialize the performance mock encoder.

        Args:
            output_shape (tuple): Shape of spike train to return.
            latency_ms (float): Artificial latency to add (for performance testing).
        """
        self.output_shape = output_shape
        self.latency_ms = latency_ms
        self.call_times = []
        self.memory_usage = []
        self.total_calls = 0

    def encode(self, raw_data: Any) -> SpikeTrain:
        """Encode with performance tracking."""
        import time

        start_time = time.time()

        # Artificial latency for testing
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

        # Generate spike train
        spike_train = torch.rand(self.output_shape)

        # Record metrics
        end_time = time.time()

        self.call_times.append(end_time - start_time)
        self.total_calls += 1

        return spike_train

    def get_config(self) -> Dict[str, Any]:
        """Get config including performance metrics."""
        avg_time = sum(self.call_times) / len(self.call_times) if self.call_times else 0

        return {
            "name": "PerformanceMockEncoder",
            "output_shape": self.output_shape,
            "latency_ms": self.latency_ms,
            "total_calls": self.total_calls,
            "avg_encode_time": avg_time
        }

    def calibrate(self, sample_data: Any) -> None:
        """Calibrate with performance tracking."""
        pass

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        return {
            "call_times": self.call_times,
            "total_calls": self.total_calls,
            "min_time": min(self.call_times) if self.call_times else 0,
            "max_time": max(self.call_times) if self.call_times else 0,
            "avg_time": sum(self.call_times) / len(self.call_times) if self.call_times else 0
        }


def create_stress_test_data(size: str = "medium") -> Any:
    """
    Create data for stress testing the framework.

    Args:
        size (str): Size of stress test data.
                   Options: "small", "medium", "large", "huge"

    Returns:
        Any: Stress test data of appropriate size.
    """
    if size == "small":
        return torch.randn(100, 10)
    elif size == "medium":
        return torch.randn(1000, 50)
    elif size == "large":
        return torch.randn(10000, 100)
    elif size == "huge":
        return torch.randn(100000, 200)
    else:
        return torch.randn(1000, 50)
