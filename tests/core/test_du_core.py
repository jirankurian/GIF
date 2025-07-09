"""
Unit Tests for DU_Core_V1 - The SNN Brain
==========================================

This module contains comprehensive unit tests for the DU_Core_V1 class,
the spiking neural network brain of the GIF framework.

Test Categories:
- Initialization and configuration validation
- Network architecture construction
- Forward pass and spike processing
- Process method (orchestrator interface)
- Configuration and utility methods
- Error handling and edge cases

The tests ensure that the DU Core correctly implements the SNN architecture
and provides the expected interface for the GIF orchestrator.
"""

import pytest
import torch
import torch.nn as nn
from typing import List

# Import the DU Core class to test
try:
    from gif_framework.core.du_core import DU_Core_V1
except ImportError:
    pytest.skip("snnTorch not available", allow_module_level=True)

from gif_framework.interfaces.base_interfaces import SpikeTrain


class TestDUCoreInitialization:
    """Test DU Core initialization and parameter validation."""
    
    def test_valid_initialization_simple(self):
        """Test basic valid initialization with simple configuration."""
        du_core = DU_Core_V1(
            input_size=10,
            hidden_sizes=[20],
            output_size=5
        )
        
        assert du_core.input_size == 10
        assert du_core.hidden_sizes == [20]
        assert du_core.output_size == 5
        assert du_core.beta == 0.95  # Default value
        assert du_core.threshold == 1.0  # Default value
        assert du_core.recurrent == False  # Default value
    
    def test_valid_initialization_complex(self):
        """Test valid initialization with complex multi-layer configuration."""
        du_core = DU_Core_V1(
            input_size=100,
            hidden_sizes=[64, 32, 16],
            output_size=10,
            beta=0.9,
            threshold=0.8,
            recurrent=False
        )
        
        assert du_core.input_size == 100
        assert du_core.hidden_sizes == [64, 32, 16]
        assert du_core.output_size == 10
        assert du_core.beta == 0.9
        assert du_core.threshold == 0.8
        assert len(du_core.linear_layers) == 4  # 3 hidden + 1 output
        assert len(du_core.lif_layers) == 4
    
    def test_valid_initialization_no_hidden_layers(self):
        """Test valid initialization with no hidden layers (direct input to output)."""
        du_core = DU_Core_V1(
            input_size=50,
            hidden_sizes=[],
            output_size=25
        )
        
        assert du_core.input_size == 50
        assert du_core.hidden_sizes == []
        assert du_core.output_size == 25
        assert len(du_core.linear_layers) == 1  # Only input to output
        assert len(du_core.lif_layers) == 1
    
    def test_invalid_input_size(self):
        """Test that invalid input_size raises ValueError."""
        with pytest.raises(ValueError, match="input_size must be positive"):
            DU_Core_V1(input_size=0, hidden_sizes=[10], output_size=5)
        
        with pytest.raises(ValueError, match="input_size must be positive"):
            DU_Core_V1(input_size=-5, hidden_sizes=[10], output_size=5)
    
    def test_invalid_output_size(self):
        """Test that invalid output_size raises ValueError."""
        with pytest.raises(ValueError, match="output_size must be positive"):
            DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=0)
        
        with pytest.raises(ValueError, match="output_size must be positive"):
            DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=-3)
    
    def test_invalid_hidden_sizes(self):
        """Test that invalid hidden_sizes raise ValueError."""
        with pytest.raises(ValueError, match="hidden_sizes.*must be positive"):
            DU_Core_V1(input_size=10, hidden_sizes=[5, 0, 3], output_size=2)
        
        with pytest.raises(ValueError, match="hidden_sizes.*must be positive"):
            DU_Core_V1(input_size=10, hidden_sizes=[5, -2], output_size=2)
    
    def test_invalid_beta(self):
        """Test that invalid beta values raise ValueError."""
        with pytest.raises(ValueError, match="beta must be in range"):
            DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=2, beta=0.0)
        
        with pytest.raises(ValueError, match="beta must be in range"):
            DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=2, beta=1.5)
        
        with pytest.raises(ValueError, match="beta must be in range"):
            DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=2, beta=-0.1)
    
    def test_invalid_threshold(self):
        """Test that invalid threshold values raise ValueError."""
        with pytest.raises(ValueError, match="threshold must be positive"):
            DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=2, threshold=0.0)
        
        with pytest.raises(ValueError, match="threshold must be positive"):
            DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=2, threshold=-0.5)
    
    def test_recurrent_not_implemented(self):
        """Test that recurrent=True raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Recurrent connections are not yet implemented"):
            DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=2, recurrent=True)


class TestDUCoreForwardPass:
    """Test DU Core forward pass and spike processing."""
    
    def test_forward_pass_shape_simple(self):
        """Test forward pass returns correct output shape for simple network."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[20], output_size=5)
        
        # Create test input: 50 time steps, batch size 1, 10 features
        input_spikes = torch.rand(50, 1, 10)
        
        # Forward pass
        output_spikes = du_core.forward(input_spikes, num_steps=50)
        
        # Check output shape
        assert output_spikes.shape == (50, 1, 5)
        assert isinstance(output_spikes, torch.Tensor)
    
    def test_forward_pass_shape_complex(self):
        """Test forward pass returns correct output shape for complex network."""
        du_core = DU_Core_V1(input_size=100, hidden_sizes=[64, 32, 16], output_size=8)
        
        # Create test input: 25 time steps, batch size 4, 100 features
        input_spikes = torch.rand(25, 4, 100)
        
        # Forward pass
        output_spikes = du_core.forward(input_spikes, num_steps=25)
        
        # Check output shape
        assert output_spikes.shape == (25, 4, 8)
        assert isinstance(output_spikes, torch.Tensor)
    
    def test_forward_pass_batch_processing(self):
        """Test forward pass handles different batch sizes correctly."""
        du_core = DU_Core_V1(input_size=20, hidden_sizes=[15], output_size=10)
        
        # Test different batch sizes
        for batch_size in [1, 2, 5, 10]:
            input_spikes = torch.rand(30, batch_size, 20)
            output_spikes = du_core.forward(input_spikes, num_steps=30)
            assert output_spikes.shape == (30, batch_size, 10)
    
    def test_forward_pass_invalid_input_shape(self):
        """Test forward pass raises error for invalid input shapes."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)
        
        # Test 2D input (missing time dimension)
        with pytest.raises(ValueError, match="input_spikes must be 3D tensor"):
            invalid_input = torch.rand(1, 10)
            du_core.forward(invalid_input, num_steps=1)
        
        # Test 4D input (too many dimensions)
        with pytest.raises(ValueError, match="input_spikes must be 3D tensor"):
            invalid_input = torch.rand(10, 1, 10, 1)
            du_core.forward(invalid_input, num_steps=10)
    
    def test_forward_pass_mismatched_num_steps(self):
        """Test forward pass raises error when num_steps doesn't match input."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)
        
        input_spikes = torch.rand(20, 1, 10)  # 20 time steps
        
        with pytest.raises(ValueError, match="input_spikes first dimension.*must match num_steps"):
            du_core.forward(input_spikes, num_steps=15)  # Mismatch: 20 vs 15
    
    def test_forward_pass_mismatched_input_size(self):
        """Test forward pass raises error when input size doesn't match network."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)
        
        input_spikes = torch.rand(20, 1, 15)  # Wrong input size: 15 instead of 10
        
        with pytest.raises(ValueError, match="input_spikes last dimension.*must match input_size"):
            du_core.forward(input_spikes, num_steps=20)


class TestDUCoreProcessMethod:
    """Test DU Core process method (orchestrator interface)."""

    def test_process_multi_step_input(self):
        """Test process method with multi-step spike train input."""
        du_core = DU_Core_V1(input_size=15, hidden_sizes=[10], output_size=5)

        # Create multi-step input
        input_spikes = torch.rand(30, 2, 15)  # 30 steps, batch 2, 15 features

        # Process through interface method
        output_spikes = du_core.process(input_spikes)

        # Check output
        assert output_spikes.shape == (30, 2, 5)
        assert isinstance(output_spikes, torch.Tensor)

    def test_process_single_step_input(self):
        """Test process method with single-step spike train input."""
        du_core = DU_Core_V1(input_size=20, hidden_sizes=[15], output_size=8)

        # Create single-step input
        input_spikes = torch.rand(3, 20)  # batch 3, 20 features (no time dimension)

        # Process through interface method
        output_spikes = du_core.process(input_spikes)

        # Check output (should remove time dimension for single-step)
        assert output_spikes.shape == (3, 8)
        assert isinstance(output_spikes, torch.Tensor)

    def test_process_invalid_input_type(self):
        """Test process method raises error for non-tensor input."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)

        with pytest.raises(TypeError, match="spike_train must be a torch.Tensor"):
            du_core.process([1, 2, 3, 4, 5])  # List instead of tensor

    def test_process_invalid_input_dimensions(self):
        """Test process method raises error for invalid tensor dimensions."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)

        # Test 1D input
        with pytest.raises(ValueError, match="spike_train must be 2D.*or 3D"):
            invalid_input = torch.rand(10)
            du_core.process(invalid_input)

        # Test 4D input
        with pytest.raises(ValueError, match="spike_train must be 2D.*or 3D"):
            invalid_input = torch.rand(5, 2, 10, 1)
            du_core.process(invalid_input)

    def test_process_mismatched_input_size(self):
        """Test process method raises error for mismatched input size."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[5], output_size=3)

        # Wrong input size
        input_spikes = torch.rand(20, 1, 15)  # 15 instead of 10

        with pytest.raises(ValueError, match="spike_train input size.*must match network input_size"):
            du_core.process(input_spikes)


class TestDUCoreConfiguration:
    """Test DU Core configuration and utility methods."""

    def test_get_config(self):
        """Test get_config method returns complete configuration."""
        du_core = DU_Core_V1(
            input_size=50,
            hidden_sizes=[32, 16],
            output_size=8,
            beta=0.9,
            threshold=0.8
        )

        config = du_core.get_config()

        # Check all configuration parameters
        assert config["input_size"] == 50
        assert config["hidden_sizes"] == [32, 16]
        assert config["output_size"] == 8
        assert config["beta"] == 0.9
        assert config["threshold"] == 0.8
        assert config["recurrent"] == False
        assert config["num_layers"] == 3  # 2 hidden + 1 output
        assert config["model_type"] == "DU_Core_V1"
        assert "total_parameters" in config
        assert "trainable_parameters" in config

    def test_get_layer_info(self):
        """Test get_layer_info method returns detailed layer information."""
        du_core = DU_Core_V1(input_size=20, hidden_sizes=[15, 10], output_size=5)

        layer_info = du_core.get_layer_info()

        # Check layer information structure
        assert len(layer_info) == 3  # 2 hidden + 1 output

        # Check first layer
        assert layer_info[0]["layer_index"] == 0
        assert layer_info[0]["input_size"] == 20
        assert layer_info[0]["output_size"] == 15
        assert layer_info[0]["layer_type"] == "Linear + LIF"

        # Check second layer
        assert layer_info[1]["input_size"] == 15
        assert layer_info[1]["output_size"] == 10

        # Check output layer
        assert layer_info[2]["input_size"] == 10
        assert layer_info[2]["output_size"] == 5

    def test_string_representation(self):
        """Test __repr__ method returns informative string."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[8, 6], output_size=4)

        repr_str = repr(du_core)

        # Check that key information is in the string representation
        assert "DU_Core_V1" in repr_str
        assert "10 → 8 → 6 → 4" in repr_str  # Architecture
        assert "beta: 0.95" in repr_str
        assert "threshold: 1.0" in repr_str
        assert "parameters:" in repr_str
