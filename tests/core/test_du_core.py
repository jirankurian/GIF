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


class TestRTLEngineIntegration:
    """Test suite for Real-Time Learning (RTL) engine integration with DU_Core."""

    def test_rtl_engine_updates_weights(self):
        """Test that RTL engine successfully updates weights during processing."""
        from gif_framework.core.rtl_mechanisms import ThreeFactor_Hebbian_Rule
        import torch

        # Create Hebbian rule with moderate learning rate for testing
        hebbian_rule = ThreeFactor_Hebbian_Rule(learning_rate=0.1)

        # Create DU_Core with RTL mechanism
        du_core = DU_Core_V1(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            plasticity_rule=hebbian_rule
        )

        # Put in training mode to enable RTL
        du_core.train()

        # Get initial weights from the first linear layer
        initial_weights = du_core.linear_layers[0].weight.data.clone()

        # Create a specific spike pattern designed to trigger STDP
        # Pre-synaptic spike followed by post-synaptic spike (should cause LTP)
        num_steps = 10
        batch_size = 1

        # Create spike train with strong activity for Hebbian learning
        spike_train = torch.ones(num_steps, batch_size, 10) * 0.8  # Strong consistent activity

        # Add some modulatory signal by setting a positive reward context
        # This helps trigger the three-factor learning

        # Process the spike train multiple times to ensure weight updates
        for _ in range(3):
            output_spikes = du_core.process(spike_train)

        # Get weights after processing
        final_weights = du_core.linear_layers[0].weight.data.clone()

        # Assert that weights have changed
        weight_difference = torch.abs(final_weights - initial_weights)
        total_change = torch.sum(weight_difference).item()

        assert total_change > 0, f"RTL engine failed to update weights. Total change: {total_change}"
        assert not torch.equal(initial_weights, final_weights), "Weights should have changed after RTL processing"

        # Verify the output has the correct shape
        assert output_spikes.shape == (num_steps, batch_size, 5), f"Expected output shape {(num_steps, batch_size, 5)}, got {output_spikes.shape}"

    def test_rtl_engine_only_active_during_training(self):
        """Test that RTL engine only updates weights during training mode."""
        from gif_framework.core.rtl_mechanisms import STDP_Rule
        import torch

        # Create STDP rule
        stdp_rule = STDP_Rule(
            learning_rate_ltp=0.01,
            learning_rate_ltd=0.005,
            tau_ltp=20.0,
            tau_ltd=20.0
        )

        # Create DU_Core with RTL mechanism
        du_core = DU_Core_V1(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            plasticity_rule=stdp_rule
        )

        # Put in evaluation mode (RTL should be disabled)
        du_core.eval()

        # Get initial weights
        initial_weights = du_core.linear_layers[0].weight.data.clone()

        # Create spike train
        spike_train = torch.rand(5, 1, 10)

        # Process the spike train (RTL should NOT be active)
        output_spikes = du_core.process(spike_train)

        # Get weights after processing
        final_weights = du_core.linear_layers[0].weight.data.clone()

        # Assert that weights have NOT changed
        assert torch.equal(initial_weights, final_weights), "Weights should not change during evaluation mode"

    def test_rtl_engine_with_three_factor_hebbian(self):
        """Test RTL engine with Three-Factor Hebbian rule."""
        from gif_framework.core.rtl_mechanisms import ThreeFactor_Hebbian_Rule
        import torch

        # Create Three-Factor Hebbian rule with higher learning rate
        hebbian_rule = ThreeFactor_Hebbian_Rule(learning_rate=0.1)

        # Create DU_Core with Hebbian RTL mechanism
        du_core = DU_Core_V1(
            input_size=8,
            hidden_sizes=[6],
            output_size=4,
            plasticity_rule=hebbian_rule
        )

        # Put in training mode
        du_core.train()

        # Get initial weights
        initial_weights = du_core.linear_layers[0].weight.data.clone()

        # Create spike train with strong activity
        spike_train = torch.ones(10, 1, 8) * 0.8  # Strong consistent activity

        # Process the spike train multiple times to ensure weight updates
        for _ in range(3):
            output_spikes = du_core.process(spike_train)

        # Get weights after processing
        final_weights = du_core.linear_layers[0].weight.data.clone()

        # Assert that weights have changed
        weight_difference = torch.abs(final_weights - initial_weights)
        total_change = torch.sum(weight_difference).item()

        assert total_change > 0, f"Three-Factor Hebbian RTL failed to update weights. Total change: {total_change}"

    def test_rtl_engine_with_du_core_v2(self):
        """Test RTL engine integration with DU_Core_V2 hybrid architecture."""
        from gif_framework.core.rtl_mechanisms import ThreeFactor_Hebbian_Rule
        from gif_framework.core.du_core_v2 import DU_Core_V2
        import torch

        # Create Hebbian rule with higher learning rate for testing
        hebbian_rule = ThreeFactor_Hebbian_Rule(learning_rate=0.5)

        # Create DU_Core_V2 with RTL mechanism
        du_core_v2 = DU_Core_V2(
            input_size=12,
            hidden_sizes=[10],
            output_size=6,
            state_dim=8,
            attention_interval=3,  # No attention layers for this simple test
            plasticity_rule=hebbian_rule
        )

        # Put in training mode
        du_core_v2.train()

        # Get initial weights from first SSM layer
        initial_weights = None
        for layer in du_core_v2.layers:
            if hasattr(layer, 'A_generator') and len(layer.A_generator) > 0:
                initial_weights = layer.A_generator[0].weight.data.clone()
                break

        assert initial_weights is not None, "Could not find SSM layer weights to test"

        # Create spike train with strong consistent activity for Hebbian learning
        spike_train = torch.ones(8, 1, 12) * 1.0  # Strong consistent activity

        # Process the spike train
        output_spikes = du_core_v2.process(spike_train)

        # Get weights after processing
        final_weights = None
        for layer in du_core_v2.layers:
            if hasattr(layer, 'A_generator') and len(layer.A_generator) > 0:
                final_weights = layer.A_generator[0].weight.data.clone()
                break

        # Assert that weights have changed
        weight_difference = torch.abs(final_weights - initial_weights)
        total_change = torch.sum(weight_difference).item()

        # Use a more lenient threshold since RTL updates are scaled down for stability
        assert total_change > 1e-6, f"DU_Core_V2 RTL failed to update weights. Total change: {total_change}"

        # Verify output shape
        assert output_spikes.shape == (8, 1, 6), f"Expected output shape {(8, 1, 6)}, got {output_spikes.shape}"

    def test_rtl_engine_error_handling(self):
        """Test that RTL engine handles errors gracefully without breaking forward pass."""
        from gif_framework.core.rtl_mechanisms import STDP_Rule
        import torch
        import warnings

        # Create a mock STDP rule that will raise an exception
        class FailingSTDP(STDP_Rule):
            def apply(self, **kwargs):
                raise RuntimeError("Intentional test failure")

        failing_rule = FailingSTDP(0.01, 0.005, 20.0, 20.0)

        # Create DU_Core with failing RTL mechanism
        du_core = DU_Core_V1(
            input_size=5,
            hidden_sizes=[4],
            output_size=3,
            plasticity_rule=failing_rule
        )

        du_core.train()

        # Process spike train - should complete despite RTL failure
        spike_train = torch.rand(3, 1, 5)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output_spikes = du_core.process(spike_train)

            # Check that a warning was issued
            assert len(w) > 0, "Expected warning for RTL failure"
            assert "RTL mechanism failed" in str(w[0].message)

        # Verify that processing completed successfully
        assert output_spikes.shape == (3, 1, 3), "Forward pass should complete despite RTL failure"


class TestDUCoreArchitecturalIntegrity:
    """Test architectural integrity and advanced validation of DU Core."""

    def test_network_layer_connectivity(self):
        """Test that network layers are properly connected."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[8, 6], output_size=4)

        # Check that we have the expected number of layers
        assert len(du_core.linear_layers) == 3  # input->hidden1, hidden1->hidden2, hidden2->output
        assert len(du_core.lif_layers) == 3     # One LIF layer for each linear layer

        # Check layer dimensions
        assert du_core.linear_layers[0].in_features == 10  # Input layer
        assert du_core.linear_layers[0].out_features == 8  # First hidden
        assert du_core.linear_layers[1].in_features == 8   # First hidden
        assert du_core.linear_layers[1].out_features == 6  # Second hidden
        assert du_core.linear_layers[2].in_features == 6   # Second hidden
        assert du_core.linear_layers[2].out_features == 4  # Output layer

    def test_weight_initialization(self):
        """Test proper weight initialization."""
        du_core = DU_Core_V1(input_size=5, hidden_sizes=[8], output_size=3)

        # Check that weights are properly initialized (not all zeros)
        for linear_layer in du_core.linear_layers:
            # Weights should not be all zeros (Xavier initialization)
            assert not torch.allclose(linear_layer.weight, torch.zeros_like(linear_layer.weight))

            # Biases should be initialized to zeros
            assert torch.allclose(linear_layer.bias, torch.zeros_like(linear_layer.bias))

            # Weights should be in reasonable range for Xavier initialization
            weight_std = torch.std(linear_layer.weight)
            assert 0.01 < weight_std < 1.0, f"Weight std {weight_std} outside expected range"

    def test_spike_timing_dynamics(self):
        """Test SNN-specific spike timing dynamics."""
        du_core = DU_Core_V1(input_size=3, hidden_sizes=[5], output_size=2, threshold=0.5)

        # Create input with specific timing pattern
        num_steps = 10
        input_spikes = torch.zeros(num_steps, 1, 3)
        input_spikes[2, 0, 0] = 1.0  # Spike at time step 2
        input_spikes[5, 0, 1] = 1.0  # Spike at time step 5
        input_spikes[8, 0, 2] = 1.0  # Spike at time step 8

        output_spikes = du_core.forward(input_spikes, num_steps=num_steps)

        # Verify output shape
        assert output_spikes.shape == (num_steps, 1, 2)

        # Check that spikes occur after input spikes (causal relationship)
        input_spike_times = torch.nonzero(input_spikes.sum(dim=2))[:, 0]
        output_spike_times = torch.nonzero(output_spikes.sum(dim=2))[:, 0]

        if len(output_spike_times) > 0:
            # Output spikes should occur at or after input spikes
            assert output_spike_times.min() >= input_spike_times.min()

    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs."""
        du_core = DU_Core_V1(input_size=5, hidden_sizes=[10], output_size=3)

        # Test with very large inputs
        large_input = torch.ones(5, 1, 5) * 1000.0
        output_large = du_core.forward(large_input, num_steps=5)
        assert torch.isfinite(output_large).all(), "Output should be finite with large inputs"

        # Test with very small inputs
        small_input = torch.ones(5, 1, 5) * 1e-10
        output_small = du_core.forward(small_input, num_steps=5)
        assert torch.isfinite(output_small).all(), "Output should be finite with small inputs"

        # Test with zero inputs
        zero_input = torch.zeros(5, 1, 5)
        output_zero = du_core.forward(zero_input, num_steps=5)
        assert torch.isfinite(output_zero).all(), "Output should be finite with zero inputs"

    def test_gradient_flow_validation(self):
        """Test that gradients flow properly through the network."""
        du_core = DU_Core_V1(input_size=4, hidden_sizes=[6], output_size=2)
        du_core.train()  # Enable training mode

        # Create input and target
        input_spikes = torch.randn(3, 1, 4, requires_grad=True)
        target = torch.randn(3, 1, 2)

        # Forward pass
        output = du_core.forward(input_spikes, num_steps=3)

        # Compute loss and backward pass
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in du_core.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert torch.isfinite(param.grad).all(), f"Invalid gradient for parameter {name}"

    def test_memory_efficiency(self):
        """Test memory efficiency with large networks."""
        import gc
        import torch

        # Force garbage collection before test
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Create moderately large network
        du_core = DU_Core_V1(input_size=100, hidden_sizes=[200, 150], output_size=50)

        # Process multiple batches
        for i in range(5):
            input_spikes = torch.randn(20, 4, 100)  # 20 time steps, batch size 4
            output = du_core.forward(input_spikes, num_steps=20)

            # Verify output shape
            assert output.shape == (20, 4, 50)

            # Clear intermediate results
            del output

        # Memory should be manageable (this is more of a smoke test)
        assert True  # If we get here without OOM, test passes


class TestDUCoreStressTesting:
    """Stress testing for DU Core under extreme conditions."""

    def test_large_network_stress(self):
        """Test DU Core with large network configurations."""
        # Create a large network
        du_core = DU_Core_V1(
            input_size=200,
            hidden_sizes=[300, 250, 200],
            output_size=100,
            beta=0.9,
            threshold=1.0
        )

        # Test with large input
        large_input = torch.randn(50, 8, 200)  # 50 time steps, batch size 8
        output = du_core.forward(large_input, num_steps=50)

        # Verify output shape and validity
        assert output.shape == (50, 8, 100)
        assert torch.isfinite(output).all()

    def test_long_sequence_processing(self):
        """Test processing of very long sequences."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[20], output_size=5)

        # Test with very long sequence
        long_sequence = torch.randn(1000, 1, 10)  # 1000 time steps
        output = du_core.forward(long_sequence, num_steps=1000)

        # Verify output shape and stability
        assert output.shape == (1000, 1, 5)
        assert torch.isfinite(output).all()

        # Check that network doesn't saturate (some variation in output)
        output_variance = torch.var(output)
        assert output_variance > 1e-6, "Network output should have some variation"

    def test_batch_processing_stress(self):
        """Test processing with large batch sizes."""
        du_core = DU_Core_V1(input_size=15, hidden_sizes=[25], output_size=8)

        # Test with large batch
        large_batch = torch.randn(20, 64, 15)  # 20 time steps, batch size 64
        output = du_core.forward(large_batch, num_steps=20)

        # Verify output shape and validity
        assert output.shape == (20, 64, 8)
        assert torch.isfinite(output).all()

    def test_parameter_extreme_values(self):
        """Test DU Core with extreme parameter values."""
        # Test with very low threshold
        du_core_low_thresh = DU_Core_V1(
            input_size=5,
            hidden_sizes=[10],
            output_size=3,
            threshold=0.01  # Very low threshold
        )

        input_spikes = torch.randn(10, 1, 5) * 0.1
        output_low = du_core_low_thresh.forward(input_spikes, num_steps=10)
        assert torch.isfinite(output_low).all()

        # Test with very high threshold
        du_core_high_thresh = DU_Core_V1(
            input_size=5,
            hidden_sizes=[10],
            output_size=3,
            threshold=10.0  # Very high threshold
        )

        output_high = du_core_high_thresh.forward(input_spikes, num_steps=10)
        assert torch.isfinite(output_high).all()

        # Test with extreme beta values
        du_core_low_beta = DU_Core_V1(
            input_size=5,
            hidden_sizes=[10],
            output_size=3,
            beta=0.01  # Very low beta (high leakage)
        )

        output_low_beta = du_core_low_beta.forward(input_spikes, num_steps=10)
        assert torch.isfinite(output_low_beta).all()

    def test_concurrent_processing_safety(self):
        """Test that DU Core is safe for concurrent processing."""
        du_core = DU_Core_V1(input_size=8, hidden_sizes=[12], output_size=4)

        # Process multiple inputs "concurrently" (sequentially but with shared state)
        inputs = [torch.randn(5, 1, 8) for _ in range(10)]
        outputs = []

        for input_data in inputs:
            output = du_core.forward(input_data, num_steps=5)
            outputs.append(output)

        # All outputs should be valid
        for i, output in enumerate(outputs):
            assert output.shape == (5, 1, 4), f"Output {i} has wrong shape"
            assert torch.isfinite(output).all(), f"Output {i} contains invalid values"

    def test_performance_benchmarking(self):
        """Benchmark DU Core performance for regression detection."""
        import time

        du_core = DU_Core_V1(input_size=50, hidden_sizes=[100, 75], output_size=25)

        # Warm up
        warmup_input = torch.randn(10, 1, 50)
        _ = du_core.forward(warmup_input, num_steps=10)

        # Benchmark
        benchmark_input = torch.randn(100, 4, 50)  # 100 time steps, batch size 4

        start_time = time.time()
        output = du_core.forward(benchmark_input, num_steps=100)
        end_time = time.time()

        processing_time = end_time - start_time

        # Verify output
        assert output.shape == (100, 4, 25)
        assert torch.isfinite(output).all()

        # Performance should be reasonable (this is a rough check)
        # Processing 100 time steps with batch size 4 should take less than 10 seconds
        assert processing_time < 10.0, f"Processing took too long: {processing_time:.2f}s"

        # Store performance metrics for potential regression detection
        throughput = (100 * 4) / processing_time  # samples per second
        assert throughput > 10, f"Throughput too low: {throughput:.2f} samples/sec"
