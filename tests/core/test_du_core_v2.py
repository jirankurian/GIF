"""
Unit Tests for DU_Core_V2 - Hybrid SNN/SSM Architecture
======================================================

This module contains comprehensive unit tests for the DU_Core_V2 class and
HybridSNNSSMLayer, the advanced hybrid architecture that combines Spiking
Neural Networks with State Space Model dynamics.

Test Categories:
- HybridSNNSSMLayer initialization and functionality
- DU_Core_V2 initialization and configuration validation
- Heterogeneous architecture construction
- Forward pass and hybrid processing
- Process method (orchestrator interface)
- Integration with GIF framework
- Error handling and edge cases

The tests ensure that the hybrid SNN/SSM architecture correctly implements
the selective state mechanism and provides seamless integration with the
GIF orchestrator.
"""

import pytest
import torch
import torch.nn as nn
from typing import List

# Import the DU Core v2 classes to test
try:
    from gif_framework.core.du_core_v2 import DU_Core_V2, HybridSNNSSMLayer
    import snntorch as snn
    DU_CORE_V2_AVAILABLE = True
except ImportError:
    DU_CORE_V2_AVAILABLE = False

from gif_framework.interfaces.base_interfaces import SpikeTrain


class TestHybridSNNSSMLayer:
    """Test HybridSNNSSMLayer initialization and functionality."""
    
    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_valid_initialization(self):
        """Test basic valid initialization of HybridSNNSSMLayer."""
        layer = HybridSNNSSMLayer(
            input_dim=10,
            output_dim=8,
            state_dim=6,
            beta=0.9,
            threshold=1.2
        )
        
        assert layer.input_dim == 10
        assert layer.output_dim == 8
        assert layer.state_dim == 6
        assert layer.beta == 0.9
        assert layer.threshold == 1.2
        
        # Check that SSM matrix generators are created
        assert hasattr(layer, 'A_generator')
        assert hasattr(layer, 'B_generator')
        assert hasattr(layer, 'C_generator')
        assert hasattr(layer, 'lif_neuron')
    
    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="All dimensions must be positive"):
            HybridSNNSSMLayer(input_dim=0, output_dim=8, state_dim=6)
        
        with pytest.raises(ValueError, match="All dimensions must be positive"):
            HybridSNNSSMLayer(input_dim=10, output_dim=-1, state_dim=6)
    
    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_invalid_parameters(self):
        """Test that invalid beta and threshold raise ValueError."""
        with pytest.raises(ValueError, match="Beta must be in range"):
            HybridSNNSSMLayer(input_dim=10, output_dim=8, state_dim=6, beta=0.0)
        
        with pytest.raises(ValueError, match="Beta must be in range"):
            HybridSNNSSMLayer(input_dim=10, output_dim=8, state_dim=6, beta=1.5)
        
        with pytest.raises(ValueError, match="Threshold must be positive"):
            HybridSNNSSMLayer(input_dim=10, output_dim=8, state_dim=6, threshold=-0.1)
    
    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_forward_pass_shape(self):
        """Test forward pass returns correct output shapes."""
        layer = HybridSNNSSMLayer(input_dim=15, output_dim=10, state_dim=8)
        
        batch_size = 3
        input_spikes = torch.rand(batch_size, 15)
        hidden_state = torch.zeros(batch_size, 8)
        
        output_spikes, new_hidden_state = layer(input_spikes, hidden_state)
        
        assert output_spikes.shape == (batch_size, 10)
        assert new_hidden_state.shape == (batch_size, 8)
        assert isinstance(output_spikes, torch.Tensor)
        assert isinstance(new_hidden_state, torch.Tensor)
    
    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_selective_state_mechanism(self):
        """Test that SSM matrices are dynamically generated from input spikes."""
        layer = HybridSNNSSMLayer(
            input_dim=5,
            output_dim=4,
            state_dim=3,
            threshold=0.1  # Lower threshold to ensure spikes
        )

        batch_size = 2
        # Use larger input values to ensure spikes are generated
        input_spikes1 = torch.rand(batch_size, 5) * 2.0 + 1.0  # Range [1, 3]
        input_spikes2 = torch.rand(batch_size, 5) * 2.0 + 1.0  # Range [1, 3]
        hidden_state = torch.zeros(batch_size, 3)

        # Forward pass with different inputs should generate different matrices
        output1, state1 = layer(input_spikes1, hidden_state)
        output2, state2 = layer(input_spikes2, hidden_state)

        # Hidden states should be different for different inputs (SSM dynamics)
        assert not torch.allclose(state1, state2, atol=1e-6), \
            "Hidden states should differ for different inputs due to selective state mechanism"

        # At least one of the outputs should be different (allowing for some randomness)
        states_different = not torch.allclose(state1, state2, atol=1e-6)
        outputs_different = not torch.allclose(output1, output2, atol=1e-6)

        assert states_different or outputs_different, \
            "Either hidden states or outputs should differ for different inputs"


class TestDUCoreV2Initialization:
    """Test DU_Core_V2 initialization and parameter validation."""
    
    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_valid_initialization_simple(self):
        """Test basic valid initialization with simple configuration."""
        du_core = DU_Core_V2(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            state_dim=6
        )
        
        assert du_core.input_size == 10
        assert du_core.hidden_sizes == [8]
        assert du_core.output_size == 5
        assert du_core.state_dim == 6
        assert du_core.attention_interval == 2  # Default value
        assert du_core.attention_heads == 4  # Default value
    
    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_valid_initialization_complex(self):
        """Test valid initialization with complex configuration."""
        du_core = DU_Core_V2(
            input_size=100,
            hidden_sizes=[64, 32, 16],
            output_size=10,
            state_dim=20,
            attention_interval=2,
            attention_heads=8,
            beta=0.9,
            threshold=1.2
        )
        
        assert du_core.input_size == 100
        assert du_core.hidden_sizes == [64, 32, 16]
        assert du_core.output_size == 10
        assert du_core.state_dim == 20
        assert du_core.attention_interval == 2
        assert du_core.attention_heads == 8
        assert du_core.beta == 0.9
        assert du_core.threshold == 1.2
    
    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_heterogeneous_architecture_construction(self):
        """Test that heterogeneous architecture is correctly constructed."""
        du_core = DU_Core_V2(
            input_size=10,
            hidden_sizes=[8, 6],
            output_size=4,
            state_dim=5,
            attention_interval=2,
            attention_heads=2
        )
        
        # Check layer types
        expected_types = ['ssm', 'ssm', 'attention', 'ssm']  # 3 layers + attention after 2
        assert du_core.layer_types == expected_types
        
        # Check that we have the right number of layers
        assert len(du_core.layers) == 4
        
        # Check layer types in the actual layers
        assert isinstance(du_core.layers[0], HybridSNNSSMLayer)
        assert isinstance(du_core.layers[1], HybridSNNSSMLayer)
        assert isinstance(du_core.layers[2], nn.MultiheadAttention)
        assert isinstance(du_core.layers[3], HybridSNNSSMLayer)
    
    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_invalid_input_size(self):
        """Test that invalid input_size raises ValueError."""
        with pytest.raises(ValueError, match="input_size must be positive"):
            DU_Core_V2(input_size=0, hidden_sizes=[10], output_size=5, state_dim=8)
        
        with pytest.raises(ValueError, match="input_size must be positive"):
            DU_Core_V2(input_size=-5, hidden_sizes=[10], output_size=5, state_dim=8)
    
    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_invalid_state_dim(self):
        """Test that invalid state_dim raises ValueError."""
        with pytest.raises(ValueError, match="state_dim must be positive"):
            DU_Core_V2(input_size=10, hidden_sizes=[8], output_size=5, state_dim=0)
        
        with pytest.raises(ValueError, match="state_dim must be positive"):
            DU_Core_V2(input_size=10, hidden_sizes=[8], output_size=5, state_dim=-3)


class TestDUCoreV2ForwardPass:
    """Test DU_Core_V2 forward pass and hybrid processing."""
    
    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_forward_pass_shape_simple(self):
        """Test forward pass returns correct output shape for simple network."""
        du_core = DU_Core_V2(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            state_dim=6
        )
        
        # Create test input: 20 time steps, batch size 2, 10 features
        input_spikes = torch.rand(20, 2, 10)
        
        # Forward pass
        output_spikes = du_core.forward(input_spikes, num_steps=20)
        
        # Check output shape
        assert output_spikes.shape == (20, 2, 5)
        assert isinstance(output_spikes, torch.Tensor)
    
    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_forward_pass_shape_complex(self):
        """Test forward pass returns correct output shape for complex hybrid network."""
        du_core = DU_Core_V2(
            input_size=50,
            hidden_sizes=[32, 16, 8],
            output_size=4,
            state_dim=12,
            attention_interval=2,
            attention_heads=4
        )
        
        # Create test input: 15 time steps, batch size 3, 50 features
        input_spikes = torch.rand(15, 3, 50)
        
        # Forward pass
        output_spikes = du_core.forward(input_spikes, num_steps=15)
        
        # Check output shape
        assert output_spikes.shape == (15, 3, 4)
        assert isinstance(output_spikes, torch.Tensor)

    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_forward_pass_with_attention_layers(self):
        """Test forward pass with attention layers processes correctly."""
        du_core = DU_Core_V2(
            input_size=8,
            hidden_sizes=[6, 4],
            output_size=3,
            state_dim=5,
            attention_interval=1,  # Attention after every SSM layer
            attention_heads=2
        )

        # Create test input: 10 time steps, batch size 1, 8 features
        input_spikes = torch.rand(10, 1, 8)

        # Forward pass should complete without errors
        output_spikes = du_core.forward(input_spikes, num_steps=10)

        # Check output shape
        assert output_spikes.shape == (10, 1, 3)
        assert isinstance(output_spikes, torch.Tensor)

    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_forward_pass_single_time_step(self):
        """Test forward pass with single time step."""
        du_core = DU_Core_V2(
            input_size=12,
            hidden_sizes=[10],
            output_size=6,
            state_dim=8
        )

        # Create single time step input
        input_spikes = torch.rand(1, 2, 12)  # 1 time step, batch 2, 12 features

        # Forward pass
        output_spikes = du_core.forward(input_spikes, num_steps=1)

        # Check output shape
        assert output_spikes.shape == (1, 2, 6)
        assert isinstance(output_spikes, torch.Tensor)

    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_forward_pass_input_validation(self):
        """Test forward pass input validation."""
        du_core = DU_Core_V2(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            state_dim=6
        )

        # Test wrong number of dimensions
        with pytest.raises(ValueError, match="input_spikes must be 3D tensor"):
            wrong_input = torch.rand(10, 5)  # Only 2D
            du_core.forward(wrong_input, num_steps=10)

        # Test mismatched num_steps
        with pytest.raises(ValueError, match="must match num_steps"):
            input_spikes = torch.rand(5, 1, 10)  # 5 time steps
            du_core.forward(input_spikes, num_steps=10)  # But asking for 10

        # Test mismatched input_size
        with pytest.raises(ValueError, match="must match input_size"):
            input_spikes = torch.rand(5, 1, 15)  # 15 features instead of 10
            du_core.forward(input_spikes, num_steps=5)


class TestDUCoreV2ProcessMethod:
    """Test DU_Core_V2 process method (orchestrator interface)."""

    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_process_multi_step_input(self):
        """Test process method with multi-step spike train input."""
        du_core = DU_Core_V2(
            input_size=15,
            hidden_sizes=[12, 8],
            output_size=5,
            state_dim=10
        )

        # Create multi-step input
        input_spikes = torch.rand(25, 2, 15)  # 25 steps, batch 2, 15 features

        # Process through interface method
        output_spikes = du_core.process(input_spikes)

        # Check output
        assert output_spikes.shape == (25, 2, 5)
        assert isinstance(output_spikes, torch.Tensor)

    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_process_single_step_input(self):
        """Test process method with single-step spike train input."""
        du_core = DU_Core_V2(
            input_size=20,
            hidden_sizes=[16],
            output_size=8,
            state_dim=12
        )

        # Create single-step input
        input_spikes = torch.rand(3, 20)  # batch 3, 20 features (no time dimension)

        # Process through interface method
        output_spikes = du_core.process(input_spikes)

        # Check output (should remove time dimension for single-step)
        assert output_spikes.shape == (3, 8)
        assert isinstance(output_spikes, torch.Tensor)

    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_process_method_compatibility_with_v1(self):
        """Test that process method is compatible with DU_Core_V1 interface."""
        # This test ensures DU_Core_V2 can be a drop-in replacement for DU_Core_V1
        du_core_v2 = DU_Core_V2(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            state_dim=6
        )

        # Test the same interface as DU_Core_V1
        input_spikes = torch.rand(30, 1, 10)
        output_spikes = du_core_v2.process(input_spikes)

        assert output_spikes.shape == (30, 1, 5)
        assert isinstance(output_spikes, torch.Tensor)

        # Test single step compatibility
        single_step = torch.rand(1, 10)
        single_output = du_core_v2.process(single_step)
        assert single_output.shape == (1, 5)


class TestDUCoreV2Integration:
    """Test DU_Core_V2 integration with GIF framework."""

    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_integration_with_gif_orchestrator(self):
        """Test integration with GIF orchestrator interface."""
        from gif_framework.orchestrator import GIF

        # Create DU_Core_V2
        du_core_v2 = DU_Core_V2(
            input_size=20,
            hidden_sizes=[16, 12],
            output_size=8,
            state_dim=10
        )

        # Create GIF orchestrator with V2 core
        gif = GIF(du_core_v2)

        # Test that the core is properly attached
        assert gif._du_core is du_core_v2
        assert hasattr(gif._du_core, 'process')  # Interface method exists

    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_parameter_count_reasonable(self):
        """Test that parameter count is reasonable for the architecture."""
        du_core = DU_Core_V2(
            input_size=100,
            hidden_sizes=[64, 32],
            output_size=10,
            state_dim=20,
            attention_interval=2,
            attention_heads=4
        )

        total_params = sum(p.numel() for p in du_core.parameters())

        # Should have a reasonable number of parameters (not too few, not too many)
        assert total_params > 1000  # At least some complexity
        assert total_params < 1000000  # But not excessive for this size

    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_gradient_flow(self):
        """Test that gradients flow properly through the hybrid architecture."""
        du_core = DU_Core_V2(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            state_dim=6
        )

        # Create input and target
        input_spikes = torch.rand(5, 1, 10, requires_grad=True)
        target = torch.rand(5, 1, 5)

        # Forward pass
        output = du_core.forward(input_spikes, num_steps=5)

        # Compute loss and backward pass
        loss = nn.MSELoss()(output, target)
        loss.backward()

        # Check that gradients exist
        assert input_spikes.grad is not None

        # Check that model parameters have gradients
        for param in du_core.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestDUCoreV2EdgeCases:
    """Test DU_Core_V2 edge cases and error handling."""

    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_no_hidden_layers(self):
        """Test DU_Core_V2 with no hidden layers (direct input to output)."""
        du_core = DU_Core_V2(
            input_size=10,
            hidden_sizes=[],
            output_size=5,
            state_dim=8
        )

        # Should have only one SSM layer (input to output)
        assert len(du_core.layers) == 1
        assert du_core.layer_types == ['ssm']

        # Test forward pass
        input_spikes = torch.rand(5, 1, 10)
        output_spikes = du_core.forward(input_spikes, num_steps=5)
        assert output_spikes.shape == (5, 1, 5)

    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_high_attention_frequency(self):
        """Test with attention after every layer."""
        du_core = DU_Core_V2(
            input_size=8,
            hidden_sizes=[6, 4],
            output_size=3,
            state_dim=5,
            attention_interval=1,  # Attention after every SSM layer
            attention_heads=2
        )

        # Should have alternating SSM and attention layers
        expected_types = ['ssm', 'attention', 'ssm', 'attention', 'ssm']
        assert du_core.layer_types == expected_types

        # Test forward pass
        input_spikes = torch.rand(3, 1, 8)
        output_spikes = du_core.forward(input_spikes, num_steps=3)
        assert output_spikes.shape == (3, 1, 3)

    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_large_batch_size(self):
        """Test with large batch size."""
        du_core = DU_Core_V2(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            state_dim=6
        )

        # Test with large batch
        batch_size = 32
        input_spikes = torch.rand(5, batch_size, 10)
        output_spikes = du_core.forward(input_spikes, num_steps=5)

        assert output_spikes.shape == (5, batch_size, 5)

    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_repr_string(self):
        """Test string representation of DU_Core_V2."""
        du_core = DU_Core_V2(
            input_size=10,
            hidden_sizes=[8, 6],
            output_size=4,
            state_dim=5,
            attention_interval=2,
            attention_heads=2
        )

        repr_str = repr(du_core)

        # Check that key information is in the representation
        assert "DU_Core_V2" in repr_str
        assert "10 → 8 → 6 → 4" in repr_str
        assert "SSM" in repr_str
        assert "Attention" in repr_str
        assert "state_dim: 5" in repr_str


class TestDUCoreV2AdvancedValidation:
    """Test advanced validation requirements for DU_Core_V2."""

    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_du_core_v2_has_heterogeneous_architecture(self):
        """Test that DU_Core_V2 has heterogeneous architecture with mix of SNN/SSM and Attention layers."""
        du_core = DU_Core_V2(
            input_size=20,
            hidden_sizes=[16, 12, 8],
            output_size=4,
            state_dim=10,
            attention_interval=2,  # Attention every 2 SSM layers
            attention_heads=4
        )

        # Verify heterogeneous architecture
        assert len(du_core.layers) > len(du_core.hidden_sizes) + 1  # Should have more layers due to attention
        assert len(du_core.layer_types) == len(du_core.layers)

        # Check that we have both SSM and attention layers
        ssm_count = du_core.layer_types.count('ssm')
        attention_count = du_core.layer_types.count('attention')

        assert ssm_count > 0, "Should have SSM layers"
        assert attention_count > 0, "Should have attention layers"

        # Verify layer types match actual layer instances
        for i, (layer, layer_type) in enumerate(zip(du_core.layers, du_core.layer_types)):
            if layer_type == 'ssm':
                assert isinstance(layer, HybridSNNSSMLayer), f"Layer {i} should be HybridSNNSSMLayer"
            elif layer_type == 'attention':
                assert isinstance(layer, torch.nn.MultiheadAttention), f"Layer {i} should be MultiheadAttention"

        # Verify attention placement follows interval pattern
        ssm_layer_indices = [i for i, t in enumerate(du_core.layer_types) if t == 'ssm']
        attention_layer_indices = [i for i, t in enumerate(du_core.layer_types) if t == 'attention']

        # Should have attention layers at appropriate intervals
        assert len(attention_layer_indices) > 0, "Should have at least one attention layer"

        # Verify architecture configuration
        config = du_core.get_config()
        assert config['total_layers'] == len(du_core.layers)
        assert config['ssm_layers'] == ssm_count
        assert config['attention_layers'] == attention_count
        assert config['architecture_type'] == 'Hybrid SNN/SSM with Attention'

    @pytest.mark.skipif(not DU_CORE_V2_AVAILABLE, reason="snnTorch not available")
    def test_du_core_v2_forward_pass_runs_without_error(self):
        """Test that complex data flow executes without error through heterogeneous architecture."""
        du_core = DU_Core_V2(
            input_size=32,
            hidden_sizes=[24, 16, 12],
            output_size=8,
            state_dim=16,
            attention_interval=1,  # Attention after every SSM layer (most complex case)
            attention_heads=4
        )

        # Test with various input configurations
        test_cases = [
            # (time_steps, batch_size, description)
            (10, 1, "short sequence, single batch"),
            (50, 2, "medium sequence, small batch"),
            (100, 1, "long sequence, single batch"),
            (20, 4, "medium sequence, larger batch"),
        ]

        for time_steps, batch_size, description in test_cases:
            with torch.no_grad():  # Disable gradients for faster execution
                # Create test input
                input_spikes = torch.rand(time_steps, batch_size, 32)

                # Forward pass should complete without error
                try:
                    output_spikes = du_core.forward(input_spikes, num_steps=time_steps)

                    # Verify output shape
                    expected_shape = (time_steps, batch_size, 8)
                    assert output_spikes.shape == expected_shape, \
                        f"Output shape mismatch for {description}: expected {expected_shape}, got {output_spikes.shape}"

                    # Verify output is valid tensor
                    assert isinstance(output_spikes, torch.Tensor), f"Output should be tensor for {description}"
                    assert not torch.isnan(output_spikes).any(), f"Output contains NaN for {description}"
                    assert torch.isfinite(output_spikes).all(), f"Output contains infinite values for {description}"

                except Exception as e:
                    pytest.fail(f"Forward pass failed for {description}: {e}")

        # Test gradient flow through complex architecture
        input_spikes = torch.rand(20, 1, 32, requires_grad=True)
        target = torch.rand(20, 1, 8)

        # Forward pass
        output_spikes = du_core.forward(input_spikes, num_steps=20)

        # Compute loss and backward pass
        loss = torch.nn.MSELoss()(output_spikes, target)
        loss.backward()

        # Verify gradients exist and are valid
        assert input_spikes.grad is not None, "Input gradients should exist"
        assert not torch.isnan(input_spikes.grad).any(), "Input gradients should not contain NaN"

        # Check gradients for all trainable parameters
        gradient_count = 0
        for name, param in du_core.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} should have gradients"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} gradients should not contain NaN"
                gradient_count += 1

        assert gradient_count > 0, "Should have at least some trainable parameters with gradients"

        # Test process method compatibility (orchestrator interface)
        input_spikes_2d = torch.rand(1, 32)  # Single time step
        output_2d = du_core.process(input_spikes_2d)
        assert output_2d.shape == (1, 8), "Process method should handle 2D input correctly"

        input_spikes_3d = torch.rand(15, 1, 32)  # Multiple time steps
        output_3d = du_core.process(input_spikes_3d)
        assert output_3d.shape == (15, 1, 8), "Process method should handle 3D input correctly"
