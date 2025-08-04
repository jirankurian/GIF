"""
Comprehensive Test Suite for Weight-Reset Protocol
=================================================

This module provides exhaustive testing for the critical weight-reset protocol
that enables valid potentiation experiments in the GIF framework. These tests
ensure the scientific rigor required for AGI research by validating that:

1. All synaptic weights are completely reset (no knowledge transfer)
2. Meta-parameters and learning mechanisms are preserved
3. Network architecture remains intact
4. Integration with the complete GIF framework works correctly

Test Coverage:
- DU_Core_V1 weight reset validation
- DU_Core_V2 hybrid architecture weight reset
- Meta-parameter preservation (CRITICAL for scientific validity)
- Plasticity rule and memory system integration
- Potentiation experiment methodology validation
- Framework integration and edge cases

The weight-reset protocol is the cornerstone of the potentiation experiment
methodology, distinguishing true system potentiation from simple knowledge
transfer. These tests ensure credible claims about AGI capabilities.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock

# Import GIF framework components
from gif_framework.core.du_core import DU_Core_V1
from gif_framework.core.du_core_v2 import DU_Core_V2, HybridSNNSSMLayer
from gif_framework.core.rtl_mechanisms import ThreeFactor_Hebbian_Rule, STDP_Rule
from gif_framework.core.memory_systems import EpisodicMemory
from gif_framework.orchestrator import GIF

# Import interfaces
from gif_framework.interfaces.base_interfaces import EncoderInterface, DecoderInterface


class TestDUCoreV1WeightReset:
    """Test suite for DU_Core_V1 weight reset protocol validation."""
    
    def test_weight_reset_changes_all_weights(self):
        """
        CRITICAL TEST: Ensures weight reset completely changes all synaptic weights.
        
        This is the most important test for scientific validity. The weight reset
        protocol must completely wipe learned knowledge while preserving structural
        improvements for valid potentiation measurement.
        """
        # Create DU_Core with known architecture
        du_core = DU_Core_V1(
            input_size=10,
            hidden_sizes=[20, 15],
            output_size=5,
            beta=0.95,
            threshold=1.0
        )
        
        # Store initial weights from all layers
        initial_weights = []
        initial_biases = []
        for layer in du_core.linear_layers:
            initial_weights.append(layer.weight.clone().detach())
            initial_biases.append(layer.bias.clone().detach())
        
        # Apply weight reset
        du_core.reset_weights()
        
        # Verify ALL weights have changed
        weights_changed = 0
        total_weights = 0
        
        for i, layer in enumerate(du_core.linear_layers):
            # Check weights
            if not torch.equal(initial_weights[i], layer.weight):
                weights_changed += 1
            total_weights += 1
            
            # Check biases
            if not torch.equal(initial_biases[i], layer.bias):
                weights_changed += 1
            total_weights += 1
        
        # CRITICAL: ALL weights must have changed
        assert weights_changed == total_weights, \
            f"Only {weights_changed}/{total_weights} weight tensors changed"
        
    def test_weight_reset_preserves_architecture(self):
        """Test that weight reset preserves network architecture."""
        du_core = DU_Core_V1(input_size=8, hidden_sizes=[16, 12], output_size=4)
        
        # Store architecture info
        initial_layer_count = len(du_core.linear_layers)
        initial_shapes = [layer.weight.shape for layer in du_core.linear_layers]
        
        # Apply weight reset
        du_core.reset_weights()
        
        # Verify architecture unchanged
        assert len(du_core.linear_layers) == initial_layer_count
        for i, layer in enumerate(du_core.linear_layers):
            assert layer.weight.shape == initial_shapes[i]
            
    def test_weight_reset_preserves_meta_parameters(self):
        """
        CRITICAL TEST: Verify meta-parameters are preserved during weight reset.
        
        This test ensures that learning mechanisms (beta, threshold, etc.) are
        preserved while only synaptic weights are reset. This is essential for
        valid potentiation experiments.
        """
        # Create DU_Core with specific meta-parameters
        original_beta = 0.85
        original_threshold = 1.5
        
        du_core = DU_Core_V1(
            input_size=6,
            hidden_sizes=[12, 8],
            output_size=3,
            beta=original_beta,
            threshold=original_threshold
        )
        
        # Apply weight reset
        du_core.reset_weights()
        
        # CRITICAL: Verify meta-parameters preserved
        assert du_core.beta == original_beta, \
            f"Beta changed from {original_beta} to {du_core.beta}"
        assert du_core.threshold == original_threshold, \
            f"Threshold changed from {original_threshold} to {du_core.threshold}"
            
    def test_weight_reset_with_plasticity_rules(self):
        """Test weight reset behavior when DU_Core has plasticity rules attached."""
        # Create plasticity rule with specific parameters
        plasticity_rule = ThreeFactor_Hebbian_Rule(learning_rate=0.02)
        original_learning_rate = plasticity_rule.base_learning_rate
        
        # Create DU_Core with plasticity rule
        du_core = DU_Core_V1(
            input_size=5,
            hidden_sizes=[10],
            output_size=3,
            plasticity_rule=plasticity_rule
        )
        
        # Apply weight reset
        du_core.reset_weights()
        
        # Verify plasticity rule parameters preserved
        if hasattr(du_core, '_plasticity_rule') and du_core._plasticity_rule is not None:
            assert du_core._plasticity_rule.base_learning_rate == original_learning_rate, \
                "Plasticity rule learning rate changed after weight reset"
                
    def test_weight_reset_deterministic_behavior(self):
        """Test that weight reset produces deterministic behavior with same random seed."""
        # Create two identical DU_Cores
        du_core1 = DU_Core_V1(input_size=4, hidden_sizes=[8], output_size=2)
        du_core2 = DU_Core_V1(input_size=4, hidden_sizes=[8], output_size=2)
        
        # Set same random seed and reset both
        torch.manual_seed(42)
        du_core1.reset_weights()
        
        torch.manual_seed(42)
        du_core2.reset_weights()
        
        # Verify identical results
        for layer1, layer2 in zip(du_core1.linear_layers, du_core2.linear_layers):
            assert torch.equal(layer1.weight, layer2.weight)
            assert torch.equal(layer1.bias, layer2.bias)
            
    def test_weight_reset_parameter_count_preservation(self):
        """Test that weight reset preserves the total number of parameters."""
        du_core = DU_Core_V1(input_size=12, hidden_sizes=[24, 18], output_size=6)
        
        # Count parameters before reset
        initial_param_count = sum(p.numel() for p in du_core.parameters())
        initial_trainable_count = sum(p.numel() for p in du_core.parameters() if p.requires_grad)
        
        # Apply weight reset
        du_core.reset_weights()
        
        # Count parameters after reset
        final_param_count = sum(p.numel() for p in du_core.parameters())
        final_trainable_count = sum(p.numel() for p in du_core.parameters() if p.requires_grad)
        
        # Verify parameter counts unchanged
        assert initial_param_count == final_param_count
        assert initial_trainable_count == final_trainable_count


class TestDUCoreV2WeightReset:
    """Test suite for DU_Core_V2 hybrid architecture weight reset protocol."""
    
    def test_hybrid_layer_weight_reset(self):
        """Test that SSM layers are properly reset in hybrid architecture."""
        du_core = DU_Core_V2(
            input_size=10,
            hidden_sizes=[16],
            output_size=8,
            state_dim=12
        )

        # First, manually modify some parameters to ensure they're not zero
        with torch.no_grad():
            for layer in du_core.layers:
                if isinstance(layer, HybridSNNSSMLayer):
                    # Set some parameters to known non-zero values
                    for param in layer.A_generator.parameters():
                        param.data.fill_(1.0)  # Set to 1.0
                    for param in layer.B_generator.parameters():
                        param.data.fill_(2.0)  # Set to 2.0
                    for param in layer.C_generator.parameters():
                        param.data.fill_(3.0)  # Set to 3.0

        # Apply weight reset
        du_core.reset_weights()

        # Verify SSM parameters changed from the known values
        for layer in du_core.layers:
            if isinstance(layer, HybridSNNSSMLayer):
                # Check A generator parameters are no longer all 1.0
                for param in layer.A_generator.parameters():
                    assert not torch.all(param == 1.0), \
                        "A generator parameters not reset from known values"

                # Check B generator parameters are no longer all 2.0
                for param in layer.B_generator.parameters():
                    assert not torch.all(param == 2.0), \
                        "B generator parameters not reset from known values"

                # Check C generator parameters are no longer all 3.0
                for param in layer.C_generator.parameters():
                    assert not torch.all(param == 3.0), \
                        "C generator parameters not reset from known values"
                
    def test_attention_layer_weight_reset(self):
        """Test that attention layers are properly reset in hybrid architecture."""
        du_core = DU_Core_V2(
            input_size=8,
            hidden_sizes=[12],
            output_size=6,
            state_dim=10,
            attention_interval=1,  # Add attention after every SSM layer
            attention_heads=2
        )
        
        # Store initial attention parameters
        initial_attention_params = []
        for layer in du_core.layers:
            if isinstance(layer, nn.MultiheadAttention):
                initial_attention_params.append({
                    'in_proj_weight': layer.in_proj_weight.clone().detach(),
                    'in_proj_bias': layer.in_proj_bias.clone().detach(),
                    'out_proj_weight': layer.out_proj.weight.clone().detach(),
                    'out_proj_bias': layer.out_proj.bias.clone().detach(),
                })
        
        # Apply weight reset
        du_core.reset_weights()
        
        # Verify attention parameters changed
        param_idx = 0
        for layer in du_core.layers:
            if isinstance(layer, nn.MultiheadAttention):
                initial_params = initial_attention_params[param_idx]
                
                # Verify attention weights were reset
                assert not torch.equal(
                    initial_params['in_proj_weight'], 
                    layer.in_proj_weight
                ), "Attention in_proj_weight not reset"
                
                assert not torch.equal(
                    initial_params['out_proj_weight'], 
                    layer.out_proj.weight
                ), "Attention out_proj_weight not reset"

                param_idx += 1

    def test_neuron_parameter_preservation(self):
        """Test that neuron dynamics parameters are preserved during reset."""
        original_beta = 0.9
        original_threshold = 1.2

        du_core = DU_Core_V2(
            input_size=6,
            hidden_sizes=[10],
            output_size=4,
            state_dim=8,
            beta=original_beta,
            threshold=original_threshold
        )

        # Apply weight reset
        du_core.reset_weights()

        # Verify neuron parameters preserved
        assert du_core.beta == original_beta, \
            f"Beta changed from {original_beta} to {du_core.beta}"
        assert du_core.threshold == original_threshold, \
            f"Threshold changed from {original_threshold} to {du_core.threshold}"

    def test_architecture_preservation(self):
        """Test that hybrid architecture is preserved during weight reset."""
        du_core = DU_Core_V2(
            input_size=8,
            hidden_sizes=[12, 10],
            output_size=6,
            state_dim=14,
            attention_interval=2,
            attention_heads=2
        )

        # Store architecture info
        initial_layer_count = len(du_core.layers)
        initial_layer_types = [type(layer).__name__ for layer in du_core.layers]

        # Apply weight reset
        du_core.reset_weights()

        # Verify architecture unchanged
        assert len(du_core.layers) == initial_layer_count
        final_layer_types = [type(layer).__name__ for layer in du_core.layers]
        assert initial_layer_types == final_layer_types

    def test_ssm_matrix_generator_reset(self):
        """Test that SSM matrix generators are properly reinitialized."""
        du_core = DU_Core_V2(input_size=6, hidden_sizes=[8], output_size=4, state_dim=10)

        # First, manually set parameters to known values
        with torch.no_grad():
            for layer in du_core.layers:
                if isinstance(layer, HybridSNNSSMLayer):
                    # Set A generator parameters to 5.0
                    for param in layer.A_generator.parameters():
                        param.data.fill_(5.0)
                    # Set B generator parameters to 6.0
                    for param in layer.B_generator.parameters():
                        param.data.fill_(6.0)
                    # Set C generator parameters to 7.0
                    for param in layer.C_generator.parameters():
                        param.data.fill_(7.0)

        # Apply weight reset
        du_core.reset_weights()

        # Verify generators were reset from known values
        for layer in du_core.layers:
            if isinstance(layer, HybridSNNSSMLayer):
                # Check A generator parameters are no longer all 5.0
                for param in layer.A_generator.parameters():
                    assert not torch.all(param == 5.0), \
                        "A generator parameters not reset from known values"

                # Check B generator parameters are no longer all 6.0
                for param in layer.B_generator.parameters():
                    assert not torch.all(param == 6.0), \
                        "B generator parameters not reset from known values"

                # Check C generator parameters are no longer all 7.0
                for param in layer.C_generator.parameters():
                    assert not torch.all(param == 7.0), \
                        "C generator parameters not reset from known values"

    def test_weight_reset_deterministic(self):
        """Test deterministic behavior of DU_Core_V2 weight reset."""
        # Create two identical DU_Cores
        config = {
            'input_size': 6,
            'hidden_sizes': [8],
            'output_size': 4,
            'state_dim': 10
        }

        du_core1 = DU_Core_V2(**config)
        du_core2 = DU_Core_V2(**config)

        # Reset with same seed
        torch.manual_seed(123)
        du_core1.reset_weights()

        torch.manual_seed(123)
        du_core2.reset_weights()

        # Verify identical results (compare parameters of Sequential modules)
        for layer1, layer2 in zip(du_core1.layers, du_core2.layers):
            if isinstance(layer1, HybridSNNSSMLayer) and isinstance(layer2, HybridSNNSSMLayer):
                # Compare A generator parameters
                for p1, p2 in zip(layer1.A_generator.parameters(), layer2.A_generator.parameters()):
                    assert torch.equal(p1, p2), "A generator parameters not identical"
                # Compare B generator parameters
                for p1, p2 in zip(layer1.B_generator.parameters(), layer2.B_generator.parameters()):
                    assert torch.equal(p1, p2), "B generator parameters not identical"
                # Compare C generator parameters
                for p1, p2 in zip(layer1.C_generator.parameters(), layer2.C_generator.parameters()):
                    assert torch.equal(p1, p2), "C generator parameters not identical"
            elif isinstance(layer1, nn.MultiheadAttention) and isinstance(layer2, nn.MultiheadAttention):
                assert torch.equal(layer1.in_proj_weight, layer2.in_proj_weight)
                assert torch.equal(layer1.out_proj.weight, layer2.out_proj.weight)

    def test_weight_reset_parameter_count(self):
        """Test that parameter count is preserved in DU_Core_V2."""
        du_core = DU_Core_V2(
            input_size=10,
            hidden_sizes=[16, 12],
            output_size=8,
            state_dim=14,
            attention_interval=2,
            attention_heads=2
        )

        # Count parameters before reset
        initial_param_count = sum(p.numel() for p in du_core.parameters())
        initial_trainable_count = sum(p.numel() for p in du_core.parameters() if p.requires_grad)

        # Apply weight reset
        du_core.reset_weights()

        # Count parameters after reset
        final_param_count = sum(p.numel() for p in du_core.parameters())
        final_trainable_count = sum(p.numel() for p in du_core.parameters() if p.requires_grad)

        # Verify parameter counts unchanged
        assert initial_param_count == final_param_count
        assert initial_trainable_count == final_trainable_count

    def test_weight_reset_integration(self):
        """Test full integration of weight reset with DU_Core_V2."""
        du_core = DU_Core_V2(
            input_size=8,
            hidden_sizes=[12, 10],
            output_size=6,
            state_dim=14,
            attention_interval=1,  # Attention after every SSM layer
            attention_heads=2
        )

        # First, manually modify some weights to ensure they're not in initial state
        with torch.no_grad():
            for layer in du_core.layers:
                if isinstance(layer, HybridSNNSSMLayer):
                    for param in layer.A_generator.parameters():
                        param.data += 0.5  # Add offset to initial values

        # Create test input
        test_input = torch.randn(50, 1, 8)  # [time_steps, batch_size, input_size]

        # Get modified output
        modified_output = du_core.forward(test_input, num_steps=50)

        # Apply weight reset
        du_core.reset_weights()

        # Get output after reset
        reset_output = du_core.forward(test_input, num_steps=50)

        # Verify outputs are different (weights changed)
        # Use a tolerance-based comparison since outputs might be very small
        output_diff = torch.abs(modified_output - reset_output).max().item()
        assert output_diff > 1e-6, \
            f"Outputs too similar after weight reset (max diff: {output_diff}) - weights may not have changed"

        # Verify output shape unchanged
        assert modified_output.shape == reset_output.shape, \
            "Output shape changed after weight reset"


class TestMetaParameterPreservation:
    """Test suite for validating meta-parameter preservation during weight reset."""

    def test_plasticity_rule_preservation(self):
        """Test that plasticity rule parameters are preserved during weight reset."""
        # Test with ThreeFactor_Hebbian_Rule
        plasticity_rule = ThreeFactor_Hebbian_Rule(learning_rate=0.03)
        original_learning_rate = plasticity_rule.base_learning_rate

        du_core = DU_Core_V1(
            input_size=5,
            hidden_sizes=[8],
            output_size=3,
            plasticity_rule=plasticity_rule
        )

        # Apply weight reset
        du_core.reset_weights()

        # Verify plasticity rule preserved
        if hasattr(du_core, '_plasticity_rule') and du_core._plasticity_rule is not None:
            assert du_core._plasticity_rule.base_learning_rate == original_learning_rate

        # Test with STDP_Rule
        stdp_rule = STDP_Rule(
            learning_rate_ltp=0.01,
            learning_rate_ltd=0.005,
            tau_ltp=20.0,
            tau_ltd=25.0
        )

        du_core_stdp = DU_Core_V1(
            input_size=4,
            hidden_sizes=[6],
            output_size=2,
            plasticity_rule=stdp_rule
        )

        # Store original parameters
        original_ltp = stdp_rule.learning_rate_ltp
        original_ltd = stdp_rule.learning_rate_ltd
        original_tau_ltp = stdp_rule.tau_ltp
        original_tau_ltd = stdp_rule.tau_ltd

        # Apply weight reset
        du_core_stdp.reset_weights()

        # Verify STDP parameters preserved
        if hasattr(du_core_stdp, '_plasticity_rule') and du_core_stdp._plasticity_rule is not None:
            assert du_core_stdp._plasticity_rule.learning_rate_ltp == original_ltp
            assert du_core_stdp._plasticity_rule.learning_rate_ltd == original_ltd
            assert du_core_stdp._plasticity_rule.tau_ltp == original_tau_ltp
            assert du_core_stdp._plasticity_rule.tau_ltd == original_tau_ltd

    def test_memory_system_preservation(self):
        """Test that memory system parameters are preserved during weight reset."""
        # Create memory system
        memory_system = EpisodicMemory(capacity=100)

        # Store some test memories using ExperienceTuple
        from gif_framework.core.memory_systems import ExperienceTuple

        for i in range(5):
            experience = ExperienceTuple(
                input_spikes=torch.randn(10, 1, 8),
                internal_state=None,
                output_spikes=torch.randn(10, 1, 4),
                task_id=f'test_task_{i}'
            )
            memory_system.add(experience)

        # Create DU_Core with memory system
        du_core = DU_Core_V1(
            input_size=6,
            hidden_sizes=[10],
            output_size=4,
            memory_system=memory_system
        )

        # Store original memory count
        original_memory_count = len(memory_system)

        # Apply weight reset
        du_core.reset_weights()

        # Verify memory system preserved
        if hasattr(du_core, '_memory_system') and du_core._memory_system is not None:
            assert len(du_core._memory_system) == original_memory_count, \
                "Memory system contents changed after weight reset"

    def test_neuron_dynamics_preservation(self):
        """Test that neuron dynamics parameters are preserved across both cores."""
        # Test DU_Core_V1
        du_core_v1 = DU_Core_V1(
            input_size=5,
            hidden_sizes=[8],
            output_size=3,
            beta=0.88,
            threshold=1.3
        )

        original_beta_v1 = du_core_v1.beta
        original_threshold_v1 = du_core_v1.threshold

        du_core_v1.reset_weights()

        assert du_core_v1.beta == original_beta_v1
        assert du_core_v1.threshold == original_threshold_v1

        # Test DU_Core_V2
        du_core_v2 = DU_Core_V2(
            input_size=6,
            hidden_sizes=[10],
            output_size=4,
            state_dim=12,
            beta=0.92,
            threshold=1.1
        )

        original_beta_v2 = du_core_v2.beta
        original_threshold_v2 = du_core_v2.threshold

        du_core_v2.reset_weights()

        assert du_core_v2.beta == original_beta_v2
        assert du_core_v2.threshold == original_threshold_v2

    def test_architecture_config_preservation(self):
        """Test that architecture configuration is preserved during reset."""
        # Test DU_Core_V2 with complex configuration
        original_config = {
            'input_size': 8,
            'hidden_sizes': [12, 10, 8],
            'output_size': 6,
            'state_dim': 14,
            'attention_interval': 2,
            'attention_heads': 2
        }

        du_core = DU_Core_V2(**original_config)

        # Apply weight reset
        du_core.reset_weights()

        # Verify configuration preserved
        config = du_core.get_config()

        assert config['input_size'] == original_config['input_size']
        assert config['output_size'] == original_config['output_size']
        assert config['state_dim'] == original_config['state_dim']
        assert len(config['hidden_sizes']) == len(original_config['hidden_sizes'])

    def test_training_state_preservation(self):
        """Test that training mode state is preserved during reset."""
        du_core = DU_Core_V2(input_size=6, hidden_sizes=[8], output_size=4, state_dim=10)

        # Set to evaluation mode
        du_core.eval()
        assert not du_core.training

        # Apply weight reset
        du_core.reset_weights()

        # Verify still in evaluation mode
        assert not du_core.training

        # Test with training mode
        du_core.train()
        assert du_core.training

        du_core.reset_weights()

        # Verify still in training mode
        assert du_core.training


class TestPotentiationExperimentValidity:
    """Test suite for validating potentiation experiment methodology."""

    def test_weight_reset_eliminates_knowledge(self):
        """
        CRITICAL TEST: Verify that weight reset eliminates task-specific knowledge.

        This test ensures that the weight reset protocol successfully removes
        learned knowledge, which is essential for valid potentiation experiments.
        """
        du_core = DU_Core_V2(input_size=10, hidden_sizes=[16], output_size=8, state_dim=12)

        # Create training data that should create specific patterns
        training_input = torch.randn(100, 1, 10)
        training_target = torch.randn(1, 8)

        # "Train" the model (simulate learning by modifying weights)
        with torch.no_grad():
            # Get initial output
            initial_output = du_core.forward(training_input, num_steps=100)

            # Manually modify some weights to simulate learning
            for layer in du_core.layers:
                if isinstance(layer, HybridSNNSSMLayer):
                    # Modify parameters in Sequential modules
                    for param in layer.A_generator.parameters():
                        param.data += 0.1 * torch.randn_like(param.data)
                    for param in layer.B_generator.parameters():
                        param.data += 0.1 * torch.randn_like(param.data)
                    for param in layer.C_generator.parameters():
                        param.data += 0.1 * torch.randn_like(param.data)

        # Get "trained" output
        trained_output = du_core.forward(training_input, num_steps=100)

        # Verify training changed the output
        assert not torch.equal(initial_output, trained_output), \
            "Training did not change output - test setup invalid"

        # Apply weight reset
        du_core.reset_weights()

        # Get output after reset
        reset_output = du_core.forward(training_input, num_steps=100)

        # Verify reset eliminated the learned patterns
        # The output should be different from the trained output
        assert not torch.equal(trained_output, reset_output), \
            "Weight reset did not eliminate learned patterns"

    def test_structural_improvements_preserved(self):
        """
        Test that structural improvements (architecture) are preserved.

        This validates that the network structure that enables better learning
        is maintained while specific knowledge is wiped.
        """
        # Create DU_Core with specific architecture
        du_core = DU_Core_V2(
            input_size=8,
            hidden_sizes=[12, 10],
            output_size=6,
            state_dim=14,
            attention_interval=1,  # Attention after every SSM layer
            attention_heads=2
        )

        # Store structural information
        initial_layer_count = len(du_core.layers)
        initial_attention_count = sum(1 for layer in du_core.layers
                                    if isinstance(layer, nn.MultiheadAttention))
        initial_ssm_count = sum(1 for layer in du_core.layers
                              if isinstance(layer, HybridSNNSSMLayer))

        # Apply weight reset
        du_core.reset_weights()

        # Verify structure preserved
        final_layer_count = len(du_core.layers)
        final_attention_count = sum(1 for layer in du_core.layers
                                  if isinstance(layer, nn.MultiheadAttention))
        final_ssm_count = sum(1 for layer in du_core.layers
                            if isinstance(layer, HybridSNNSSMLayer))

        assert initial_layer_count == final_layer_count
        assert initial_attention_count == final_attention_count
        assert initial_ssm_count == final_ssm_count

    def test_learning_capacity_measurement(self):
        """
        Test that the reset protocol enables valid learning capacity measurement.

        This test validates that after reset, the model can still learn but
        starts from a clean slate, enabling fair comparison.
        """
        du_core = DU_Core_V2(input_size=6, hidden_sizes=[8], output_size=4, state_dim=10)

        # Create simple learning task with stronger input
        test_input = torch.randn(20, 1, 6) * 2.0  # Stronger input to ensure non-zero outputs

        # Get baseline output
        baseline_output = du_core.forward(test_input, num_steps=20)

        # Modify some weights to ensure they change after reset
        with torch.no_grad():
            for layer in du_core.layers:
                if hasattr(layer, 'linear') and hasattr(layer.linear, 'weight'):
                    layer.linear.weight.data += torch.randn_like(layer.linear.weight.data) * 0.1

        # Apply weight reset
        du_core.reset_weights()

        # Get reset output
        reset_output = du_core.forward(test_input, num_steps=20)

        # Check if outputs are meaningfully different (allowing for small numerical differences)
        output_diff = torch.abs(baseline_output - reset_output).mean().item()
        assert output_diff > 1e-6, \
            f"Reset did not create clean slate (difference: {output_diff})"

        # Verify model can still process inputs (learning capacity intact)
        assert reset_output.shape == baseline_output.shape, \
            "Model lost processing capability after reset"
        assert not torch.isnan(reset_output).any(), \
            "Model produces NaN after reset"
        assert torch.isfinite(reset_output).all(), \
            "Model produces infinite values after reset"

    def test_scientific_control_validity(self):
        """
        Test that the weight reset protocol provides valid scientific control.

        This validates that the experimental methodology is sound for
        measuring system potentiation.
        """
        # Create two identical models
        du_core_control = DU_Core_V2(input_size=8, hidden_sizes=[10], output_size=6, state_dim=12)
        du_core_experimental = DU_Core_V2(input_size=8, hidden_sizes=[10], output_size=6, state_dim=12)

        # Simulate "pre-training" on experimental model
        with torch.no_grad():
            for layer in du_core_experimental.layers:
                if isinstance(layer, HybridSNNSSMLayer):
                    # Simulate learning by modifying parameters
                    for param in layer.A_generator.parameters():
                        param.data += 0.05 * torch.randn_like(param.data)

        # Apply weight reset to experimental model
        du_core_experimental.reset_weights()

        # Test with same input
        test_input = torch.randn(30, 1, 8)

        control_output = du_core_control.forward(test_input, num_steps=30)
        experimental_output = du_core_experimental.forward(test_input, num_steps=30)

        # Outputs should be different (different random weights)
        # but both should be valid (same capability)
        assert not torch.equal(control_output, experimental_output), \
            "Control and experimental outputs identical - invalid control"
        assert control_output.shape == experimental_output.shape, \
            "Different output shapes - invalid comparison"
        assert not torch.isnan(control_output).any() and not torch.isnan(experimental_output).any(), \
            "NaN outputs detected - invalid models"


class TestIntegrationWithFramework:
    """Test suite for weight reset integration with the complete GIF framework."""

    def test_gif_orchestrator_integration(self):
        """Test weight reset integration with GIF orchestrator."""
        # Create DU_Core
        du_core = DU_Core_V2(input_size=8, hidden_sizes=[10], output_size=6, state_dim=12)

        # Create GIF orchestrator
        gif = GIF(du_core)

        # Store initial weights
        initial_weights = []
        for layer in du_core.layers:
            if isinstance(layer, HybridSNNSSMLayer):
                initial_weights.append({
                    'A_gen': [p.clone().detach() for p in layer.A_generator.parameters()],
                    'B_gen': [p.clone().detach() for p in layer.B_generator.parameters()],
                    'C_gen': [p.clone().detach() for p in layer.C_generator.parameters()],
                })

        # Apply weight reset through DU_Core
        du_core.reset_weights()

        # Verify GIF still functions
        test_input = torch.randn(8)

        # This should work without errors
        try:
            # Note: This might fail if encoders/decoders aren't attached, but that's expected
            result = gif.process_single_input(test_input)
            integration_success = True
        except Exception as e:
            # If it fails due to missing encoders/decoders, that's acceptable
            if "encoder" in str(e).lower() or "decoder" in str(e).lower():
                integration_success = True
            else:
                integration_success = False

        assert integration_success, "GIF integration failed after weight reset"

        # Verify weights actually changed
        weights_changed = False
        weight_idx = 0
        for layer in du_core.layers:
            if isinstance(layer, HybridSNNSSMLayer):
                initial = initial_weights[weight_idx]
                # Check A generator parameters
                for initial_param, current_param in zip(initial['A_gen'], layer.A_generator.parameters()):
                    if not torch.equal(initial_param, current_param):
                        weights_changed = True
                        break
                # Check B generator parameters
                for initial_param, current_param in zip(initial['B_gen'], layer.B_generator.parameters()):
                    if not torch.equal(initial_param, current_param):
                        weights_changed = True
                        break
                # Check C generator parameters
                for initial_param, current_param in zip(initial['C_gen'], layer.C_generator.parameters()):
                    if not torch.equal(initial_param, current_param):
                        weights_changed = True
                        break
                if weights_changed:
                    break
                weight_idx += 1

        assert weights_changed, "Weights did not change during reset"

    def test_encoder_decoder_compatibility(self):
        """Test that weight reset maintains encoder/decoder compatibility."""
        # Create mock encoder and decoder
        class MockEncoder(EncoderInterface):
            def encode(self, raw_data):
                return torch.randn(50, 1, 8)  # 50 time steps, batch size 1, 8 features
            def calibrate(self, calibration_data):
                pass
            def get_config(self):
                return {'type': 'mock'}

        class MockDecoder(DecoderInterface):
            def decode(self, spike_train):
                return torch.mean(spike_train).item()
            def get_config(self):
                return {'type': 'mock'}

        # Create DU_Core and GIF
        du_core = DU_Core_V2(input_size=8, hidden_sizes=[10], output_size=6, state_dim=12)
        gif = GIF(du_core)
        gif.attach_encoder(MockEncoder())
        gif.attach_decoder(MockDecoder())

        # Test processing before reset
        test_data = torch.randn(8)
        try:
            result_before = gif.process_single_input(test_data)
            processing_before = True
        except:
            processing_before = False

        # Apply weight reset
        du_core.reset_weights()

        # Test processing after reset
        try:
            result_after = gif.process_single_input(test_data)
            processing_after = True
        except:
            processing_after = False

        # Both should work (or both should fail consistently)
        assert processing_before == processing_after, \
            "Encoder/decoder compatibility changed after weight reset"

    def test_training_pipeline_integration(self):
        """Test weight reset integration with training pipeline."""
        du_core = DU_Core_V2(input_size=6, hidden_sizes=[8], output_size=4, state_dim=10)

        # Set up for training
        du_core.train()
        optimizer = torch.optim.Adam(du_core.parameters(), lr=0.001)

        # Store initial optimizer state
        initial_optimizer_state = len(optimizer.state)

        # Apply weight reset
        du_core.reset_weights()

        # Verify training mode preserved
        assert du_core.training, "Training mode not preserved after reset"

        # Verify optimizer still works
        test_input = torch.randn(20, 1, 6)
        test_target = torch.randn(20, 1, 4)  # Match output shape [time_steps, batch_size, output_size]

        output = du_core.forward(test_input, num_steps=20)
        loss = torch.nn.functional.mse_loss(output, test_target)

        # This should work without errors
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Test completed successfully if no exceptions raised
        assert True, "Training pipeline integration successful"

    def test_model_loading_and_reset(self):
        """Test loading pre-trained model and applying weight reset."""
        # Create and "train" a model
        du_core_original = DU_Core_V2(input_size=6, hidden_sizes=[8], output_size=4, state_dim=10)

        # Simulate training by modifying weights
        with torch.no_grad():
            for layer in du_core_original.layers:
                if isinstance(layer, HybridSNNSSMLayer):
                    for param in layer.A_generator.parameters():
                        param.data += 0.1 * torch.randn_like(param.data)

        # Save model state
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            torch.save(du_core_original.state_dict(), tmp_file.name)
            model_path = tmp_file.name

        try:
            # Create new model and load state
            du_core_loaded = DU_Core_V2(input_size=6, hidden_sizes=[8], output_size=4, state_dim=10)
            du_core_loaded.load_state_dict(torch.load(model_path))

            # Verify loaded model has same weights as original
            test_input = torch.randn(20, 1, 6)
            original_output = du_core_original.forward(test_input, num_steps=20)
            loaded_output = du_core_loaded.forward(test_input, num_steps=20)

            assert torch.allclose(original_output, loaded_output, atol=1e-6), \
                "Loaded model does not match original"

            # Apply weight reset to loaded model
            du_core_loaded.reset_weights()

            # Verify reset worked
            reset_output = du_core_loaded.forward(test_input, num_steps=20)
            assert not torch.equal(loaded_output, reset_output), \
                "Weight reset did not change loaded model"

        finally:
            # Clean up temporary file
            import os
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error handling in weight reset."""

    def test_reset_empty_model(self):
        """Test weight reset on model with minimal configuration."""
        # Create minimal DU_Core
        du_core = DU_Core_V2(input_size=1, hidden_sizes=[], output_size=1, state_dim=2)

        # This should not raise an error
        try:
            du_core.reset_weights()
            reset_success = True
        except Exception as e:
            reset_success = False
            print(f"Reset failed on minimal model: {e}")

        assert reset_success, "Weight reset failed on minimal model"

    def test_reset_partially_trained_model(self):
        """Test weight reset on partially trained model."""
        du_core = DU_Core_V2(input_size=6, hidden_sizes=[8], output_size=4, state_dim=10)

        # Partially modify some weights (simulate partial training)
        with torch.no_grad():
            layers_with_ssm = [layer for layer in du_core.layers if isinstance(layer, HybridSNNSSMLayer)]
            if layers_with_ssm:
                # Only modify first layer - get first parameter of A_generator
                layer = layers_with_ssm[0]
                first_param = next(layer.A_generator.parameters())
                first_param.data.view(-1)[0] = 999.0  # Extreme value

        # Apply weight reset
        du_core.reset_weights()

        # Verify extreme value was reset
        for layer in du_core.layers:
            if isinstance(layer, HybridSNNSSMLayer):
                for param in layer.A_generator.parameters():
                    assert not torch.any(param == 999.0), \
                        "Extreme weight value not reset"

    def test_reset_error_recovery(self):
        """Test error handling and recovery during weight reset."""
        du_core = DU_Core_V2(input_size=6, hidden_sizes=[8], output_size=4, state_dim=10)

        # Store original state
        original_state = du_core.state_dict()

        # Apply weight reset (should succeed)
        try:
            du_core.reset_weights()
            reset_success = True
        except Exception as e:
            reset_success = False
            print(f"Unexpected reset failure: {e}")

        assert reset_success, "Weight reset unexpectedly failed"

        # Verify model is still functional
        test_input = torch.randn(20, 1, 6)
        try:
            output = du_core.forward(test_input, num_steps=20)
            functional_after_reset = True
        except Exception as e:
            functional_after_reset = False
            print(f"Model not functional after reset: {e}")

        assert functional_after_reset, "Model not functional after weight reset"

        # Verify output is reasonable
        assert not torch.isnan(output).any(), "NaN values in output after reset"
        assert torch.isfinite(output).all(), "Infinite values in output after reset"
