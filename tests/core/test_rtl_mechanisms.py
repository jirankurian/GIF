"""
Test Suite for Real-Time Learning (RTL) Mechanisms
==================================================

This module contains comprehensive tests for the synaptic plasticity engine that drives
Real-Time Learning within the GIF framework. It validates the implementation of all
plasticity rules and their integration with the DU Core.

Test Coverage:
- PlasticityRuleInterface contract enforcement
- STDP_Rule implementation and parameter validation
- ThreeFactor_Hebbian_Rule implementation and parameter validation
- Integration with DU_Core_V1
- Error handling and edge cases
- Biological plausibility of learning rules

Author: GIF Development Team
Phase: 3.1 - RTL Mechanisms Testing
"""

import pytest
import torch
import numpy as np
from abc import ABC
from typing import Any

# Import the RTL mechanisms to test
from gif_framework.core.rtl_mechanisms import (
    PlasticityRuleInterface,
    STDP_Rule,
    ThreeFactor_Hebbian_Rule
)

# Import DU Core for integration testing
from gif_framework.core.du_core import DU_Core_V1


class TestPlasticityRuleInterface:
    """Test suite for the PlasticityRuleInterface abstract base class."""

    def test_interface_is_abstract(self):
        """Test that PlasticityRuleInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PlasticityRuleInterface()

    def test_interface_inheritance(self):
        """Test that PlasticityRuleInterface properly inherits from ABC."""
        assert issubclass(PlasticityRuleInterface, ABC)

    def test_apply_method_is_abstract(self):
        """Test that apply method is properly marked as abstract."""
        # Check that apply method exists and is abstract
        assert hasattr(PlasticityRuleInterface, 'apply')
        assert getattr(PlasticityRuleInterface.apply, '__isabstractmethod__', False)

    def test_concrete_implementation_works(self):
        """Test that concrete implementations of the interface work correctly."""
        
        class TestRule(PlasticityRuleInterface):
            def apply(self, **kwargs) -> torch.Tensor:
                return torch.zeros(2, 2)
        
        # Should be able to instantiate concrete implementation
        rule = TestRule()
        result = rule.apply()
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2)

    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementations cannot be instantiated."""
        
        class IncompleteRule(PlasticityRuleInterface):
            pass  # Missing apply method
        
        with pytest.raises(TypeError):
            IncompleteRule()


class TestSTDP_Rule:
    """Test suite for the STDP_Rule implementation."""

    def test_initialization_valid_parameters(self):
        """Test STDP_Rule initialization with valid parameters."""
        rule = STDP_Rule(
            learning_rate_ltp=0.01,
            learning_rate_ltd=0.005,
            tau_ltp=20.0,
            tau_ltd=20.0
        )
        
        assert rule.learning_rate_ltp == 0.01
        assert rule.learning_rate_ltd == 0.005
        assert rule.tau_ltp == 20.0
        assert rule.tau_ltd == 20.0
        assert isinstance(rule, PlasticityRuleInterface)

    def test_initialization_invalid_parameters(self):
        """Test STDP_Rule initialization with invalid parameters."""
        
        # Test negative learning rates
        with pytest.raises(ValueError, match="learning_rate_ltp must be positive"):
            STDP_Rule(-0.01, 0.005, 20.0, 20.0)
        
        with pytest.raises(ValueError, match="learning_rate_ltd must be positive"):
            STDP_Rule(0.01, -0.005, 20.0, 20.0)
        
        # Test zero learning rates
        with pytest.raises(ValueError, match="learning_rate_ltp must be positive"):
            STDP_Rule(0.0, 0.005, 20.0, 20.0)
        
        # Test negative time constants
        with pytest.raises(ValueError, match="tau_ltp must be positive"):
            STDP_Rule(0.01, 0.005, -20.0, 20.0)
        
        with pytest.raises(ValueError, match="tau_ltd must be positive"):
            STDP_Rule(0.01, 0.005, 20.0, -20.0)

    def test_apply_missing_parameters(self):
        """Test STDP apply method with missing required parameters."""
        rule = STDP_Rule(0.01, 0.005, 20.0, 20.0)
        
        # Missing pre_spikes
        with pytest.raises(ValueError, match="STDP rule requires 'pre_spikes' parameter"):
            rule.apply(post_spikes=torch.rand(10, 2, 5))
        
        # Missing post_spikes
        with pytest.raises(ValueError, match="STDP rule requires 'post_spikes' parameter"):
            rule.apply(pre_spikes=torch.rand(10, 2, 5))

    def test_apply_invalid_tensor_shapes(self):
        """Test STDP apply method with invalid tensor shapes."""
        rule = STDP_Rule(0.01, 0.005, 20.0, 20.0)
        
        # Test mismatched time dimensions
        with pytest.raises(ValueError, match="Time dimensions must match"):
            rule.apply(
                pre_spikes=torch.rand(5, 2, 5),  # 5 time steps
                post_spikes=torch.rand(10, 2, 5)  # 10 time steps
            )
        
        # Test mismatched batch dimensions
        with pytest.raises(ValueError, match="Batch dimensions must match"):
            rule.apply(
                pre_spikes=torch.rand(10, 2, 5),  # batch_size=2
                post_spikes=torch.rand(10, 3, 5)  # batch_size=3
            )
        
        # Test mismatched time dimensions
        with pytest.raises(ValueError, match="Time dimensions must match"):
            rule.apply(
                pre_spikes=torch.rand(10, 2, 5),
                post_spikes=torch.rand(15, 2, 5)  # Different time dimension
            )
        
        # Test mismatched batch dimensions
        with pytest.raises(ValueError, match="Batch dimensions must match"):
            rule.apply(
                pre_spikes=torch.rand(10, 2, 5),
                post_spikes=torch.rand(10, 3, 5)  # Different batch dimension
            )

    def test_apply_valid_computation(self):
        """Test STDP apply method with valid inputs produces correct output."""
        rule = STDP_Rule(0.01, 0.005, 20.0, 20.0)
        
        # Create test spike data
        time_steps, batch_size, num_pre, num_post = 5, 2, 3, 4
        pre_spikes = torch.zeros(time_steps, batch_size, num_pre)
        post_spikes = torch.zeros(time_steps, batch_size, num_post)
        
        # Add some spikes
        pre_spikes[1, 0, 0] = 1.0  # Pre-neuron 0 spikes at time 1
        post_spikes[2, 0, 1] = 1.0  # Post-neuron 1 spikes at time 2 (should cause LTP)
        
        weight_updates = rule.apply(pre_spikes=pre_spikes, post_spikes=post_spikes)
        
        # Check output shape
        assert weight_updates.shape == (num_pre, num_post)
        assert isinstance(weight_updates, torch.Tensor)
        
        # Check that some learning occurred (non-zero updates)
        assert not torch.allclose(weight_updates, torch.zeros_like(weight_updates))

    def test_apply_with_custom_dt(self):
        """Test STDP apply method with custom time step."""
        rule = STDP_Rule(0.01, 0.005, 20.0, 20.0)

        # Create spike patterns with guaranteed spikes
        pre_spikes = torch.zeros(5, 2, 3, dtype=torch.bool)
        post_spikes = torch.zeros(5, 2, 4, dtype=torch.bool)

        # Add some spikes at specific times
        pre_spikes[1, 0, 0] = True  # Pre-spike at time 1
        post_spikes[2, 0, 0] = True  # Post-spike at time 2 (causal)
        pre_spikes[3, 1, 1] = True  # Pre-spike at time 3
        post_spikes[1, 1, 1] = True  # Post-spike at time 1 (anti-causal)

        # Test with different dt values
        updates_dt1 = rule.apply(pre_spikes=pre_spikes, post_spikes=post_spikes, dt=1.0)
        updates_dt2 = rule.apply(pre_spikes=pre_spikes, post_spikes=post_spikes, dt=2.0)

        assert updates_dt1.shape == (3, 4)
        assert updates_dt2.shape == (3, 4)
        # Different dt should produce different results
        assert not torch.allclose(updates_dt1, updates_dt2)


class TestThreeFactorHebbianRule:
    """Test suite for the ThreeFactor_Hebbian_Rule implementation."""

    def test_initialization_valid_parameters(self):
        """Test ThreeFactor_Hebbian_Rule initialization with valid parameters."""
        rule = ThreeFactor_Hebbian_Rule(learning_rate=0.01)

        assert rule.base_learning_rate == 0.01
        assert rule.current_learning_rate == 0.01  # Starts at base rate
        assert rule.meta_learning_rate == 0.1  # Default value
        assert isinstance(rule, PlasticityRuleInterface)

    def test_initialization_invalid_parameters(self):
        """Test ThreeFactor_Hebbian_Rule initialization with invalid parameters."""
        
        # Test negative learning rate
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            ThreeFactor_Hebbian_Rule(-0.01)
        
        # Test zero learning rate
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            ThreeFactor_Hebbian_Rule(0.0)

    def test_apply_missing_parameters(self):
        """Test ThreeFactor_Hebbian apply method with missing required parameters."""
        rule = ThreeFactor_Hebbian_Rule(0.01)
        
        # Missing pre_activity
        with pytest.raises(ValueError, match="Three-Factor Hebbian rule requires either 'pre_activity' or 'pre_spikes' parameter"):
            rule.apply(post_activity=torch.rand(2, 5), modulatory_signal=0.5)
        
        # Missing post_activity
        with pytest.raises(ValueError, match="Three-Factor Hebbian rule requires either 'post_activity' or 'post_spikes' parameter"):
            rule.apply(pre_activity=torch.rand(2, 3), modulatory_signal=0.5)
        
        # Missing modulatory_signal
        with pytest.raises(ValueError, match="Three-Factor Hebbian rule requires 'modulatory_signal' parameter"):
            rule.apply(pre_activity=torch.rand(2, 3), post_activity=torch.rand(2, 5))

    def test_apply_invalid_parameter_types(self):
        """Test ThreeFactor_Hebbian apply method with invalid parameter types."""
        rule = ThreeFactor_Hebbian_Rule(0.01)

        # Test non-tensor pre_activity
        with pytest.raises(ValueError, match="pre_activity must be a torch.Tensor"):
            rule.apply(
                pre_activity=[1, 2, 3],  # List instead of tensor
                post_activity=torch.rand(2, 5),
                modulatory_signal=0.5
            )

        # Test non-tensor post_activity
        with pytest.raises(ValueError, match="post_activity must be a torch.Tensor"):
            rule.apply(
                pre_activity=torch.rand(2, 3),
                post_activity=[1, 2, 3],  # List instead of tensor
                modulatory_signal=0.5
            )

        # Test non-scalar modulatory_signal
        with pytest.raises(ValueError, match="modulatory_signal must be a scalar number"):
            rule.apply(
                pre_activity=torch.rand(2, 3),
                post_activity=torch.rand(2, 5),
                modulatory_signal=torch.tensor([0.5])  # Tensor instead of scalar
            )

    def test_apply_invalid_tensor_shapes(self):
        """Test ThreeFactor_Hebbian apply method with invalid tensor shapes."""
        rule = ThreeFactor_Hebbian_Rule(0.01)

        # Test 1D tensors (should be 2D)
        with pytest.raises(ValueError, match="pre_activity must be 2D tensor"):
            rule.apply(
                pre_activity=torch.rand(5),  # 1D instead of 2D
                post_activity=torch.rand(2, 5),
                modulatory_signal=0.5
            )

        with pytest.raises(ValueError, match="post_activity must be 2D tensor"):
            rule.apply(
                pre_activity=torch.rand(2, 3),
                post_activity=torch.rand(5),  # 1D instead of 2D
                modulatory_signal=0.5
            )

        # Test mismatched batch dimensions
        with pytest.raises(ValueError, match="Batch dimensions must match"):
            rule.apply(
                pre_activity=torch.rand(2, 3),
                post_activity=torch.rand(3, 5),  # Different batch dimension
                modulatory_signal=0.5
            )

    def test_apply_valid_computation(self):
        """Test ThreeFactor_Hebbian apply method with valid inputs produces correct output."""
        rule = ThreeFactor_Hebbian_Rule(0.01)

        # Create test activity data
        batch_size, num_pre, num_post = 2, 3, 4
        pre_activity = torch.rand(batch_size, num_pre)
        post_activity = torch.rand(batch_size, num_post)
        modulatory_signal = 0.5

        weight_updates = rule.apply(
            pre_activity=pre_activity,
            post_activity=post_activity,
            modulatory_signal=modulatory_signal
        )

        # Check output shape
        assert weight_updates.shape == (num_pre, num_post)
        assert isinstance(weight_updates, torch.Tensor)

        # Check that learning occurred proportional to modulatory signal
        expected_magnitude = rule.current_learning_rate * modulatory_signal
        assert torch.all(torch.abs(weight_updates) <= expected_magnitude * 2)  # Allow some variance

    def test_apply_modulatory_signal_effects(self):
        """Test that modulatory signal properly gates learning."""
        rule = ThreeFactor_Hebbian_Rule(0.01)

        pre_activity = torch.ones(2, 3)
        post_activity = torch.ones(2, 4)

        # Test with zero modulatory signal (no learning)
        updates_zero = rule.apply(
            pre_activity=pre_activity,
            post_activity=post_activity,
            modulatory_signal=0.0
        )
        assert torch.allclose(updates_zero, torch.zeros_like(updates_zero))

        # Test with positive modulatory signal (positive learning)
        updates_pos = rule.apply(
            pre_activity=pre_activity,
            post_activity=post_activity,
            modulatory_signal=1.0
        )
        assert torch.all(updates_pos > 0)

        # Test with negative modulatory signal (negative learning)
        updates_neg = rule.apply(
            pre_activity=pre_activity,
            post_activity=post_activity,
            modulatory_signal=-1.0
        )
        assert torch.all(updates_neg < 0)

        # Test proportionality
        assert torch.allclose(updates_neg, -updates_pos)


class TestDUCoreIntegration:
    """Test suite for RTL mechanisms integration with DU_Core_V1."""

    def test_du_core_plasticity_rule_injection(self):
        """Test that DU_Core_V1 can accept plasticity rules via dependency injection."""

        # Test with STDP rule
        stdp_rule = STDP_Rule(0.01, 0.005, 20.0, 20.0)
        du_core = DU_Core_V1(
            input_size=10,
            hidden_sizes=[5],
            output_size=3,
            plasticity_rule=stdp_rule
        )

        assert du_core._plasticity_rule is stdp_rule

        # Test with Hebbian rule
        hebbian_rule = ThreeFactor_Hebbian_Rule(0.01)
        du_core = DU_Core_V1(
            input_size=10,
            hidden_sizes=[5],
            output_size=3,
            plasticity_rule=hebbian_rule
        )

        assert du_core._plasticity_rule is hebbian_rule

    def test_du_core_invalid_plasticity_rule_type(self):
        """Test that DU_Core_V1 rejects invalid plasticity rule types."""

        class InvalidRule:
            def apply(self, **kwargs):
                return torch.zeros(2, 2)

        with pytest.raises(TypeError, match="plasticity_rule must implement PlasticityRuleInterface"):
            DU_Core_V1(
                input_size=10,
                hidden_sizes=[5],
                output_size=3,
                plasticity_rule=InvalidRule()  # Doesn't implement interface
            )

    def test_du_core_apply_learning_with_stdp(self):
        """Test DU_Core_V1 apply_learning method with STDP rule."""
        stdp_rule = STDP_Rule(0.01, 0.005, 20.0, 20.0)
        du_core = DU_Core_V1(
            input_size=5,
            hidden_sizes=[3],
            output_size=2,
            plasticity_rule=stdp_rule
        )

        # Create test spike data
        pre_spikes = torch.rand(10, 1, 5) > 0.9
        post_spikes = torch.rand(10, 1, 2) > 0.9

        weight_updates = du_core.apply_learning(
            pre_spikes=pre_spikes,
            post_spikes=post_spikes
        )

        assert isinstance(weight_updates, torch.Tensor)
        assert weight_updates.shape == (5, 2)

    def test_du_core_apply_learning_with_hebbian(self):
        """Test DU_Core_V1 apply_learning method with Hebbian rule."""
        hebbian_rule = ThreeFactor_Hebbian_Rule(0.01)
        du_core = DU_Core_V1(
            input_size=5,
            hidden_sizes=[3],
            output_size=2,
            plasticity_rule=hebbian_rule
        )

        # Create test activity data
        pre_activity = torch.rand(2, 5)
        post_activity = torch.rand(2, 2)
        modulatory_signal = 0.5

        weight_updates = du_core.apply_learning(
            pre_activity=pre_activity,
            post_activity=post_activity,
            modulatory_signal=modulatory_signal
        )

        assert isinstance(weight_updates, torch.Tensor)
        assert weight_updates.shape == (5, 2)

    def test_du_core_apply_learning_without_plasticity_rule(self):
        """Test DU_Core_V1 apply_learning method without plasticity rule raises error."""
        du_core = DU_Core_V1(
            input_size=5,
            hidden_sizes=[3],
            output_size=2
            # No plasticity_rule provided
        )

        with pytest.raises(RuntimeError, match="No plasticity rule has been injected"):
            du_core.apply_learning(pre_activity=torch.rand(2, 5), post_activity=torch.rand(2, 2))

    def test_du_core_apply_learning_invalid_parameters(self):
        """Test DU_Core_V1 apply_learning method with invalid parameters."""
        stdp_rule = STDP_Rule(0.01, 0.005, 20.0, 20.0)
        du_core = DU_Core_V1(
            input_size=5,
            hidden_sizes=[3],
            output_size=2,
            plasticity_rule=stdp_rule
        )

        # Test with missing required parameters for STDP
        with pytest.raises(ValueError, match="Plasticity rule STDP_Rule failed to apply"):
            du_core.apply_learning()  # Missing pre_spikes and post_spikes

    def test_du_core_repr_with_plasticity_rule(self):
        """Test DU_Core_V1 string representation includes plasticity rule information."""
        stdp_rule = STDP_Rule(0.01, 0.005, 20.0, 20.0)
        du_core = DU_Core_V1(
            input_size=5,
            hidden_sizes=[3],
            output_size=2,
            plasticity_rule=stdp_rule
        )

        repr_str = repr(du_core)
        assert "plasticity_rule: STDP_Rule" in repr_str

        # Test without plasticity rule
        du_core_no_rule = DU_Core_V1(
            input_size=5,
            hidden_sizes=[3],
            output_size=2
        )

        repr_str_no_rule = repr(du_core_no_rule)
        assert "plasticity_rule: None" in repr_str_no_rule


# Additional tests specified in Prompt 3.5
def test_three_factor_rule_gating():
    """Tests that the modulatory signal correctly gates learning."""
    rule = ThreeFactor_Hebbian_Rule(learning_rate=0.1)
    # Need 2D tensors [batch_size, num_neurons]
    pre_activity = torch.tensor([[0.0, 1.0, 1.0]])  # [1, 3]
    post_activity = torch.tensor([[1.0, 0.0, 1.0]])  # [1, 3]

    # Test with zero modulatory signal (should be no learning)
    delta_w_zero = rule.apply(
        pre_activity=pre_activity,
        post_activity=post_activity,
        modulatory_signal=0.0
    )
    assert torch.all(delta_w_zero == 0)

    # Test with positive modulatory signal (should be Hebbian learning)
    delta_w_positive = rule.apply(
        pre_activity=pre_activity,
        post_activity=post_activity,
        modulatory_signal=1.0
    )
    # Expected: learning_rate * pre_activity.T @ post_activity
    expected_dw = 0.1 * pre_activity.T @ post_activity
    assert torch.allclose(delta_w_positive, expected_dw)
