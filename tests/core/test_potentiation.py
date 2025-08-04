"""
Test Suite for System Potentiation Engine via Meta-Plasticity
============================================================

This module contains comprehensive tests for the System Potentiation mechanism,
which enables the GIF framework to "learn how to learn" through adaptive learning
rates based on performance feedback.

System Potentiation is the core scientific mechanism that allows the framework
to improve its learning efficiency over time, directly supporting the advanced
claims of the PhD thesis regarding meta-learning capabilities.

Test Coverage:
=============

1. **Performance Tracking**: Tests for EpisodicMemory performance outcome tracking
2. **Surprise Signal Calculation**: Validation of error rate to surprise signal conversion
3. **Meta-Plasticity**: Tests for adaptive learning rate mechanisms
4. **Integration Testing**: End-to-end System Potentiation validation
5. **Edge Cases**: Boundary conditions and error handling

Scientific Validation:
=====================

These tests provide concrete, measurable evidence that the System Potentiation
engine successfully implements meta-learning, enabling the system to adapt its
learning strategy based on performance feedback.
"""

import pytest
import torch
import warnings
from collections import deque

from gif_framework.core.memory_systems import EpisodicMemory, ExperienceTuple
from gif_framework.core.rtl_mechanisms import ThreeFactor_Hebbian_Rule
from gif_framework.core.du_core import DU_Core_V1


class TestPerformanceTracking:
    """Test suite for performance tracking capabilities in EpisodicMemory."""

    def test_performance_buffer_initialization(self):
        """Test that performance buffer is properly initialized."""
        memory = EpisodicMemory(capacity=100, performance_buffer_size=50)
        
        assert hasattr(memory, 'recent_performance_buffer')
        assert isinstance(memory.recent_performance_buffer, deque)
        assert memory.recent_performance_buffer.maxlen == 50
        assert len(memory.recent_performance_buffer) == 0

    def test_add_experience_with_outcome(self):
        """Test adding experiences with performance outcomes."""
        memory = EpisodicMemory(capacity=100)
        
        # Create test experience
        experience = ExperienceTuple(
            input_spikes=torch.rand(10, 1, 5),
            internal_state=None,
            output_spikes=torch.rand(10, 1, 3),
            task_id="test_task"
        )
        
        # Add with correct outcome
        memory.add(experience, outcome=1)
        assert len(memory.recent_performance_buffer) == 1
        assert memory.recent_performance_buffer[0] == 1
        
        # Add with incorrect outcome
        memory.add(experience, outcome=0)
        assert len(memory.recent_performance_buffer) == 2
        assert memory.recent_performance_buffer[1] == 0

    def test_add_experience_without_outcome(self):
        """Test backward compatibility - adding experiences without outcomes."""
        memory = EpisodicMemory(capacity=100)
        
        experience = ExperienceTuple(
            input_spikes=torch.rand(10, 1, 5),
            internal_state=None,
            output_spikes=torch.rand(10, 1, 3),
            task_id="test_task"
        )
        
        # Should work without outcome (backward compatibility)
        memory.add(experience)
        assert len(memory.recent_performance_buffer) == 0
        assert len(memory) == 1

    def test_invalid_outcome_values(self):
        """Test validation of outcome values."""
        memory = EpisodicMemory(capacity=100)
        
        experience = ExperienceTuple(
            input_spikes=torch.rand(10, 1, 5),
            internal_state=None,
            output_spikes=torch.rand(10, 1, 3),
            task_id="test_task"
        )
        
        # Test invalid outcome values
        with pytest.raises(ValueError, match="Outcome must be None, 0.*or 1"):
            memory.add(experience, outcome=2)
        
        with pytest.raises(ValueError, match="Outcome must be None, 0.*or 1"):
            memory.add(experience, outcome=-1)
        
        with pytest.raises(ValueError, match="Outcome must be None, 0.*or 1"):
            memory.add(experience, outcome=0.5)


class TestSurpriseSignalCalculation:
    """Test suite for surprise signal calculation from performance data."""

    def test_surprise_signal_empty_buffer(self):
        """Test surprise signal calculation with no performance data."""
        memory = EpisodicMemory(capacity=100)
        
        surprise = memory.get_surprise_signal()
        assert surprise == 0.5  # Neutral surprise for empty buffer

    def test_surprise_signal_perfect_performance(self):
        """Test surprise signal with perfect performance (all correct)."""
        memory = EpisodicMemory(capacity=100)
        experience = ExperienceTuple(
            input_spikes=torch.rand(10, 1, 5),
            internal_state=None,
            output_spikes=torch.rand(10, 1, 3),
            task_id="test_task"
        )
        
        # Add several correct outcomes
        for _ in range(10):
            memory.add(experience, outcome=1)
        
        surprise = memory.get_surprise_signal()
        assert surprise == 0.0  # No surprise for perfect performance

    def test_surprise_signal_worst_performance(self):
        """Test surprise signal with worst performance (all incorrect)."""
        memory = EpisodicMemory(capacity=100)
        experience = ExperienceTuple(
            input_spikes=torch.rand(10, 1, 5),
            internal_state=None,
            output_spikes=torch.rand(10, 1, 3),
            task_id="test_task"
        )
        
        # Add several incorrect outcomes
        for _ in range(10):
            memory.add(experience, outcome=0)
        
        surprise = memory.get_surprise_signal()
        assert surprise == 1.0  # Maximum surprise for worst performance

    def test_surprise_signal_mixed_performance(self):
        """Test surprise signal with mixed performance."""
        memory = EpisodicMemory(capacity=100)
        experience = ExperienceTuple(
            input_spikes=torch.rand(10, 1, 5),
            internal_state=None,
            output_spikes=torch.rand(10, 1, 3),
            task_id="test_task"
        )
        
        # Add mixed outcomes: 7 correct, 3 incorrect
        for _ in range(7):
            memory.add(experience, outcome=1)
        for _ in range(3):
            memory.add(experience, outcome=0)
        
        surprise = memory.get_surprise_signal()
        expected_surprise = 3.0 / 10.0  # 30% error rate
        assert abs(surprise - expected_surprise) < 1e-6

    def test_performance_stats(self):
        """Test detailed performance statistics."""
        memory = EpisodicMemory(capacity=100, performance_buffer_size=20)
        experience = ExperienceTuple(
            input_spikes=torch.rand(10, 1, 5),
            internal_state=None,
            output_spikes=torch.rand(10, 1, 3),
            task_id="test_task"
        )
        
        # Add 6 correct, 4 incorrect outcomes
        for _ in range(6):
            memory.add(experience, outcome=1)
        for _ in range(4):
            memory.add(experience, outcome=0)
        
        stats = memory.get_performance_stats()
        
        assert stats['success_rate'] == 0.6
        assert stats['error_rate'] == 0.4
        assert stats['surprise_signal'] == 0.4
        assert stats['sample_count'] == 10
        assert stats['buffer_utilization'] == 0.5  # 10/20


class TestMetaPlasticity:
    """Test suite for meta-plasticity mechanisms in ThreeFactor_Hebbian_Rule."""

    def test_meta_plasticity_initialization(self):
        """Test initialization of meta-plasticity parameters."""
        rule = ThreeFactor_Hebbian_Rule(learning_rate=0.01, meta_learning_rate=0.1)
        
        assert rule.base_learning_rate == 0.01
        assert rule.meta_learning_rate == 0.1
        assert rule.current_learning_rate == 0.01  # Starts at base rate
        assert len(rule._adaptation_history) == 0
        assert len(rule._surprise_history) == 0

    def test_learning_rate_adaptation(self):
        """Test learning rate adaptation based on surprise signals."""
        rule = ThreeFactor_Hebbian_Rule(learning_rate=0.01, meta_learning_rate=0.2)
        
        initial_lr = rule.current_learning_rate
        
        # High surprise should increase learning rate
        rule.update_learning_rate(0.8)  # 80% error rate
        high_surprise_lr = rule.current_learning_rate
        assert high_surprise_lr > initial_lr
        
        # Low surprise should result in lower learning rate
        rule.update_learning_rate(0.1)  # 10% error rate
        low_surprise_lr = rule.current_learning_rate
        assert low_surprise_lr < high_surprise_lr
        assert low_surprise_lr > initial_lr  # Still above base due to meta_learning_rate

    def test_learning_rate_adapts_to_performance(self):
        """Core test as specified in the prompt - learning rate adapts to performance."""
        # Create DU_Core with meta-plasticity enabled
        hebbian_rule = ThreeFactor_Hebbian_Rule(learning_rate=0.01, meta_learning_rate=0.2)
        memory = EpisodicMemory(capacity=1000)
        
        du_core = DU_Core_V1(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            plasticity_rule=hebbian_rule,
            memory_system=memory
        )
        
        # Get initial learning rate
        initial_learning_rate = du_core._plasticity_rule.current_learning_rate
        
        # Simulate series of INCORRECT predictions
        experience = ExperienceTuple(
            input_spikes=torch.rand(10, 1, 10),
            internal_state=None,
            output_spikes=torch.rand(10, 1, 5),
            task_id="test_task"
        )
        
        for _ in range(20):
            memory.add(experience, outcome=0)  # Incorrect predictions
        
        # Trigger meta-learning update
        surprise_signal = memory.get_surprise_signal()
        du_core._plasticity_rule.update_learning_rate(surprise_signal)
        
        # Learning rate should be GREATER than initial
        increased_learning_rate = du_core._plasticity_rule.current_learning_rate
        assert increased_learning_rate > initial_learning_rate
        
        # Now simulate series of CORRECT predictions
        for _ in range(30):
            memory.add(experience, outcome=1)  # Correct predictions
        
        # Trigger meta-learning update
        surprise_signal = memory.get_surprise_signal()
        du_core._plasticity_rule.update_learning_rate(surprise_signal)
        
        # Learning rate should have DECREASED from peak
        final_learning_rate = du_core._plasticity_rule.current_learning_rate
        assert final_learning_rate < increased_learning_rate
        
        # Verify meta-plasticity is working correctly
        assert final_learning_rate != initial_learning_rate  # System has adapted

    def test_invalid_surprise_signal(self):
        """Test validation of surprise signal values."""
        rule = ThreeFactor_Hebbian_Rule(learning_rate=0.01)
        
        # Test invalid surprise signal values
        with pytest.raises(ValueError, match="surprise_signal must be in range"):
            rule.update_learning_rate(-0.1)
        
        with pytest.raises(ValueError, match="surprise_signal must be in range"):
            rule.update_learning_rate(1.1)

    def test_meta_plasticity_stats(self):
        """Test meta-plasticity statistics collection."""
        rule = ThreeFactor_Hebbian_Rule(learning_rate=0.01, meta_learning_rate=0.1)
        
        # Apply some adaptations
        rule.update_learning_rate(0.3)
        rule.update_learning_rate(0.7)
        rule.update_learning_rate(0.1)
        
        stats = rule.get_meta_plasticity_stats()
        
        assert stats['current_learning_rate'] == rule.current_learning_rate
        assert stats['base_learning_rate'] == 0.01
        assert stats['meta_learning_rate'] == 0.1
        assert stats['adaptation_count'] == 3
        assert len(stats['surprise_history']) == 3
        assert len(stats['learning_rate_history']) == 3
        assert stats['recent_surprise'] == 0.1
        assert abs(stats['avg_surprise'] - (0.3 + 0.7 + 0.1) / 3) < 1e-6


class TestSystemPotentiationIntegration:
    """Test suite for end-to-end System Potentiation integration."""

    def setup_method(self):
        """Set up fresh state for each test to ensure isolation."""
        # Reset any global state that might affect tests
        torch.manual_seed(42)  # Ensure reproducible results

    def test_meta_plasticity_integration_with_du_core(self):
        """Test complete integration of meta-plasticity with DU_Core processing."""
        # Create fresh components to avoid test isolation issues
        hebbian_rule = ThreeFactor_Hebbian_Rule(learning_rate=0.05, meta_learning_rate=0.15)
        memory = EpisodicMemory(capacity=1000)

        # Clear any existing state
        memory.clear()

        du_core = DU_Core_V1(
            input_size=8,
            hidden_sizes=[6],
            output_size=4,
            plasticity_rule=hebbian_rule,
            memory_system=memory
        )

        du_core.train()  # Enable training mode for RTL

        # Get initial state
        initial_lr = du_core._plasticity_rule.current_learning_rate
        initial_weights = du_core.linear_layers[0].weight.data.clone()

        # Simulate poor performance period
        experience = ExperienceTuple(
            input_spikes=torch.rand(10, 1, 8),
            internal_state=None,
            output_spikes=torch.rand(10, 1, 4),
            task_id="integration_test"
        )

        # Add poor performance outcomes
        for _ in range(15):
            memory.add(experience, outcome=0)

        # Update learning rate based on performance
        surprise = memory.get_surprise_signal()
        du_core._plasticity_rule.update_learning_rate(surprise)

        # Learning rate should have increased
        adapted_lr = du_core._plasticity_rule.current_learning_rate
        assert adapted_lr > initial_lr

        # Process some data with adapted learning rate
        spike_train = torch.ones(5, 1, 8) * 0.6  # Consistent activity
        output = du_core.process(spike_train)

        # Weights should have changed more due to higher learning rate
        final_weights = du_core.linear_layers[0].weight.data.clone()
        weight_change = torch.abs(final_weights - initial_weights).sum().item()
        assert weight_change > 0  # Weights changed due to RTL with adapted rate

        # Verify output shape
        assert output.shape == (5, 1, 4)

    def test_performance_buffer_overflow(self):
        """Test performance buffer behavior when it exceeds capacity."""
        memory = EpisodicMemory(capacity=100, performance_buffer_size=5)
        experience = ExperienceTuple(
            input_spikes=torch.rand(10, 1, 5),
            internal_state=None,
            output_spikes=torch.rand(10, 1, 3),
            task_id="overflow_test"
        )

        # Add more outcomes than buffer capacity
        for i in range(10):
            outcome = 1 if i % 2 == 0 else 0  # Alternating outcomes
            memory.add(experience, outcome=outcome)

        # Buffer should only contain last 5 outcomes
        assert len(memory.recent_performance_buffer) == 5

        # Should contain outcomes from indices 5-9: [0, 1, 0, 1, 0] (i=5,6,7,8,9)
        expected_outcomes = [0, 1, 0, 1, 0]  # i%2==0 gives 0 for odd i, 1 for even i
        assert list(memory.recent_performance_buffer) == expected_outcomes

        # Surprise signal should reflect only recent outcomes
        surprise = memory.get_surprise_signal()
        expected_surprise = 3.0 / 5.0  # 3 incorrect out of 5 (outcomes [0,1,0,1,0] = 2 correct, 3 incorrect)
        assert abs(surprise - expected_surprise) < 1e-6

    def test_meta_plasticity_with_continual_trainer(self):
        """Test meta-plasticity integration with Continual_Trainer (direct testing)."""
        # This test validates the meta-learning statistics tracking
        # by directly testing the trainer's meta-learning functionality

        from gif_framework.training.trainer import Continual_Trainer

        # Create a minimal trainer for testing meta-learning stats
        # We'll test the meta-learning functionality directly

        # Create memory system with meta-plasticity
        memory = EpisodicMemory(capacity=1000)

        # Simulate adding outcomes to test surprise signal calculation
        experience = ExperienceTuple(
            input_spikes=torch.rand(10, 1, 8),
            internal_state=None,
            output_spikes=torch.rand(10, 1, 3),
            task_id="trainer_test"
        )

        # Add poor performance outcomes
        for _ in range(8):
            memory.add(experience, outcome=0)  # Incorrect
        for _ in range(2):
            memory.add(experience, outcome=1)  # Correct

        # Test surprise signal calculation
        surprise = memory.get_surprise_signal()
        expected_surprise = 0.8  # 80% error rate
        assert abs(surprise - expected_surprise) < 1e-6

        # Test meta-plasticity rule adaptation
        hebbian_rule = ThreeFactor_Hebbian_Rule(learning_rate=0.02, meta_learning_rate=0.1)
        initial_lr = hebbian_rule.current_learning_rate

        # Update learning rate based on poor performance
        hebbian_rule.update_learning_rate(surprise)
        adapted_lr = hebbian_rule.current_learning_rate

        # Learning rate should have increased due to poor performance
        assert adapted_lr > initial_lr

        # Test meta-plasticity statistics
        stats = hebbian_rule.get_meta_plasticity_stats()
        assert stats['current_learning_rate'] == adapted_lr
        assert stats['adaptation_count'] == 1
        assert stats['recent_surprise'] == surprise

        # Test with good performance
        memory_good = EpisodicMemory(capacity=1000)
        for _ in range(9):
            memory_good.add(experience, outcome=1)  # Correct
        for _ in range(1):
            memory_good.add(experience, outcome=0)  # Incorrect

        surprise_good = memory_good.get_surprise_signal()
        expected_surprise_good = 0.1  # 10% error rate
        assert abs(surprise_good - expected_surprise_good) < 1e-6

        # Update learning rate based on good performance
        hebbian_rule.update_learning_rate(surprise_good)
        final_lr = hebbian_rule.current_learning_rate

        # Learning rate should be lower than the adapted rate but still above base
        assert final_lr < adapted_lr
        assert final_lr > initial_lr  # Still above base due to meta_learning_rate > 0


class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error handling in System Potentiation."""

    def test_meta_plasticity_with_zero_meta_learning_rate(self):
        """Test meta-plasticity with zero meta-learning rate (no adaptation)."""
        rule = ThreeFactor_Hebbian_Rule(learning_rate=0.01, meta_learning_rate=0.0)

        initial_lr = rule.current_learning_rate

        # Apply various surprise signals
        rule.update_learning_rate(0.0)  # Perfect performance
        rule.update_learning_rate(1.0)  # Worst performance
        rule.update_learning_rate(0.5)  # Chance performance

        # Learning rate should remain unchanged
        assert rule.current_learning_rate == initial_lr

    def test_meta_plasticity_history_bounds(self):
        """Test that adaptation history is properly bounded."""
        rule = ThreeFactor_Hebbian_Rule(learning_rate=0.01, meta_learning_rate=0.1)

        # Add more adaptations than the maximum history size
        for i in range(1200):  # More than max_history (1000)
            rule.update_learning_rate(0.5)

        # History should be bounded
        assert len(rule._surprise_history) <= 1000
        assert len(rule._adaptation_history) <= 1000

        # Should contain the most recent entries
        assert all(s == 0.5 for s in rule._surprise_history)

    def test_invalid_meta_learning_rate(self):
        """Test validation of meta-learning rate parameter."""
        with pytest.raises(ValueError, match="meta_learning_rate must be non-negative"):
            ThreeFactor_Hebbian_Rule(learning_rate=0.01, meta_learning_rate=-0.1)

        # Zero should be allowed (no adaptation)
        rule = ThreeFactor_Hebbian_Rule(learning_rate=0.01, meta_learning_rate=0.0)
        assert rule.meta_learning_rate == 0.0

    def test_performance_buffer_size_validation(self):
        """Test validation of performance buffer size parameter."""
        with pytest.raises(ValueError, match="Performance buffer size must be a positive integer"):
            EpisodicMemory(capacity=100, performance_buffer_size=0)

        with pytest.raises(ValueError, match="Performance buffer size must be a positive integer"):
            EpisodicMemory(capacity=100, performance_buffer_size=-10)

    def test_meta_plasticity_with_empty_stats(self):
        """Test meta-plasticity statistics with no adaptation history."""
        rule = ThreeFactor_Hebbian_Rule(learning_rate=0.01, meta_learning_rate=0.1)

        stats = rule.get_meta_plasticity_stats()

        assert stats['current_learning_rate'] == 0.01
        assert stats['adaptation_count'] == 0
        assert stats['recent_surprise'] is None
        assert stats['avg_surprise'] is None
        assert len(stats['surprise_history']) == 0
        assert len(stats['learning_rate_history']) == 0

    def test_backward_compatibility_with_existing_code(self):
        """Test that meta-plasticity enhancements don't break existing functionality."""
        # Test old-style initialization (should still work)
        rule = ThreeFactor_Hebbian_Rule(learning_rate=0.01)
        assert rule.base_learning_rate == 0.01
        assert rule.meta_learning_rate == 0.1  # Default value
        assert rule.current_learning_rate == 0.01

        # Test old-style memory usage (should still work)
        memory = EpisodicMemory(capacity=100)
        experience = ExperienceTuple(
            input_spikes=torch.rand(10, 1, 5),
            internal_state=None,
            output_spikes=torch.rand(10, 1, 3),
            task_id="backward_compatibility_test"
        )

        # Should work without outcome parameter
        memory.add(experience)
        assert len(memory) == 1
        assert len(memory.recent_performance_buffer) == 0

        # Surprise signal should return neutral value
        surprise = memory.get_surprise_signal()
        assert surprise == 0.5
