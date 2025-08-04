"""
Test Suite for Episodic Memory Systems
======================================

This module contains comprehensive tests for the episodic memory system that serves as the
foundation for continual learning within the GIF framework. It validates the implementation
of ExperienceTuple and EpisodicMemory classes and their integration with the DU Core.

Test Coverage:
- ExperienceTuple data structure validation and immutability
- EpisodicMemory capacity management and FIFO behavior
- Random sampling functionality for continual learning algorithms
- Integration with DU_Core_V1 dependency injection
- Error handling and edge cases
- Performance characteristics and memory efficiency

Author: GIF Development Team
Phase: 3.2 - Episodic Memory System Testing
"""

import pytest
import torch
import random
from collections import deque
from typing import Any, List

# Import the memory systems to test
from gif_framework.core.memory_systems import (
    ExperienceTuple,
    EpisodicMemory
)

# Import DU Core for integration testing
from gif_framework.core.du_core import DU_Core_V1


class TestExperienceTuple:
    """Test suite for the ExperienceTuple data structure."""

    def test_experience_tuple_creation(self):
        """Test that ExperienceTuple can be created with valid data."""
        input_spikes = torch.randn(50, 1, 100)
        output_spikes = torch.randn(50, 1, 10)
        
        experience = ExperienceTuple(
            input_spikes=input_spikes,
            internal_state=None,
            output_spikes=output_spikes,
            task_id="test_task"
        )
        
        assert experience.input_spikes is input_spikes
        assert experience.internal_state is None
        assert experience.output_spikes is output_spikes
        assert experience.task_id == "test_task"

    def test_experience_tuple_immutability(self):
        """Test that ExperienceTuple is immutable (NamedTuple behavior)."""
        experience = ExperienceTuple(
            input_spikes=torch.randn(10, 1, 5),
            internal_state=None,
            output_spikes=torch.randn(10, 1, 3),
            task_id="immutable_test"
        )
        
        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            experience.task_id = "modified_task"
        
        with pytest.raises(AttributeError):
            experience.input_spikes = torch.randn(5, 1, 5)

    def test_experience_tuple_field_access(self):
        """Test that ExperienceTuple fields can be accessed by name."""
        input_data = torch.ones(20, 2, 50)
        output_data = torch.zeros(20, 2, 8)
        internal_data = {"hidden_state": torch.randn(2, 32)}
        
        experience = ExperienceTuple(
            input_spikes=input_data,
            internal_state=internal_data,
            output_spikes=output_data,
            task_id="field_access_test"
        )
        
        # Test named field access
        assert torch.equal(experience.input_spikes, input_data)
        assert experience.internal_state == internal_data
        assert torch.equal(experience.output_spikes, output_data)
        assert experience.task_id == "field_access_test"
        
        # Test that it's still a tuple (can be unpacked) - now with 5 fields including conceptual_state
        inp, state, out, task, conceptual = experience
        assert torch.equal(inp, input_data)
        assert state == internal_data
        assert torch.equal(out, output_data)
        assert task == "field_access_test"
        assert conceptual is None  # Default value for conceptual_state

    def test_experience_tuple_with_different_internal_states(self):
        """Test ExperienceTuple with various internal state types."""
        base_input = torch.randn(10, 1, 20)
        base_output = torch.randn(10, 1, 5)
        
        # Test with None
        exp_none = ExperienceTuple(base_input, None, base_output, "none_state")
        assert exp_none.internal_state is None
        
        # Test with dictionary
        state_dict = {"membrane_potential": torch.randn(1, 20), "spike_count": 5}
        exp_dict = ExperienceTuple(base_input, state_dict, base_output, "dict_state")
        assert exp_dict.internal_state == state_dict
        
        # Test with tensor
        state_tensor = torch.randn(1, 64)
        exp_tensor = ExperienceTuple(base_input, state_tensor, base_output, "tensor_state")
        assert torch.equal(exp_tensor.internal_state, state_tensor)
        
        # Test with custom object
        class CustomState:
            def __init__(self, value):
                self.value = value
        
        custom_state = CustomState(42)
        exp_custom = ExperienceTuple(base_input, custom_state, base_output, "custom_state")
        assert exp_custom.internal_state.value == 42


class TestEpisodicMemory:
    """Test suite for the EpisodicMemory class."""

    def test_memory_initialization_valid_capacity(self):
        """Test EpisodicMemory initialization with valid capacity values."""
        # Test various valid capacities
        for capacity in [1, 10, 100, 1000, 10000]:
            memory = EpisodicMemory(capacity=capacity)
            assert memory.capacity == capacity
            assert len(memory) == 0
            assert memory.is_empty()
            assert not memory.is_full()

    def test_memory_initialization_invalid_capacity(self):
        """Test EpisodicMemory initialization with invalid capacity values."""
        # Test negative capacity
        with pytest.raises(ValueError, match="Capacity must be a positive integer"):
            EpisodicMemory(capacity=-1)
        
        # Test zero capacity
        with pytest.raises(ValueError, match="Capacity must be a positive integer"):
            EpisodicMemory(capacity=0)
        
        # Test non-integer capacity
        with pytest.raises(ValueError, match="Capacity must be a positive integer"):
            EpisodicMemory(capacity=10.5)
        
        # Test string capacity
        with pytest.raises(ValueError, match="Capacity must be a positive integer"):
            EpisodicMemory(capacity="100")

    def test_memory_add_valid_experience(self):
        """Test adding valid experiences to memory."""
        memory = EpisodicMemory(capacity=5)
        
        # Create and add a valid experience
        experience = ExperienceTuple(
            input_spikes=torch.randn(30, 1, 50),
            internal_state=None,
            output_spikes=torch.randn(30, 1, 10),
            task_id="add_test"
        )
        
        memory.add(experience)
        
        assert len(memory) == 1
        assert not memory.is_empty()
        assert not memory.is_full()

    def test_memory_add_invalid_experience_type(self):
        """Test adding invalid experience types to memory."""
        memory = EpisodicMemory(capacity=5)
        
        # Test with non-ExperienceTuple
        with pytest.raises(TypeError, match="Expected ExperienceTuple"):
            memory.add("not_an_experience")
        
        with pytest.raises(TypeError, match="Expected ExperienceTuple"):
            memory.add({"input": torch.randn(10, 1, 5)})
        
        with pytest.raises(TypeError, match="Expected ExperienceTuple"):
            memory.add([torch.randn(10, 1, 5), None, torch.randn(10, 1, 3), "task"])

    def test_memory_add_invalid_experience_fields(self):
        """Test adding experiences with invalid field types."""
        memory = EpisodicMemory(capacity=5)
        
        # Test with non-tensor input_spikes
        with pytest.raises(ValueError, match="input_spikes must be a torch.Tensor"):
            invalid_exp = ExperienceTuple(
                input_spikes=[1, 2, 3],  # List instead of tensor
                internal_state=None,
                output_spikes=torch.randn(10, 1, 3),
                task_id="invalid_input"
            )
            memory.add(invalid_exp)
        
        # Test with non-tensor output_spikes
        with pytest.raises(ValueError, match="output_spikes must be a torch.Tensor"):
            invalid_exp = ExperienceTuple(
                input_spikes=torch.randn(10, 1, 5),
                internal_state=None,
                output_spikes="not_a_tensor",  # String instead of tensor
                task_id="invalid_output"
            )
            memory.add(invalid_exp)
        
        # Test with non-string task_id
        with pytest.raises(ValueError, match="task_id must be a string"):
            invalid_exp = ExperienceTuple(
                input_spikes=torch.randn(10, 1, 5),
                internal_state=None,
                output_spikes=torch.randn(10, 1, 3),
                task_id=123  # Integer instead of string
            )
            memory.add(invalid_exp)

    def test_memory_fifo_behavior(self):
        """Test that memory follows FIFO (First-In-First-Out) behavior when at capacity."""
        capacity = 3
        memory = EpisodicMemory(capacity=capacity)
        
        # Create experiences with identifiable task_ids
        experiences = []
        for i in range(5):  # More than capacity
            exp = ExperienceTuple(
                input_spikes=torch.randn(10, 1, 5),
                internal_state=None,
                output_spikes=torch.randn(10, 1, 3),
                task_id=f"task_{i}"
            )
            experiences.append(exp)
            memory.add(exp)
        
        # Memory should be at capacity
        assert len(memory) == capacity
        assert memory.is_full()
        
        # Should contain only the last 3 experiences (FIFO)
        sampled = memory.sample(capacity)
        task_ids = {exp.task_id for exp in sampled}
        expected_task_ids = {"task_2", "task_3", "task_4"}
        assert task_ids == expected_task_ids

    def test_memory_sample_valid_batch_sizes(self):
        """Test sampling with various valid batch sizes."""
        memory = EpisodicMemory(capacity=10)
        
        # Add some experiences
        for i in range(5):
            exp = ExperienceTuple(
                input_spikes=torch.randn(10, 1, 8),
                internal_state=None,
                output_spikes=torch.randn(10, 1, 4),
                task_id=f"sample_task_{i}"
            )
            memory.add(exp)
        
        # Test sampling smaller than available
        batch = memory.sample(batch_size=3)
        assert len(batch) == 3
        assert all(isinstance(exp, ExperienceTuple) for exp in batch)
        
        # Test sampling equal to available
        batch = memory.sample(batch_size=5)
        assert len(batch) == 5
        
        # Test sampling larger than available (should return all)
        batch = memory.sample(batch_size=10)
        assert len(batch) == 5  # Only 5 available

    def test_memory_sample_invalid_batch_size(self):
        """Test sampling with invalid batch sizes."""
        memory = EpisodicMemory(capacity=5)
        
        # Add one experience
        exp = ExperienceTuple(
            input_spikes=torch.randn(10, 1, 5),
            internal_state=None,
            output_spikes=torch.randn(10, 1, 3),
            task_id="sample_test"
        )
        memory.add(exp)
        
        # Test negative batch size
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            memory.sample(batch_size=-1)
        
        # Test zero batch size
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            memory.sample(batch_size=0)
        
        # Test non-integer batch size
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            memory.sample(batch_size=2.5)

    def test_memory_sample_empty_memory(self):
        """Test sampling from empty memory raises appropriate error."""
        memory = EpisodicMemory(capacity=5)
        
        with pytest.raises(RuntimeError, match="Cannot sample from empty memory"):
            memory.sample(batch_size=1)

    def test_memory_sample_uniqueness(self):
        """Test that sampled experiences are unique (no duplicates)."""
        memory = EpisodicMemory(capacity=10)
        
        # Add experiences
        for i in range(8):
            exp = ExperienceTuple(
                input_spikes=torch.randn(10, 1, 5),
                internal_state=None,
                output_spikes=torch.randn(10, 1, 3),
                task_id=f"unique_task_{i}"
            )
            memory.add(exp)
        
        # Sample and check uniqueness
        batch = memory.sample(batch_size=5)
        task_ids = [exp.task_id for exp in batch]
        assert len(task_ids) == len(set(task_ids))  # No duplicates

    def test_memory_sample_randomness(self):
        """Test that sampling is actually random."""
        memory = EpisodicMemory(capacity=20)
        
        # Add many experiences
        for i in range(15):
            exp = ExperienceTuple(
                input_spikes=torch.randn(10, 1, 5),
                internal_state=None,
                output_spikes=torch.randn(10, 1, 3),
                task_id=f"random_task_{i}"
            )
            memory.add(exp)
        
        # Sample multiple times and check for variation
        samples = []
        for _ in range(10):
            batch = memory.sample(batch_size=5)
            task_ids = sorted([exp.task_id for exp in batch])
            samples.append(tuple(task_ids))
        
        # Should have some variation in samples (not all identical)
        unique_samples = set(samples)
        assert len(unique_samples) > 1  # At least some variation

    def test_memory_utility_methods(self):
        """Test utility methods like __len__, is_empty, is_full."""
        memory = EpisodicMemory(capacity=3)

        # Test empty memory
        assert len(memory) == 0
        assert memory.is_empty()
        assert not memory.is_full()

        # Add one experience
        exp1 = ExperienceTuple(
            input_spikes=torch.randn(10, 1, 5),
            internal_state=None,
            output_spikes=torch.randn(10, 1, 3),
            task_id="util_test_1"
        )
        memory.add(exp1)

        assert len(memory) == 1
        assert not memory.is_empty()
        assert not memory.is_full()

        # Add two more experiences (reach capacity)
        for i in range(2, 4):
            exp = ExperienceTuple(
                input_spikes=torch.randn(10, 1, 5),
                internal_state=None,
                output_spikes=torch.randn(10, 1, 3),
                task_id=f"util_test_{i}"
            )
            memory.add(exp)

        assert len(memory) == 3
        assert not memory.is_empty()
        assert memory.is_full()

    def test_memory_get_task_count(self):
        """Test task-specific experience counting."""
        memory = EpisodicMemory(capacity=10)

        # Add experiences for different tasks
        task_counts = {"task_A": 3, "task_B": 2, "task_C": 1}

        for task_id, count in task_counts.items():
            for i in range(count):
                exp = ExperienceTuple(
                    input_spikes=torch.randn(10, 1, 5),
                    internal_state=None,
                    output_spikes=torch.randn(10, 1, 3),
                    task_id=task_id
                )
                memory.add(exp)

        # Verify task counts
        for task_id, expected_count in task_counts.items():
            assert memory.get_task_count(task_id) == expected_count

        # Test non-existent task
        assert memory.get_task_count("non_existent_task") == 0

        # Test invalid task_id type
        with pytest.raises(ValueError, match="task_id must be a string"):
            memory.get_task_count(123)

    def test_memory_clear(self):
        """Test memory clearing functionality."""
        memory = EpisodicMemory(capacity=5)

        # Add some experiences
        for i in range(3):
            exp = ExperienceTuple(
                input_spikes=torch.randn(10, 1, 5),
                internal_state=None,
                output_spikes=torch.randn(10, 1, 3),
                task_id=f"clear_test_{i}"
            )
            memory.add(exp)

        assert len(memory) == 3
        assert not memory.is_empty()

        # Clear memory
        memory.clear()

        assert len(memory) == 0
        assert memory.is_empty()
        assert not memory.is_full()

        # Should be able to add new experiences after clearing
        new_exp = ExperienceTuple(
            input_spikes=torch.randn(10, 1, 5),
            internal_state=None,
            output_spikes=torch.randn(10, 1, 3),
            task_id="after_clear"
        )
        memory.add(new_exp)
        assert len(memory) == 1

    def test_memory_get_memory_info(self):
        """Test comprehensive memory information retrieval."""
        memory = EpisodicMemory(capacity=5)

        # Add experiences for different tasks
        tasks = ["task_A", "task_A", "task_B", "task_C"]
        for i, task_id in enumerate(tasks):
            exp = ExperienceTuple(
                input_spikes=torch.randn(10, 1, 5),
                internal_state=None,
                output_spikes=torch.randn(10, 1, 3),
                task_id=task_id
            )
            memory.add(exp)

        info = memory.get_memory_info()

        assert info["current_size"] == 4
        assert info["capacity"] == 5
        assert info["utilization"] == 0.8  # 4/5
        assert info["is_empty"] == False
        assert info["is_full"] == False

        expected_distribution = {"task_A": 2, "task_B": 1, "task_C": 1}
        assert info["task_distribution"] == expected_distribution


class TestDUCoreIntegration:
    """Test suite for episodic memory integration with DU_Core_V1."""

    def test_du_core_memory_system_injection(self):
        """Test that DU_Core_V1 can accept episodic memory via dependency injection."""
        memory = EpisodicMemory(capacity=1000)

        du_core = DU_Core_V1(
            input_size=10,
            hidden_sizes=[5],
            output_size=3,
            memory_system=memory
        )

        assert du_core._memory_system is memory

    def test_du_core_invalid_memory_system_type(self):
        """Test that DU_Core_V1 rejects invalid memory system types."""

        class InvalidMemory:
            def add(self, experience):
                pass
            def sample(self, batch_size):
                return []

        with pytest.raises(TypeError, match="memory_system must be an EpisodicMemory instance"):
            DU_Core_V1(
                input_size=10,
                hidden_sizes=[5],
                output_size=3,
                memory_system=InvalidMemory()  # Doesn't inherit from EpisodicMemory
            )

    def test_du_core_memory_system_integration_workflow(self):
        """Test complete workflow of DU Core with episodic memory."""
        memory = EpisodicMemory(capacity=100)

        du_core = DU_Core_V1(
            input_size=20,
            hidden_sizes=[10],
            output_size=5,
            memory_system=memory
        )

        # Simulate processing and memory storage
        input_spikes = (torch.rand(30, 1, 20) > 0.8).float()  # Sparse spikes as float
        output_spikes = du_core.process(input_spikes)

        # Create and store experience
        experience = ExperienceTuple(
            input_spikes=input_spikes,
            internal_state=None,
            output_spikes=output_spikes,
            task_id="integration_test"
        )
        memory.add(experience)

        # Verify memory contains the experiences
        # Note: DU_Core now automatically stores experiences during processing,
        # so we expect 2 experiences: one from du_core.process() and one from manual add()
        assert len(memory) == 2
        assert memory.get_task_count("integration_test") == 1
        assert memory.get_task_count("du_core_processing") == 1

        # Sample experiences back
        sampled = memory.sample(batch_size=2)
        assert len(sampled) == 2

        # Check that both task IDs are present
        task_ids = {exp.task_id for exp in sampled}
        assert "integration_test" in task_ids
        assert "du_core_processing" in task_ids
        assert torch.equal(sampled[0].input_spikes, input_spikes)
        assert torch.equal(sampled[0].output_spikes, output_spikes)

    def test_du_core_without_memory_system(self):
        """Test that DU_Core_V1 works normally without memory system."""
        du_core = DU_Core_V1(
            input_size=15,
            hidden_sizes=[8],
            output_size=4
            # No memory_system provided
        )

        assert du_core._memory_system is None

        # Should still process normally
        input_spikes = (torch.rand(25, 1, 15) > 0.9).float()
        output_spikes = du_core.process(input_spikes)

        assert output_spikes.shape == (25, 1, 4)

    def test_memory_system_with_realistic_spike_data(self):
        """Test memory system with realistic spike train data."""
        memory = EpisodicMemory(capacity=50)

        # Create realistic sparse spike data
        time_steps, batch_size, input_size, output_size = 100, 2, 50, 10

        # Generate sparse spikes (typical 1-5% firing rate)
        input_spikes = torch.rand(time_steps, batch_size, input_size) < 0.02
        output_spikes = torch.rand(time_steps, batch_size, output_size) < 0.03

        # Convert to float for processing
        input_spikes = input_spikes.float()
        output_spikes = output_spikes.float()

        # Store multiple realistic experiences
        for i in range(10):
            experience = ExperienceTuple(
                input_spikes=input_spikes,
                internal_state={"step": i, "metadata": f"realistic_data_{i}"},
                output_spikes=output_spikes,
                task_id=f"realistic_task_{i % 3}"  # 3 different tasks
            )
            memory.add(experience)

        assert len(memory) == 10

        # Test sampling with realistic data
        batch = memory.sample(batch_size=5)
        assert len(batch) == 5

        for exp in batch:
            assert exp.input_spikes.shape == (time_steps, batch_size, input_size)
            assert exp.output_spikes.shape == (time_steps, batch_size, output_size)
            assert exp.task_id.startswith("realistic_task_")
            assert isinstance(exp.internal_state, dict)

    def test_memory_performance_characteristics(self):
        """Test memory system performance with larger datasets."""
        import time

        memory = EpisodicMemory(capacity=1000)

        # Test insertion performance
        start_time = time.time()
        for i in range(500):
            exp = ExperienceTuple(
                input_spikes=torch.randn(50, 1, 100),
                internal_state=None,
                output_spikes=torch.randn(50, 1, 20),
                task_id=f"perf_task_{i % 10}"
            )
            memory.add(exp)
        insertion_time = time.time() - start_time

        # Should be fast (less than 1 second for 500 insertions)
        assert insertion_time < 1.0
        assert len(memory) == 500

        # Test sampling performance
        start_time = time.time()
        for _ in range(100):
            batch = memory.sample(batch_size=32)
            assert len(batch) == 32
        sampling_time = time.time() - start_time

        # Should be fast (less than 1 second for 100 samples)
        assert sampling_time < 1.0


# Additional tests specified in Prompt 3.5
def test_memory_capacity_and_fifo():
    """Tests that the memory respects its capacity via FIFO."""
    memory = EpisodicMemory(capacity=3)
    exp1 = ExperienceTuple(
        input_spikes=torch.randn(10, 1, 5),
        internal_state=None,
        output_spikes=torch.randn(10, 1, 3),
        task_id="task_A"
    )
    exp2 = ExperienceTuple(
        input_spikes=torch.randn(10, 1, 5),
        internal_state=None,
        output_spikes=torch.randn(10, 1, 3),
        task_id="task_A"
    )
    exp3 = ExperienceTuple(
        input_spikes=torch.randn(10, 1, 5),
        internal_state=None,
        output_spikes=torch.randn(10, 1, 3),
        task_id="task_B"
    )
    exp4 = ExperienceTuple(
        input_spikes=torch.randn(10, 1, 5),
        internal_state=None,
        output_spikes=torch.randn(10, 1, 3),
        task_id="task_C"
    )

    memory.add(exp1)
    memory.add(exp2)
    memory.add(exp3)
    assert len(memory._experiences) == 3

    # Adding the 4th experience should push out the 1st one
    memory.add(exp4)
    assert len(memory._experiences) == 3

    # Check that the oldest experience (exp1) is gone and exp4 is present
    # Use task_id to verify FIFO behavior since tensor comparison is problematic
    task_ids = [exp.task_id for exp in memory._experiences]
    assert "task_A" in task_ids  # exp2 should still be there
    assert "task_B" in task_ids  # exp3 should still be there
    assert "task_C" in task_ids  # exp4 should be there
    assert task_ids.count("task_A") == 1  # Only one task_A experience (exp2, not exp1)
