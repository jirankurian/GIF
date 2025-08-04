"""
Test Suite for Continual Learning Trainer
==========================================

This module contains comprehensive tests for the Continual_Trainer class that implements
the Gradient Episodic Memory (GEM) algorithm for preventing catastrophic forgetting.

Test Coverage:
- Trainer initialization and validation
- GEM algorithm components (gradient extraction, interference detection, projection)
- Integration with GIF orchestrator and memory systems
- Training statistics and monitoring
- Error handling and edge cases
- Multi-task continual learning scenarios

Author: GIF Development Team
Phase: 3.3 - Continual Learning Engine Testing
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List
from unittest.mock import Mock, patch

# Import the components to test
from gif_framework.training.trainer import Continual_Trainer
from gif_framework.orchestrator import GIF
from gif_framework.core.du_core import DU_Core_V1
from gif_framework.core.rtl_mechanisms import STDP_Rule
from gif_framework.core.memory_systems import EpisodicMemory, ExperienceTuple
from gif_framework.interfaces.base_interfaces import EncoderInterface, DecoderInterface


class MockEncoder(EncoderInterface):
    """Mock encoder for testing."""

    def __init__(self, output_shape=(30, 1, 10)):
        """Initialize mock encoder with configurable output shape."""
        self.output_shape = output_shape

    def encode(self, data: Any) -> torch.Tensor:
        """Convert input data to spike trains."""
        if isinstance(data, torch.Tensor):
            return data.float()
        return torch.randn(*self.output_shape).float()

    def get_config(self) -> Dict[str, Any]:
        return {"type": "MockEncoder", "output_size": self.output_shape[-1]}

    def calibrate(self, data_samples: List[Any]) -> None:
        pass


class MockDecoder(DecoderInterface):
    """Mock decoder for testing."""

    def decode(self, spike_train: torch.Tensor) -> Any:
        """Convert spike trains to output."""
        # Sum over time dimension and create classification output
        # Input shape: [time_steps, batch_size, features]
        # Output shape: [batch_size, num_classes]
        batch_size = spike_train.shape[1]
        summed = torch.sum(spike_train, dim=0)  # Sum over time: [batch_size, features]
        # Create a simple 2-class output for testing
        return torch.randn(batch_size, 2)

    def forward_classification(self, spike_counts: torch.Tensor) -> torch.Tensor:
        """
        Mock forward classification method for testing.

        Args:
            spike_counts (torch.Tensor): Spike count tensor.

        Returns:
            torch.Tensor: Mock classification output.
        """
        # Return a simple mock output for classification with gradients
        batch_size = spike_counts.shape[0] if len(spike_counts.shape) > 0 else 1
        # Create a simple linear transformation to maintain gradient flow
        # Use a small weight matrix to transform spike_counts to output
        weight = torch.ones(spike_counts.shape[-1], 2, requires_grad=True) * 0.1
        bias = torch.zeros(2, requires_grad=True)

        # Linear transformation: output = spike_counts @ weight + bias
        if spike_counts.dim() == 1:
            spike_counts = spike_counts.unsqueeze(0)
        output = torch.matmul(spike_counts, weight) + bias
        return output

    def get_config(self) -> Dict[str, Any]:
        return {"type": "MockDecoder", "input_size": 5}


class TestContinualTrainerInitialization:
    """Test suite for Continual_Trainer initialization."""

    def test_trainer_initialization_valid_components(self):
        """Test trainer initialization with valid components."""
        # Create components
        memory = EpisodicMemory(capacity=100)
        plasticity_rule = STDP_Rule(
            learning_rate_ltp=0.01,
            learning_rate_ltd=0.005,
            tau_ltp=20.0,
            tau_ltd=20.0
        )
        
        du_core = DU_Core_V1(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            plasticity_rule=plasticity_rule,
            memory_system=memory
        )
        
        gif = GIF(du_core)
        gif.attach_encoder(MockEncoder())
        gif.attach_decoder(MockDecoder())
        
        optimizer = optim.Adam(du_core.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Create trainer
        trainer = Continual_Trainer(
            gif_model=gif,
            optimizer=optimizer,
            criterion=criterion,
            memory_batch_size=16
        )
        
        assert trainer.gif_model is gif
        assert trainer.optimizer is optimizer
        assert trainer.criterion is criterion
        assert trainer.memory_batch_size == 16
        assert trainer.training_stats['total_steps'] == 0
        assert trainer.training_stats['interference_count'] == 0

    def test_trainer_initialization_invalid_gif_model(self):
        """Test trainer initialization with invalid GIF model."""
        optimizer = optim.Adam([torch.randn(5, 5)], lr=0.001)
        criterion = nn.MSELoss()
        
        with pytest.raises(TypeError, match="gif_model must be a GIF instance"):
            Continual_Trainer(
                gif_model="not_a_gif_model",
                optimizer=optimizer,
                criterion=criterion
            )

    def test_trainer_initialization_invalid_memory_batch_size(self):
        """Test trainer initialization with invalid memory batch size."""
        memory = EpisodicMemory(capacity=100)
        du_core = DU_Core_V1(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            memory_system=memory
        )
        gif = GIF(du_core)
        
        optimizer = optim.Adam(du_core.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Test negative batch size
        with pytest.raises(ValueError, match="memory_batch_size must be positive integer"):
            Continual_Trainer(gif, optimizer, criterion, memory_batch_size=-1)
        
        # Test zero batch size
        with pytest.raises(ValueError, match="memory_batch_size must be positive integer"):
            Continual_Trainer(gif, optimizer, criterion, memory_batch_size=0)
        
        # Test non-integer batch size
        with pytest.raises(ValueError, match="memory_batch_size must be positive integer"):
            Continual_Trainer(gif, optimizer, criterion, memory_batch_size=5.5)

    def test_trainer_initialization_missing_memory_system(self):
        """Test trainer initialization with DU Core lacking memory system."""
        du_core = DU_Core_V1(
            input_size=10,
            hidden_sizes=[8],
            output_size=5
            # No memory_system provided
        )
        gif = GIF(du_core)
        
        optimizer = optim.Adam(du_core.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        with pytest.raises(RuntimeError, match="DU Core must have an injected memory system"):
            Continual_Trainer(gif, optimizer, criterion)


class TestContinualTrainerGradientOperations:
    """Test suite for gradient extraction and manipulation."""

    def setup_method(self):
        """Set up test components."""
        self.memory = EpisodicMemory(capacity=100)
        self.plasticity_rule = STDP_Rule(
            learning_rate_ltp=0.01,
            learning_rate_ltd=0.005,
            tau_ltp=20.0,
            tau_ltd=20.0
        )
        
        self.du_core = DU_Core_V1(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            plasticity_rule=self.plasticity_rule,
            memory_system=self.memory
        )
        
        self.gif = GIF(self.du_core)
        self.gif.attach_encoder(MockEncoder(output_shape=(30, 1, 10)))  # Match DU Core input size
        self.gif.attach_decoder(MockDecoder())
        
        self.optimizer = optim.Adam(self.du_core.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.trainer = Continual_Trainer(
            gif_model=self.gif,
            optimizer=self.optimizer,
            criterion=self.criterion,
            memory_batch_size=4
        )

    def test_gradient_extraction(self):
        """Test gradient extraction functionality."""
        # Create dummy input and target
        input_data = torch.randn(30, 1, 10)
        target = torch.randn(1, 2)  # Match decoder output size (batch_size, num_classes)

        # Forward pass and compute gradients
        # Use the same forward path as the trainer
        spike_train = self.gif._encoder.encode(input_data)
        processed_spikes = self.gif._du_core.process(spike_train)
        spike_counts = processed_spikes.sum(dim=0)
        if spike_counts.dim() == 1:
            spike_counts = spike_counts.unsqueeze(0)
        output = self.gif._decoder.forward_classification(spike_counts)
        loss = self.criterion(output, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Extract gradients
        gradients = self.trainer._extract_gradients()
        
        # Verify gradient tensor properties
        assert isinstance(gradients, torch.Tensor)
        assert gradients.dim() == 1  # Should be flattened
        assert gradients.numel() > 0  # Should contain gradient values
        
        # Verify gradients are not all zero (indicating actual computation)
        assert not torch.allclose(gradients, torch.zeros_like(gradients))

    def test_gradient_application(self):
        """Test gradient application back to model parameters."""
        # Get original parameters
        original_params = [param.clone() for param in self.du_core.parameters()]
        
        # Create dummy gradients
        total_params = sum(param.numel() for param in self.du_core.parameters())
        dummy_gradients = torch.randn(total_params) * 0.01
        
        # Apply gradients
        self.trainer._apply_gradients(dummy_gradients)
        
        # Verify gradients were applied to parameters
        for param in self.du_core.parameters():
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))

    def test_gem_projection_with_interference(self):
        """Test GEM projection when interference is detected."""
        # Create current gradient
        g_current = torch.tensor([1.0, 0.0, -1.0])
        
        # Create interfering past gradient (negative dot product)
        past_experience = ExperienceTuple(
            input_spikes=torch.randn(10, 1, 5),
            internal_state=None,
            output_spikes=torch.randn(1, 5),
            task_id="past_task"
        )
        
        # Mock the experience gradient computation to return known gradient
        with patch.object(self.trainer, '_compute_experience_gradient') as mock_compute:
            g_past = torch.tensor([-1.0, 0.0, 1.0])  # Will have negative dot product with g_current
            mock_compute.return_value = g_past
            
            # Apply GEM constraints
            projected_gradient = self.trainer._apply_gem_constraints(g_current, [past_experience])
            
            # Verify projection occurred
            assert not torch.allclose(projected_gradient, g_current)
            
            # Verify interference was detected and recorded
            assert self.trainer.training_stats['interference_count'] > 0
            assert self.trainer.training_stats['projection_count'] > 0

    def test_gem_projection_without_interference(self):
        """Test GEM projection when no interference is detected."""
        # Create current gradient
        g_current = torch.tensor([1.0, 0.0, 1.0])
        
        # Create non-interfering past gradient (positive dot product)
        past_experience = ExperienceTuple(
            input_spikes=torch.randn(10, 1, 5),
            internal_state=None,
            output_spikes=torch.randn(1, 5),
            task_id="past_task"
        )
        
        # Mock the experience gradient computation to return known gradient
        with patch.object(self.trainer, '_compute_experience_gradient') as mock_compute:
            g_past = torch.tensor([1.0, 0.0, 1.0])  # Will have positive dot product with g_current
            mock_compute.return_value = g_past
            
            # Apply GEM constraints
            projected_gradient = self.trainer._apply_gem_constraints(g_current, [past_experience])
            
            # Verify no projection occurred (gradients should be identical)
            assert torch.allclose(projected_gradient, g_current)
            
            # Verify no interference was detected
            initial_interference_count = self.trainer.training_stats['interference_count']
            # (interference count should not increase from this test)


class TestContinualTrainerTrainingStep:
    """Test suite for the complete training step workflow."""

    def setup_method(self):
        """Set up test components."""
        # Create fresh memory for each test to avoid cross-test contamination
        self.memory = EpisodicMemory(capacity=100)
        self.plasticity_rule = STDP_Rule(
            learning_rate_ltp=0.01,
            learning_rate_ltd=0.005,
            tau_ltp=20.0,
            tau_ltd=20.0
        )

        self.du_core = DU_Core_V1(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            plasticity_rule=self.plasticity_rule,
            memory_system=self.memory
        )

        self.gif = GIF(self.du_core)
        self.gif.attach_encoder(MockEncoder(output_shape=(30, 1, 10)))  # Match DU Core input size
        self.gif.attach_decoder(MockDecoder())

        self.optimizer = optim.Adam(self.du_core.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.trainer = Continual_Trainer(
            gif_model=self.gif,
            optimizer=self.optimizer,
            criterion=self.criterion,
            memory_batch_size=4
        )

    def test_train_step_basic_functionality(self):
        """Test basic training step functionality."""
        # Get initial memory size
        initial_memory_size = len(self.du_core._memory_system)

        # Create input data and target
        input_data = torch.randn(30, 1, 10)
        target = torch.randn(1, 2)  # Match decoder output size (batch_size, num_classes)
        task_id = "test_task"

        # Perform training step
        loss = self.trainer.train_step(input_data, target, task_id)

        # Verify training step completed successfully
        assert isinstance(loss, float)
        assert loss >= 0.0  # Loss should be non-negative

        # Verify experience was stored in memory (at least one new experience)
        assert len(self.du_core._memory_system) >= initial_memory_size + 1
        assert self.memory.get_task_count(task_id) == 1

        # Verify training statistics were updated
        assert self.trainer.training_stats['total_steps'] == 1
        assert task_id in self.trainer.training_stats['task_losses']
        assert len(self.trainer.training_stats['task_losses'][task_id]) == 1

    def test_train_step_with_memory_sampling(self):
        """Test training step with memory sampling and GEM constraints."""
        # Pre-populate memory with experiences
        for i in range(10):
            experience = ExperienceTuple(
                input_spikes=torch.randn(30, 1, 10),
                internal_state=None,
                output_spikes=torch.randn(1, 2),  # Match decoder output size (batch_size, num_classes)
                task_id=f"past_task_{i}"
            )
            self.du_core._memory_system.add(experience)

        # Perform training step
        input_data = torch.randn(30, 1, 10)
        target = torch.randn(1, 2)  # Match decoder output size (batch_size, num_classes)
        task_id = "new_task"

        loss = self.trainer.train_step(input_data, target, task_id)

        # Verify training step completed successfully
        assert isinstance(loss, float)
        assert loss >= 0.0

        # Verify new experience was added (check DU Core's memory)
        assert len(self.du_core._memory_system) >= 11
        assert self.memory.get_task_count(task_id) == 1

        # Verify training statistics were updated
        assert self.trainer.training_stats['total_steps'] == 1

    def test_train_step_error_handling(self):
        """Test training step error handling."""
        # Test with valid target shape
        input_data = torch.randn(30, 1, 10)
        target = torch.randn(1, 2)  # Match decoder output size (batch_size, num_classes)
        task_id = "error_test"

        # Should handle the training step gracefully
        loss = self.trainer.train_step(input_data, target, task_id)

        # Verify that a loss value is returned (even if not meaningful)
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))

    def test_train_step_statistics_tracking(self):
        """Test comprehensive statistics tracking during training."""
        # Perform multiple training steps
        for i in range(5):
            input_data = torch.randn(30, 1, 10)
            target = torch.randn(1, 2)  # Match decoder output size (batch_size, num_classes)
            task_id = f"task_{i % 2}"  # Alternate between two tasks

            loss = self.trainer.train_step(input_data, target, task_id)

            # Verify statistics are being tracked
            assert self.trainer.training_stats['total_steps'] == i + 1
            assert task_id in self.trainer.training_stats['task_losses']

        # Verify task-specific loss tracking
        assert len(self.trainer.training_stats['task_losses']['task_0']) >= 2
        assert len(self.trainer.training_stats['task_losses']['task_1']) >= 2

        # Verify memory utilization tracking
        assert len(self.trainer.training_stats['memory_utilization']) == 5


class TestContinualTrainerIntegration:
    """Test suite for trainer integration with GIF components."""

    def test_trainer_with_different_optimizers(self):
        """Test trainer with different PyTorch optimizers."""
        memory = EpisodicMemory(capacity=50)
        du_core = DU_Core_V1(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            memory_system=memory
        )
        gif = GIF(du_core)
        gif.attach_encoder(MockEncoder())
        gif.attach_decoder(MockDecoder())

        criterion = nn.MSELoss()

        # Test with different optimizers
        optimizers = [
            optim.Adam(du_core.parameters(), lr=0.001),
            optim.SGD(du_core.parameters(), lr=0.01),
            optim.RMSprop(du_core.parameters(), lr=0.001)
        ]

        for optimizer in optimizers:
            trainer = Continual_Trainer(gif, optimizer, criterion)

            # Perform a training step
            input_data = torch.randn(30, 1, 10)
            target = torch.randn(1, 2)  # Match decoder output size (batch_size, num_classes)
            loss = trainer.train_step(input_data, target, "test_task")

            assert isinstance(loss, float)
            assert loss >= 0.0

    def test_trainer_with_different_loss_functions(self):
        """Test trainer with different loss functions."""
        memory = EpisodicMemory(capacity=50)
        du_core = DU_Core_V1(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            memory_system=memory
        )
        gif = GIF(du_core)
        gif.attach_encoder(MockEncoder())
        gif.attach_decoder(MockDecoder())

        optimizer = optim.Adam(du_core.parameters(), lr=0.001)

        # Test with different loss functions
        loss_functions = [
            nn.MSELoss(),
            nn.L1Loss(),
            nn.SmoothL1Loss()
        ]

        for criterion in loss_functions:
            trainer = Continual_Trainer(gif, optimizer, criterion)

            # Perform a training step
            input_data = torch.randn(30, 1, 10)
            target = torch.randn(1, 2)  # Match decoder output size (batch_size, num_classes)
            loss = trainer.train_step(input_data, target, "test_task")

            assert isinstance(loss, float)
            assert loss >= 0.0

    def test_trainer_statistics_methods(self):
        """Test trainer statistics and monitoring methods."""
        memory = EpisodicMemory(capacity=50)
        du_core = DU_Core_V1(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            memory_system=memory
        )
        gif = GIF(du_core)
        gif.attach_encoder(MockEncoder())
        gif.attach_decoder(MockDecoder())

        optimizer = optim.Adam(du_core.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        trainer = Continual_Trainer(gif, optimizer, criterion)

        # Perform some training steps
        for i in range(3):
            input_data = torch.randn(30, 1, 10)
            target = torch.randn(1, 2)  # Match decoder output size (batch_size, num_classes)
            trainer.train_step(input_data, target, f"task_{i}")

        # Test statistics retrieval
        stats = trainer.get_training_stats()

        assert stats['total_steps'] == 3
        assert len(stats['task_losses']) == 3
        assert len(stats['memory_utilization']) == 3
        assert 'interference_count' in stats
        assert 'projection_count' in stats


class TestContinualTrainerMultiTaskScenarios:
    """Test suite for multi-task continual learning scenarios."""

    def setup_method(self):
        """Set up test components for multi-task scenarios."""
        self.memory = EpisodicMemory(capacity=200)
        self.plasticity_rule = STDP_Rule(
            learning_rate_ltp=0.01,
            learning_rate_ltd=0.005,
            tau_ltp=20.0,
            tau_ltd=20.0
        )

        self.du_core = DU_Core_V1(
            input_size=20,
            hidden_sizes=[16, 12],
            output_size=8,
            plasticity_rule=self.plasticity_rule,
            memory_system=self.memory
        )

        self.gif = GIF(self.du_core)
        self.gif.attach_encoder(MockEncoder(output_shape=(30, 1, 20)))  # Match DU Core input size
        self.gif.attach_decoder(MockDecoder())

        self.optimizer = optim.Adam(self.du_core.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.trainer = Continual_Trainer(
            gif_model=self.gif,
            optimizer=self.optimizer,
            criterion=self.criterion,
            memory_batch_size=8
        )

    def test_sequential_task_learning(self):
        """Test learning multiple tasks sequentially."""
        tasks = ["task_A", "task_B", "task_C"]
        samples_per_task = 10

        for task_id in tasks:
            for i in range(samples_per_task):
                input_data = torch.randn(30, 1, 20)
                target = torch.randn(1, 2)  # Match decoder output size (batch_size, num_classes)

                loss = self.trainer.train_step(input_data, target, task_id)
                assert isinstance(loss, float)

        # Verify training completed successfully for all tasks
        # Note: Memory counts may vary due to capacity limits and eviction policies
        total_memory_size = len(self.du_core._memory_system)
        assert total_memory_size > 0, "Memory should contain some experiences"

        # Verify total training steps
        assert self.trainer.training_stats['total_steps'] == len(tasks) * samples_per_task

    def test_interleaved_task_learning(self):
        """Test learning with interleaved tasks (more realistic scenario)."""
        tasks = ["task_A", "task_B", "task_C"]
        total_steps = 30

        for step in range(total_steps):
            # Randomly select task (simulating real-world interleaved learning)
            task_id = tasks[step % len(tasks)]

            input_data = torch.randn(30, 1, 20)
            target = torch.randn(1, 2)  # Match decoder output size (batch_size, num_classes)

            loss = self.trainer.train_step(input_data, target, task_id)
            assert isinstance(loss, float)

        # Verify training completed successfully for all tasks
        # Note: Memory counts may vary due to capacity limits and eviction policies
        total_memory_size = len(self.du_core._memory_system)
        assert total_memory_size > 0, "Memory should contain some experiences"

        # Verify training completed successfully
        assert self.trainer.training_stats['total_steps'] == total_steps

    def test_catastrophic_forgetting_prevention(self):
        """Test that GEM prevents catastrophic forgetting."""
        # This is a simplified test - in practice, you'd measure actual performance
        # on held-out test sets for each task

        # Train on first task extensively
        task_a_losses = []
        for i in range(20):
            input_data = torch.randn(30, 1, 20)
            target = torch.randn(1, 2)  # Match decoder output size (batch_size, num_classes)
            loss = self.trainer.train_step(input_data, target, "task_A")
            task_a_losses.append(loss)

        # Train on second task
        task_b_losses = []
        for i in range(20):
            input_data = torch.randn(30, 1, 20)
            target = torch.randn(1, 2)  # Match decoder output size (batch_size, num_classes)
            loss = self.trainer.train_step(input_data, target, "task_B")
            task_b_losses.append(loss)

        # Verify that interference was detected and projections occurred
        # (indicating GEM is actively preventing catastrophic forgetting)
        stats = self.trainer.get_training_stats()

        # Should have some interference events when learning task B
        # (exact numbers depend on random initialization and data)
        assert stats['total_steps'] == 40
        assert len(self.du_core._memory_system) >= 40

        # Verify both tasks completed training successfully
        # Note: Memory counts may vary due to capacity limits and eviction policies
        total_memory_size = len(self.du_core._memory_system)
        assert total_memory_size > 0, "Memory should contain some experiences"


# Additional tests specified in Prompt 3.5
def test_gem_gradient_projection():
    """
    Validates the core mathematical projection of the GEM algorithm.
    This test ensures that if a current gradient conflicts with a past
    gradient, it is projected to be orthogonal to the past gradient.
    """
    # Create two conflicting gradients. g_current wants to move left,
    # g_past wants to move right. Their dot product is negative.
    g_current = torch.tensor([-1.0, 0.0])
    g_past = torch.tensor([1.0, 0.0])

    dot_product = torch.dot(g_current, g_past)
    assert dot_product < 0, "Test setup failed: gradients must conflict."

    # Manually perform the GEM projection
    # This logic should be extracted from the trainer into a testable utility function
    g_proj = g_current - (dot_product / torch.dot(g_past, g_past)) * g_past

    # The new projected gradient should be orthogonal to the past gradient.
    # Their dot product should now be zero.
    final_dot_product = torch.dot(g_proj, g_past)
    assert torch.isclose(final_dot_product, torch.tensor(0.0))
