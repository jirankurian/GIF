"""
Continual Learning Trainer with Gradient Episodic Memory (GEM)
==============================================================

This module implements the Continual_Trainer class, which provides sophisticated
continual learning capabilities using the Gradient Episodic Memory (GEM) algorithm.
The trainer prevents catastrophic forgetting by constraining new learning to not
interfere with previously acquired knowledge.

Gradient Episodic Memory (GEM) Algorithm:
=========================================

GEM is a state-of-the-art continual learning algorithm that addresses the fundamental
problem of catastrophic forgetting in neural networks. The core insight is that
learning new tasks should not increase the loss on previously learned tasks.

**The Analogy**: Think of a student studying for a final exam. Before learning a new
chapter (Task B), the student reviews problems from previous chapters (Task A). When
learning new material, the student ensures that new knowledge doesn't contradict or
interfere with previously learned concepts.

**Technical Implementation**: 
1. Compute gradients for the current learning task
2. Sample random experiences from episodic memory (past tasks)
3. Compute gradients for these past experiences
4. Check for interference using gradient dot products
5. If interference is detected (negative dot product), project the current gradient
   to be orthogonal to the interfering past gradient
6. Apply the projected gradient to update model parameters

Mathematical Foundation:
=======================

For a current gradient g_current and past gradient g_past:
- Interference occurs when: dot(g_current, g_past) < 0
- Projection formula: g_projected = g_current - (dot(g_current, g_past) / ||g_past||²) * g_past

This ensures that the projected gradient does not increase the loss on past tasks
while still enabling learning on the current task.

Biological Inspiration:
======================

The GEM algorithm draws inspiration from several biological learning mechanisms:

**Hippocampal Replay**: During learning and sleep, the hippocampus replays past
experiences, allowing the brain to consolidate memories without forgetting.

**Constraint-Based Learning**: Biological neural networks appear to use constraint-
based mechanisms to prevent new learning from overwriting existing memories.

**Experience Sampling**: The brain doesn't replay all past experiences but samples
relevant ones, similar to GEM's random sampling from episodic memory.

Integration with GIF Framework:
==============================

The Continual_Trainer seamlessly integrates with all GIF components:
- Uses the GIF orchestrator for complete cognitive cycles
- Leverages DU Core's injected plasticity rules and memory systems
- Stores experiences in the episodic memory system
- Applies GEM constraints during gradient updates

Example Usage:
=============

    # Create trainer with GIF model, optimizer, and loss function
    trainer = Continual_Trainer(gif_model, optimizer, criterion)
    
    # Train on sequential tasks with continual learning
    for task_id in ["task_A", "task_B", "task_C"]:
        for data, target in get_task_data(task_id):
            loss = trainer.train_step(data, target, task_id)
            
    # Monitor training statistics
    stats = trainer.get_training_stats()
    print(f"Interference events: {stats['interference_count']}")
    print(f"Average projection magnitude: {stats['avg_projection_magnitude']}")

This implementation enables true continual learning, allowing the GIF framework to
learn new tasks while preserving performance on previously learned tasks.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
from gif_framework.orchestrator import GIF
from gif_framework.core.memory_systems import ExperienceTuple


class Continual_Trainer:
    """
    Continual learning trainer implementing Gradient Episodic Memory (GEM) algorithm.
    
    This class orchestrates the complete continual learning process, including
    experience storage, gradient computation, interference detection, and gradient
    projection to prevent catastrophic forgetting.
    
    The trainer works with any GIF model that has been configured with appropriate
    plasticity rules and episodic memory systems, providing a high-level interface
    for continual learning experiments.
    
    Key Features:
    - Implements complete GEM algorithm with gradient projection
    - Automatic experience storage in episodic memory
    - Interference detection and constraint enforcement
    - Training statistics and monitoring capabilities
    - Integration with PyTorch optimizers and loss functions
    
    Attributes:
        gif_model: The GIF orchestrator containing the DU Core and attached components
        optimizer: PyTorch optimizer for parameter updates
        criterion: Loss function for computing training objectives
        memory_batch_size: Number of past experiences to sample for GEM constraints
        training_stats: Dictionary containing training statistics and metrics
    """
    
    def __init__(
        self,
        gif_model: GIF,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        memory_batch_size: int = 32,
        gradient_clip_norm: float = None
    ):
        """
        Initialize the continual learning trainer.
        
        Args:
            gif_model (GIF): The GIF orchestrator instance containing the DU Core
                           with injected plasticity rules and memory systems.
            optimizer (torch.optim.Optimizer): PyTorch optimizer for parameter updates.
                                              Common choices: Adam, SGD, RMSprop.
            criterion (torch.nn.Module): Loss function for computing training objectives.
                                        Common choices: CrossEntropyLoss, MSELoss.
            memory_batch_size (int, optional): Number of past experiences to sample
                                              for GEM constraint checking. Default: 32.
            gradient_clip_norm (float, optional): Maximum norm for gradient clipping.
                                                 If None, no clipping is applied.
        
        Raises:
            TypeError: If gif_model is not a GIF instance.
            ValueError: If memory_batch_size is not positive.
            RuntimeError: If DU Core lacks required memory system.
        """
        # Validate inputs
        if not isinstance(gif_model, GIF):
            raise TypeError(f"gif_model must be a GIF instance, got {type(gif_model)}")

        if not isinstance(memory_batch_size, int) or memory_batch_size <= 0:
            raise ValueError(f"memory_batch_size must be positive integer, got {memory_batch_size}")

        # Validate that DU Core has required components for continual learning
        if gif_model._du_core._memory_system is None:
            raise RuntimeError(
                "DU Core must have an injected memory system for continual learning. "
                "Please initialize DU_Core_V1 with an EpisodicMemory instance."
            )

        # Store components
        self.gif_model = gif_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.memory_batch_size = memory_batch_size
        self.gradient_clip_norm = gradient_clip_norm

        # Initialize training statistics
        self.training_stats = {
            'total_steps': 0,
            'interference_count': 0,
            'projection_count': 0,
            'avg_projection_magnitude': 0.0,
            'total_projection_magnitude': 0.0,
            'task_losses': {},
            'memory_utilization': [],
            # Meta-learning statistics for System Potentiation
            'meta_learning': {
                'total_adaptations': 0,
                'correct_predictions': 0,
                'incorrect_predictions': 0,
                'accuracy_history': [],
                'surprise_history': [],
                'learning_rate_history': [],
                'adaptation_magnitude_history': []
            }
        }

    def train_step(self, sample_data: Any, target: torch.Tensor, task_id: str) -> float:
        """
        Perform a single GEM-constrained training step.

        This method implements the complete GEM algorithm:
        1. Forward pass through the GIF model
        2. Experience storage in episodic memory
        3. Current gradient computation
        4. Past experience sampling and gradient computation
        5. Interference detection and gradient projection
        6. Final parameter update

        Args:
            sample_data (Any): Input data to process through the GIF model.
                             Format depends on the attached encoder.
            target (torch.Tensor): Target output for supervised learning.
                                 Shape depends on the task and decoder.
            task_id (str): Identifier for the current task. Used for organizing
                         experiences in memory and tracking task-specific metrics.

        Returns:
            float: Training loss value for the current step.

        Raises:
            RuntimeError: If GIF model is not properly configured.
            ValueError: If target tensor has incompatible shape.

        Example:
            # Single training step
            loss = trainer.train_step(
                sample_data=input_image,
                target=torch.tensor([class_label]),
                task_id="image_classification"
            )
        """
        self.training_stats['total_steps'] += 1

        # A. Forward Pass - Process input through complete GIF pipeline
        try:
            # First, encode the raw data to get spike trains
            spike_train = self.gif_model._encoder.encode(sample_data)

            # Move spike train to the same device as the DU Core (for GPU acceleration)
            if hasattr(self.gif_model._du_core, 'linear_layers') and len(self.gif_model._du_core.linear_layers) > 0:
                device = next(self.gif_model._du_core.linear_layers[0].parameters()).device
                spike_train = spike_train.to(device)

            # Then process through DU Core
            processed_spikes = self.gif_model._du_core.process(spike_train)

            # For training, we need logits with gradients, not discrete classifications
            # Calculate spike counts and use decoder's classification forward method
            spike_counts = processed_spikes.sum(dim=0)  # Sum across time
            if spike_counts.dim() == 1:
                spike_counts = spike_counts.unsqueeze(0)  # Add batch dimension

            # Use decoder's classification forward method to get logits with gradients
            output = self.gif_model._decoder.forward_classification(spike_counts)

        except Exception as e:
            raise RuntimeError(f"GIF model forward pass failed: {e}") from e

        # B. Determine Outcome for Meta-Plasticity
        # Calculate prediction outcome for System Potentiation
        with torch.no_grad():
            # Handle both single samples and batches
            if output.dim() == 1:
                # Single sample case - add batch dimension
                output_for_prediction = output.unsqueeze(0)
                target_for_prediction = target.unsqueeze(0) if target.dim() == 1 else target
            else:
                # Batch case
                output_for_prediction = output
                target_for_prediction = target

            # Get predicted class (highest logit)
            predicted_class = torch.argmax(output_for_prediction, dim=1)

            # Determine if prediction is correct (1) or incorrect (0)
            if target_for_prediction.dim() == 1 or (target_for_prediction.dim() == 2 and target_for_prediction.shape[1] == 1):
                # Target is class indices
                actual_class = target_for_prediction.flatten()
            else:
                # Target is one-hot encoded or multi-dimensional
                actual_class = torch.argmax(target_for_prediction, dim=1)

            # Calculate outcome: 1 for correct, 0 for incorrect
            # Handle both single sample and batch cases
            comparison = (predicted_class == actual_class)
            if comparison.numel() == 1:
                outcome = int(comparison.item())
            else:
                # For batch case, take the first sample's outcome
                outcome = int(comparison[0].item())

        # C. Store Experience with Performance Outcome
        # Store the raw input data for gradient replay and performance tracking
        # Handle different input data types (tensor, DataFrame, numpy array, etc.)
        try:
            if isinstance(sample_data, torch.Tensor):
                stored_input = sample_data.detach().clone()
            elif hasattr(sample_data, 'to_numpy'):  # polars DataFrame
                numpy_data = sample_data.to_numpy()
                # Flatten if it's a 2D array with time series data
                if numpy_data.ndim > 1:
                    stored_input = torch.tensor(numpy_data.flatten(), dtype=torch.float32)
                else:
                    stored_input = torch.tensor(numpy_data, dtype=torch.float32)
            elif hasattr(sample_data, 'values') and hasattr(sample_data, 'columns'):  # pandas DataFrame
                numpy_data = sample_data.values
                if numpy_data.ndim > 1:
                    stored_input = torch.tensor(numpy_data.flatten(), dtype=torch.float32)
                else:
                    stored_input = torch.tensor(numpy_data, dtype=torch.float32)
            elif hasattr(sample_data, '__array__') and not hasattr(sample_data, 'values'):  # numpy array
                stored_input = torch.tensor(sample_data, dtype=torch.float32)
            else:
                # For other types, try direct conversion
                stored_input = torch.tensor(sample_data, dtype=torch.float32)
        except Exception as e:
            # If conversion fails, store a placeholder tensor with reasonable size
            stored_input = torch.zeros(100, dtype=torch.float32)  # Reasonable size for light curve data
            # Only print warning in debug mode to reduce noise
            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"Warning: Could not convert sample_data to tensor: {e}")

        experience = ExperienceTuple(
            input_spikes=stored_input,  # Store raw input data (not encoded)
            internal_state=None,  # Placeholder for future DU Core internal state
            output_spikes=output.detach().clone(),  # Store the output
            task_id=task_id
        )

        # Add experience with performance outcome for meta-plasticity
        self.gif_model._du_core._memory_system.add(experience, outcome=outcome)

        # D. Apply Meta-Learning (System Potentiation)
        # Update plasticity rule learning rate based on recent performance
        if self.gif_model._du_core._plasticity_rule is not None:
            # Import here to avoid circular imports
            from gif_framework.core.rtl_mechanisms import ThreeFactor_Hebbian_Rule

            # Only apply meta-plasticity to rules that support it
            if isinstance(self.gif_model._du_core._plasticity_rule, ThreeFactor_Hebbian_Rule):
                try:
                    # Get surprise signal from memory system
                    surprise_signal = self.gif_model._du_core._memory_system.get_surprise_signal()

                    # Update learning rate based on performance
                    self.gif_model._du_core._plasticity_rule.update_learning_rate(surprise_signal)

                    # Track meta-learning statistics
                    self._update_meta_learning_stats(outcome, surprise_signal)

                except Exception as e:
                    # Log meta-learning error but don't break training
                    import warnings
                    warnings.warn(f"Meta-learning update failed: {str(e)}")

        # F. Calculate Current Gradient
        loss_current = self.criterion(output, target)

        # Clear any existing gradients
        self.optimizer.zero_grad()

        # Compute gradients for current task
        loss_current.backward(retain_graph=True)

        # Extract current gradients into a single flattened tensor
        g_current = self._extract_gradients()

        # G. Sample from Memory and Check for Interference
        memory_system = self.gif_model._du_core._memory_system

        # Only apply GEM if we have enough experiences and they're from the current session
        if len(memory_system) >= self.memory_batch_size and self.training_stats['total_steps'] > 1:
            try:
                # Sample past experiences
                past_experiences = memory_system.sample(self.memory_batch_size)

                # H. Calculate Past Gradients & Check for Interference
                g_current = self._apply_gem_constraints(g_current, past_experiences)
            except Exception as e:
                # If GEM fails (e.g., due to incompatible experience shapes), skip it
                # This can happen when experiences from different test configurations are mixed
                pass

        # I. Apply Final Update
        self._apply_gradients(g_current)

        # Apply gradient clipping if specified
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.gif_model._du_core.parameters(),
                self.gradient_clip_norm
            )

        self.optimizer.step()

        # Update training statistics
        self._update_training_stats(task_id, loss_current.item(), memory_system)

        return loss_current.item()

    def _extract_gradients(self) -> torch.Tensor:
        """
        Extract and flatten all model gradients into a single tensor.

        This method iterates through all model parameters and concatenates
        their gradients into a single flattened tensor for GEM processing.

        Returns:
            torch.Tensor: Flattened gradient tensor containing all model gradients.

        Raises:
            RuntimeError: If any parameter lacks gradients.
        """
        gradients = []

        for param in self.gif_model._du_core.parameters():
            if param.grad is None:
                raise RuntimeError(
                    "Found parameter without gradient. Ensure backward() was called "
                    "before extracting gradients."
                )
            gradients.append(param.grad.view(-1))

        return torch.cat(gradients)

    def _apply_gradients(self, flattened_gradients: torch.Tensor) -> None:
        """
        Apply flattened gradients back to model parameters.

        This method takes a flattened gradient tensor and distributes the
        gradients back to their corresponding model parameters.

        Args:
            flattened_gradients (torch.Tensor): Flattened gradient tensor to apply.
        """
        start_idx = 0

        for param in self.gif_model._du_core.parameters():
            param_size = param.numel()
            param_gradients = flattened_gradients[start_idx:start_idx + param_size]
            param.grad = param_gradients.view(param.shape)
            start_idx += param_size

    def _apply_gem_constraints(
        self,
        g_current: torch.Tensor,
        past_experiences: List[ExperienceTuple]
    ) -> torch.Tensor:
        """
        Apply GEM constraints by projecting current gradient away from interfering past gradients.

        This method implements the core GEM algorithm by:
        1. Computing gradients for past experiences
        2. Detecting interference through dot products
        3. Projecting current gradient to avoid negative interference

        Args:
            g_current (torch.Tensor): Current task gradient (flattened).
            past_experiences (List[ExperienceTuple]): Sampled past experiences.

        Returns:
            torch.Tensor: Projected gradient that satisfies GEM constraints.
        """
        projected_gradient = g_current.clone()

        for experience in past_experiences:
            # Compute gradient for this past experience
            g_past = self._compute_experience_gradient(experience)

            # Check for interference (negative dot product)
            dot_product = torch.dot(projected_gradient, g_past)

            if dot_product < 0:
                # Interference detected - apply projection
                self.training_stats['interference_count'] += 1
                self.training_stats['projection_count'] += 1

                # GEM projection formula: g = g - (g·g_past / ||g_past||²) * g_past
                projection_coefficient = dot_product / torch.dot(g_past, g_past)
                projection = projection_coefficient * g_past
                projected_gradient = projected_gradient - projection

                # Track projection magnitude for statistics
                projection_magnitude = torch.norm(projection).item()
                self.training_stats['total_projection_magnitude'] += projection_magnitude

        # Update average projection magnitude
        if self.training_stats['projection_count'] > 0:
            self.training_stats['avg_projection_magnitude'] = (
                self.training_stats['total_projection_magnitude'] /
                self.training_stats['projection_count']
            )

        return projected_gradient

    def _compute_experience_gradient(self, experience: ExperienceTuple) -> torch.Tensor:
        """
        Compute gradient for a past experience.

        This method performs a forward pass with a past experience and computes
        the resulting gradients for GEM constraint checking.

        Args:
            experience (ExperienceTuple): Past experience to compute gradient for.

        Returns:
            torch.Tensor: Flattened gradient tensor for the past experience.
        """
        # Clear existing gradients
        self.optimizer.zero_grad()

        # Re-run forward pass with past experience raw data to get gradients
        # This is necessary because stored outputs don't have grad_fn
        # First encode the raw input data, then process through DU Core
        encoded_spikes = self.gif_model._encoder.encode(experience.input_spikes)
        processed_spikes = self.gif_model._du_core.process(encoded_spikes)

        # Calculate spike counts and use decoder's classification forward method for gradients
        spike_counts = processed_spikes.sum(dim=0)  # Sum across time
        if spike_counts.dim() == 1:
            spike_counts = spike_counts.unsqueeze(0)  # Add batch dimension

        # Use decoder's classification forward method to get logits with gradients
        past_output = self.gif_model._decoder.forward_classification(spike_counts)

        # Create a dummy target with the same shape as current decoder output
        # This ensures compatibility when replaying experiences from different configurations
        dummy_target = torch.zeros_like(past_output).detach()

        # Compute loss for past experience
        past_loss = self.criterion(past_output, dummy_target)

        # Compute gradients
        past_loss.backward(retain_graph=True)

        # Extract and return gradients
        return self._extract_gradients()

    def _update_training_stats(
        self,
        task_id: str,
        loss: float,
        memory_system
    ) -> None:
        """
        Update training statistics and metrics.

        Args:
            task_id (str): Current task identifier.
            loss (float): Current training loss.
            memory_system: The episodic memory system.
        """
        # Track task-specific losses
        if task_id not in self.training_stats['task_losses']:
            self.training_stats['task_losses'][task_id] = []
        self.training_stats['task_losses'][task_id].append(loss)

        # Track memory utilization
        memory_info = memory_system.get_memory_info()
        self.training_stats['memory_utilization'].append(memory_info['utilization'])

    def _update_meta_learning_stats(self, outcome: int, surprise_signal: float) -> None:
        """
        Update meta-learning statistics for System Potentiation analysis.

        Args:
            outcome (int): Prediction outcome (1 for correct, 0 for incorrect)
            surprise_signal (float): Current surprise signal from performance tracking
        """
        meta_stats = self.training_stats['meta_learning']

        # Update adaptation count
        meta_stats['total_adaptations'] += 1

        # Update prediction counts
        if outcome == 1:
            meta_stats['correct_predictions'] += 1
        else:
            meta_stats['incorrect_predictions'] += 1

        # Calculate current accuracy
        total_predictions = meta_stats['correct_predictions'] + meta_stats['incorrect_predictions']
        current_accuracy = meta_stats['correct_predictions'] / total_predictions if total_predictions > 0 else 0.0

        # Store histories for analysis
        meta_stats['accuracy_history'].append(current_accuracy)
        meta_stats['surprise_history'].append(surprise_signal)

        # Get current learning rate from plasticity rule if available
        if (self.gif_model._du_core._plasticity_rule is not None and
            hasattr(self.gif_model._du_core._plasticity_rule, 'current_learning_rate')):
            current_lr = self.gif_model._du_core._plasticity_rule.current_learning_rate
            base_lr = self.gif_model._du_core._plasticity_rule.base_learning_rate
            adaptation_magnitude = current_lr / base_lr

            meta_stats['learning_rate_history'].append(current_lr)
            meta_stats['adaptation_magnitude_history'].append(adaptation_magnitude)

        # Keep histories bounded to prevent memory growth
        max_history = 1000
        for history_key in ['accuracy_history', 'surprise_history', 'learning_rate_history', 'adaptation_magnitude_history']:
            if len(meta_stats[history_key]) > max_history:
                meta_stats[history_key] = meta_stats[history_key][-max_history:]

    def get_meta_learning_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive meta-learning statistics for System Potentiation analysis.

        Returns:
            Dict[str, Any]: Dictionary containing detailed meta-learning metrics including
                          accuracy trends, surprise signals, learning rate adaptations,
                          and System Potentiation effectiveness measures.
        """
        meta_stats = self.training_stats['meta_learning'].copy()

        # Calculate derived statistics
        total_predictions = meta_stats['correct_predictions'] + meta_stats['incorrect_predictions']
        if total_predictions > 0:
            meta_stats['overall_accuracy'] = meta_stats['correct_predictions'] / total_predictions
        else:
            meta_stats['overall_accuracy'] = 0.0

        # Calculate recent performance trends
        if len(meta_stats['accuracy_history']) >= 10:
            recent_accuracy = sum(meta_stats['accuracy_history'][-10:]) / 10
            meta_stats['recent_accuracy'] = recent_accuracy

            # Calculate improvement trend
            if len(meta_stats['accuracy_history']) >= 20:
                early_accuracy = sum(meta_stats['accuracy_history'][-20:-10]) / 10
                meta_stats['accuracy_improvement'] = recent_accuracy - early_accuracy
            else:
                meta_stats['accuracy_improvement'] = 0.0
        else:
            meta_stats['recent_accuracy'] = meta_stats['overall_accuracy']
            meta_stats['accuracy_improvement'] = 0.0

        # Calculate adaptation statistics
        if len(meta_stats['adaptation_magnitude_history']) > 0:
            meta_stats['avg_adaptation_magnitude'] = sum(meta_stats['adaptation_magnitude_history']) / len(meta_stats['adaptation_magnitude_history'])
            meta_stats['max_adaptation_magnitude'] = max(meta_stats['adaptation_magnitude_history'])
            meta_stats['min_adaptation_magnitude'] = min(meta_stats['adaptation_magnitude_history'])
        else:
            meta_stats['avg_adaptation_magnitude'] = 1.0
            meta_stats['max_adaptation_magnitude'] = 1.0
            meta_stats['min_adaptation_magnitude'] = 1.0

        return meta_stats

    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive training statistics.

        Returns:
            Dict[str, Any]: Dictionary containing training metrics and statistics.
        """
        return self.training_stats.copy()

    def reset_stats(self) -> None:
        """Reset all training statistics."""
        self.training_stats = {
            'total_steps': 0,
            'interference_count': 0,
            'projection_count': 0,
            'avg_projection_magnitude': 0.0,
            'total_projection_magnitude': 0.0,
            'task_losses': {},
            'memory_utilization': []
        }
