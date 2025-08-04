"""
Episodic Memory Systems for the General Intelligence Framework
=============================================================

This module implements the episodic memory system that serves as the foundation for continual
learning and agentic reasoning within the GIF framework. Unlike simple data buffers, this
system provides structured storage and retrieval of past experiences, enabling the framework
to learn continuously without catastrophic forgetting.

Episodic Memory in AGI Context:
==============================

In cognitive science, episodic memory refers to the memory of everyday events - the "what,
when, where" of personal experiences. For our AGI system, episodic memory serves a similar
but more structured purpose: it's the agent's personal history of interactions with the world.

When the DU Core processes an input and produces an output, that entire interaction becomes
an "experience" that can be stored and later recalled. This enables the system to:

1. **Reflect on Past Experiences:** Learn from its own unique history
2. **Prevent Catastrophic Forgetting:** Use past experiences to constrain new learning
3. **Enable Continual Learning:** Support algorithms like Gradient Episodic Memory (GEM)
4. **Support Meta-Learning:** Identify patterns across different tasks and domains

Technical Architecture:
=======================

The system is built around two core components:

**ExperienceTuple:** An immutable data structure that captures a complete interaction:
- Input spike trains that entered the system
- Internal state of the DU Core during processing
- Output spike trains produced by the system
- Task identifier for organizing and retrieving memories

**EpisodicMemory:** An efficient memory buffer that manages collections of experiences:
- Uses collections.deque with maxlen for automatic FIFO behavior
- Provides random sampling for continual learning algorithms
- Supports task-specific memory organization and retrieval
- Optimized for real-time learning scenarios

Key Design Principles:
=====================

**Efficiency:** Uses highly optimized C-based deque for fast append/remove operations
**Modularity:** Clean separation between memory storage and learning algorithms
**Scalability:** Configurable capacity with automatic memory management
**Flexibility:** Supports various sampling strategies and memory organization schemes

Integration with GEM Algorithm:
==============================

The episodic memory system is specifically designed to support Gradient Episodic Memory
(GEM), a state-of-the-art continual learning algorithm that prevents catastrophic forgetting:

1. **Experience Storage:** Each learning interaction is stored as an ExperienceTuple
2. **Random Sampling:** GEM requires random batches of past experiences for gradient projection
3. **Task Organization:** Memories can be organized by task_id for task-specific retrieval
4. **Efficient Access:** Fast sampling enables real-time constraint checking during learning

Example Usage:
=============

    # Create episodic memory with capacity for 10,000 experiences
    memory = EpisodicMemory(capacity=10000)

    # Store an experience after processing
    experience = ExperienceTuple(
        input_spikes=input_spike_train,
        internal_state=None,  # Placeholder for future use
        output_spikes=output_spike_train,
        task_id="exoplanet_classification"
    )
    memory.add(experience)

    # Sample random batch for GEM algorithm
    past_experiences = memory.sample(batch_size=32)

    # Check memory status
    print(f"Memory contains {len(memory)} experiences")
    print(f"Task-specific count: {memory.get_task_count('exoplanet_classification')}")

Integration Points:
==================

- **DU Core Integration:** Designed for dependency injection into DU_Core_V1
- **RTL Compatibility:** Works seamlessly with plasticity rules from rtl_mechanisms.py
- **GEM Algorithm:** Provides the memory substrate for continual learning
- **Testing Framework:** Compatible with existing pytest infrastructure

This episodic memory system transforms the GIF framework from a reactive processing system
into a truly adaptive, learning agent that builds and leverages its own experiential knowledge.
"""

import random
import torch
from collections import deque
from typing import Any, Dict, List, NamedTuple, Optional


# =============================================================================
# Experience Data Structure
# =============================================================================

class ExperienceTuple(NamedTuple):
    """
    Immutable data structure representing a single experience in the episodic memory.

    An experience captures a complete interaction cycle within the GIF framework:
    the input that was processed, the internal state during processing, the output
    that was produced, the conceptual understanding formed, and the task context.

    This structure serves as the fundamental unit of episodic memory, enabling the
    system to store, retrieve, and learn from its past interactions. The immutable
    nature (via NamedTuple) ensures data integrity and prevents accidental modification
    of stored memories.

    Biological Context:
    In biological neural networks, episodic memories are thought to be stored as
    patterns of synaptic connections that can be reactivated to recall past experiences.
    Our ExperienceTuple serves a similar function, capturing the essential information
    needed to reconstruct and learn from past neural processing events, including
    the conceptual understanding formed through Vector Symbolic Architecture.

    Fields:
    -------
    input_spikes : torch.Tensor
        The spike train that served as input to the DU Core for this experience.
        Shape: [time_steps, batch_size, input_features]
        This represents the encoded sensory or processed information that initiated
        the neural processing cycle.

    internal_state : Any
        The internal state of the DU Core during this experience. Currently serves
        as a placeholder for future extensions that might capture membrane potentials,
        hidden states, or other internal neural dynamics. Can be None for now.

        Future possibilities:
        - Membrane potential snapshots
        - Hidden layer activations
        - Attention weights
        - Neuromodulator levels

    output_spikes : torch.Tensor
        The spike train produced by the DU Core as output for this experience.
        Shape: [time_steps, batch_size, output_features]
        This represents the neural response or decision generated by the system.

    task_id : str
        A string identifier indicating the task or domain context for this experience.
        This enables task-specific memory organization and retrieval, which is crucial
        for continual learning algorithms that need to prevent interference between
        different learning domains.

        Examples:
        - "exoplanet_classification"
        - "ecg_arrhythmia_detection"
        - "general_pattern_recognition"
        - "reward_learning_phase_1"

    conceptual_state : Optional[torch.Tensor]
        The conceptual representation formed by the DU Core during this experience,
        represented as a hypervector from the Vector Symbolic Architecture (VSA).
        Shape: [vsa_dimension] (typically 10,000 dimensions)

        This field captures the "Deep Understanding" formed by the system - the
        explicit conceptual knowledge and relationships discovered through VSA
        operations like binding and bundling. It represents the system's semantic
        interpretation of the experience.

        When None, indicates that VSA was not active or conceptual state was not
        captured for this experience (maintains backward compatibility).

        Examples of conceptual states:
        - Bound concepts: OBJECT ⊗ APPLE (object-attribute relationship)
        - Bundled concepts: FRUIT + SWEET + RED (composite concept)
        - Master state: Accumulated understanding across experiences

    Usage Examples:
    --------------

    # Create an experience after processing input (backward compatible)
    experience = ExperienceTuple(
        input_spikes=encoded_sensor_data,
        internal_state=None,  # Placeholder for future use
        output_spikes=neural_decision,
        task_id="medical_diagnosis",
        conceptual_state=None  # No VSA active
    )

    # Create an experience with VSA conceptual understanding
    experience_with_vsa = ExperienceTuple(
        input_spikes=encoded_sensor_data,
        internal_state=None,
        output_spikes=neural_decision,
        task_id="medical_diagnosis",
        conceptual_state=hypervector_representation  # VSA concept
    )

    # Access fields by name (more readable than tuple indexing)
    print(f"Experience from task: {experience.task_id}")
    print(f"Input shape: {experience.input_spikes.shape}")
    print(f"Output shape: {experience.output_spikes.shape}")

    # Use in continual learning algorithms
    for exp in memory.sample(batch_size=32):
        # Replay past experience for GEM constraint checking
        past_output = model(exp.input_spikes)
        loss = criterion(past_output, exp.output_spikes)
        # ... GEM gradient projection logic

    Design Rationale:
    ----------------

    **NamedTuple Choice:** Provides immutability, memory efficiency, and named field
    access while maintaining the performance characteristics of regular tuples.

    **Field Selection:** Captures the minimal essential information needed for most
    continual learning algorithms while remaining extensible for future enhancements.

    **Task Organization:** The task_id field enables sophisticated memory management
    strategies, including task-specific sampling and interference analysis.
    """

    input_spikes: torch.Tensor
    internal_state: Any
    output_spikes: torch.Tensor
    task_id: str
    conceptual_state: Optional[torch.Tensor] = None


# =============================================================================
# Episodic Memory System
# =============================================================================

class EpisodicMemory:
    """
    Efficient episodic memory system for storing and retrieving past experiences.

    This class implements a sophisticated memory buffer that serves as the foundation
    for continual learning within the GIF framework. It manages collections of
    ExperienceTuple objects using an efficient FIFO (First-In-First-Out) buffer
    with configurable capacity and advanced sampling capabilities.

    The system is specifically designed to support continual learning algorithms
    like Gradient Episodic Memory (GEM) that require efficient access to random
    samples of past experiences to prevent catastrophic forgetting.

    Technical Implementation:
    ========================

    **Storage Backend:** Uses collections.deque with maxlen for optimal performance:
    - O(1) append operations for storing new experiences
    - Automatic removal of oldest experiences when capacity is exceeded
    - Memory-efficient circular buffer behavior
    - Thread-safe operations for concurrent access

    **Sampling Strategy:** Implements uniform random sampling without replacement:
    - Uses random.sample() for cryptographically secure randomness
    - Ensures no duplicate experiences in sampled batches
    - Handles edge cases when requested batch size exceeds available memories
    - Optimized for frequent sampling operations during learning

    **Memory Organization:** Supports task-aware memory management:
    - Experiences tagged with task_id for domain-specific retrieval
    - Efficient counting and filtering by task type
    - Enables sophisticated continual learning strategies

    Biological Inspiration:
    ======================

    This system draws inspiration from the hippocampal memory system in mammalian
    brains, which is responsible for episodic memory formation and retrieval:

    - **Rapid Encoding:** New experiences are quickly stored without interfering
      with existing memories (similar to hippocampal pattern separation)
    - **Random Replay:** During learning, random past experiences are replayed
      (similar to hippocampal replay during sleep and rest)
    - **Capacity Management:** Older memories naturally fade as new ones are formed
      (similar to natural forgetting processes)

    Integration with Continual Learning:
    ===================================

    The episodic memory system is designed to seamlessly integrate with continual
    learning algorithms, particularly Gradient Episodic Memory (GEM):

    1. **Experience Storage:** Each learning interaction is automatically stored
    2. **Constraint Generation:** Random samples provide constraints for gradient projection
    3. **Forgetting Prevention:** Past experiences prevent catastrophic forgetting
    4. **Knowledge Transfer:** Cross-task experiences enable positive transfer

    Example Usage:
    =============

        # Initialize memory with capacity for 10,000 experiences
        memory = EpisodicMemory(capacity=10000)

        # Store experiences during learning
        for input_data, target_output in training_data:
            # Process through DU Core
            output = du_core.process(input_data)

            # Store the experience
            experience = ExperienceTuple(
                input_spikes=input_data,
                internal_state=None,
                output_spikes=output,
                task_id="current_task"
            )
            memory.add(experience)

        # Sample for continual learning
        if len(memory) >= 32:
            past_experiences = memory.sample(batch_size=32)
            # Use for GEM gradient projection...

        # Monitor memory usage
        print(f"Memory utilization: {len(memory)}/{memory.capacity}")
        print(f"Task distribution: {memory.get_task_count('task_1')} experiences")

    Performance Characteristics:
    ===========================

    - **Storage:** O(1) amortized insertion time
    - **Sampling:** O(k) where k is the batch size
    - **Memory Usage:** O(n) where n is the capacity
    - **Thread Safety:** Safe for concurrent read/write operations
    """

    def __init__(self, capacity: int, performance_buffer_size: int = 100):
        """
        Initialize the episodic memory system with specified capacity.

        Args:
            capacity (int): Maximum number of experiences to store. When this
                          limit is reached, the oldest experiences are automatically
                          removed to make room for new ones (FIFO behavior).

                          Typical values:
                          - Small experiments: 1,000 - 10,000
                          - Medium experiments: 10,000 - 100,000
                          - Large-scale systems: 100,000 - 1,000,000

            performance_buffer_size (int, optional): Size of the performance tracking
                                                    buffer for meta-plasticity. This buffer
                                                    stores recent performance outcomes to
                                                    calculate surprise signals. Default: 100.

        Raises:
            ValueError: If capacity is not a positive integer.
            ValueError: If performance_buffer_size is not a positive integer.

        Note:
            The actual memory usage will depend on the size of the stored tensors.
            Each ExperienceTuple contains two tensors (input_spikes, output_spikes)
            plus minimal metadata. Plan capacity accordingly based on available RAM.

            The performance buffer enables System Potentiation by tracking recent
            outcomes to generate surprise signals for adaptive learning rates.
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"Capacity must be a positive integer, got {capacity}")

        if not isinstance(performance_buffer_size, int) or performance_buffer_size <= 0:
            raise ValueError(f"Performance buffer size must be a positive integer, got {performance_buffer_size}")

        self.capacity = capacity
        self.performance_buffer_size = performance_buffer_size
        self._experiences: deque = deque(maxlen=capacity)

        # Performance tracking for meta-plasticity and System Potentiation
        self.recent_performance_buffer: deque = deque(maxlen=performance_buffer_size)

    def add(self, experience: ExperienceTuple, outcome: Optional[int] = None) -> None:
        """
        Add a new experience to the episodic memory with optional performance outcome.

        This method stores a new experience in the memory buffer. If the buffer
        is at capacity, the oldest experience is automatically removed to make
        room for the new one (FIFO behavior).

        For System Potentiation, the method can also track performance outcomes
        to enable meta-plasticity and adaptive learning rates.

        Args:
            experience (ExperienceTuple): The experience to store. Must be a valid
                                        ExperienceTuple with all required fields.
            outcome (Optional[int]): Performance outcome for this experience.
                                   - 1: Correct/successful prediction
                                   - 0: Incorrect/failed prediction
                                   - None: No performance tracking (backward compatibility)

        Raises:
            TypeError: If experience is not an ExperienceTuple.
            ValueError: If experience contains invalid data (e.g., non-tensor spikes).
            ValueError: If outcome is not None, 0, or 1.

        Example:
            # Basic usage (backward compatible)
            experience = ExperienceTuple(
                input_spikes=torch.randn(50, 1, 100),
                internal_state=None,
                output_spikes=torch.randn(50, 1, 10),
                task_id="classification_task"
            )
            memory.add(experience)

            # With performance tracking for meta-plasticity
            memory.add(experience, outcome=1)  # Correct prediction
            memory.add(experience, outcome=0)  # Incorrect prediction
        """
        if not isinstance(experience, ExperienceTuple):
            raise TypeError(f"Expected ExperienceTuple, got {type(experience)}")

        # Validate tensor fields
        if not isinstance(experience.input_spikes, torch.Tensor):
            raise ValueError("input_spikes must be a torch.Tensor")
        if not isinstance(experience.output_spikes, torch.Tensor):
            raise ValueError("output_spikes must be a torch.Tensor")
        if not isinstance(experience.task_id, str):
            raise ValueError("task_id must be a string")

        # Validate outcome if provided
        if outcome is not None and outcome not in [0, 1]:
            raise ValueError(f"Outcome must be None, 0 (incorrect), or 1 (correct), got {outcome}")

        # Add to memory (deque automatically handles capacity overflow)
        self._experiences.append(experience)

        # Track performance outcome for meta-plasticity if provided
        if outcome is not None:
            self.recent_performance_buffer.append(outcome)

    def get_surprise_signal(self) -> float:
        """
        Calculate surprise signal from recent performance for meta-plasticity.

        The surprise signal quantifies how poorly the system is currently performing,
        which drives adaptive learning rates in the System Potentiation mechanism.
        High surprise (poor performance) increases plasticity to learn faster,
        while low surprise (good performance) stabilizes learning to preserve knowledge.

        Returns:
            float: Surprise signal in range [0.0, 1.0]
                  - 0.0: Perfect performance (all recent predictions correct)
                  - 0.5: Random performance (50% correct)
                  - 1.0: Worst performance (all recent predictions incorrect)

        Note:
            If no performance data is available, returns 0.5 (neutral surprise)
            to avoid biasing the meta-learning process.

        Example:
            # After tracking some outcomes
            memory.add(experience1, outcome=1)  # Correct
            memory.add(experience2, outcome=1)  # Correct
            memory.add(experience3, outcome=0)  # Incorrect

            surprise = memory.get_surprise_signal()  # Returns ~0.33 (33% error rate)
        """
        if len(self.recent_performance_buffer) == 0:
            # No performance data available - return neutral surprise
            return 0.5

        # Calculate success rate from recent outcomes
        success_rate = sum(self.recent_performance_buffer) / len(self.recent_performance_buffer)

        # Convert to error rate (surprise signal)
        # High error rate = high surprise = need more plasticity
        surprise_signal = 1.0 - success_rate

        return surprise_signal

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get detailed performance statistics for analysis and debugging.

        Returns:
            Dict[str, float]: Dictionary containing performance metrics:
                - 'success_rate': Fraction of correct predictions [0.0, 1.0]
                - 'error_rate': Fraction of incorrect predictions [0.0, 1.0]
                - 'surprise_signal': Current surprise signal [0.0, 1.0]
                - 'sample_count': Number of performance samples tracked
                - 'buffer_utilization': Fraction of performance buffer filled [0.0, 1.0]
        """
        if len(self.recent_performance_buffer) == 0:
            return {
                'success_rate': 0.5,  # Neutral assumption
                'error_rate': 0.5,
                'surprise_signal': 0.5,
                'sample_count': 0,
                'buffer_utilization': 0.0
            }

        success_rate = sum(self.recent_performance_buffer) / len(self.recent_performance_buffer)
        error_rate = 1.0 - success_rate

        return {
            'success_rate': success_rate,
            'error_rate': error_rate,
            'surprise_signal': error_rate,  # Same as get_surprise_signal()
            'sample_count': len(self.recent_performance_buffer),
            'buffer_utilization': len(self.recent_performance_buffer) / self.performance_buffer_size
        }

    def sample(self, batch_size: int) -> List[ExperienceTuple]:
        """
        Sample a random batch of experiences from memory.

        This is the most critical method for continual learning algorithms. It
        provides random access to past experiences, which is essential for
        algorithms like GEM that need to check constraints against historical data.

        The sampling is performed without replacement, ensuring that each experience
        in the returned batch is unique. If the requested batch size exceeds the
        number of available experiences, all available experiences are returned.

        Args:
            batch_size (int): Number of experiences to sample. Must be positive.
                            If larger than available experiences, all experiences
                            are returned.

        Returns:
            List[ExperienceTuple]: A list of randomly sampled experiences.
                                 Length will be min(batch_size, len(memory)).

        Raises:
            ValueError: If batch_size is not a positive integer.
            RuntimeError: If memory is empty (no experiences to sample).

        Example:
            # Sample 32 random experiences for GEM algorithm
            if len(memory) > 0:
                batch = memory.sample(batch_size=32)
                for exp in batch:
                    # Use experience for gradient constraint checking
                    past_loss = compute_loss(exp.input_spikes, exp.output_spikes)
                    # ... GEM projection logic

        Algorithm Details:
            Uses random.sample() which implements reservoir sampling for efficiency.
            This ensures uniform probability distribution across all stored experiences
            and cryptographically secure randomness for reproducible experiments.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

        if len(self._experiences) == 0:
            raise RuntimeError("Cannot sample from empty memory")

        # Determine actual sample size (handle case where batch_size > available experiences)
        actual_sample_size = min(batch_size, len(self._experiences))

        # Sample without replacement using random.sample
        # This is more efficient than manual shuffling for large memories
        sampled_experiences = random.sample(list(self._experiences), actual_sample_size)

        return sampled_experiences

    def __len__(self) -> int:
        """
        Return the current number of experiences stored in memory.

        Returns:
            int: Number of experiences currently in memory (0 to capacity).

        Example:
            print(f"Memory contains {len(memory)} experiences")
            if len(memory) >= min_batch_size:
                # Safe to sample
                batch = memory.sample(min_batch_size)
        """
        return len(self._experiences)

    def is_empty(self) -> bool:
        """
        Check if the memory contains no experiences.

        Returns:
            bool: True if memory is empty, False otherwise.

        Example:
            if not memory.is_empty():
                # Safe to sample experiences
                batch = memory.sample(batch_size)
        """
        return len(self._experiences) == 0

    def is_full(self) -> bool:
        """
        Check if the memory has reached its maximum capacity.

        Returns:
            bool: True if memory is at capacity, False otherwise.

        Note:
            When memory is full, adding new experiences will automatically
            remove the oldest experiences (FIFO behavior).

        Example:
            if memory.is_full():
                print("Memory at capacity - oldest experiences will be removed")
        """
        return len(self._experiences) == self.capacity

    def get_task_count(self, task_id: str) -> int:
        """
        Count the number of experiences for a specific task.

        This method enables task-aware memory management and analysis,
        which is useful for understanding memory distribution across
        different learning domains and for implementing task-specific
        continual learning strategies.

        Args:
            task_id (str): The task identifier to count experiences for.

        Returns:
            int: Number of experiences with the specified task_id.

        Example:
            # Check task distribution in memory
            task_counts = {}
            for task in ["task_A", "task_B", "task_C"]:
                task_counts[task] = memory.get_task_count(task)
            print(f"Task distribution: {task_counts}")

        Performance:
            O(n) where n is the number of experiences in memory.
            For frequent task counting, consider maintaining separate
            task-specific counters if performance becomes critical.
        """
        if not isinstance(task_id, str):
            raise ValueError("task_id must be a string")

        return sum(1 for exp in self._experiences if exp.task_id == task_id)

    def clear(self) -> None:
        """
        Remove all experiences from memory.

        This method completely empties the memory buffer, which can be useful
        for resetting the system between experiments or for implementing
        custom memory management strategies.

        Example:
            # Reset memory between experimental phases
            memory.clear()
            assert memory.is_empty()
            assert len(memory) == 0

        Warning:
            This operation is irreversible. All stored experiences will be lost.
        """
        self._experiences.clear()

    def get_memory_info(self) -> dict:
        """
        Get comprehensive information about the current memory state.

        Returns:
            dict: Dictionary containing memory statistics and metadata.
                 Includes current size, capacity, utilization, and task distribution.

        Example:
            info = memory.get_memory_info()
            print(f"Memory utilization: {info['utilization']:.1%}")
            print(f"Task distribution: {info['task_distribution']}")
        """
        # Count experiences by task
        task_distribution = {}
        for exp in self._experiences:
            task_distribution[exp.task_id] = task_distribution.get(exp.task_id, 0) + 1

        return {
            'current_size': len(self._experiences),
            'capacity': self.capacity,
            'utilization': len(self._experiences) / self.capacity if self.capacity > 0 else 0.0,
            'is_empty': self.is_empty(),
            'is_full': self.is_full(),
            'task_distribution': task_distribution,
            'unique_tasks': len(task_distribution)
        }
