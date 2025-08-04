"""
Real-Time Learning (RTL) Mechanisms
===================================

This module implements the synaptic plasticity engine that drives Real-Time Learning (RTL)
within the General Intelligence Framework. It provides a modular library of biologically
plausible learning rules that govern how synaptic weights change based on neural activity.

Unlike traditional deep learning which uses global backpropagation during separate training
phases, RTL implements local, continuous learning where synapses strengthen or weaken based
only on the activity of connected neurons and optional modulatory signals. This approach
enables online, adaptive learning that mirrors biological neural networks.

Key Concepts:
============

**Synaptic Plasticity:** The ability of synapses to strengthen or weaken over time based on
neural activity patterns. This is the fundamental mechanism of learning and memory formation
in both biological brains and our DU Core.

**Local Learning Rules:** Unlike backpropagation, these rules only use information available
locally at each synapse (pre-synaptic activity, post-synaptic activity, and optional
modulatory signals). This makes them biologically plausible and computationally efficient.

**Spike-Timing Dependent Plasticity (STDP):** A learning rule where the precise timing of
spikes matters. If a pre-synaptic neuron fires just before a post-synaptic neuron (causing
it to fire), the connection strengthens (LTP). If it fires just after (didn't contribute),
the connection weakens (LTD).

**Three-Factor Learning:** Extends basic Hebbian learning ("neurons that fire together, wire
together") with a third modulatory signal representing reward, surprise, or attention. This
enables goal-directed and context-dependent learning.

Architecture:
============

The module follows a strategy pattern with a common interface (`PlasticityRuleInterface`)
that allows different learning rules to be used interchangeably within the DU Core. This
modular design enables experimentation with different learning mechanisms.

Components:
- PlasticityRuleInterface: Abstract base class defining the contract for all learning rules
- STDP_Rule: Implements spike-timing dependent plasticity with exponential traces
- ThreeFactor_Hebbian_Rule: Implements modulatory signal-gated Hebbian learning

Example Usage:
=============

    # Create an STDP learning rule
    stdp_rule = STDP_Rule(
        learning_rate_ltp=0.01,    # Long-term potentiation rate
        learning_rate_ltd=0.005,   # Long-term depression rate
        tau_ltp=20.0,              # LTP time constant (ms)
        tau_ltd=20.0               # LTD time constant (ms)
    )

    # Apply the rule to update weights
    weight_updates = stdp_rule.apply(
        pre_spikes=pre_synaptic_spikes,
        post_spikes=post_synaptic_spikes
    )

    # Create a three-factor rule for reward-based learning
    hebbian_rule = ThreeFactor_Hebbian_Rule(learning_rate=0.01)

    # Apply with modulatory signal (e.g., reward)
    weight_updates = hebbian_rule.apply(
        pre_activity=pre_activity,
        post_activity=post_activity,
        modulatory_signal=reward_signal
    )

Technical Implementation:
========================

All learning rules return weight update tensors (Δw) that have the same shape as the
weight matrices they apply to. The DU Core can then apply these updates to its synaptic
weights, enabling continuous adaptation during operation.

The rules are designed for efficient batch processing and are fully compatible with
PyTorch's automatic differentiation system, enabling integration with gradient-based
optimization when needed.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


# =============================================================================
# Abstract Base Class - Plasticity Rule Interface
# =============================================================================

class PlasticityRuleInterface(ABC):
    """
    Abstract base class defining the contract for all synaptic plasticity rules.

    This interface ensures that all learning rules in the RTL engine are interchangeable
    and can be used as "plug-in" components within the DU Core. It enforces a consistent
    API while allowing flexibility in the specific parameters each rule requires.

    The interface follows the strategy pattern, enabling the DU Core to switch between
    different learning mechanisms without code changes. This modularity is essential
    for experimentation and optimization of learning algorithms.

    Design Principles:
    - **Modularity:** All rules implement the same interface for interchangeability
    - **Flexibility:** Uses **kwargs to accommodate different parameter requirements
    - **Efficiency:** Returns tensor updates for batch processing
    - **Compatibility:** Integrates seamlessly with PyTorch ecosystem

    Contract Requirements:
    - Must implement the apply() method
    - Must return weight update tensor with same shape as target weights
    - Should handle batch processing efficiently
    - Should validate input parameters appropriately

    Example Implementation:
        class MyCustomRule(PlasticityRuleInterface):
            def __init__(self, learning_rate: float):
                self.learning_rate = learning_rate

            def apply(self, **kwargs) -> torch.Tensor:
                # Extract required parameters
                pre_activity = kwargs['pre_activity']
                post_activity = kwargs['post_activity']

                # Calculate weight updates
                delta_w = self.learning_rate * pre_activity * post_activity
                return delta_w
    """

    @abstractmethod
    def apply(self, **kwargs) -> torch.Tensor:
        """
        Calculate synaptic weight updates based on neural activity.

        This is the core method that all plasticity rules must implement. It takes
        neural activity data and computes the corresponding changes to synaptic
        weights (Δw) according to the specific learning rule.

        The method uses **kwargs to provide maximum flexibility, as different
        learning rules require different types of input data:
        - STDP rules need pre/post spike timing information
        - Hebbian rules need activity levels and modulatory signals
        - Homeostatic rules might need long-term activity statistics

        Args:
            **kwargs: Flexible keyword arguments containing the neural activity
                     data required by the specific learning rule. Common parameters:
                     - pre_spikes: Pre-synaptic spike trains
                     - post_spikes: Post-synaptic spike trains
                     - pre_activity: Pre-synaptic activity levels
                     - post_activity: Post-synaptic activity levels
                     - modulatory_signal: Third-factor modulatory signal
                     - time_step: Current simulation time step

        Returns:
            torch.Tensor: Weight update tensor (Δw) with the same shape as the
                         synaptic weight matrix it applies to. Positive values
                         indicate strengthening (LTP), negative values indicate
                         weakening (LTD).

        Raises:
            ValueError: If required parameters are missing or have invalid shapes.
            RuntimeError: If the rule is not properly configured.

        Note:
            The returned tensor should be added to the current weights:
            new_weights = current_weights + learning_rule.apply(**activity_data)
        """
        pass


# =============================================================================
# STDP Rule - Spike-Timing Dependent Plasticity
# =============================================================================

class STDP_Rule(PlasticityRuleInterface):
    """
    Implements Spike-Timing Dependent Plasticity (STDP) learning rule.

    STDP is a fundamental biological learning mechanism where the precise timing
    of pre- and post-synaptic spikes determines whether a synapse is strengthened
    or weakened. This rule captures the biological principle that "neurons that
    fire together, wire together" with crucial timing sensitivity.

    Biological Context:
    - **Long-Term Potentiation (LTP):** When a pre-synaptic spike precedes and
      contributes to a post-synaptic spike, the synapse strengthens. This
      reinforces causal relationships in neural circuits.
    - **Long-Term Depression (LTD):** When a pre-synaptic spike follows a
      post-synaptic spike (didn't contribute to firing), the synapse weakens.
      This eliminates spurious connections.

    Technical Implementation:
    The rule uses exponential traces to efficiently track the timing relationships
    between spikes across all synapses simultaneously. For each neuron, we maintain
    a trace that represents its recent spiking history with exponential decay.

    Mathematical Formulation:
    - Pre-synaptic trace: x_pre(t) = Σ exp(-(t-t_spike)/τ_ltp)
    - Post-synaptic trace: x_post(t) = Σ exp(-(t-t_spike)/τ_ltd)
    - LTP update: Δw_ltp = A_ltp * x_pre * post_spike
    - LTD update: Δw_ltd = -A_ltd * x_post * pre_spike

    Parameters:
    - learning_rate_ltp: Strength of long-term potentiation (A_ltp)
    - learning_rate_ltd: Strength of long-term depression (A_ltd)
    - tau_ltp: Time constant for LTP trace decay (typically 10-20ms)
    - tau_ltd: Time constant for LTD trace decay (typically 10-20ms)

    Example Usage:
        # Create STDP rule with typical biological parameters
        stdp = STDP_Rule(
            learning_rate_ltp=0.01,    # 1% weight change for LTP
            learning_rate_ltd=0.005,   # 0.5% weight change for LTD
            tau_ltp=20.0,              # 20ms LTP time window
            tau_ltd=20.0               # 20ms LTD time window
        )

        # Apply to spike data (batch_size=32, num_neurons=100, time_steps=50)
        pre_spikes = torch.rand(50, 32, 100) > 0.95  # Sparse spikes
        post_spikes = torch.rand(50, 32, 100) > 0.95

        weight_updates = stdp.apply(
            pre_spikes=pre_spikes,
            post_spikes=post_spikes
        )
    """

    def __init__(
        self,
        learning_rate_ltp: float,
        learning_rate_ltd: float,
        tau_ltp: float,
        tau_ltd: float
    ):
        """
        Initialize the STDP learning rule with specified parameters.

        Args:
            learning_rate_ltp (float): Learning rate for Long-Term Potentiation.
                                     Controls how much synapses strengthen when
                                     pre-synaptic spikes precede post-synaptic spikes.
                                     Typical range: 0.001 - 0.1

            learning_rate_ltd (float): Learning rate for Long-Term Depression.
                                     Controls how much synapses weaken when
                                     pre-synaptic spikes follow post-synaptic spikes.
                                     Typically smaller than LTP rate.
                                     Typical range: 0.0005 - 0.05

            tau_ltp (float): Time constant for LTP trace decay in milliseconds.
                           Determines the time window for LTP interactions.
                           Typical range: 10-50ms

            tau_ltd (float): Time constant for LTD trace decay in milliseconds.
                           Determines the time window for LTD interactions.
                           Typical range: 10-50ms

        Raises:
            ValueError: If any parameter is negative or zero.
        """
        if learning_rate_ltp <= 0:
            raise ValueError(f"learning_rate_ltp must be positive, got {learning_rate_ltp}")
        if learning_rate_ltd <= 0:
            raise ValueError(f"learning_rate_ltd must be positive, got {learning_rate_ltd}")
        if tau_ltp <= 0:
            raise ValueError(f"tau_ltp must be positive, got {tau_ltp}")
        if tau_ltd <= 0:
            raise ValueError(f"tau_ltd must be positive, got {tau_ltd}")

        self.learning_rate_ltp = learning_rate_ltp
        self.learning_rate_ltd = learning_rate_ltd
        self.tau_ltp = tau_ltp
        self.tau_ltd = tau_ltd

        # Pre-compute decay factors for efficiency
        # Note: These will be adjusted based on actual time step when apply() is called
        self.decay_ltp = None
        self.decay_ltd = None

        # Initialize trace variables (will be set during first apply() call)
        self.pre_trace = None
        self.post_trace = None

    def apply(self, **kwargs) -> torch.Tensor:
        """
        Apply STDP learning rule to calculate weight updates.

        This method implements the core STDP algorithm using exponential traces
        to efficiently compute timing-dependent weight changes for all synapses
        simultaneously.

        Algorithm:
        1. Update exponential traces for pre- and post-synaptic activity
        2. Calculate LTP updates: strengthen synapses when pre precedes post
        3. Calculate LTD updates: weaken synapses when post precedes pre
        4. Combine updates and return weight change tensor

        Args:
            **kwargs: Must contain:
                pre_spikes (torch.Tensor): Pre-synaptic spike trains with shape
                                         [time_steps, batch_size, num_pre_neurons]
                post_spikes (torch.Tensor): Post-synaptic spike trains with shape
                                          [time_steps, batch_size, num_post_neurons]
                dt (float, optional): Time step size in milliseconds. Default: 1.0

        Returns:
            torch.Tensor: Weight update matrix with shape
                         [num_pre_neurons, num_post_neurons]. Positive values
                         indicate strengthening (LTP), negative values indicate
                         weakening (LTD).

        Raises:
            ValueError: If required parameters are missing or have incompatible shapes.
        """
        # Extract required parameters
        if 'pre_spikes' not in kwargs:
            raise ValueError("STDP rule requires 'pre_spikes' parameter")
        if 'post_spikes' not in kwargs:
            raise ValueError("STDP rule requires 'post_spikes' parameter")

        pre_spikes = kwargs['pre_spikes']
        post_spikes = kwargs['post_spikes']
        dt = kwargs.get('dt', 1.0)  # Default time step: 1ms

        # Handle both 2D (single time step) and 3D (multi time step) inputs
        if pre_spikes.dim() == 2:
            # Single time step: add time dimension
            pre_spikes = pre_spikes.unsqueeze(0)  # [1, batch, neurons]
        if post_spikes.dim() == 2:
            # Single time step: add time dimension
            post_spikes = post_spikes.unsqueeze(0)  # [1, batch, neurons]

        # Validate input shapes
        if pre_spikes.dim() != 3:
            raise ValueError(f"pre_spikes must be 2D [batch, neurons] or 3D [time, batch, neurons], got shape {pre_spikes.shape}")
        if post_spikes.dim() != 3:
            raise ValueError(f"post_spikes must be 2D [batch, neurons] or 3D [time, batch, neurons], got shape {post_spikes.shape}")
        if pre_spikes.shape[0] != post_spikes.shape[0]:
            raise ValueError(f"Time dimensions must match: pre={pre_spikes.shape[0]}, post={post_spikes.shape[0]}")
        if pre_spikes.shape[1] != post_spikes.shape[1]:
            raise ValueError(f"Batch dimensions must match: pre={pre_spikes.shape[1]}, post={post_spikes.shape[1]}")

        time_steps, batch_size, num_pre = pre_spikes.shape
        _, _, num_post = post_spikes.shape
        device = pre_spikes.device

        # Initialize or update decay factors based on time step
        self.decay_ltp = torch.exp(torch.tensor(-dt / self.tau_ltp, device=device))
        self.decay_ltd = torch.exp(torch.tensor(-dt / self.tau_ltd, device=device))

        # Initialize traces if first call
        if self.pre_trace is None:
            self.pre_trace = torch.zeros(batch_size, num_pre, device=device)
            self.post_trace = torch.zeros(batch_size, num_post, device=device)

        # Ensure traces have correct shape for current batch
        if self.pre_trace.shape != (batch_size, num_pre):
            self.pre_trace = torch.zeros(batch_size, num_pre, device=device)
        if self.post_trace.shape != (batch_size, num_post):
            self.post_trace = torch.zeros(batch_size, num_post, device=device)

        # Initialize weight update accumulator
        weight_updates = torch.zeros(num_pre, num_post, device=device)

        # Process each time step
        for t in range(time_steps):
            current_pre = pre_spikes[t]  # [batch_size, num_pre]
            current_post = post_spikes[t]  # [batch_size, num_post]

            # LTP: Strengthen synapses when pre-trace is high and post spikes occur
            # This captures cases where pre-synaptic activity preceded post-synaptic firing
            ltp_update = torch.outer(
                self.pre_trace.mean(dim=0),  # Average across batch
                current_post.float().mean(dim=0)     # Average across batch (convert bool to float)
            ) * self.learning_rate_ltp

            # LTD: Weaken synapses when post-trace is high and pre spikes occur
            # This captures cases where post-synaptic activity preceded pre-synaptic firing
            ltd_update = torch.outer(
                current_pre.float().mean(dim=0),     # Average across batch (convert bool to float)
                self.post_trace.mean(dim=0)  # Average across batch
            ) * self.learning_rate_ltd

            # Accumulate weight updates
            weight_updates += ltp_update - ltd_update

            # Update traces with exponential decay and new spikes
            self.pre_trace = self.pre_trace * self.decay_ltp + current_pre.float()
            self.post_trace = self.post_trace * self.decay_ltd + current_post.float()

        return weight_updates


# =============================================================================
# Three-Factor Hebbian Rule - Modulatory Signal-Gated Learning
# =============================================================================

class ThreeFactor_Hebbian_Rule(PlasticityRuleInterface):
    """
    Implements a Three-Factor Hebbian learning rule with modulatory signal gating.

    This rule extends the basic Hebbian principle ("neurons that fire together, wire
    together") with a crucial third factor: a modulatory signal that gates when
    learning should occur. This enables goal-directed, context-dependent learning
    that goes beyond simple correlation-based adaptation.

    Biological Context:
    In biological neural networks, neuromodulators like dopamine, acetylcholine,
    and norepinephrine provide contextual signals that indicate when learning
    should be enhanced or suppressed. These signals can represent:
    - **Reward/Punishment:** Dopamine signals reward prediction errors
    - **Attention/Salience:** Acetylcholine indicates important stimuli
    - **Arousal/Context:** Norepinephrine modulates learning based on behavioral state

    Technical Implementation:
    The rule computes weight updates as the product of three factors:
    1. Pre-synaptic activity (correlation factor)
    2. Post-synaptic activity (correlation factor)
    3. Modulatory signal (gating factor)

    Mathematical Formulation:
    Δw = η * pre_activity * post_activity * modulatory_signal

    Where:
    - η (eta) is the base learning rate
    - pre_activity and post_activity can be spike rates, membrane potentials, or other activity measures
    - modulatory_signal is a scalar that scales learning (positive for enhancement, negative for suppression)

    Key Advantages:
    - **Context Sensitivity:** Learning only occurs when the modulatory signal indicates it's appropriate
    - **Goal-Directed Learning:** Can implement reward-based learning by using reward as the modulatory signal
    - **Computational Efficiency:** Simple element-wise operations make it very fast
    - **Biological Plausibility:** Directly models known neuromodulatory mechanisms

    Example Usage:
        # Create three-factor rule for reward-based learning
        hebbian = ThreeFactor_Hebbian_Rule(learning_rate=0.01)

        # Apply with reward signal (positive for good outcomes)
        weight_updates = hebbian.apply(
            pre_activity=pre_neuron_rates,      # [batch_size, num_pre]
            post_activity=post_neuron_rates,    # [batch_size, num_post]
            modulatory_signal=reward_value      # scalar reward signal
        )

        # Apply with attention signal (high when stimulus is important)
        weight_updates = hebbian.apply(
            pre_activity=sensory_input,
            post_activity=cortical_response,
            modulatory_signal=attention_level
        )
    """

    def __init__(self, learning_rate: float, meta_learning_rate: float = 0.1):
        """
        Initialize the Three-Factor Hebbian learning rule with meta-plasticity support.

        This enhanced version supports System Potentiation through adaptive learning rates
        that change based on the system's performance. Poor performance increases plasticity
        to learn faster, while good performance stabilizes learning to preserve knowledge.

        Args:
            learning_rate (float): Base learning rate (η₀) that serves as the foundation
                                 for weight updates. This is the learning rate used when
                                 the system performs at chance level (50% accuracy).
                                 Typical range: 0.001 - 0.1

            meta_learning_rate (float, optional): Meta-plasticity rate (α) that controls
                                                 how strongly the learning rate adapts to
                                                 performance feedback. Higher values make
                                                 the system more responsive to performance
                                                 changes. Typical range: 0.01 - 0.5
                                                 Default: 0.1

        Raises:
            ValueError: If learning_rate is not positive.
            ValueError: If meta_learning_rate is not positive.

        Note:
            The current learning rate is calculated as:
            η_current = η₀ × (1 + α × surprise_signal)

            Where surprise_signal ∈ [0, 1] represents the error rate:
            - surprise_signal = 0.0 → η_current = η₀ (perfect performance)
            - surprise_signal = 0.5 → η_current = η₀ × 1.05 (chance performance)
            - surprise_signal = 1.0 → η_current = η₀ × 1.1 (worst performance)
        """
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")

        if meta_learning_rate < 0:
            raise ValueError(f"meta_learning_rate must be non-negative, got {meta_learning_rate}")

        # Base learning rate (foundation for adaptation)
        self.base_learning_rate = learning_rate

        # Meta-plasticity rate (controls adaptation strength)
        self.meta_learning_rate = meta_learning_rate

        # Current adaptive learning rate (starts at base rate)
        self.current_learning_rate = learning_rate

        # Statistics for research and debugging
        self._adaptation_history = []
        self._surprise_history = []

    def update_learning_rate(self, surprise_signal: float) -> None:
        """
        Update the learning rate based on performance feedback (meta-plasticity).

        This method implements the core System Potentiation mechanism by adapting
        the learning rate based on how well the system is currently performing.
        This enables the system to "learn how to learn" more effectively.

        Algorithm:
        η_current = η₀ × (1 + α × surprise_signal)

        Where:
        - η₀ is the base learning rate
        - α is the meta-learning rate
        - surprise_signal ∈ [0, 1] is the current error rate

        Args:
            surprise_signal (float): Current surprise signal from performance tracking.
                                   Range: [0.0, 1.0]
                                   - 0.0: Perfect performance (all predictions correct)
                                   - 0.5: Chance performance (50% correct)
                                   - 1.0: Worst performance (all predictions wrong)

        Raises:
            ValueError: If surprise_signal is not in valid range [0.0, 1.0].

        Example:
            # System performing poorly (80% error rate)
            rule.update_learning_rate(0.8)  # Increases learning rate

            # System performing well (10% error rate)
            rule.update_learning_rate(0.1)  # Slightly increases learning rate

            # Perfect performance
            rule.update_learning_rate(0.0)  # Maintains base learning rate
        """
        if not (0.0 <= surprise_signal <= 1.0):
            raise ValueError(f"surprise_signal must be in range [0.0, 1.0], got {surprise_signal}")

        # Calculate new adaptive learning rate
        # Higher surprise (poor performance) → higher learning rate
        # Lower surprise (good performance) → learning rate closer to base
        self.current_learning_rate = self.base_learning_rate * (
            1.0 + self.meta_learning_rate * surprise_signal
        )

        # Store statistics for analysis
        self._surprise_history.append(surprise_signal)
        self._adaptation_history.append(self.current_learning_rate)

        # Keep history bounded to prevent memory growth
        max_history = 1000
        if len(self._surprise_history) > max_history:
            self._surprise_history = self._surprise_history[-max_history:]
            self._adaptation_history = self._adaptation_history[-max_history:]

    def get_meta_plasticity_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about meta-plasticity adaptation for analysis.

        Returns:
            Dict[str, Any]: Dictionary containing meta-plasticity metrics:
                - 'current_learning_rate': Current adaptive learning rate
                - 'base_learning_rate': Base (initial) learning rate
                - 'meta_learning_rate': Meta-plasticity adaptation rate
                - 'adaptation_factor': Current learning rate / base learning rate
                - 'recent_surprise': Most recent surprise signal (if available)
                - 'avg_surprise': Average surprise over recent history
                - 'adaptation_count': Number of adaptations performed
                - 'surprise_history': List of recent surprise signals
                - 'learning_rate_history': List of recent learning rates
        """
        stats = {
            'current_learning_rate': self.current_learning_rate,
            'base_learning_rate': self.base_learning_rate,
            'meta_learning_rate': self.meta_learning_rate,
            'adaptation_factor': self.current_learning_rate / self.base_learning_rate,
            'adaptation_count': len(self._adaptation_history)
        }

        if len(self._surprise_history) > 0:
            stats.update({
                'recent_surprise': self._surprise_history[-1],
                'avg_surprise': sum(self._surprise_history) / len(self._surprise_history),
                'surprise_history': self._surprise_history.copy(),
                'learning_rate_history': self._adaptation_history.copy()
            })
        else:
            stats.update({
                'recent_surprise': None,
                'avg_surprise': None,
                'surprise_history': [],
                'learning_rate_history': []
            })

        return stats

    def apply(self, **kwargs) -> torch.Tensor:
        """
        Apply Three-Factor Hebbian learning rule to calculate weight updates.

        This method implements the core three-factor learning algorithm by computing
        the element-wise product of pre-synaptic activity, post-synaptic activity,
        and the modulatory signal, scaled by the learning rate.

        The modulatory signal acts as a "gate" that determines when and how much
        learning occurs:
        - Positive values enhance learning (strengthen correlations)
        - Negative values can implement "anti-learning" (weaken correlations)
        - Zero values prevent any learning from occurring
        - Magnitude controls the strength of the learning effect

        Args:
            **kwargs: Must contain:
                pre_activity (torch.Tensor): Pre-synaptic activity levels with shape
                                           [batch_size, num_pre_neurons]. Can represent
                                           firing rates, membrane potentials, or other
                                           activity measures.

                                           OR

                pre_spikes (torch.Tensor): Pre-synaptic spike trains (alternative to pre_activity)

                post_activity (torch.Tensor): Post-synaptic activity levels with shape
                                             [batch_size, num_post_neurons]. Should use
                                             the same activity measure as pre_activity.

                                             OR

                post_spikes (torch.Tensor): Post-synaptic spike trains (alternative to post_activity)

                modulatory_signal (float): Scalar modulatory signal that gates learning.
                                         Represents contextual information like reward,
                                         attention, or behavioral state.

        Returns:
            torch.Tensor: Weight update matrix with shape [num_pre_neurons, num_post_neurons].
                         Each element represents the change in synaptic strength between
                         the corresponding pre- and post-synaptic neurons.

        Raises:
            ValueError: If required parameters are missing or have incompatible shapes.
        """
        # Extract required parameters with flexibility for spike or activity inputs
        if 'modulatory_signal' not in kwargs:
            raise ValueError("Three-Factor Hebbian rule requires 'modulatory_signal' parameter")

        # Handle pre-synaptic input (activity or spikes)
        if 'pre_activity' in kwargs:
            pre_activity = kwargs['pre_activity']
        elif 'pre_spikes' in kwargs:
            pre_activity = kwargs['pre_spikes'].float()  # Convert spikes to activity
        else:
            raise ValueError("Three-Factor Hebbian rule requires either 'pre_activity' or 'pre_spikes' parameter")

        # Handle post-synaptic input (activity or spikes)
        if 'post_activity' in kwargs:
            post_activity = kwargs['post_activity']
        elif 'post_spikes' in kwargs:
            post_activity = kwargs['post_spikes'].float()  # Convert spikes to activity
        else:
            raise ValueError("Three-Factor Hebbian rule requires either 'post_activity' or 'post_spikes' parameter")

        modulatory_signal = kwargs['modulatory_signal']

        # Validate input shapes and types
        if not isinstance(pre_activity, torch.Tensor):
            raise ValueError(f"pre_activity must be a torch.Tensor, got {type(pre_activity)}")
        if not isinstance(post_activity, torch.Tensor):
            raise ValueError(f"post_activity must be a torch.Tensor, got {type(post_activity)}")
        if not isinstance(modulatory_signal, (int, float)):
            raise ValueError(f"modulatory_signal must be a scalar number, got {type(modulatory_signal)}")

        if pre_activity.dim() != 2:
            raise ValueError(f"pre_activity must be 2D tensor [batch, neurons], got shape {pre_activity.shape}")
        if post_activity.dim() != 2:
            raise ValueError(f"post_activity must be 2D tensor [batch, neurons], got shape {post_activity.shape}")
        if pre_activity.shape[0] != post_activity.shape[0]:
            raise ValueError(f"Batch dimensions must match: pre={pre_activity.shape[0]}, post={post_activity.shape[0]}")

        # Extract dimensions
        batch_size, num_pre = pre_activity.shape
        _, num_post = post_activity.shape

        # Compute weight updates using three-factor rule
        # Δw = η * <pre_activity> * <post_activity> * modulatory_signal
        # where <·> denotes average across batch dimension

        # Average activity across batch to get population-level correlations
        avg_pre_activity = pre_activity.mean(dim=0)    # [num_pre]
        avg_post_activity = post_activity.mean(dim=0)  # [num_post]

        # Compute outer product to get all pairwise correlations
        correlation_matrix = torch.outer(avg_pre_activity, avg_post_activity)  # [num_pre, num_post]

        # Apply three-factor rule: scale by current adaptive learning rate and modulatory signal
        # Uses current_learning_rate which adapts based on performance (meta-plasticity)
        weight_updates = self.current_learning_rate * correlation_matrix * modulatory_signal

        return weight_updates
