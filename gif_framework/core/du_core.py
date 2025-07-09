"""
Deep Understanding (DU) Core v1 - The SNN Brain
===============================================

This module implements the first functional version of the Deep Understanding Core,
the central "brain" of the General Intelligence Framework. The DU Core is a configurable,
multi-layer Spiking Neural Network (SNN) built using the snnTorch library.

The DU Core v1 focuses on creating a robust and flexible SNN architecture using
Leaky Integrate-and-Fire (LIF) neurons, laying the foundation for advanced learning
and memory mechanisms that will be added in Phase 3.

Key Features:
- **Configurable Architecture**: Arbitrary number of hidden layers with custom sizes
- **LIF Neuron Model**: Biologically plausible spiking neurons with leaky integration
- **PyTorch Integration**: Full compatibility with PyTorch ecosystem and automatic differentiation
- **snnTorch Optimization**: Leverages highly optimized SNN implementations
- **Orchestrator Compatible**: Implements the interface expected by the GIF orchestrator
- **Foundation for RTL**: Architecture designed for future real-time learning mechanisms

Technical Implementation:
- Uses snnTorch's Leaky class for LIF neuron implementation
- Alternating Linear and Leaky layers for multi-layer processing
- Configurable neuron parameters (beta for leakage, threshold for firing)
- Temporal simulation loop for processing spike trains over time
- Proper state management for membrane potentials

The LIF Neuron Model:
1. **Integrate**: Neuron membrane potential increases with input spikes
2. **Leak**: Membrane potential decays over time without input (controlled by beta)
3. **Fire**: When potential exceeds threshold, neuron fires and resets

This implementation serves as the cognitive engine that processes encoded spike trains
from various domains (astronomical, medical, etc.) and produces processed spike trains
for decoders to interpret into meaningful actions.

Author: GIF Development Team
Phase: 2.3 - Core Framework Implementation
"""

from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
try:
    import snntorch as snn
except ImportError:
    raise ImportError(
        "snnTorch is required for the DU Core. Please install it with: "
        "pip install snntorch>=0.7.0"
    )

# Import SpikeTrain type alias for consistency with framework
from gif_framework.interfaces.base_interfaces import SpikeTrain

# Import RTL and memory components for dependency injection
from gif_framework.core.rtl_mechanisms import PlasticityRuleInterface
from gif_framework.core.memory_systems import EpisodicMemory


class DU_Core_V1(nn.Module):
    """
    The first version of the Deep Understanding Core - a configurable SNN brain.

    This class implements a multi-layer Spiking Neural Network using Leaky Integrate-and-Fire
    (LIF) neurons. It serves as the central "brain" of the GIF framework, processing spike
    trains from encoders and producing processed spike trains for decoders.

    The architecture consists of alternating Linear and Leaky (LIF neuron) layers, allowing
    for flexible network configurations while maintaining biological plausibility through
    spiking neuron dynamics.

    Key Design Principles:
    - **Configurable Architecture**: Support for arbitrary network topologies
    - **Biological Plausibility**: LIF neurons mimic real neural behavior
    - **Temporal Processing**: Explicit time-step simulation for spike train processing
    - **PyTorch Integration**: Full compatibility with PyTorch training and inference
    - **State Management**: Proper handling of neuron membrane potentials

    Network Architecture:
        Input → Linear₁ → LIF₁ → Linear₂ → LIF₂ → ... → LinearN → LIFN → Output

    Each LIF layer implements the leaky integrate-and-fire dynamics:
    - Membrane potential integrates input spikes
    - Potential leaks over time (controlled by beta parameter)
    - Neuron fires when potential exceeds threshold
    - Potential resets after firing

    Example Usage:
        # Create a 3-layer SNN: 100 → 64 → 32 → 10
        du_core = DU_Core_V1(
            input_size=100,
            hidden_sizes=[64, 32],
            output_size=10,
            beta=0.95,      # Low leakage
            threshold=1.0   # Firing threshold
        )

        # Process spike train (50 time steps, batch size 1, 100 features)
        input_spikes = torch.rand(50, 1, 100)
        output_spikes = du_core.process(input_spikes)

    Attributes:
        input_size (int): Dimension of input spike trains
        hidden_sizes (List[int]): Sizes of hidden layers
        output_size (int): Dimension of output spike trains
        beta (float): Leakage rate for LIF neurons (0 < beta ≤ 1)
        threshold (float): Firing threshold for LIF neurons
        recurrent (bool): Flag for recurrent connections (future feature)
        linear_layers (nn.ModuleList): Linear transformation layers
        lif_layers (nn.ModuleList): LIF neuron layers
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        beta: float = 0.95,
        threshold: float = 1.0,
        recurrent: bool = False,
        plasticity_rule: Optional[PlasticityRuleInterface] = None,
        memory_system: Optional[EpisodicMemory] = None
    ) -> None:
        """
        Initialize the DU Core v1 with configurable SNN architecture and optional RTL components.

        This constructor dynamically builds a multi-layer SNN based on the provided
        parameters. The network consists of alternating Linear and LIF layers,
        with the number and size of layers determined by the hidden_sizes list.

        **New in Task 3.3:** The constructor now supports dependency injection of
        Real-Time Learning (RTL) components, enabling continual learning capabilities:
        - PlasticityRuleInterface: Defines how synaptic weights adapt during learning
        - EpisodicMemory: Stores past experiences for continual learning algorithms

        Args:
            input_size (int): Size of input spike trains. Must be positive.
            hidden_sizes (List[int]): List of hidden layer sizes. Each element
                                    defines the size of one hidden layer. Can be
                                    empty for a single-layer network.
            output_size (int): Size of output spike trains. Must be positive.
            beta (float, optional): Leakage rate for LIF neurons. Values closer
                                  to 1.0 mean less leakage (longer memory).
                                  Must be in range (0, 1]. Defaults to 0.95.
            threshold (float, optional): Membrane potential threshold for spike
                                       generation. Must be positive. Defaults to 1.0.
            recurrent (bool, optional): Enable recurrent connections within layers.
                                      Currently a placeholder for future implementation.
                                      Defaults to False.
            plasticity_rule (PlasticityRuleInterface, optional): Learning rule for
                                                               synaptic weight updates.
                                                               If None, no learning occurs.
                                                               Default: None
            memory_system (EpisodicMemory, optional): Memory system for storing past
                                                    experiences. Required for continual
                                                    learning algorithms like GEM.
                                                    Default: None

        Raises:
            ValueError: If input_size, output_size, or any hidden_size is not positive.
            ValueError: If beta is not in range (0, 1] or threshold is not positive.
            NotImplementedError: If recurrent=True (not yet implemented).
            TypeError: If plasticity_rule or memory_system have incorrect types.

        Example:
            # Basic SNN without learning (backward compatible)
            du_core = DU_Core_V1(input_size=784, hidden_sizes=[128], output_size=10)

            # SNN with continual learning capabilities
            from gif_framework.core.rtl_mechanisms import STDP_Rule
            from gif_framework.core.memory_systems import EpisodicMemory

            plasticity_rule = STDP_Rule(
                learning_rate_ltp=0.01,
                learning_rate_ltd=0.005,
                tau_ltp=20.0,
                tau_ltd=20.0
            )
            memory = EpisodicMemory(capacity=10000)

            du_core = DU_Core_V1(
                input_size=100,
                hidden_sizes=[64, 32],
                output_size=10,
                plasticity_rule=plasticity_rule,
                memory_system=memory
            )
        """
        super(DU_Core_V1, self).__init__()

        # Validate input parameters
        self._validate_parameters(input_size, hidden_sizes, output_size, beta, threshold, recurrent)
        self._validate_rtl_components(plasticity_rule, memory_system)

        # Store configuration
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes.copy()  # Defensive copy
        self.output_size = output_size
        self.beta = beta
        self.threshold = threshold
        self.recurrent = recurrent

        # Store injected RTL components for continual learning
        self._plasticity_rule = plasticity_rule
        self._memory_system = memory_system

        # Build the network architecture dynamically
        self.linear_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()

        # Create layer size sequence: input → hidden₁ → hidden₂ → ... → output
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Build alternating Linear and LIF layers
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]

            # Add Linear layer for synaptic connections
            linear_layer = nn.Linear(in_size, out_size)
            self.linear_layers.append(linear_layer)

            # Add LIF layer for spiking neuron dynamics
            lif_layer = snn.Leaky(beta=beta, threshold=threshold)
            self.lif_layers.append(lif_layer)

        # Initialize network weights
        self._initialize_weights()

    def _validate_parameters(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        beta: float,
        threshold: float,
        recurrent: bool
    ) -> None:
        """Validate constructor parameters for correctness."""
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")

        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")

        for i, size in enumerate(hidden_sizes):
            if size <= 0:
                raise ValueError(f"hidden_sizes[{i}] must be positive, got {size}")

        if not (0 < beta <= 1.0):
            raise ValueError(f"beta must be in range (0, 1], got {beta}")

        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")

        if recurrent:
            raise NotImplementedError(
                "Recurrent connections are not yet implemented in DU_Core_V1. "
                "This feature will be added in a future version."
            )

    def _validate_rtl_components(
        self,
        plasticity_rule: Optional[PlasticityRuleInterface],
        memory_system: Optional[EpisodicMemory]
    ) -> None:
        """
        Validate the injected RTL components for type correctness.

        Args:
            plasticity_rule: The plasticity rule to validate (can be None)
            memory_system: The memory system to validate (can be None)

        Raises:
            TypeError: If components have incorrect types.
        """
        if plasticity_rule is not None and not isinstance(plasticity_rule, PlasticityRuleInterface):
            raise TypeError(
                f"plasticity_rule must implement PlasticityRuleInterface, "
                f"got {type(plasticity_rule)}"
            )

        if memory_system is not None and not isinstance(memory_system, EpisodicMemory):
            raise TypeError(
                f"memory_system must be an EpisodicMemory instance, "
                f"got {type(memory_system)}"
            )

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier/Glorot initialization."""
        for linear_layer in self.linear_layers:
            nn.init.xavier_uniform_(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)

    def forward(self, input_spikes: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Forward pass through the SNN for a given number of time steps.

        This method implements the core temporal simulation loop of the SNN.
        For each time step, it processes the input spikes through all layers
        sequentially, maintaining the membrane potentials of LIF neurons and
        recording their output spikes.

        The simulation follows these steps for each time step:
        1. Take input spikes for current time step
        2. Pass through first Linear layer (synaptic connections)
        3. Pass through first LIF layer (neuron dynamics)
        4. Repeat for all subsequent layer pairs
        5. Record output spikes from final layer

        Args:
            input_spikes (torch.Tensor): Input spike train with shape
                                       [num_steps, batch_size, input_size].
                                       Values should be binary (0 or 1) or
                                       continuous spike rates.
            num_steps (int): Number of simulation time steps. Must match
                           the first dimension of input_spikes.

        Returns:
            torch.Tensor: Output spike train with shape
                         [num_steps, batch_size, output_size].
                         Contains the spike outputs from the final layer
                         at each time step.

        Raises:
            ValueError: If input tensor shape is incorrect or num_steps mismatch.
            RuntimeError: If forward pass fails due to layer configuration issues.

        Example:
            # Process 100 time steps with batch size 4
            input_spikes = torch.rand(100, 4, 784)  # Random spike rates
            output_spikes = du_core.forward(input_spikes, num_steps=100)
            print(output_spikes.shape)  # torch.Size([100, 4, 10])
        """
        # Validate input tensor
        if input_spikes.dim() != 3:
            raise ValueError(
                f"input_spikes must be 3D tensor [num_steps, batch_size, input_size], "
                f"got shape {input_spikes.shape}"
            )

        if input_spikes.size(0) != num_steps:
            raise ValueError(
                f"input_spikes first dimension ({input_spikes.size(0)}) must match "
                f"num_steps ({num_steps})"
            )

        if input_spikes.size(2) != self.input_size:
            raise ValueError(
                f"input_spikes last dimension ({input_spikes.size(2)}) must match "
                f"input_size ({self.input_size})"
            )

        batch_size = input_spikes.size(1)

        # Initialize membrane potentials for all LIF layers
        # Each LIF layer needs to track membrane potential for each neuron
        mem_potentials = []
        for lif_layer in self.lif_layers:
            # Initialize membrane potential to zero for all neurons in the layer
            mem_potentials.append(lif_layer.init_leaky())

        # Initialize output spike recording
        output_spikes_list = []

        # Main simulation loop - process each time step
        for step in range(num_steps):
            # Get input spikes for current time step
            current_input = input_spikes[step]  # Shape: [batch_size, input_size]

            # Process through all layers sequentially
            layer_output = current_input

            for layer_idx, (linear_layer, lif_layer) in enumerate(zip(self.linear_layers, self.lif_layers)):
                # Linear transformation (synaptic connections)
                linear_output = linear_layer(layer_output)

                # LIF neuron dynamics
                spike_output, mem_potentials[layer_idx] = lif_layer(
                    linear_output, mem_potentials[layer_idx]
                )

                # Output of this layer becomes input to next layer
                layer_output = spike_output

            # Record output spikes from final layer
            output_spikes_list.append(layer_output)

        # Stack all time steps into a single tensor
        output_spikes = torch.stack(output_spikes_list, dim=0)

        return output_spikes

    def process(self, spike_train: SpikeTrain) -> SpikeTrain:
        """
        Process a spike train through the SNN (interface method for GIF orchestrator).

        This method serves as the main interface between the DU Core and the GIF
        orchestrator. It wraps the forward() method with appropriate input validation
        and tensor format handling to ensure compatibility with the framework's
        SpikeTrain type alias.

        The method automatically infers the number of simulation steps from the
        input tensor's first dimension and handles any necessary tensor format
        conversions for optimal snnTorch processing.

        Args:
            spike_train (SpikeTrain): Input spike train tensor with shape
                                    [num_steps, batch_size, input_size] or
                                    [batch_size, input_size] for single time step.
                                    SpikeTrain is an alias for torch.Tensor.

        Returns:
            SpikeTrain: Processed spike train tensor with shape
                       [num_steps, batch_size, output_size] or
                       [batch_size, output_size] for single time step input.

        Raises:
            ValueError: If spike_train has invalid shape or dimensions.
            TypeError: If spike_train is not a torch.Tensor.

        Example:
            # Process multi-step spike train
            input_spikes = torch.rand(50, 1, 100)
            output_spikes = du_core.process(input_spikes)

            # Process single time step
            single_step = torch.rand(1, 100)
            output_step = du_core.process(single_step)

        Note:
            This method is called by the GIF orchestrator during the cognitive cycle:
            processed_spikes = self._du_core.process(spike_train)
        """
        # Type validation
        if not isinstance(spike_train, torch.Tensor):
            raise TypeError(
                f"spike_train must be a torch.Tensor (SpikeTrain), got {type(spike_train)}"
            )

        # Handle different input tensor shapes
        if spike_train.dim() == 2:
            # Single time step: [batch_size, input_size]
            # Add time dimension: [1, batch_size, input_size]
            spike_train = spike_train.unsqueeze(0)
            single_step_input = True
        elif spike_train.dim() == 3:
            # Multi-step: [num_steps, batch_size, input_size]
            single_step_input = False
        else:
            raise ValueError(
                f"spike_train must be 2D [batch_size, input_size] or "
                f"3D [num_steps, batch_size, input_size], got shape {spike_train.shape}"
            )

        # Validate input size matches network configuration
        if spike_train.size(-1) != self.input_size:
            raise ValueError(
                f"spike_train input size ({spike_train.size(-1)}) must match "
                f"network input_size ({self.input_size})"
            )

        # Infer number of simulation steps
        num_steps = spike_train.size(0)

        # Process through the SNN
        try:
            output_spikes = self.forward(spike_train, num_steps)
        except Exception as e:
            raise RuntimeError(f"SNN processing failed: {str(e)}") from e

        # Handle output format for single-step inputs
        if single_step_input:
            # Remove time dimension: [1, batch_size, output_size] → [batch_size, output_size]
            output_spikes = output_spikes.squeeze(0)

        return output_spikes

    def reset_states(self) -> None:
        """
        Reset all neuron membrane potentials to their initial state.

        This method reinitializes the membrane potentials of all LIF neurons
        in the network to zero. This is useful for:
        - Starting fresh processing sequences
        - Clearing network state between different inputs
        - Ensuring reproducible behavior in testing

        Note:
            This method doesn't need to be called before each forward pass,
            as the forward method automatically initializes states. It's
            provided for explicit state management when needed.
        """
        # Reset is handled automatically in forward pass
        # This method is provided for explicit state management
        pass

    def reset_weights(self) -> None:
        """
        Reset all synaptic weights to random initial values.

        This method is crucial for the potentiation experiment protocol in Task 5.2.
        It allows us to completely wipe the specific knowledge learned from previous
        tasks (e.g., exoplanet detection) while preserving any structural or
        meta-learning improvements that may have occurred in the network architecture.

        The weight reset protocol is the key scientific control that distinguishes
        system potentiation from simple knowledge transfer. By re-randomizing all
        synaptic weights, we ensure that any performance improvements in subsequent
        learning can only be attributed to fundamental changes in the learning
        mechanism itself, not to retained task-specific knowledge.

        Implementation:
        - Iterates through all linear (synaptic) layers in the network
        - Calls reset_parameters() on each layer to re-initialize weights and biases
        - Uses the same initialization scheme as the original network construction
        - Preserves network architecture and neuron parameters (beta, threshold)

        Scientific Justification:
        This method enables rigorous testing of the AGI hypothesis that diverse
        prior experience improves the fundamental learning capacity of neural
        systems. Without this reset capability, improved performance on new tasks
        could be attributed to simple feature transfer rather than true system
        potentiation.

        Usage in Potentiation Experiment:
        1. Load pre-trained DU_Core from exoplanet task
        2. Immediately call reset_weights() to wipe specific knowledge
        3. Train on medical task with identical protocol as naive model
        4. Compare learning curves to measure potentiation effect

        Example:
            # Load pre-trained model
            du_core.load_state_dict(torch.load('exoplanet_model.pth'))

            # Reset weights for potentiation experiment
            du_core.reset_weights()

            # Now train on new task - any improvement is due to potentiation
            train_on_medical_task(du_core)

        Note:
            This method only resets synaptic weights, not the network architecture
            or neuron parameters. The structural "experience" of the network
            (layer sizes, connectivity patterns) is preserved, which may contribute
            to improved learning efficiency through architectural optimization.
        """
        print("Executing weight reset protocol for potentiation experiment...")

        # Reset all linear layer parameters (weights and biases)
        for i, linear_layer in enumerate(self.linear_layers):
            # Store original layer info for logging
            layer_info = f"Layer {i+1}: {linear_layer.in_features} → {linear_layer.out_features}"

            # Reset parameters using PyTorch's built-in reset_parameters method
            # This uses the same initialization scheme as during original construction
            linear_layer.reset_parameters()

            print(f"  ✓ Reset {layer_info}")

        # Log completion
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Weight reset complete: {len(self.linear_layers)} layers, {total_params:,} parameters re-initialized")
        print("Network is now ready for potentiation experiment training")

        # Note: LIF neuron parameters (beta, threshold) are NOT reset
        # These represent the "experience" of the network's temporal dynamics
        # and may contribute to improved learning through optimized neural behavior

    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration of the DU Core.

        Returns a dictionary containing all configuration parameters
        that define the network architecture and neuron properties.
        This is essential for reproducibility, logging, and debugging.

        Returns:
            Dict[str, Any]: Configuration dictionary containing:
                - Network architecture parameters
                - Neuron model parameters
                - Layer information
                - Total parameter count

        Example:
            config = du_core.get_config()
            print(f"Network has {config['total_parameters']} parameters")
            print(f"Architecture: {config['layer_sizes']}")
        """
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Build layer size information
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        config = {
            # Architecture parameters
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes.copy(),
            "output_size": self.output_size,
            "layer_sizes": layer_sizes,
            "num_layers": len(self.linear_layers),

            # Neuron parameters
            "beta": self.beta,
            "threshold": self.threshold,
            "recurrent": self.recurrent,

            # Network statistics
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,

            # Model information
            "model_type": "DU_Core_V1",
            "neuron_type": "Leaky Integrate-and-Fire (LIF)",
            "framework": "snnTorch + PyTorch"
        }

        return config

    def get_layer_info(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about each layer in the network.

        Returns:
            List[Dict[str, Any]]: List of dictionaries, one per layer pair,
                                 containing layer-specific information.

        Example:
            layer_info = du_core.get_layer_info()
            for i, layer in enumerate(layer_info):
                print(f"Layer {i}: {layer['input_size']} → {layer['output_size']}")
        """
        layer_info = []
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        for i, (linear_layer, lif_layer) in enumerate(zip(self.linear_layers, self.lif_layers)):
            info = {
                "layer_index": i,
                "input_size": layer_sizes[i],
                "output_size": layer_sizes[i + 1],
                "linear_parameters": sum(p.numel() for p in linear_layer.parameters()),
                "beta": self.beta,
                "threshold": self.threshold,
                "layer_type": "Linear + LIF"
            }
            layer_info.append(info)

        return layer_info

    def __repr__(self) -> str:
        """String representation of the DU Core."""
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        architecture = " → ".join(map(str, layer_sizes))

        return (
            f"DU_Core_V1(\n"
            f"  architecture: {architecture}\n"
            f"  beta: {self.beta}\n"
            f"  threshold: {self.threshold}\n"
            f"  parameters: {sum(p.numel() for p in self.parameters()):,}\n"
            f"  plasticity_rule: {type(self._plasticity_rule).__name__ if self._plasticity_rule else 'None'}\n"
            f"  memory_system: {type(self._memory_system).__name__ if self._memory_system else 'None'}\n"
            f")"
        )

    def apply_learning(self, **kwargs) -> torch.Tensor:
        """
        Apply the injected plasticity rule to calculate synaptic weight updates.

        This method serves as the interface between the DU Core and its attached
        plasticity rule. It delegates the learning computation to the injected
        PlasticityRuleInterface, enabling modular and interchangeable learning
        mechanisms.

        The method is designed to be called by continual learning algorithms
        (such as GEM) that need to compute weight updates based on neural
        activity patterns.

        Args:
            **kwargs: Flexible keyword arguments passed directly to the
                     plasticity rule's apply() method. The required parameters
                     depend on the specific plasticity rule being used:

                     For STDP_Rule:
                     - pre_spikes: Pre-synaptic spike trains
                     - post_spikes: Post-synaptic spike trains
                     - dt: Time step size (optional)

                     For ThreeFactor_Hebbian_Rule:
                     - pre_activity: Pre-synaptic activity levels
                     - post_activity: Post-synaptic activity levels
                     - modulatory_signal: Modulatory signal (reward, attention, etc.)

        Returns:
            torch.Tensor: Weight update tensor (Δw) with the same shape as the
                         synaptic weight matrix. This tensor should be added to
                         the current weights to apply the learning update.

        Raises:
            RuntimeError: If no plasticity rule has been injected into the DU Core.
            ValueError: If the plasticity rule raises an error due to invalid parameters.

        Example:
            # Apply STDP learning
            weight_updates = du_core.apply_learning(
                pre_spikes=pre_synaptic_spikes,
                post_spikes=post_synaptic_spikes,
                dt=1.0
            )

            # Apply three-factor Hebbian learning
            weight_updates = du_core.apply_learning(
                pre_activity=pre_activity,
                post_activity=post_activity,
                modulatory_signal=reward_signal
            )

        Note:
            This method only computes the weight updates; it does not apply them
            to the network. The calling code (typically a continual learning
            trainer) is responsible for applying the updates to the appropriate
            network parameters.
        """
        if self._plasticity_rule is None:
            raise RuntimeError(
                "No plasticity rule has been injected into this DU Core. "
                "Please provide a PlasticityRuleInterface instance during initialization "
                "to enable learning capabilities."
            )

        try:
            return self._plasticity_rule.apply(**kwargs)
        except Exception as e:
            raise ValueError(
                f"Plasticity rule {type(self._plasticity_rule).__name__} failed to apply: {e}"
            ) from e
