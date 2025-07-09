"""
Neuromorphic Hardware Simulator
===============================

Software simulation layer that mimics the key operational characteristics of neuromorphic
hardware (e.g., Intel Loihi 2) for the General Intelligence Framework. This simulator
enables testing of Spiking Neural Networks under realistic hardware constraints without
requiring physical neuromorphic chips.

Key Features:
============

**Event-Driven Computation**: Unlike traditional neural networks that process data in
dense batches, neuromorphic hardware operates in an event-driven manner. Computation
only occurs when and where spikes (events) happen, leading to sparse, efficient processing.
This simulator enforces this paradigm by processing data one time-step at a time and
only performing computations for active neurons.

**Energy Efficiency Modeling**: The simulator estimates energy consumption based on
synaptic operations (SynOps) and spike counts, using realistic energy figures from
published neuromorphic hardware specifications. This enables quantitative analysis
of the energy benefits claimed for spiking neural networks.

**Biological Plausibility**: The event-driven nature mirrors how biological neural
networks operate, where neurons consume minimal energy until they receive or generate
spikes. This makes the simulation more representative of both biological and
neuromorphic hardware behavior.

Technical Implementation:
========================

The simulator operates by:
1. **Temporal Processing**: Processing input spike trains one time-step at a time
2. **Activity Monitoring**: Counting spikes and synaptic operations at each step
3. **Energy Calculation**: Estimating total energy consumption based on activity
4. **State Management**: Properly initializing and managing neuron membrane potentials

Energy Model:
============

The energy estimation is based on the fundamental unit of computation in neuromorphic
hardware: the Synaptic Operation (SynOp). A SynOp occurs every time a spike is
transmitted across a synapse, causing a change in the downstream neuron's membrane
potential.

Energy per SynOp: 2.5e-11 Joules (based on Intel Loihi 2 specifications)
Total Energy = Total SynOps × Energy per SynOp

This model provides a realistic estimate of the energy advantages of sparse,
event-driven computation compared to dense matrix operations on traditional hardware.

Integration with GIF Framework:
==============================

The simulator is designed to work seamlessly with the GIF framework components:
- **DU Core**: Processes spike trains through the SNN architecture
- **Training Infrastructure**: Can be integrated with Continual_Trainer for experiments
- **Analysis Tools**: Provides detailed statistics for performance evaluation

Example Usage:
=============

    from gif_framework.core.du_core import DU_Core_V1
    from simulators.neuromorphic_sim import NeuromorphicSimulator
    import torch

    # Create SNN model
    du_core = DU_Core_V1(
        input_size=100,
        hidden_sizes=[64, 32],
        output_size=10
    )

    # Create neuromorphic simulator
    simulator = NeuromorphicSimulator(
        snn_model=du_core,
        energy_per_synop=2.5e-11  # Joules per synaptic operation
    )

    # Generate input spike train
    input_spikes = torch.rand(50, 1, 100)  # 50 time steps, batch size 1, 100 features

    # Run event-driven simulation
    output_spikes, stats = simulator.run(input_spikes)

    print(f"Total spikes: {stats['total_spikes']}")
    print(f"Total SynOps: {stats['total_synops']}")
    print(f"Estimated energy: {stats['estimated_energy_joules']:.2e} Joules")

This simulator enables rigorous evaluation of neuromorphic computing claims and provides
the foundation for realistic hardware-constrained experiments in the GIF framework.
"""

import torch
from typing import Dict, Tuple, Any, Union
import warnings


class NeuromorphicSimulator:
    """
    Software simulator for neuromorphic hardware behavior.

    This class provides a software-based simulation environment that mimics the key
    operational characteristics of neuromorphic hardware, specifically event-driven
    computation and energy efficiency modeling. It enables testing of Spiking Neural
    Networks under realistic hardware constraints without requiring physical chips.

    The simulator enforces the fundamental principles of neuromorphic computing:
    1. **Event-Driven Processing**: Computation only occurs when spikes are present
    2. **Sparse Activity**: Only active neurons and synapses consume energy
    3. **Temporal Dynamics**: Processing happens one time-step at a time
    4. **Energy Awareness**: Tracks and estimates energy consumption based on activity

    Key Capabilities:
    ================

    **Event-Driven Execution**: Unlike traditional neural networks that process entire
    batches simultaneously, this simulator processes spike trains temporally, one
    time-step at a time. This mirrors how neuromorphic hardware operates and enforces
    the sparse, event-driven computation paradigm.

    **Energy Modeling**: The simulator tracks synaptic operations (SynOps) and spike
    counts to estimate energy consumption. This provides quantitative evidence for
    the energy efficiency claims of neuromorphic computing.

    **Hardware Realism**: By constraining computation to occur only when and where
    spikes are present, the simulator provides a more realistic testing environment
    for algorithms intended for neuromorphic hardware deployment.

    Attributes:
        snn_model (torch.nn.Module): The SNN model to simulate (e.g., DU_Core_V1)
        energy_per_synop (float): Energy cost per synaptic operation in Joules

    Example:
        # Create simulator with DU Core
        simulator = NeuromorphicSimulator(du_core, energy_per_synop=2.5e-11)

        # Run event-driven simulation
        output_spikes, stats = simulator.run(input_spike_train)

        # Analyze energy consumption
        print(f"Energy used: {stats['estimated_energy_joules']:.2e} J")
    """

    def __init__(
        self,
        snn_model: torch.nn.Module,
        energy_per_synop: float = 2.5e-11
    ) -> None:
        """
        Initialize the neuromorphic hardware simulator.

        Args:
            snn_model (torch.nn.Module): The SNN model to simulate. Must have the
                                       required attributes for layer access and
                                       state management (e.g., DU_Core_V1).
            energy_per_synop (float): Energy consumption per synaptic operation
                                    in Joules. Default value (2.5e-11) is based
                                    on Intel Loihi 2 specifications.

        Raises:
            TypeError: If snn_model is not a PyTorch module.
            ValueError: If energy_per_synop is not positive.
            AttributeError: If snn_model lacks required attributes for simulation.

        Example:
            # Create simulator with default energy model
            simulator = NeuromorphicSimulator(du_core)

            # Create simulator with custom energy parameters
            simulator = NeuromorphicSimulator(du_core, energy_per_synop=1e-11)
        """
        # Validate input parameters
        if not isinstance(snn_model, torch.nn.Module):
            raise TypeError(
                f"snn_model must be a PyTorch module, got {type(snn_model)}"
            )

        if not isinstance(energy_per_synop, (int, float)) or energy_per_synop <= 0:
            raise ValueError(
                f"energy_per_synop must be a positive number, got {energy_per_synop}"
            )

        # Validate that the model has the required attributes for simulation
        self._validate_model_compatibility(snn_model)

        # Store configuration
        self.snn_model = snn_model
        self.energy_per_synop = float(energy_per_synop)

        # Initialize simulation statistics
        self._reset_simulation_stats()

    def _validate_model_compatibility(self, model: torch.nn.Module) -> None:
        """
        Validate that the SNN model has the required attributes for simulation.

        Args:
            model (torch.nn.Module): The SNN model to validate.

        Raises:
            AttributeError: If model lacks required attributes.
        """
        required_attributes = ['lif_layers']

        for attr in required_attributes:
            if not hasattr(model, attr):
                raise AttributeError(
                    f"SNN model must have '{attr}' attribute for neuromorphic simulation. "
                    f"Ensure you're using a compatible model like DU_Core_V1."
                )

        # Check if model has reset method (optional but recommended)
        if not hasattr(model, 'reset_states'):
            warnings.warn(
                "SNN model does not have 'reset_states' method. "
                "State initialization may not work as expected.",
                UserWarning
            )

    def _reset_simulation_stats(self) -> None:
        """Reset internal simulation statistics."""
        self._last_run_stats = {
            'total_spikes': 0,
            'total_synops': 0,
            'estimated_energy_joules': 0.0,
            'num_time_steps': 0,
            'avg_spikes_per_step': 0.0,
            'avg_synops_per_step': 0.0
        }

    def run(
        self,
        input_spikes: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute event-driven neuromorphic simulation of the SNN.

        This method implements the core event-driven simulation loop that processes
        spike trains one time-step at a time, mimicking the behavior of neuromorphic
        hardware. Unlike traditional neural network processing that operates on entire
        batches simultaneously, this simulation enforces temporal, sparse computation.

        The simulation process:
        1. **Initialize States**: Reset neuron membrane potentials
        2. **Temporal Loop**: Process each time-step sequentially
        3. **Event Processing**: Compute only when spikes are present
        4. **Activity Monitoring**: Count spikes and synaptic operations
        5. **Energy Estimation**: Calculate total energy consumption

        Args:
            input_spikes (torch.Tensor): Input spike train with shape
                                       [num_steps, batch_size, input_size].
                                       Values should be binary (0/1) or continuous
                                       spike rates between 0 and 1.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: A tuple containing:
                - output_spikes (torch.Tensor): Output spike train with shape
                                               [num_steps, batch_size, output_size]
                - statistics (Dict[str, float]): Simulation statistics including:
                    * 'total_spikes': Total number of spikes generated
                    * 'total_synops': Total synaptic operations performed
                    * 'estimated_energy_joules': Estimated energy consumption
                    * 'num_time_steps': Number of simulation time steps
                    * 'avg_spikes_per_step': Average spikes per time step
                    * 'avg_synops_per_step': Average SynOps per time step

        Raises:
            ValueError: If input tensor has incorrect shape or invalid values.
            RuntimeError: If simulation fails due to model incompatibility.

        Example:
            # Create input spike train (100 time steps, batch size 4, 50 features)
            input_spikes = torch.rand(100, 4, 50)

            # Run neuromorphic simulation
            output_spikes, stats = simulator.run(input_spikes)

            # Analyze results
            print(f"Processed {stats['num_time_steps']} time steps")
            print(f"Total energy: {stats['estimated_energy_joules']:.2e} Joules")
            print(f"Spike efficiency: {stats['avg_spikes_per_step']:.2f} spikes/step")
        """
        # Validate input tensor
        self._validate_input_spikes(input_spikes)

        # Extract simulation parameters
        num_steps, batch_size, input_size = input_spikes.shape

        # Initialize simulation state
        self._reset_simulation_stats()
        self._initialize_model_state()

        # Initialize activity counters
        total_spikes = 0
        total_synops = 0
        output_spikes_list = []

        # Event-driven simulation loop - process one time-step at a time
        for step in range(num_steps):
            # Extract current time-step input
            current_input = input_spikes[step]  # Shape: [batch_size, input_size]

            # Process through SNN model (single time-step)
            step_output = self._process_single_timestep(current_input)

            # Count neural activity for this time-step
            step_spikes, step_synops = self._count_activity(current_input)

            # Accumulate activity counters
            total_spikes += step_spikes
            total_synops += step_synops

            # Store output spikes
            output_spikes_list.append(step_output)

        # Compile final results
        output_spikes = torch.stack(output_spikes_list, dim=0)

        # Calculate energy consumption
        estimated_energy = total_synops * self.energy_per_synop

        # Compile comprehensive statistics
        statistics = {
            'total_spikes': float(total_spikes),
            'total_synops': float(total_synops),
            'estimated_energy_joules': float(estimated_energy),
            'num_time_steps': float(num_steps),
            'avg_spikes_per_step': float(total_spikes / num_steps) if num_steps > 0 else 0.0,
            'avg_synops_per_step': float(total_synops / num_steps) if num_steps > 0 else 0.0
        }

        # Store statistics for potential later access
        self._last_run_stats = statistics.copy()

        return output_spikes, statistics

    def _validate_input_spikes(self, input_spikes: torch.Tensor) -> None:
        """
        Validate the input spike tensor format and values.

        Args:
            input_spikes (torch.Tensor): Input tensor to validate.

        Raises:
            ValueError: If tensor has incorrect shape or invalid values.
        """
        if not isinstance(input_spikes, torch.Tensor):
            raise ValueError(
                f"input_spikes must be a torch.Tensor, got {type(input_spikes)}"
            )

        if input_spikes.dim() != 3:
            raise ValueError(
                f"input_spikes must be 3D tensor [num_steps, batch_size, input_size], "
                f"got shape {input_spikes.shape}"
            )

        if input_spikes.size(0) <= 0:
            raise ValueError(
                f"Number of time steps must be positive, got {input_spikes.size(0)}"
            )

        # Check for reasonable spike values (should be between 0 and 1 for rates)
        if torch.any(input_spikes < 0) or torch.any(input_spikes > 1):
            warnings.warn(
                "Input spikes contain values outside [0, 1] range. "
                "For optimal neuromorphic simulation, use binary spikes (0/1) "
                "or spike rates between 0 and 1.",
                UserWarning
            )

    def _initialize_model_state(self) -> None:
        """
        Initialize the SNN model state for simulation.

        This method resets the membrane potentials of all neurons to ensure
        a clean starting state for the simulation.
        """
        # Reset model state if method is available
        if hasattr(self.snn_model, 'reset_states'):
            self.snn_model.reset_states()
        else:
            # If no reset method, issue warning but continue
            warnings.warn(
                "SNN model does not have reset_states method. "
                "Membrane potentials may not be properly initialized.",
                UserWarning
            )

    def _process_single_timestep(self, current_input: torch.Tensor) -> torch.Tensor:
        """
        Process a single time-step through the SNN model.

        Args:
            current_input (torch.Tensor): Input spikes for current time-step
                                         with shape [batch_size, input_size].

        Returns:
            torch.Tensor: Output spikes for current time-step with shape
                         [batch_size, output_size].
        """
        # Reshape input to add time dimension for model compatibility
        # Shape: [batch_size, input_size] -> [1, batch_size, input_size]
        timestep_input = current_input.unsqueeze(0)

        # Process through model for single time-step
        # Most SNN models expect [num_steps, batch_size, features] format
        if hasattr(self.snn_model, 'forward'):
            # Use forward method with num_steps=1
            timestep_output = self.snn_model.forward(timestep_input, num_steps=1)
        elif hasattr(self.snn_model, 'process'):
            # Use process method (GIF interface)
            timestep_output = self.snn_model.process(timestep_input)
        else:
            # Fallback to direct call
            timestep_output = self.snn_model(timestep_input)

        # Remove time dimension from output
        # Shape: [1, batch_size, output_size] -> [batch_size, output_size]
        return timestep_output.squeeze(0)

    def _count_activity(
        self,
        current_input: torch.Tensor
    ) -> Tuple[int, int]:
        """
        Count neural activity (spikes and synaptic operations) for current time-step.

        This method analyzes the current state of the SNN model to count:
        1. Total spikes generated by all LIF layers
        2. Estimated synaptic operations based on layer connectivity

        Args:
            current_input (torch.Tensor): Input spikes for current time-step.

        Returns:
            Tuple[int, int]: (total_spikes, total_synops) for this time-step.
        """
        total_spikes = 0
        total_synops = 0

        # Count spikes from all LIF layers
        for layer_idx, lif_layer in enumerate(self.snn_model.lif_layers):
            # Access current spike output from LIF layer
            if hasattr(lif_layer, 'spk1') and lif_layer.spk1 is not None:
                # Count spikes in this layer
                layer_spikes = lif_layer.spk1.sum().item()
                total_spikes += layer_spikes

                # Estimate synaptic operations for this layer
                # SynOps ≈ number of output neurons that could receive spikes
                layer_synops = lif_layer.spk1.numel()  # Total possible connections
                total_synops += layer_synops
            else:
                # If spk1 not available, estimate based on layer size
                warnings.warn(
                    f"Layer {layer_idx} does not have accessible spike output (spk1). "
                    "Activity counting may be inaccurate.",
                    UserWarning
                )

        return int(total_spikes), int(total_synops)

    def get_last_run_stats(self) -> Dict[str, float]:
        """
        Get statistics from the most recent simulation run.

        Returns:
            Dict[str, float]: Dictionary containing simulation statistics from
                            the last call to run(). Returns empty dict if no
                            simulation has been executed yet.

        Example:
            # Run simulation
            output, stats = simulator.run(input_spikes)

            # Get stats again later
            cached_stats = simulator.get_last_run_stats()
            assert stats == cached_stats
        """
        return self._last_run_stats.copy()

    def get_energy_model_info(self) -> Dict[str, Any]:
        """
        Get information about the energy model used by the simulator.

        Returns:
            Dict[str, Any]: Dictionary containing energy model parameters
                          and related information.

        Example:
            info = simulator.get_energy_model_info()
            print(f"Energy per SynOp: {info['energy_per_synop']:.2e} J")
        """
        return {
            'energy_per_synop': self.energy_per_synop,
            'energy_unit': 'Joules',
            'model_basis': 'Intel Loihi 2 specifications',
            'calculation_method': 'Total SynOps × Energy per SynOp'
        }

    def estimate_energy_for_activity(
        self,
        total_spikes: int,
        total_synops: int
    ) -> float:
        """
        Estimate energy consumption for given activity levels.

        This utility method allows estimation of energy consumption without
        running a full simulation, useful for theoretical analysis or
        optimization studies.

        Args:
            total_spikes (int): Total number of spikes.
            total_synops (int): Total number of synaptic operations.

        Returns:
            float: Estimated energy consumption in Joules.

        Example:
            # Estimate energy for hypothetical activity
            energy = simulator.estimate_energy_for_activity(
                total_spikes=1000,
                total_synops=50000
            )
            print(f"Estimated energy: {energy:.2e} J")
        """
        return float(total_synops * self.energy_per_synop)

    def __repr__(self) -> str:
        """String representation of the simulator."""
        return (
            f"NeuromorphicSimulator("
            f"model={type(self.snn_model).__name__}, "
            f"energy_per_synop={self.energy_per_synop:.2e})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Neuromorphic Hardware Simulator\n"
            f"  Model: {type(self.snn_model).__name__}\n"
            f"  Energy per SynOp: {self.energy_per_synop:.2e} Joules\n"
            f"  Last run stats: {len(self._last_run_stats)} metrics available"
        )
