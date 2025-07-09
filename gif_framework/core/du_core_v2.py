"""
Deep Understanding (DU) Core v2 - Hybrid SNN/SSM Architecture
============================================================

This module implements the second generation of the Deep Understanding Core,
featuring a revolutionary hybrid architecture that combines Spiking Neural Networks
with State Space Model dynamics. This represents a significant advancement over
the v1 LIF-only implementation, incorporating cutting-edge selective state
mechanisms inspired by the Mamba architecture.

The DU Core v2 introduces two major innovations:
1. **HybridSNNSSMLayer**: A novel neuron model where membrane dynamics follow
   State Space Model equations with selective state mechanisms
2. **Heterogeneous Architecture**: Strategic mixing of efficient SSM layers
   with attention layers for global reasoning capabilities

Key Features:
- **Selective State Mechanism**: Dynamic A, B, C matrices generated from input spikes
- **SSM-Driven Neurons**: Membrane potential evolution via discretized SSM equations
- **Hybrid Processing**: Efficient sequential processing + global attention reasoning
- **Neuromorphic Compatibility**: Maintains event-driven computation model
- **Backward Compatibility**: Drop-in replacement for DU_Core_V1

Technical Innovation:
The core breakthrough is the HybridSNNSSMLayer, which replaces traditional LIF
dynamics with SSM state updates:
- Traditional LIF: V(t+1) = β*V(t) + I(t)
- Hybrid SSM: h(t+1) = A(x)*h(t) + B(x)*x(t), where A,B,C = f(input_spikes)

This allows the network to maintain long-term dependencies while preserving
the energy efficiency and biological plausibility of spiking computation.

Architecture Philosophy:
Following the success of models like NVIDIA's Nemotron-H, this implementation
uses a heterogeneous approach:
- Majority SSM layers for efficient temporal processing
- Strategic attention layers for complex reasoning and global context
- Configurable mixing ratio based on task requirements

Author: GIF Development Team
Phase: 6.1 - Advanced Features Implementation
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError(
        "PyTorch is required for the DU Core v2. Please install it with: "
        "pip install torch>=2.0.0"
    )

try:
    import snntorch as snn
except ImportError:
    raise ImportError(
        "snnTorch is required for the DU Core v2. Please install it with: "
        "pip install snntorch>=0.7.0"
    )

# Import SpikeTrain type alias for consistency with framework
from gif_framework.interfaces.base_interfaces import SpikeTrain

# Import RTL and memory components for dependency injection
from gif_framework.core.rtl_mechanisms import PlasticityRuleInterface
from gif_framework.core.memory_systems import EpisodicMemory

# Import knowledge augmentation (with graceful fallback)
try:
    from gif_framework.core.knowledge_augmenter import KnowledgeAugmenter
except ImportError:
    # Graceful fallback if knowledge dependencies are not installed
    KnowledgeAugmenter = None


class HybridSNNSSMLayer(nn.Module):
    """
    A novel spiking neural network layer with State Space Model dynamics.
    
    This layer represents the core innovation of DU_Core_V2, combining the efficiency
    of State Space Models with the biological plausibility of spiking neurons.
    Unlike traditional LIF neurons with fixed dynamics, this layer uses selective
    state mechanisms where the SSM matrices (A, B, C) are dynamically generated
    based on the input spike patterns.
    
    The layer implements the following computation:
    1. Generate dynamic SSM matrices from input spikes: A, B, C = f(input_spikes)
    2. Update hidden state via SSM equation: h(t+1) = A * h(t) + B * input_spikes
    3. Compute pre-synaptic potential: potential = C * h(t+1)
    4. Generate output spikes via LIF neuron: spikes = LIF(potential)
    
    This design enables:
    - Long-term memory through SSM state dynamics
    - Adaptive processing via selective state mechanism
    - Energy-efficient spike-based computation
    - Compatibility with neuromorphic hardware
    
    Args:
        input_dim (int): Dimension of input spike trains
        output_dim (int): Dimension of output spike trains  
        state_dim (int): Dimension of internal SSM hidden state
        beta (float): Leakage parameter for LIF neuron (0 < beta ≤ 1)
        threshold (float): Firing threshold for LIF neuron
        
    Example:
        layer = HybridSNNSSMLayer(input_dim=100, output_dim=64, state_dim=32)
        hidden_state = torch.zeros(batch_size, 32)
        
        for t in range(num_steps):
            spikes, hidden_state = layer(input_spikes[t], hidden_state)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dim: int,
        beta: float = 0.95,
        threshold: float = 1.0
    ) -> None:
        super(HybridSNNSSMLayer, self).__init__()
        
        # Validate parameters
        if input_dim <= 0 or output_dim <= 0 or state_dim <= 0:
            raise ValueError("All dimensions must be positive")
        if not (0 < beta <= 1):
            raise ValueError("Beta must be in range (0, 1]")
        if threshold <= 0:
            raise ValueError("Threshold must be positive")
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.beta = beta
        self.threshold = threshold
        
        # Selective state mechanism: dynamic matrix generators
        # These small networks generate the SSM matrices based on input spikes
        self.A_generator = nn.Sequential(
            nn.Linear(input_dim, state_dim),
            nn.Tanh(),
            nn.Linear(state_dim, state_dim * state_dim)
        )
        
        self.B_generator = nn.Sequential(
            nn.Linear(input_dim, state_dim),
            nn.Tanh(), 
            nn.Linear(state_dim, state_dim * input_dim)
        )
        
        self.C_generator = nn.Sequential(
            nn.Linear(input_dim, state_dim),
            nn.Tanh(),
            nn.Linear(state_dim, output_dim * state_dim)
        )
        
        # LIF neuron for spike generation
        self.lif_neuron = snn.Leaky(beta=beta, threshold=threshold, init_hidden=True)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in [self.A_generator, self.B_generator, self.C_generator]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    
        # Initialize A_generator to produce near-identity matrices for stability
        with torch.no_grad():
            # Bias the final layer of A_generator toward identity matrix
            identity_bias = torch.eye(self.state_dim).flatten()
            self.A_generator[-1].bias.copy_(identity_bias * 0.1)

    def forward(
        self,
        input_spikes: torch.Tensor,
        hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the hybrid SNN/SSM layer.

        This method implements the core innovation of the layer: using input spikes
        to dynamically generate SSM matrices, updating the hidden state via SSM
        dynamics, and generating output spikes through a LIF neuron.

        The computation follows these steps:
        1. Generate dynamic A, B, C matrices from input spikes (selective state)
        2. Update hidden state: h(t+1) = A * h(t) + B * input_spikes
        3. Compute pre-synaptic potential: potential = C * h(t+1)
        4. Generate spikes via LIF neuron: spikes = LIF(potential)

        Args:
            input_spikes (torch.Tensor): Input spike train for current time step
                                       Shape: [batch_size, input_dim]
            hidden_state (torch.Tensor): Previous hidden state from SSM
                                       Shape: [batch_size, state_dim]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output_spikes: Generated spikes [batch_size, output_dim]
                - new_hidden_state: Updated hidden state [batch_size, state_dim]

        Example:
            # Single time step processing
            spikes = torch.rand(4, 100)  # batch_size=4, input_dim=100
            hidden = torch.zeros(4, 32)  # state_dim=32

            output_spikes, new_hidden = layer(spikes, hidden)
        """
        batch_size = input_spikes.size(0)

        # Step 1: Generate dynamic SSM matrices using selective state mechanism
        # This is the key innovation - matrices adapt to input spike patterns
        A_flat = self.A_generator(input_spikes)  # [batch_size, state_dim^2]
        B_flat = self.B_generator(input_spikes)  # [batch_size, state_dim * input_dim]
        C_flat = self.C_generator(input_spikes)  # [batch_size, output_dim * state_dim]

        # Reshape to proper matrix dimensions
        A = A_flat.view(batch_size, self.state_dim, self.state_dim)
        B = B_flat.view(batch_size, self.state_dim, self.input_dim)
        C = C_flat.view(batch_size, self.output_dim, self.state_dim)

        # Step 2: Update hidden state via SSM equation
        # h(t+1) = A * h(t) + B * input_spikes
        # Using batch matrix multiplication for efficiency
        state_transition = torch.bmm(A, hidden_state.unsqueeze(-1)).squeeze(-1)
        input_projection = torch.bmm(B, input_spikes.unsqueeze(-1)).squeeze(-1)
        new_hidden_state = state_transition + input_projection

        # Step 3: Compute pre-synaptic potential via output equation
        # potential = C * h(t+1)
        potential = torch.bmm(C, new_hidden_state.unsqueeze(-1)).squeeze(-1)

        # Step 4: Generate spikes via LIF neuron
        # The LIF neuron maintains its own membrane potential state
        output_spikes, _ = self.lif_neuron(potential)

        return output_spikes, new_hidden_state

    def reset_state(self) -> None:
        """Reset the internal state of the LIF neuron."""
        # Reset LIF neuron membrane potential
        if hasattr(self.lif_neuron, 'reset_mem'):
            self.lif_neuron.reset_mem()

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization and debugging."""
        return {
            "layer_type": "HybridSNNSSMLayer",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "state_dim": self.state_dim,
            "beta": self.beta,
            "threshold": self.threshold,
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class DU_Core_V2(nn.Module):
    """
    The second generation Deep Understanding Core with hybrid SNN/SSM architecture.

    This class represents a major advancement in neuromorphic computing, implementing
    a heterogeneous architecture that strategically combines:
    1. HybridSNNSSMLayer blocks for efficient temporal processing
    2. Multi-head attention layers for global reasoning and context integration

    The architecture follows the successful pattern of models like NVIDIA's Nemotron-H,
    using mostly efficient SSM-based layers with strategic attention placement for
    complex reasoning tasks. This provides the best of both worlds:
    - Efficient, stateful processing for long sequences (SSM layers)
    - Powerful global reasoning capabilities (attention layers)
    - Neuromorphic compatibility and energy efficiency (spiking computation)

    Key Innovations:
    - **Heterogeneous Layer Mixing**: Configurable ratio of SSM to attention layers
    - **Adaptive State Management**: Different processing modes for different layer types
    - **Sequence Accumulation**: Intelligent buffering for attention operations
    - **Backward Compatibility**: Drop-in replacement for DU_Core_V1

    Architecture Pattern:
    The network alternates between efficient SSM processing and global attention:
    [Input] → [SSM] → [SSM] → [Attention] → [SSM] → [SSM] → [Attention] → [Output]

    This pattern allows the model to:
    1. Process sequences efficiently with SSM layers
    2. Periodically perform global reasoning with attention
    3. Maintain long-term dependencies through SSM state
    4. Handle complex reasoning tasks through attention mechanisms

    Args:
        input_size (int): Dimension of input spike trains
        hidden_sizes (List[int]): Sizes of hidden layers in the network
        output_size (int): Dimension of output spike trains
        state_dim (int): Dimension of SSM hidden states
        attention_interval (int): Insert attention layer every N SSM layers
        attention_heads (int): Number of attention heads for MultiheadAttention
        beta (float): Leakage parameter for LIF neurons
        threshold (float): Firing threshold for LIF neurons
        plasticity_rule (Optional): Learning rule for continual learning
        memory_system (Optional): Episodic memory system for experience replay
        knowledge_augmenter (Optional): Knowledge augmentation system for RAG/CAG

    Example:
        # Create hybrid architecture: 100 → 64 → 32 → 10
        du_core_v2 = DU_Core_V2(
            input_size=100,
            hidden_sizes=[64, 32],
            output_size=10,
            state_dim=32,
            attention_interval=2,  # Attention every 2 SSM layers
            attention_heads=4
        )

        # Process spike train
        input_spikes = torch.rand(50, 1, 100)  # 50 time steps
        output_spikes = du_core_v2.process(input_spikes)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        state_dim: int = 64,
        attention_interval: int = 2,
        attention_heads: int = 4,
        beta: float = 0.95,
        threshold: float = 1.0,
        plasticity_rule: Optional[PlasticityRuleInterface] = None,
        memory_system: Optional[EpisodicMemory] = None,
        knowledge_augmenter: Optional['KnowledgeAugmenter'] = None
    ) -> None:
        """Initialize the DU Core v2 with hybrid SNN/SSM architecture."""
        super(DU_Core_V2, self).__init__()

        # Validate parameters
        self._validate_parameters(
            input_size, hidden_sizes, output_size, state_dim,
            attention_interval, attention_heads, beta, threshold
        )

        # Store configuration
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes.copy()
        self.output_size = output_size
        self.state_dim = state_dim
        self.attention_interval = attention_interval
        self.attention_heads = attention_heads
        self.beta = beta
        self.threshold = threshold

        # Store RTL components for continual learning
        self._plasticity_rule = plasticity_rule
        self._memory_system = memory_system

        # Store knowledge augmentation component
        self._knowledge_augmenter = knowledge_augmenter
        if knowledge_augmenter and KnowledgeAugmenter is None:
            raise ImportError(
                "KnowledgeAugmenter dependencies not installed. Please install with: "
                "pip install pymilvus>=2.3.0 neo4j>=5.0.0 sentence-transformers>=2.2.0"
            )

        # Build heterogeneous architecture
        self.layers = nn.ModuleList()
        self.layer_types = []  # Track layer types for forward pass

        # Create layer size sequence
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Build mixed architecture
        layer_count = 0
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]

            # Add SSM layer
            ssm_layer = HybridSNNSSMLayer(
                input_dim=in_dim,
                output_dim=out_dim,
                state_dim=state_dim,
                beta=beta,
                threshold=threshold
            )
            self.layers.append(ssm_layer)
            self.layer_types.append('ssm')
            layer_count += 1

            # Add attention layer at specified intervals (except for last layer)
            if (layer_count % attention_interval == 0 and
                i < len(layer_sizes) - 2):  # Not the last layer

                attention_layer = nn.MultiheadAttention(
                    embed_dim=out_dim,
                    num_heads=attention_heads,
                    batch_first=True
                )
                self.layers.append(attention_layer)
                self.layer_types.append('attention')

        # Initialize weights
        self._initialize_weights()

    def _validate_parameters(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        state_dim: int,
        attention_interval: int,
        attention_heads: int,
        beta: float,
        threshold: float
    ) -> None:
        """Validate all initialization parameters."""
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if attention_interval <= 0:
            raise ValueError("attention_interval must be positive")
        if attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if not (0 < beta <= 1):
            raise ValueError("beta must be in range (0, 1]")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if not isinstance(hidden_sizes, list):
            raise TypeError("hidden_sizes must be a list")
        if any(size <= 0 for size in hidden_sizes):
            raise ValueError("All hidden layer sizes must be positive")

    def _initialize_weights(self) -> None:
        """Initialize weights for attention layers (SSM layers self-initialize)."""
        for layer in self.layers:
            if isinstance(layer, nn.MultiheadAttention):
                # Initialize attention weights
                nn.init.xavier_uniform_(layer.in_proj_weight)
                nn.init.zeros_(layer.in_proj_bias)
                nn.init.xavier_uniform_(layer.out_proj.weight)
                nn.init.zeros_(layer.out_proj.bias)

    def forward(self, input_spikes: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Forward pass through the hybrid SNN/SSM architecture.

        This method implements the complex temporal processing required for the
        heterogeneous architecture. It handles two different processing modes:

        1. **Sequential Processing (SSM layers)**: Process spikes one time step
           at a time, maintaining hidden states across time steps
        2. **Parallel Processing (Attention layers)**: Accumulate sequence history
           and perform global self-attention operations

        The forward pass alternates between these modes based on the layer type,
        allowing the model to efficiently process temporal sequences while
        periodically performing global reasoning operations.

        Processing Flow:
        1. Initialize hidden states for all SSM layers
        2. For each time step:
           a. Process through SSM layers sequentially
           b. When encountering attention layers, accumulate sequence
           c. Apply attention to full sequence history
           d. Continue with subsequent SSM layers
        3. Return final output spike sequence

        Args:
            input_spikes (torch.Tensor): Input spike trains
                                       Shape: [num_steps, batch_size, input_size]
            num_steps (int): Number of time steps to simulate

        Returns:
            torch.Tensor: Output spike trains
                         Shape: [num_steps, batch_size, output_size]

        Example:
            # Process 100 time steps
            input_spikes = torch.rand(100, 4, 784)  # batch_size=4
            output_spikes = du_core_v2.forward(input_spikes, num_steps=100)
        """
        # Validate input
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

        # Initialize hidden states for all SSM layers
        hidden_states = []
        ssm_layer_idx = 0
        for layer_type in self.layer_types:
            if layer_type == 'ssm':
                # Get the actual SSM layer to determine state_dim
                ssm_layer = self.layers[ssm_layer_idx]
                hidden_state = torch.zeros(
                    batch_size, ssm_layer.state_dim,
                    device=input_spikes.device, dtype=input_spikes.dtype
                )
                hidden_states.append(hidden_state)
                ssm_layer_idx += 1
            else:
                hidden_states.append(None)  # Placeholder for attention layers

        # Storage for output spikes and sequence history for attention
        output_spikes_list = []
        sequence_history = []  # For attention layers

        # Main temporal simulation loop
        for step in range(num_steps):
            current_input = input_spikes[step]  # [batch_size, input_size]
            layer_output = current_input

            # Process through all layers
            layer_idx = 0
            ssm_state_idx = 0

            for layer_type in self.layer_types:
                layer = self.layers[layer_idx]

                if layer_type == 'ssm':
                    # Sequential SSM processing
                    layer_output, hidden_states[ssm_state_idx] = layer(
                        layer_output, hidden_states[ssm_state_idx]
                    )
                    ssm_state_idx += 1

                elif layer_type == 'attention':
                    # Parallel attention processing
                    # Accumulate current output for sequence
                    sequence_history.append(layer_output.unsqueeze(1))  # Add time dim

                    if len(sequence_history) > 1:
                        # Create sequence tensor for attention
                        sequence_tensor = torch.cat(sequence_history, dim=1)
                        # Shape: [batch_size, sequence_length, features]

                        # Apply self-attention
                        attended_output, _ = layer(
                            sequence_tensor, sequence_tensor, sequence_tensor
                        )

                        # Use the last time step output
                        layer_output = attended_output[:, -1, :]
                    # If only one time step, skip attention (need at least 2 for context)

                layer_idx += 1

            # Store output for this time step
            output_spikes_list.append(layer_output)

        # Stack all time steps
        output_spikes = torch.stack(output_spikes_list, dim=0)

        return output_spikes

    def _detect_uncertainty(
        self,
        hidden_states: List[torch.Tensor],
        attention_weights: Optional[torch.Tensor] = None
    ) -> Tuple[bool, float]:
        """
        Detect uncertainty in the network's internal state to trigger knowledge augmentation.

        This method analyzes the network's internal dynamics to identify situations
        where external knowledge might be beneficial. It uses multiple uncertainty
        indicators to make this determination.

        Args:
            hidden_states (List[torch.Tensor]): Hidden states from SSM layers
            attention_weights (Optional[torch.Tensor]): Attention weights if available

        Returns:
            Tuple[bool, float]: (should_augment, uncertainty_score)
        """
        if not self._knowledge_augmenter:
            return False, 0.0

        try:
            uncertainty_indicators = []

            # Indicator 1: Hidden state variance (low variance = high uncertainty)
            for hidden_state in hidden_states:
                if hidden_state is not None:
                    state_variance = torch.var(hidden_state, dim=-1).mean().item()
                    # Normalize and invert (low variance = high uncertainty)
                    uncertainty_indicators.append(1.0 / (1.0 + state_variance))

            # Indicator 2: Attention entropy (if available)
            if attention_weights is not None:
                # Calculate entropy of attention weights
                attention_probs = F.softmax(attention_weights, dim=-1)
                attention_entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1).mean().item()
                # High entropy = high uncertainty
                uncertainty_indicators.append(attention_entropy / math.log(attention_weights.size(-1)))

            # Combine uncertainty indicators
            if uncertainty_indicators:
                uncertainty_score = sum(uncertainty_indicators) / len(uncertainty_indicators)
                # Threshold for triggering knowledge augmentation
                uncertainty_threshold = 0.7
                should_augment = uncertainty_score > uncertainty_threshold

                return should_augment, uncertainty_score
            else:
                return False, 0.0

        except Exception as e:
            # Graceful fallback on error
            return False, 0.0

    def _formulate_knowledge_query(
        self,
        current_input: torch.Tensor,
        layer_output: torch.Tensor
    ) -> str:
        """
        Formulate a knowledge query based on current processing context.

        This method converts the network's current internal state into a
        natural language query that can be used for knowledge retrieval.

        Args:
            current_input (torch.Tensor): Current input spikes
            layer_output (torch.Tensor): Current layer output

        Returns:
            str: Natural language query for knowledge retrieval
        """
        try:
            # Simple heuristic: use spike patterns to generate query
            # In a real implementation, this could be much more sophisticated

            # Calculate activation patterns
            input_activity = torch.sum(current_input, dim=-1).mean().item()
            output_activity = torch.sum(layer_output, dim=-1).mean().item()

            # Generate context-aware query based on activity patterns
            if input_activity > 0.5 and output_activity < 0.3:
                # High input, low output suggests need for processing knowledge
                return "processing methods analysis techniques"
            elif input_activity < 0.3 and output_activity > 0.5:
                # Low input, high output suggests internal generation
                return "pattern recognition classification methods"
            elif input_activity > 0.7:
                # Very high input activity
                return "signal processing feature extraction"
            else:
                # Default general query
                return "domain knowledge context information"

        except Exception:
            # Fallback query
            return "relevant context information"

    def _integrate_knowledge_context(
        self,
        layer_output: torch.Tensor,
        knowledge_context: List[str]
    ) -> torch.Tensor:
        """
        Integrate retrieved knowledge context into the current processing.

        This method combines the network's current output with external
        knowledge context to enhance processing capabilities.

        Args:
            layer_output (torch.Tensor): Current layer output
            knowledge_context (List[str]): Retrieved knowledge text

        Returns:
            torch.Tensor: Enhanced output with integrated knowledge
        """
        if not knowledge_context or not self._knowledge_augmenter:
            return layer_output

        try:
            # Simple integration: modulate output based on knowledge availability
            # In a real implementation, this would be much more sophisticated

            # Knowledge availability factor
            knowledge_factor = min(len(knowledge_context) / 5.0, 1.0)  # Normalize to [0, 1]

            # Enhance output with knowledge-informed modulation
            # This is a placeholder - real implementation would use embeddings
            enhanced_output = layer_output * (1.0 + 0.1 * knowledge_factor)

            return enhanced_output

        except Exception:
            # Fallback to original output
            return layer_output

    def process(self, spike_train: SpikeTrain) -> SpikeTrain:
        """
        Process a spike train through the hybrid SNN/SSM network.

        This method serves as the main interface for the GIF orchestrator,
        providing backward compatibility with DU_Core_V1 while leveraging
        the advanced hybrid architecture of v2.

        Args:
            spike_train (SpikeTrain): Input spike train tensor
                                    Shape: [num_steps, batch_size, input_size]

        Returns:
            SpikeTrain: Processed output spike train
                       Shape: [num_steps, batch_size, output_size]
        """
        if spike_train.dim() != 3:
            raise ValueError(
                f"spike_train must be 3D tensor [num_steps, batch_size, input_size], "
                f"got shape {spike_train.shape}"
            )

        num_steps = spike_train.size(0)
        return self.forward(spike_train, num_steps)

    def reset_weights(self) -> None:
        """
        Reset all network weights for potentiation experiments.

        This method enables rigorous testing of system potentiation by completely
        re-randomizing all learnable parameters while preserving the network
        architecture. This is crucial for distinguishing true learning capacity
        improvements from simple knowledge transfer.

        The method resets:
        - All SSM matrix generators (A, B, C generators)
        - All attention layer weights
        - Maintains neuron parameters (beta, threshold) and architecture

        Scientific Justification:
        By resetting weights after pre-training, any performance improvement on
        new tasks can be attributed to enhanced learning mechanisms rather than
        transferred features, providing evidence for true system potentiation.
        """
        # Reset SSM layers
        for layer in self.layers:
            if isinstance(layer, HybridSNNSSMLayer):
                layer._initialize_weights()
            elif isinstance(layer, nn.MultiheadAttention):
                # Reset attention weights
                nn.init.xavier_uniform_(layer.in_proj_weight)
                nn.init.zeros_(layer.in_proj_bias)
                nn.init.xavier_uniform_(layer.out_proj.weight)
                nn.init.zeros_(layer.out_proj.bias)

    def get_config(self) -> Dict[str, Any]:
        """
        Get comprehensive configuration of the DU Core v2.

        Returns detailed information about the hybrid architecture,
        including layer composition, parameter counts, and configuration.

        Returns:
            Dict[str, Any]: Complete configuration dictionary
        """
        # Calculate layer statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Analyze layer composition
        ssm_layers = sum(1 for t in self.layer_types if t == 'ssm')
        attention_layers = sum(1 for t in self.layer_types if t == 'attention')

        # Get layer size information
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        config = {
            # Architecture parameters
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes.copy(),
            "output_size": self.output_size,
            "layer_sizes": layer_sizes,
            "state_dim": self.state_dim,

            # Hybrid architecture composition
            "total_layers": len(self.layers),
            "ssm_layers": ssm_layers,
            "attention_layers": attention_layers,
            "layer_types": self.layer_types.copy(),
            "attention_interval": self.attention_interval,
            "attention_heads": self.attention_heads,

            # Neuron parameters
            "beta": self.beta,
            "threshold": self.threshold,

            # Network statistics
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,

            # Model information
            "model_type": "DU_Core_V2",
            "architecture_type": "Hybrid SNN/SSM with Attention",
            "framework": "snnTorch + PyTorch + Custom SSM",

            # RTL components
            "plasticity_rule": type(self._plasticity_rule).__name__ if self._plasticity_rule else None,
            "memory_system": type(self._memory_system).__name__ if self._memory_system else None,

            # Knowledge augmentation
            "knowledge_augmenter": type(self._knowledge_augmenter).__name__ if self._knowledge_augmenter else None,
            "rag_available": self._knowledge_augmenter.get_status()["rag_available"] if self._knowledge_augmenter else False,
            "cag_available": self._knowledge_augmenter.get_status()["cag_available"] if self._knowledge_augmenter else False
        }

        return config

    def apply_learning(self, **kwargs) -> Optional[Dict[str, torch.Tensor]]:
        """
        Apply continual learning mechanisms (placeholder for future RTL integration).

        This method provides the interface for real-time learning mechanisms
        to be applied to the hybrid architecture. Future implementations will
        extend this to work with the SSM dynamics and attention mechanisms.

        Args:
            **kwargs: Learning parameters (implementation dependent)

        Returns:
            Optional[Dict[str, torch.Tensor]]: Weight updates if plasticity rule exists

        Raises:
            RuntimeError: If no plasticity rule has been injected
        """
        if self._plasticity_rule is None:
            raise RuntimeError(
                "No plasticity rule has been injected into DU_Core_V2. "
                "Please provide a plasticity rule during initialization."
            )

        # Future implementation will integrate with SSM dynamics
        # For now, delegate to the plasticity rule
        return self._plasticity_rule.apply(**kwargs)

    def __repr__(self) -> str:
        """String representation of the DU Core v2."""
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        architecture = " → ".join(map(str, layer_sizes))

        ssm_count = sum(1 for t in self.layer_types if t == 'ssm')
        attention_count = sum(1 for t in self.layer_types if t == 'attention')

        return (
            f"DU_Core_V2(\n"
            f"  architecture: {architecture}\n"
            f"  hybrid_composition: {ssm_count} SSM + {attention_count} Attention layers\n"
            f"  state_dim: {self.state_dim}\n"
            f"  attention_heads: {self.attention_heads}\n"
            f"  beta: {self.beta}, threshold: {self.threshold}\n"
            f"  parameters: {sum(p.numel() for p in self.parameters()):,}\n"
            f"  plasticity_rule: {type(self._plasticity_rule).__name__ if self._plasticity_rule else 'None'}\n"
            f"  memory_system: {type(self._memory_system).__name__ if self._memory_system else 'None'}\n"
            f"  knowledge_augmenter: {type(self._knowledge_augmenter).__name__ if self._knowledge_augmenter else 'None'}\n"
            f")"
        )
