# DU Core v2: Hybrid SNN/SSM Architecture

## Overview

The DU Core v2 represents a revolutionary advancement in neuromorphic computing, implementing a novel hybrid architecture that combines Spiking Neural Networks (SNNs) with State Space Model (SSM) dynamics. This implementation introduces cutting-edge selective state mechanisms inspired by the Mamba architecture, strategically interspersed with attention layers for global reasoning capabilities.

## Key Innovations

### 1. HybridSNNSSMLayer

The core innovation is the `HybridSNNSSMLayer`, which replaces traditional Leaky Integrate-and-Fire (LIF) dynamics with SSM-driven state updates:

**Traditional LIF Neuron:**
```
V(t+1) = β * V(t) + I(t)
```

**Hybrid SSM Neuron:**
```
A, B, C = f(input_spikes)  # Selective state mechanism
h(t+1) = A * h(t) + B * input_spikes  # SSM state update
potential = C * h(t+1)  # Output projection
spikes = LIF(potential)  # Spike generation
```

### 2. Selective State Mechanism

Following the Mamba architecture, the SSM matrices (A, B, C) are dynamically generated based on input spike patterns:

- **A_generator**: Creates state transition matrices
- **B_generator**: Creates input projection matrices  
- **C_generator**: Creates output projection matrices

This allows the network to adapt its internal dynamics based on the input patterns, enabling more flexible and powerful sequence modeling.

### 3. Heterogeneous Architecture

The DU Core v2 uses a strategic mix of layer types:

```
[Input] → [SSM] → [SSM] → [Attention] → [SSM] → [SSM] → [Attention] → [Output]
```

- **SSM Layers**: Efficient temporal processing with long-term memory
- **Attention Layers**: Global reasoning and context integration
- **Configurable Ratio**: Attention frequency controlled by `attention_interval`

## Architecture Benefits

### Efficiency
- **Sequential Processing**: SSM layers process one time step at a time
- **Parallel Reasoning**: Attention layers operate on accumulated sequences
- **Memory Efficiency**: Hidden states maintain long-term dependencies

### Biological Plausibility
- **Spike-Based Computation**: Maintains event-driven processing model
- **Neuromorphic Compatibility**: Compatible with neuromorphic hardware
- **Energy Efficiency**: Sparse spike-based computation

### Performance
- **Long Sequences**: SSM dynamics handle long-term dependencies
- **Complex Reasoning**: Attention mechanisms enable global context
- **Adaptive Processing**: Selective state adapts to input patterns

## Implementation Details

### HybridSNNSSMLayer

```python
class HybridSNNSSMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, state_dim, beta=0.95, threshold=1.0):
        # Dynamic matrix generators (selective state)
        self.A_generator = nn.Sequential(...)
        self.B_generator = nn.Sequential(...)
        self.C_generator = nn.Sequential(...)
        
        # LIF neuron for spike generation
        self.lif_neuron = snn.Leaky(beta=beta, threshold=threshold)
        
    def forward(self, input_spikes, hidden_state):
        # Generate dynamic SSM matrices
        A = self.A_generator(input_spikes).view(batch_size, state_dim, state_dim)
        B = self.B_generator(input_spikes).view(batch_size, state_dim, input_dim)
        C = self.C_generator(input_spikes).view(batch_size, output_dim, state_dim)
        
        # Update hidden state via SSM equation
        new_hidden_state = torch.bmm(A, hidden_state.unsqueeze(-1)).squeeze(-1) + \
                          torch.bmm(B, input_spikes.unsqueeze(-1)).squeeze(-1)
        
        # Generate spikes via LIF neuron
        potential = torch.bmm(C, new_hidden_state.unsqueeze(-1)).squeeze(-1)
        output_spikes, _ = self.lif_neuron(potential)
        
        return output_spikes, new_hidden_state
```

### DU_Core_V2

```python
class DU_Core_V2(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, 
                 state_dim=64, attention_interval=2, attention_heads=4):
        # Build heterogeneous architecture
        self.layers = nn.ModuleList()
        self.layer_types = []
        
        # Mix SSM and attention layers
        for i, (in_dim, out_dim) in enumerate(layer_pairs):
            # Add SSM layer
            self.layers.append(HybridSNNSSMLayer(...))
            self.layer_types.append('ssm')
            
            # Add attention layer at intervals
            if should_add_attention:
                self.layers.append(nn.MultiheadAttention(...))
                self.layer_types.append('attention')
```

## Usage Examples

### Basic Usage

```python
from gif_framework.core import DU_Core_V2

# Create hybrid architecture
du_core_v2 = DU_Core_V2(
    input_size=100,
    hidden_sizes=[64, 32],
    output_size=10,
    state_dim=32,
    attention_interval=2,
    attention_heads=4
)

# Process spike train
input_spikes = torch.rand(50, 1, 100)  # 50 time steps
output_spikes = du_core_v2.process(input_spikes)
```

### Advanced Configuration

```python
# High-frequency attention for complex reasoning
du_core_v2 = DU_Core_V2(
    input_size=784,
    hidden_sizes=[128, 64, 32],
    output_size=10,
    state_dim=64,
    attention_interval=1,  # Attention after every SSM layer
    attention_heads=8,
    beta=0.9,
    threshold=1.2
)

# Low-frequency attention for efficiency
du_core_v2 = DU_Core_V2(
    input_size=100,
    hidden_sizes=[64, 32],
    output_size=10,
    state_dim=32,
    attention_interval=3,  # Attention every 3 SSM layers
    attention_heads=2
)
```

## Interface Compatibility

The DU Core v2 maintains full backward compatibility with DU Core v1:

- **Same Interface**: `process()`, `get_config()`, `reset_weights()`
- **Drop-in Replacement**: Can replace DU_Core_V1 in existing code
- **GIF Integration**: Works with existing GIF orchestrator
- **RTL Support**: Compatible with plasticity rules and memory systems

## Performance Characteristics

### Parameter Scaling
- **SSM Layers**: O(state_dim²) parameters per layer
- **Attention Layers**: O(embed_dim²) parameters per layer
- **Total**: Depends on SSM/attention ratio

### Computational Complexity
- **SSM Forward**: O(batch_size × state_dim²) per time step
- **Attention Forward**: O(sequence_length² × embed_dim) per attention layer
- **Memory**: O(batch_size × state_dim) for hidden states

### Energy Efficiency
- **Spike-Based**: Only active neurons consume energy
- **Event-Driven**: Processing triggered by spike events
- **Neuromorphic**: Compatible with low-power neuromorphic chips

## Research Impact

This implementation represents several significant contributions:

1. **Novel Neuron Model**: First implementation of SSM-driven spiking neurons
2. **Selective State in SNNs**: Dynamic matrix generation based on spike patterns
3. **Hybrid Architecture**: Strategic mixing of efficient SSM and powerful attention
4. **Neuromorphic Innovation**: Maintains biological plausibility while adding SSM power

## Future Extensions

- **Hardware Optimization**: Specialized kernels for neuromorphic chips
- **Learning Rules**: Integration with SSM-aware plasticity mechanisms
- **Multi-Modal**: Extension to handle multiple input modalities
- **Scaling**: Optimization for very large state dimensions

## References

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- NVIDIA Nemotron-H: Hybrid Architecture Design
- snnTorch: Deep Learning with Spiking Neural Networks
- General Intelligence Framework Documentation
