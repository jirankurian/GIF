# POC/gif_poc_modules/architecture/core_du.py

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SnnCoreDU(nn.Module):
    """
    Spiking Neural Network (SNN) Core implementing the Deep Understanding (DU) proxy
    using snnTorch Leaky Integrate-and-Fire (LIF) neurons.
    Processes features from the encoder over a defined number of time steps.
    Uses manual state initialization (init_hidden=False).
    """
    def __init__(self, cfg, input_features): # Accepts config object (cfg)
        super().__init__()
        self.cfg = cfg
        self.input_features = input_features

        # --- SNN Parameters from Config ---
        self.hidden_size = cfg.SNN_HIDDEN_SIZE
        self.num_layers = cfg.SNN_NUM_LAYERS # Recommendation: Keep num_layers=1 for this simple structure
        self.num_time_steps = cfg.NUM_TIME_STEPS
        self.output_features = self.hidden_size # Output features = hidden size of the last SNN layer

        # --- Surrogate Gradient ---
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # --- LIF Neuron Parameters ---
        beta = cfg.SPIKE_BETA          # Membrane potential decay rate
        threshold = cfg.SPIKE_THRESHOLD # Firing threshold

        # --- Input Projection Layer ---
        if self.input_features != self.hidden_size:
            self.input_projection = nn.Linear(self.input_features, self.hidden_size)
        else:
            self.input_projection = nn.Identity()
            print("SnnCoreDU: Input features match hidden size, using Identity projection.")

        # --- SNN Layers ---
        # Create a list of LIF layers
        snn_core_layers = []
        if self.num_layers > 1:
             print(f"Warning: SnnCoreDU with num_layers={self.num_layers} uses independent LIF layers. "
                   "For stacked SNNs, modify forward pass logic.")
        for i in range(self.num_layers):
            snn_core_layers.append(
                snn.Leaky(
                    beta=beta,
                    threshold=threshold,
                    spike_grad=spike_grad,
                    init_hidden=False,   # <<< Set to False for manual initialization
                    output=True,         # Output spikes and membrane potential
                    # reset_mechanism="subtract" # Consider desired neuron reset behavior
                )
            )
        self.snn_layers = nn.ModuleList(snn_core_layers)

    def forward(self, x):
        """
        Processes the input features through the SNN over time steps.

        Args:
            x (torch.Tensor): Input features from the encoder projection layer.
                              Shape: (batch_size, input_features for SNN core).

        Returns:
            tuple: (aggregated_output, total_spikes_per_sample)
                   - aggregated_output (torch.Tensor): Sum of output spikes over time steps from the last layer.
                                                      Shape: (batch_size, self.output_features).
                   - total_spikes_per_sample (torch.Tensor): Total spikes fired per sample over all layers/steps.
                                                             Shape: (batch_size,).
        """
        batch_size = x.size(0)
        device = x.device

        # 1. Project encoder features to SNN input dimension
        snn_input_static = self.input_projection(x) # Shape: (batch_size, self.hidden_size)

        # Initialize membrane potential list manually since init_hidden=False
        # Use the layer's init_leaky() method which returns tensor of zeros with correct shape/device
        mem_states = [layer.init_leaky() for layer in self.snn_layers]
        # Ensure states are on the correct device (init_leaky might default to CPU)
        mem_states = [mem.to(device) for mem in mem_states]


        # Store spikes from the *last* layer over time
        last_layer_spk_rec = []
        # Track total spikes per sample across all layers and time steps
        total_spikes_per_sample = torch.zeros(batch_size, device=device)

        # 2. Simulate SNN over time steps
        for step in range(self.num_time_steps):
            current_input_for_step = snn_input_static # Apply the same projected input at each step
            new_mem_states = [] # Store updated membrane potentials for the next step

            for layer_idx, snn_layer in enumerate(self.snn_layers):
                # Pass input and *current* membrane potential state
                # Since init_hidden=False, we MUST provide the state manually.
                spk, mem = snn_layer(current_input_for_step, mem_states[layer_idx])

                # Store the *returned* membrane potential for the *next* time step
                new_mem_states.append(mem)

                # Accumulate total spikes
                total_spikes_per_sample += spk.sum(dim=1) # Sum spikes across neurons for each sample

                # Store spikes from the *last* layer for aggregation
                if layer_idx == self.num_layers - 1:
                    last_layer_spk_rec.append(spk)

                # --- Handling multiple layers (if designing a stacked SNN) ---
                # if layer_idx < self.num_layers - 1:
                #     current_input_for_step = spk

            # Update membrane states for the next time step
            mem_states = new_mem_states

        # 3. Aggregate results from the last layer's spikes
        if not last_layer_spk_rec:
            aggregated_output = torch.zeros(batch_size, self.output_features, device=device)
        else:
            spk_rec_tensor = torch.stack(last_layer_spk_rec, dim=0)
            aggregated_output = torch.sum(spk_rec_tensor, dim=0)

        return aggregated_output, total_spikes_per_sample

    def get_output_features(self):
        """Returns the number of output features from the SNN core."""
        return self.output_features
