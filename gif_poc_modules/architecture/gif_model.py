# POC/gif_poc_modules/architecture/gif_model.py

import torch
import torch.nn as nn

# Import components using RELATIVE paths within the same package level
# This makes the package structure more robust.
from .encoder import get_encoder, CNNEncoder, LSTMEncoder # Use '.' for relative import
from .core_du import SnnCoreDU # Use '.' for relative import
from .decoder import ClassificationDecoder, RegressionDecoder # Use '.' for relative import

class GIFModel(nn.Module):
    """
    Assembled General Intelligence Framework (GIF) Proof-of-Concept Model.

    Connects the selected Encoder (CNN or LSTM) to the SNN Core (DU proxy)
    via a linear projection layer, and then feeds the SNN output to
    separate Decoder heads for classification and regression tasks.
    """
    def __init__(self, cfg): # Accepts the global config object
        super().__init__()
        self.config = cfg # Store config for potential future use within the model if needed

        # 1. Instantiate the Encoder based on config
        #    The get_encoder function handles selecting CNN or LSTM
        #    and calculates its output feature size automatically.
        self.encoder = get_encoder(cfg=cfg)
        encoder_output_features = self.encoder.get_output_features()
        print(f"GIFModel: Encoder instantiated ({cfg.ANN_TYPE}) with output features: {encoder_output_features}")

        # 2. Linear Layer to bridge Encoder output to SNN Core input size
        #    This allows flexibility if the encoder output size doesn't match
        #    the desired SNN hidden size.
        self.ann_to_snn_input = nn.Linear(encoder_output_features, cfg.SNN_HIDDEN_SIZE)
        print(f"GIFModel: ANN->SNN projection layer: {encoder_output_features} -> {cfg.SNN_HIDDEN_SIZE}")

        # 3. Instantiate the SNN Core (DU Proxy)
        #    Input features must match the output of the ann_to_snn_input layer.
        self.core_du = SnnCoreDU(cfg=cfg, input_features=cfg.SNN_HIDDEN_SIZE)
        core_output_features = self.core_du.get_output_features()
        print(f"GIFModel: SNN Core (DU) instantiated with output features: {core_output_features}")


        # 4. Instantiate Decoder Heads
        #    These take the aggregated output from the SNN core.
        self.decoder_cls = ClassificationDecoder(cfg=cfg, input_features=core_output_features)
        self.decoder_reg = RegressionDecoder(cfg=cfg, input_features=core_output_features)
        print(f"GIFModel: Decoders instantiated.")

    def forward(self, x):
        """
        Defines the forward pass of the GIF model.

        Args:
            x (torch.Tensor): Input batch tensor. Shape depends on the encoder type:
                              - CNN: Expects (N, C, L) e.g., (64, 1, 100)
                              - LSTM: Expects (N, L, F) e.g., (64, 100, 1)

        Returns:
            tuple: (cls_output, reg_output, total_spikes_per_sample)
                   - cls_output (torch.Tensor): Logits from the classification decoder.
                   - reg_output (torch.Tensor): Predictions from the regression decoder.
                   - total_spikes_per_sample (torch.Tensor): Total spikes from the SNN core per sample.
        """
        # --- Input Shape Handling ---
        # Ensure input shape matches the expected format for the chosen encoder
        # Data loader should ideally handle this, but double-check here.
        if isinstance(self.encoder, LSTMEncoder):
            # LSTM expects (N, L, F)
            if x.ndim == 2 and self.encoder.input_features == 1:
                # If input is (N, L) and LSTM expects 1 feature, add feature dim
                x = x.unsqueeze(-1) # (N, L) -> (N, L, 1)
            elif x.ndim != 3 or x.shape[-1] != self.encoder.input_features:
                 raise ValueError(f"LSTMEncoder requires input shape (N, L, F={self.encoder.input_features}), got {x.shape}")
        elif isinstance(self.encoder, CNNEncoder):
             # CNN expects (N, C, L)
             if x.ndim == 2 and self.encoder.input_channels == 1:
                 # If input is (N, L) and CNN expects 1 channel, add channel dim
                 x = x.unsqueeze(1) # (N, L) -> (N, C=1, L)
             elif x.ndim == 3 and x.shape[1] != self.encoder.input_channels:
                 # Allow permute if shape is (N, L, C) instead of (N, C, L)
                 if x.shape[2] == self.encoder.input_channels:
                     x = x.permute(0, 2, 1)
                 else:
                    raise ValueError(f"CNNEncoder requires input shape (N, C={self.encoder.input_channels}, L), got {x.shape}")
             elif x.ndim != 3:
                  raise ValueError(f"CNNEncoder requires 3D input (N, C, L), got {x.ndim}D shape {x.shape}")


        # --- Forward Pass through Modules ---
        # 1. Encoder: Extract features from input sequence
        #    Output shape: (batch_size, encoder_output_features)
        encoded_features = self.encoder(x)

        # 2. Projection Layer: Map encoder features to SNN input size
        #    Output shape: (batch_size, cfg.SNN_HIDDEN_SIZE)
        snn_input_features = self.ann_to_snn_input(encoded_features)

        # 3. SNN Core (DU): Process features over time steps
        #    Output: aggregated features (sum of spikes) and total spike count
        #    aggregated_output shape: (batch_size, core_output_features)
        #    total_spikes shape: (batch_size,)
        core_output_aggregated, total_spikes = self.core_du(snn_input_features)

        # 4. Decoders: Generate task-specific outputs from SNN core output
        #    cls_output shape: (batch_size, cfg.OUTPUT_CLS_SIZE)
        #    reg_output shape: (batch_size, cfg.OUTPUT_REG_SIZE)
        cls_output = self.decoder_cls(core_output_aggregated)
        reg_output = self.decoder_reg(core_output_aggregated)

        # Return all relevant outputs
        return cls_output, reg_output, total_spikes
