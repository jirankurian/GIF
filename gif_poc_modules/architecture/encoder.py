# POC/gif_poc_modules/architecture/encoder.py

import torch
import torch.nn as nn
import abc # Abstract Base Classes

class BaseEncoder(nn.Module, abc.ABC):
    """Abstract base class for encoders to ensure consistent interface."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg # Store config

    @abc.abstractmethod
    def forward(self, x):
        """Processes the input tensor."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_output_features(self):
        """Returns the number of features in the output tensor."""
        raise NotImplementedError

class CNNEncoder(BaseEncoder):
    """
    1D Convolutional Neural Network Encoder for time series data.
    Expects input shape: (N, C, L) where N=batch, C=channels/features, L=sequence length.
    """
    def __init__(self, cfg, input_channels=1): # Accepts cfg
        super().__init__(cfg=cfg)
        self.input_channels = input_channels

        # Use parameters from the config object
        self.seq_length = cfg.SEQ_LENGTH
        cnn_out_channels = cfg.CNN_OUT_CHANNELS
        cnn_kernel_sizes = cfg.CNN_KERNEL_SIZES

        if len(cnn_out_channels) != len(cnn_kernel_sizes):
            raise ValueError("CNN_OUT_CHANNELS and CNN_KERNEL_SIZES must have the same length in config.")

        cnn_layers = []
        in_channels = self.input_channels
        current_seq_len = self.seq_length # Keep track of seq length changes if padding='valid' was used

        for i in range(len(cnn_out_channels)):
            # Use padding='same' equivalent for Conv1d by setting padding = kernel_size // 2 (assuming stride=1)
            # This keeps the sequence length the same after convolution.
            padding_val = cnn_kernel_sizes[i] // 2
            cnn_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=cnn_out_channels[i],
                    kernel_size=cnn_kernel_sizes[i],
                    padding=padding_val, # Maintain sequence length
                    stride=1 # Default stride
                )
            )
            cnn_layers.append(nn.ReLU())
            # Optional: Add Pooling layers if needed, but be mindful of sequence length changes
            # cnn_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            # current_seq_len = current_seq_len // 2 # Update seq len if pooling
            in_channels = cnn_out_channels[i]

        self.cnn_module = nn.Sequential(*cnn_layers)

        # Calculate the output feature size dynamically
        # We need to run a dummy input through the CNN layers
        with torch.no_grad():
            # Ensure dummy input matches expected shape (N, C, L)
            dummy_input = torch.zeros(1, self.input_channels, self.seq_length)
            dummy_output = self.cnn_module(dummy_input)
            # The output shape will be (N, last_out_channels, final_seq_len)
            # We flatten the spatial dimensions (C * L) for the linear layer input
            self._output_features = dummy_output.view(1, -1).shape[1]
            print(f"CNNEncoder calculated output features: {self._output_features} (from shape {dummy_output.shape})")

    def forward(self, x):
        """
        Processes the input tensor x.
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, L).
        Returns:
            torch.Tensor: Output features of shape (N, self._output_features).
        """
        if x.ndim != 3 or x.shape[1] != self.input_channels:
             # Try to fix common shape issues if input is (N, L, C=1) or (N, L)
             if x.ndim == 3 and x.shape[2] == self.input_channels:
                 x = x.permute(0, 2, 1) # Reshape (N, L, C) to (N, C, L)
             elif x.ndim == 2 and self.input_channels == 1:
                 x = x.unsqueeze(1) # Reshape (N, L) to (N, C=1, L)
             else:
                raise ValueError(f"CNNEncoder input shape mismatch. Expected (N, {self.input_channels}, {self.seq_length}), got {x.shape}")

        features = self.cnn_module(x)
        # Flatten the output for the subsequent linear layer
        features = features.view(x.size(0), -1)
        return features

    def get_output_features(self):
        """Returns the calculated number of output features after flattening."""
        return self._output_features

class LSTMEncoder(BaseEncoder):
    """
    LSTM Encoder for time series data.
    Expects input shape: (N, L, F) where N=batch, L=sequence length, F=input features per step.
    Outputs the last hidden state.
    """
    def __init__(self, cfg, input_features=1): # Accepts cfg
        super().__init__(cfg=cfg)
        self.input_features = input_features

        # Use parameters from the config object
        self.hidden_size = cfg.LSTM_HIDDEN_SIZE
        self.num_layers = cfg.LSTM_NUM_LAYERS

        # LSTM Layer
        self.lstm_module = nn.LSTM(
            input_size=self.input_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True, # Input/output tensors are provided as (batch, seq, feature)
            bidirectional=False # Set to True if bidirectional needed, output_features doubles
        )
        self._output_features = self.hidden_size * (2 if self.lstm_module.bidirectional else 1)
        print(f"LSTMEncoder output features: {self._output_features}")

    def forward(self, x):
        """
        Processes the input tensor x.
        Args:
            x (torch.Tensor): Input tensor of shape (N, L, F).
        Returns:
            torch.Tensor: Last hidden state of shape (N, self._output_features).
        """
        # Ensure input shape is (N, L, F)
        if x.ndim != 3 or x.shape[-1] != self.input_features:
            # Try to fix common shape issue if input is (N, L) and features=1
            if x.ndim == 2 and self.input_features == 1:
                x = x.unsqueeze(-1) # Reshape (N, L) to (N, L, F=1)
            else:
                raise ValueError(f"LSTMEncoder input feature dimension mismatch. Expected F={self.input_features} in shape (N, L, F), got input shape {x.shape}")

        # LSTM outputs: output_seq, (h_n, c_n)
        # output_seq shape: (N, L, D * H_out) where D=2 if bidirectional else 1
        # h_n shape: (D * num_layers, N, H_out) - hidden state for t = seq_len
        # c_n shape: (D * num_layers, N, H_out) - cell state for t = seq_len
        lstm_out, (h_n, c_n) = self.lstm_module(x)

        # We typically use the hidden state of the last layer (and last time step)
        # h_n is stacked (D*num_layers, N, H_out). We want the final layer's state.
        # If bidirectional, the last two entries [-2, :, :] and [-1, :, :] are the forward and backward final states.
        if self.lstm_module.bidirectional:
             # Concatenate final forward and backward hidden states
             last_hidden_state = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
             # Get the hidden state of the last layer
             last_hidden_state = h_n[-1, :, :]

        return last_hidden_state # Shape: (N, D * H_out)

    def get_output_features(self):
        """Returns the size of the LSTM's hidden state (doubled if bidirectional)."""
        return self._output_features

def get_encoder(cfg):
    """
    Factory function to create the appropriate encoder based on the config.

    Args:
        cfg (module): The configuration module.

    Returns:
        BaseEncoder: An instance of CNNEncoder or LSTMEncoder.
    """
    ann_type = cfg.ANN_TYPE
    print(f"Creating encoder of type: {ann_type}")

    if ann_type == 'CNN':
        # CNN expects input_channels=1 for our (N, L, 1) -> (N, 1, L) data
        return CNNEncoder(cfg=cfg, input_channels=1)
    elif ann_type == 'LSTM':
        # LSTM expects input_features=1 for our (N, L, 1) data
        return LSTMEncoder(cfg=cfg, input_features=1)
    else:
        raise ValueError(f"Unsupported ANN_TYPE '{ann_type}' specified in config.")
