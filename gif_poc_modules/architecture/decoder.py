# POC/gif_poc_modules/architecture/decoder.py

import torch
import torch.nn as nn

class ClassificationDecoder(nn.Module):
    """
    Simple Linear Decoder head for the binary classification task.
    Takes features from the SNN core and outputs raw logits.
    """
    def __init__(self, cfg, input_features): # Accepts config object (cfg)
        super().__init__()
        self.cfg = cfg
        self.input_features = input_features
        # Output size for classification from config
        self.output_size = cfg.OUTPUT_CLS_SIZE

        # A single linear layer is often sufficient for a decoder head
        self.linear = nn.Linear(self.input_features, self.output_size)
        print(f"ClassificationDecoder initialized: Input={input_features}, Output={self.output_size}")

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Aggregated features from the SNN core.
                              Shape: (batch_size, input_features).
        Returns:
            torch.Tensor: Raw output logits for classification.
                          Shape: (batch_size, output_size).
        """
        logits = self.linear(x)
        return logits

class RegressionDecoder(nn.Module):
    """
    Simple Linear Decoder head for the regression task.
    Takes features from the SNN core and outputs predicted values.
    """
    def __init__(self, cfg, input_features): # Accepts config object (cfg)
        super().__init__()
        self.cfg = cfg
        self.input_features = input_features
        # Output size for regression from config
        self.output_size = cfg.OUTPUT_REG_SIZE

        # A single linear layer
        self.linear = nn.Linear(self.input_features, self.output_size)
        print(f"RegressionDecoder initialized: Input={input_features}, Output={self.output_size}")

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Aggregated features from the SNN core.
                              Shape: (batch_size, input_features).
        Returns:
            torch.Tensor: Predicted values for regression.
                          Shape: (batch_size, output_size).
        """
        predictions = self.linear(x)
        # No activation function typically needed for regression outputs,
        # unless you want to constrain the output range (e.g., Sigmoid for [0,1], ReLU for non-negative).
        # The physics penalty handles the [0,1] constraint in the loss for this POC.
        return predictions
