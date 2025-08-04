"""
Exoplanet Decoder for Classification and Regression Tasks
=========================================================

This module implements the ExoplanetDecoder class, which converts spike trains from the
DU Core into meaningful exoplanet detection outputs. This decoder represents the first
concrete implementation of the DecoderInterface, demonstrating the plug-and-play
architecture of the GIF framework for domain-specific applications.

The decoder supports dual-mode operation to fulfill the research validation requirements:
1. **Classification Mode**: Binary exoplanet detection (Planet/No Planet)
2. **Regression Mode**: Continuous physical parameter estimation (e.g., planet radius)

This dual capability enables comprehensive evaluation of the GIF framework's ability to
perform both discrete decision-making and continuous parameter estimation tasks.

Key Features:
============

**Population Coding for Classification**: The decoder uses population coding principles
to interpret spike patterns. The neuron population's collective activity is analyzed
to make classification decisions, mimicking how biological neural networks process
information through distributed representations.

**Linear Readout for Regression**: For continuous parameter estimation, the decoder
employs a trainable linear layer that maps spike count patterns to physical values.
This approach enables the system to learn complex mappings from neural activity to
real-world parameters.

**Dual-Mode Architecture**: The same decoder can operate in both classification and
regression modes, demonstrating the flexibility of the spike-based representation
and the decoder's ability to extract different types of information from the same
neural activity patterns.

Technical Implementation:
========================

**Classification Algorithm**:
1. Sum spikes across time for each output neuron
2. Identify neuron with maximum spike count
3. Map neuron index to class label

**Regression Algorithm**:
1. Sum spikes across time to create activity vector
2. Pass through trainable linear transformation
3. Output continuous physical parameters

**Biological Inspiration**: Both approaches are inspired by neuroscience research on
how biological neural networks decode information from spike patterns, providing
biological plausibility to the artificial system.

Integration with GIF Framework:
==============================

The ExoplanetDecoder seamlessly integrates with the GIF orchestrator through the
DecoderInterface contract. As a PyTorch module, it supports gradient-based learning
for the regression components while maintaining compatibility with the spike-based
processing pipeline.

Example Usage:
=============

    from applications.poc_exoplanet.decoders.exoplanet_decoder import ExoplanetDecoder
    
    # Create decoder for binary classification
    decoder = ExoplanetDecoder(input_size=10, output_size=1)
    
    # Classification mode
    class_prediction = decoder.decode(output_spikes, mode='classification')
    print(f"Prediction: {'Planet' if class_prediction == 1 else 'No Planet'}")
    
    # Regression mode for parameter estimation
    radius_estimate = decoder.decode(output_spikes, mode='regression')
    print(f"Estimated planet radius ratio: {radius_estimate:.4f}")
    
    # Integration with GIF framework
    from gif_framework.orchestrator import GIF
    gif_model = GIF(du_core)
    gif_model.attach_decoder(decoder)

This decoder enables the GIF framework to produce meaningful scientific outputs from
spike-based neural processing, bridging the gap between neuromorphic computation and
real-world astronomical applications.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Union, Optional, Literal
import warnings

from gif_framework.interfaces.base_interfaces import DecoderInterface, SpikeTrain, Action


class ExoplanetDecoder(DecoderInterface, nn.Module):
    """
    Dual-mode decoder for exoplanet detection and parameter estimation.
    
    This decoder converts spike trains from the DU Core into meaningful outputs for
    exoplanet science applications. It supports both classification (binary planet
    detection) and regression (continuous parameter estimation) modes, enabling
    comprehensive evaluation of the GIF framework's capabilities.
    
    The decoder implements two complementary decoding strategies:
    
    **Classification Mode (Population Coding)**:
    - Analyzes collective spike activity across neuron populations
    - Uses winner-take-all principle for binary classification
    - Biologically inspired by cortical decision-making mechanisms
    
    **Regression Mode (Linear Readout)**:
    - Employs trainable linear transformation of spike patterns
    - Maps neural activity to continuous physical parameters
    - Enables learning of complex activity-to-parameter relationships
    
    Key Advantages:
    - **Dual Functionality**: Single decoder handles multiple task types
    - **Biological Plausibility**: Based on neuroscience decoding principles
    - **Trainable Components**: Supports gradient-based learning for regression
    - **Flexible Output**: Adapts to different scientific requirements
    
    Attributes:
        input_size (int): Number of input neurons from DU Core
        output_size (int): Number of output parameters for regression mode
        regression_layer (nn.Linear): Trainable layer for parameter estimation
        
    Example:
        # Create decoder for exoplanet detection
        decoder = ExoplanetDecoder(input_size=32, output_size=2)
        
        # Binary classification
        is_planet = decoder.decode(spike_train, mode='classification')
        
        # Parameter estimation
        params = decoder.decode(spike_train, mode='regression')
    """
    
    def __init__(
        self, 
        input_size: int, 
        output_size: int = 1,
        classification_threshold: float = 0.5
    ) -> None:
        """
        Initialize the exoplanet decoder.
        
        Args:
            input_size (int): Number of input neurons (output size of DU Core).
                            Must match the output dimension of the connected SNN.
            output_size (int): Number of continuous parameters to estimate in
                             regression mode. Default 1 for single parameter
                             estimation (e.g., planet radius ratio).
            classification_threshold (float): Decision threshold for binary
                                            classification when using probabilistic
                                            interpretation. Default 0.5.
                                            
        Raises:
            ValueError: If input_size or output_size are not positive integers.
            
        Example:
            # Decoder for binary classification + radius estimation
            decoder = ExoplanetDecoder(input_size=64, output_size=1)
            
            # Decoder for multi-parameter estimation
            decoder = ExoplanetDecoder(input_size=64, output_size=3)  # radius, period, inclination
        """
        # Initialize both parent classes
        DecoderInterface.__init__(self)
        nn.Module.__init__(self)
        
        # Validate parameters
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(f"input_size must be a positive integer, got {input_size}")
            
        if not isinstance(output_size, int) or output_size <= 0:
            raise ValueError(f"output_size must be a positive integer, got {output_size}")
            
        if not isinstance(classification_threshold, (int, float)) or not (0 < classification_threshold < 1):
            raise ValueError(f"classification_threshold must be between 0 and 1, got {classification_threshold}")
        
        # Store configuration
        self.input_size = input_size
        self.output_size = output_size
        self.classification_threshold = classification_threshold
        
        # Create trainable linear layer for regression mode
        self.regression_layer = nn.Linear(input_size, output_size)

        # Create trainable linear layer for classification mode (outputs logits)
        self.classification_layer = nn.Linear(input_size, 2)  # Binary classification: 2 classes

        # Initialize weights using specialized initialization for spike-based inputs
        # Use smaller weights since spike counts can be large
        self._initialize_spike_optimized_weights()
        
        # Statistics for monitoring and debugging
        self._last_decoding_stats = {
            'input_shape': None,
            'total_input_spikes': 0,
            'spike_rate': 0.0,
            'mode_used': None,
            'output_value': None
        }
    
    def decode(
        self, 
        spike_train: SpikeTrain, 
        mode: Literal['classification', 'regression'] = 'classification'
    ) -> Action:
        """
        Decode spike train into meaningful exoplanet detection output.
        
        This method implements the core decoding logic, supporting both classification
        and regression modes. The choice of mode determines the interpretation of the
        spike patterns and the type of output generated.
        
        Args:
            spike_train (SpikeTrain): Input spike train from DU Core with shape
                                    [num_time_steps, batch_size, num_neurons] or
                                    [num_time_steps, num_neurons]. The spike train
                                    represents the temporal neural activity patterns
                                    encoding the DU Core's analysis.
            mode (str): Decoding mode, either 'classification' or 'regression'.
                       - 'classification': Binary planet detection using population coding
                       - 'regression': Continuous parameter estimation using linear readout
                       
        Returns:
            Action: Decoded output in format appropriate for the specified mode:
                   - Classification: Integer class label (0=No Planet, 1=Planet)
                   - Regression: Float or tensor with estimated parameter values
                   
        Raises:
            ValueError: If spike_train has invalid shape or mode is not supported.
            RuntimeError: If decoding process fails.
            
        Example:
            # Binary classification
            prediction = decoder.decode(spike_train, mode='classification')
            print(f"Planet detected: {prediction == 1}")
            
            # Parameter estimation
            radius_ratio = decoder.decode(spike_train, mode='regression')
            print(f"Estimated radius ratio: {radius_ratio:.4f}")
        """
        # Validate inputs
        spike_train = self._validate_spike_train(spike_train)
        self._validate_mode(mode)
        
        # Calculate spike counts for population analysis
        spike_counts = self._calculate_spike_counts(spike_train)
        
        # Update decoding statistics
        self._update_decoding_stats(spike_train, spike_counts, mode)
        
        # Perform mode-specific decoding
        if mode == 'classification':
            return self._decode_classification(spike_counts)
        elif mode == 'regression':
            return self._decode_regression(spike_counts)
        else:
            raise ValueError(f"Unsupported decoding mode: {mode}")
    
    def _validate_spike_train(self, spike_train: SpikeTrain) -> torch.Tensor:
        """
        Validate and format input spike train.
        
        Args:
            spike_train: Input spike train tensor.
            
        Returns:
            torch.Tensor: Validated and properly formatted spike train.
            
        Raises:
            ValueError: If spike train format is invalid.
        """
        if not isinstance(spike_train, torch.Tensor):
            raise ValueError(f"spike_train must be a torch.Tensor, got {type(spike_train)}")
        
        # Handle different input shapes
        if spike_train.dim() == 2:
            # Shape: [num_time_steps, num_neurons]
            if spike_train.size(1) != self.input_size:
                raise ValueError(
                    f"spike_train last dimension ({spike_train.size(1)}) must match "
                    f"decoder input_size ({self.input_size})"
                )
        elif spike_train.dim() == 3:
            # Shape: [num_time_steps, batch_size, num_neurons]
            if spike_train.size(2) != self.input_size:
                raise ValueError(
                    f"spike_train last dimension ({spike_train.size(2)}) must match "
                    f"decoder input_size ({self.input_size})"
                )
            # For now, handle only single batch
            if spike_train.size(1) != 1:
                warnings.warn(
                    f"Decoder currently handles single batch only, "
                    f"using first batch from {spike_train.size(1)} batches"
                )
            spike_train = spike_train[:, 0, :]  # Extract first batch
        else:
            raise ValueError(
                f"spike_train must be 2D or 3D tensor, got {spike_train.dim()}D"
            )
        
        return spike_train

    def _validate_mode(self, mode: str) -> None:
        """Validate decoding mode parameter."""
        valid_modes = ['classification', 'regression']
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

    def _calculate_spike_counts(self, spike_train: torch.Tensor) -> torch.Tensor:
        """
        Calculate total spike counts for each neuron across time.

        Args:
            spike_train: Input spike train [num_time_steps, num_neurons].

        Returns:
            torch.Tensor: Spike counts for each neuron [num_neurons].
        """
        # Sum spikes across time dimension for population coding
        spike_counts = spike_train.sum(dim=0)  # Shape: [num_neurons]
        return spike_counts

    def _decode_classification(self, spike_counts: torch.Tensor) -> int:
        """
        Perform binary classification using population coding.

        This method implements the winner-take-all principle where the neuron
        with the highest spike count determines the classification decision.
        For binary classification, we interpret the relative activity levels
        to make the planet/no-planet decision.

        Args:
            spike_counts: Spike counts for each neuron [num_neurons].

        Returns:
            int: Classification result (0=No Planet, 1=Planet).
        """
        if len(spike_counts) == 0:
            warnings.warn("Empty spike counts, returning default classification (0)")
            return 0

        # For binary classification, we can use different strategies:
        if self.input_size == 1:
            # Single output neuron: threshold-based decision
            spike_rate = spike_counts[0].item()
            return 1 if spike_rate > self.classification_threshold else 0

        elif self.input_size == 2:
            # Two neurons: winner-take-all between planet/no-planet neurons
            # Neuron 0 = No Planet, Neuron 1 = Planet
            neuron_0_spikes = spike_counts[0].item()
            neuron_1_spikes = spike_counts[1].item()

            # Winner-take-all: return index of neuron with most spikes
            return 1 if neuron_1_spikes > neuron_0_spikes else 0

        else:
            # Multiple neurons: use total activity level
            total_activity = spike_counts.sum().item()
            # Normalize by number of neurons and use threshold
            avg_activity = total_activity / self.input_size
            return 1 if avg_activity > self.classification_threshold else 0

    def _decode_regression(self, spike_counts: torch.Tensor) -> Union[float, torch.Tensor]:
        """
        Perform regression using linear readout layer.

        This method passes the spike count vector through a trainable linear
        layer to estimate continuous physical parameters. The linear layer
        learns the mapping from neural activity patterns to parameter values.

        Args:
            spike_counts: Spike counts for each neuron [num_neurons].

        Returns:
            Union[float, torch.Tensor]: Estimated parameter value(s).
                                      Single float for output_size=1,
                                      tensor for multiple parameters.
        """
        # Ensure spike_counts is the right shape for linear layer
        if spike_counts.dim() == 1:
            spike_counts = spike_counts.unsqueeze(0)  # Add batch dimension

        # Pass through linear layer
        output = self.regression_layer(spike_counts)

        # Remove batch dimension and return appropriate format
        output = output.squeeze(0)

        if self.output_size == 1:
            return float(output.item())
        else:
            return output

    def _update_decoding_stats(
        self,
        spike_train: torch.Tensor,
        spike_counts: torch.Tensor,
        mode: str
    ) -> None:
        """Update internal decoding statistics."""
        total_spikes = spike_train.sum().item()
        num_time_steps = spike_train.size(0)

        self._last_decoding_stats = {
            'input_shape': list(spike_train.shape),
            'total_input_spikes': int(total_spikes),
            'spike_rate': float(total_spikes / (num_time_steps * self.input_size)),
            'mode_used': mode,
            'spike_counts': spike_counts.tolist()
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Get decoder configuration parameters.

        Returns:
            Dict[str, Any]: Configuration dictionary containing all decoder parameters
                          and recent decoding statistics.

        Example:
            config = decoder.get_config()
            print(f"Input size: {config['input_size']}")
            print(f"Last spike rate: {config['last_decoding_stats']['spike_rate']:.3f}")
        """
        return {
            'decoder_type': 'ExoplanetDecoder',
            'input_size': self.input_size,
            'output_size': self.output_size,
            'classification_threshold': self.classification_threshold,
            'supported_modes': ['classification', 'regression'],
            'classification_method': 'population_coding',
            'regression_method': 'linear_readout',
            'trainable_parameters': sum(p.numel() for p in self.parameters()),
            'last_decoding_stats': self._last_decoding_stats.copy()
        }

    def get_decoding_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the most recent decoding operation.

        Returns:
            Dict[str, Any]: Statistics from last decode() call.
        """
        return self._last_decoding_stats.copy()

    def set_classification_threshold(self, threshold: float) -> None:
        """
        Update the classification threshold.

        Args:
            threshold (float): New threshold value between 0 and 1.

        Raises:
            ValueError: If threshold is not in valid range.
        """
        if not isinstance(threshold, (int, float)) or not (0 < threshold < 1):
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
        self.classification_threshold = threshold

    def reset_regression_layer(self) -> None:
        """
        Reset the regression layer weights to random initialization.

        This method is useful for retraining or transfer learning scenarios.
        """
        nn.init.xavier_uniform_(self.regression_layer.weight)
        nn.init.zeros_(self.regression_layer.bias)

    def forward(self, spike_counts: torch.Tensor) -> torch.Tensor:
        """
        PyTorch forward pass for regression mode (used during training).

        Args:
            spike_counts: Spike count tensor [batch_size, input_size].

        Returns:
            torch.Tensor: Regression outputs [batch_size, output_size].
        """
        return self.regression_layer(spike_counts)

    def forward_classification(self, spike_counts: torch.Tensor) -> torch.Tensor:
        """
        PyTorch forward pass for classification mode (returns logits for training).

        Args:
            spike_counts: Spike count tensor [batch_size, input_size].

        Returns:
            torch.Tensor: Classification logits [batch_size, 2] for binary classification.
        """
        # Normalize spike counts to prevent extreme values
        # Use adaptive normalization based on the current batch statistics
        normalized_counts = self._normalize_spike_counts(spike_counts)
        return self.classification_layer(normalized_counts)

    def _normalize_spike_counts(self, spike_counts: torch.Tensor) -> torch.Tensor:
        """
        Normalize spike counts to reasonable range for neural network processing.

        Args:
            spike_counts: Raw spike counts [batch_size, input_size]

        Returns:
            torch.Tensor: Normalized spike counts in range suitable for linear layers
        """
        # Method 1: Divide by time steps to get spike rates (more biologically plausible)
        # Assume typical time series length of ~1000-5000 steps
        typical_time_steps = 4000.0
        spike_rates = spike_counts / typical_time_steps

        # Method 2: Additional scaling to keep values in reasonable range for neural networks
        # Scale to roughly [-1, 1] range for better gradient flow
        scaled_rates = spike_rates * 10.0  # Scale up to make differences more pronounced

        return scaled_rates

    def _initialize_spike_optimized_weights(self) -> None:
        """
        Initialize weights optimized for spike-based inputs.

        Spike counts can be much larger than typical neural network inputs,
        so we use smaller initial weights to prevent gradient explosion.
        """
        # For classification layer: use smaller weights since spike counts are normalized
        # but can still be larger than typical inputs
        nn.init.normal_(self.classification_layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classification_layer.bias)

        # For regression layer: use Xavier but with smaller scale
        nn.init.xavier_uniform_(self.regression_layer.weight, gain=0.1)
        nn.init.zeros_(self.regression_layer.bias)

    def __repr__(self) -> str:
        """String representation of the decoder."""
        return (
            f"ExoplanetDecoder(input_size={self.input_size}, "
            f"output_size={self.output_size}, "
            f"threshold={self.classification_threshold})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Exoplanet Decoder (Dual-Mode)\n"
            f"  Input neurons: {self.input_size}\n"
            f"  Output parameters: {self.output_size}\n"
            f"  Classification: Population coding (threshold={self.classification_threshold})\n"
            f"  Regression: Linear readout ({sum(p.numel() for p in self.parameters())} parameters)\n"
            f"  Last decoding: {self._last_decoding_stats.get('mode_used', 'none')} mode"
        )
