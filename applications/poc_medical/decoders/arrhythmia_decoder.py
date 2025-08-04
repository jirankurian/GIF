"""
Arrhythmia Decoder for Multi-Class Cardiac Classification
=========================================================

This module implements the Arrhythmia_Decoder class, which converts spike trains from the
DU Core into meaningful arrhythmia classification outputs. This decoder represents the medical
domain implementation of the DecoderInterface, demonstrating the plug-and-play architecture
of the GIF framework for clinical diagnostic applications.

The decoder uses a trainable linear readout layer that interprets the population code of
the SNN's output neurons to classify cardiac arrhythmias according to established medical
standards. This approach separates the complex temporal pattern learning (handled by the SNN)
from the final classification decision (handled by the linear layer).

Key Features:
============

**Spike Integration and Population Decoding:**
The decoder first integrates spikes over the entire simulation window to extract the
"final vote" from each output neuron. This population vector represents the DU Core's
learned representation of the cardiac pattern.

**Trainable Linear Readout:**
A torch.nn.Linear layer maps the integrated spike counts to class logits for each
possible arrhythmia type. This layer is trainable and learns the optimal mapping
from neural population codes to diagnostic categories.

**AAMI Standard Classification:**
The decoder supports classification according to the Association for the Advancement
of Medical Instrumentation (AAMI) standard arrhythmia classes:
- Normal Sinus Rhythm (N)
- Atrial Fibrillation (AFIB)
- Atrial Flutter (AFL)
- Supraventricular Tachycardia (SVT)
- Ventricular Tachycardia (VT)
- Ventricular Fibrillation (VF)
- Premature Ventricular Contraction (PVC)
- Premature Atrial Contraction (PAC)

**Biological Inspiration:**
This decoding approach mimics cortical decision-making mechanisms where populations
of neurons collectively encode information, and downstream areas integrate this
activity to make categorical decisions.

Integration with GIF Framework:
==============================

The Arrhythmia_Decoder seamlessly integrates with the GIF orchestrator through the
DecoderInterface contract, enabling plug-and-play compatibility with any SNN architecture
in the DU Core. The decoder's linear layer can be trained end-to-end with the SNN
using standard backpropagation through time.

Example Usage:
=============

    from applications.poc_medical.decoders.arrhythmia_decoder import Arrhythmia_Decoder
    
    # Create decoder for 8-class arrhythmia classification
    decoder = Arrhythmia_Decoder(input_size=100, num_classes=8)
    
    # Decode spike train to arrhythmia class
    prediction = decoder.decode(output_spikes)
    print(f"Predicted arrhythmia: {prediction}")
    
    # Training mode for end-to-end learning
    decoder.train()
    logits = decoder.forward(output_spikes)  # For loss calculation
    
    # Integration with GIF framework
    from gif_framework.orchestrator import GIF
    gif_model = GIF(du_core)
    gif_model.attach_decoder(decoder)

This decoder enables the GIF framework to perform clinical-grade arrhythmia detection
with the efficiency and biological plausibility of spiking neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings

from gif_framework.interfaces.base_interfaces import DecoderInterface, SpikeTrain, Action


class Arrhythmia_Decoder(DecoderInterface, nn.Module):
    """
    Trainable linear readout decoder for arrhythmia classification.
    
    This decoder converts spike trains from the DU Core into arrhythmia class predictions
    using a two-stage process: spike integration followed by linear classification.
    The approach separates temporal pattern learning (SNN) from decision making (linear layer).
    
    The decoder supports both inference mode (returning class names) and training mode
    (returning logits for loss calculation), making it suitable for end-to-end learning
    with the GIF framework.
    
    Architecture:
    - Spike Integration: Temporal summation of spike counts per neuron
    - Linear Readout: Fully connected layer mapping spike counts to class logits
    - Softmax Classification: Probability distribution over arrhythmia classes
    
    Attributes:
        input_size (int): Number of input neurons from DU Core
        num_classes (int): Number of arrhythmia classes to classify
        class_names (List[str]): Names of arrhythmia classes (AAMI standard)
        linear_readout (nn.Linear): Trainable linear classification layer
        dropout_rate (float): Dropout probability for regularization
        
    Example:
        >>> decoder = Arrhythmia_Decoder(input_size=100, num_classes=8)
        >>> prediction = decoder.decode(spike_train)
        >>> print(f"Predicted: {prediction}")
    """
    
    # AAMI standard arrhythmia class names
    DEFAULT_CLASS_NAMES = [
        "Normal Sinus Rhythm",
        "Atrial Fibrillation", 
        "Atrial Flutter",
        "Supraventricular Tachycardia",
        "Ventricular Tachycardia",
        "Ventricular Fibrillation",
        "Premature Ventricular Contraction",
        "Premature Atrial Contraction"
    ]
    
    def __init__(
        self,
        input_size: int,
        num_classes: int = 8,
        class_names: Optional[List[str]] = None,
        dropout_rate: float = 0.1,
        bias: bool = True
    ) -> None:
        """
        Initialize the arrhythmia decoder with specified architecture.
        
        Args:
            input_size: Number of input neurons from DU Core
            num_classes: Number of arrhythmia classes to classify
            class_names: Optional custom class names (defaults to AAMI standard)
            dropout_rate: Dropout probability for regularization during training
            bias: Whether to include bias terms in linear layer
        """
        super(Arrhythmia_Decoder, self).__init__()
        
        # Validate parameters
        if input_size <= 0:
            raise ValueError(f"Input size must be positive, got {input_size}")
        if num_classes <= 1:
            raise ValueError(f"Number of classes must be > 1, got {num_classes}")
        if not 0 <= dropout_rate <= 1:
            raise ValueError(f"Dropout rate must be in [0,1], got {dropout_rate}")
            
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Set class names
        if class_names is None:
            if num_classes <= len(self.DEFAULT_CLASS_NAMES):
                self.class_names = self.DEFAULT_CLASS_NAMES[:num_classes]
            else:
                # Generate generic names for additional classes
                base_names = self.DEFAULT_CLASS_NAMES.copy()
                for i in range(len(base_names), num_classes):
                    base_names.append(f"Arrhythmia_Class_{i+1}")
                self.class_names = base_names
        else:
            if len(class_names) != num_classes:
                raise ValueError(f"Number of class names ({len(class_names)}) must match num_classes ({num_classes})")
            self.class_names = class_names.copy()
        
        # Define neural network layers
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_readout = nn.Linear(input_size, num_classes, bias=bias)
        
        # Initialize weights using Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.linear_readout.weight)
        if bias:
            nn.init.zeros_(self.linear_readout.bias)
        
        # Decoding statistics for monitoring
        self.decoding_stats = {
            'total_decodings': 0,
            'class_predictions': {name: 0 for name in self.class_names},
            'average_confidence': 0.0,
            'average_spike_count': 0.0
        }
    
    def decode(self, spike_train: SpikeTrain) -> Action:
        """
        Decode spike train into arrhythmia class prediction.
        
        This method implements the complete decoding pipeline:
        1. Validate and process input spike train
        2. Integrate spikes over time to get population vector
        3. Apply linear readout to get class logits
        4. Convert to class prediction via softmax + argmax
        
        Args:
            spike_train: Input spike train from DU Core with shape
                        [num_time_steps, batch_size, num_neurons] or
                        [num_time_steps, num_neurons]
                        
        Returns:
            String name of predicted arrhythmia class
            
        Raises:
            ValueError: If spike train has invalid shape or contains invalid values
            RuntimeError: If decoding process fails
            
        Example:
            >>> prediction = decoder.decode(output_spikes)
            >>> print(f"Detected arrhythmia: {prediction}")
        """
        # Set to evaluation mode for inference
        self.eval()
        
        with torch.no_grad():
            # Validate and process spike train
            spike_counts = self._integrate_spikes(spike_train)
            
            # Apply linear readout
            logits = self.linear_readout(spike_counts)
            
            # Convert to class prediction
            class_probabilities = F.softmax(logits, dim=-1)
            predicted_class_idx = torch.argmax(class_probabilities, dim=-1).item()
            
            # Get class name
            predicted_class = self.class_names[predicted_class_idx]
            
            # Update statistics
            confidence = float(torch.max(class_probabilities).item())
            self._update_decoding_stats(predicted_class, confidence, spike_counts)
            
            return predicted_class
    
    def forward(self, spike_train: SpikeTrain) -> torch.Tensor:
        """
        Forward pass for training (returns logits instead of class names).
        
        This method is used during training to get raw logits that can be
        used with loss functions like CrossEntropyLoss.
        
        Args:
            spike_train: Input spike train tensor
            
        Returns:
            Class logits tensor with shape [batch_size, num_classes]
        """
        # Integrate spikes over time
        spike_counts = self._integrate_spikes(spike_train)
        
        # Apply dropout during training
        if self.training:
            spike_counts = self.dropout(spike_counts)
        
        # Linear readout to class logits
        logits = self.linear_readout(spike_counts)
        
        return logits

    def _integrate_spikes(self, spike_train: SpikeTrain) -> torch.Tensor:
        """
        Integrate spikes over time to create population vector.

        This method sums spikes for each neuron across the entire time window,
        creating a vector of total spike counts that represents the DU Core's
        final "vote" for the input pattern.

        Args:
            spike_train: Input spike train tensor

        Returns:
            Integrated spike counts tensor with shape [batch_size, num_neurons]
            or [num_neurons] for single samples
        """
        # Validate input tensor
        spike_train = self._validate_spike_train(spike_train)

        # Handle different input shapes
        if spike_train.ndim == 2:
            # Shape: [num_time_steps, num_neurons]
            spike_counts = torch.sum(spike_train, dim=0)  # Sum over time

        elif spike_train.ndim == 3:
            # Shape: [num_time_steps, batch_size, num_neurons]
            spike_counts = torch.sum(spike_train, dim=0)  # Sum over time, keep batch

        else:
            raise ValueError(f"Unsupported spike train shape: {spike_train.shape}")

        # Ensure we have the right number of input features
        expected_features = self.input_size
        actual_features = spike_counts.shape[-1]

        if actual_features != expected_features:
            raise ValueError(f"Expected {expected_features} input features, got {actual_features}")

        return spike_counts

    def _validate_spike_train(self, spike_train: SpikeTrain) -> torch.Tensor:
        """
        Validate and convert spike train to appropriate tensor format.

        Args:
            spike_train: Input spike train (tensor, numpy array, or list)

        Returns:
            Validated torch tensor
        """
        # Convert to tensor if needed
        if isinstance(spike_train, np.ndarray):
            spike_train = torch.from_numpy(spike_train).float()
        elif isinstance(spike_train, list):
            spike_train = torch.tensor(spike_train, dtype=torch.float32)
        elif not isinstance(spike_train, torch.Tensor):
            raise ValueError(f"Unsupported spike train type: {type(spike_train)}")

        # Validate tensor properties
        if spike_train.ndim < 2 or spike_train.ndim > 3:
            raise ValueError(f"Spike train must be 2D or 3D, got shape {spike_train.shape}")

        if not torch.all(torch.isfinite(spike_train)):
            raise ValueError("Spike train contains non-finite values")

        if torch.any(spike_train < 0):
            warnings.warn("Spike train contains negative values, clipping to zero")
            spike_train = torch.clamp(spike_train, min=0.0)

        return spike_train

    def _update_decoding_stats(
        self,
        predicted_class: str,
        confidence: float,
        spike_counts: torch.Tensor
    ) -> None:
        """Update decoding statistics for monitoring and analysis."""
        self.decoding_stats['total_decodings'] += 1
        self.decoding_stats['class_predictions'][predicted_class] += 1

        # Update running averages
        total = self.decoding_stats['total_decodings']
        old_conf_avg = self.decoding_stats['average_confidence']
        old_spike_avg = self.decoding_stats['average_spike_count']

        # Exponential moving average for confidence
        self.decoding_stats['average_confidence'] = (old_conf_avg * (total - 1) + confidence) / total

        # Average spike count
        current_spike_count = float(torch.mean(spike_counts).item())
        self.decoding_stats['average_spike_count'] = (old_spike_avg * (total - 1) + current_spike_count) / total

    def get_config(self) -> Dict[str, Any]:
        """
        Return decoder configuration for reproducibility and analysis.

        Returns:
            Dictionary containing decoder parameters and current statistics
        """
        return {
            'decoder_type': 'Arrhythmia_Decoder',
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'class_names': self.class_names.copy(),
            'dropout_rate': self.dropout_rate,
            'has_bias': self.linear_readout.bias is not None,
            'decoding_statistics': self.decoding_stats.copy(),
            'model_parameters': {
                'total_parameters': sum(p.numel() for p in self.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
            }
        }

    def get_class_probabilities(self, spike_train: SpikeTrain) -> Dict[str, float]:
        """
        Get probability distribution over all arrhythmia classes.

        This method provides detailed probability information for each class,
        useful for uncertainty quantification and confidence assessment.

        Args:
            spike_train: Input spike train from DU Core

        Returns:
            Dictionary mapping class names to their predicted probabilities
        """
        self.eval()

        with torch.no_grad():
            spike_counts = self._integrate_spikes(spike_train)
            logits = self.linear_readout(spike_counts)
            probabilities = F.softmax(logits, dim=-1)

            # Convert to dictionary
            prob_dict = {}
            for i, class_name in enumerate(self.class_names):
                prob_dict[class_name] = float(probabilities[i].item())

            return prob_dict

    def reset_stats(self) -> None:
        """Reset decoding statistics for fresh monitoring."""
        self.decoding_stats = {
            'total_decodings': 0,
            'class_predictions': {name: 0 for name in self.class_names},
            'average_confidence': 0.0,
            'average_spike_count': 0.0
        }

    def get_top_k_predictions(self, spike_train: SpikeTrain, k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top-k most likely arrhythmia classes with their probabilities.

        Args:
            spike_train: Input spike train from DU Core
            k: Number of top predictions to return

        Returns:
            List of (class_name, probability) tuples sorted by probability (descending)
        """
        prob_dict = self.get_class_probabilities(spike_train)

        # Sort by probability and return top k
        sorted_predictions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:k]
