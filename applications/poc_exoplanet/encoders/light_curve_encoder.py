"""
Light Curve Encoder for Exoplanet Detection
===========================================

This module implements the LightCurveEncoder class, which converts astronomical light curve
data into spike trains suitable for processing by Spiking Neural Networks. This encoder
represents the first concrete implementation of the EncoderInterface, demonstrating the
plug-and-play architecture of the GIF framework.

The encoder uses Delta Modulation, an efficient encoding scheme particularly well-suited
for time-series data like astronomical light curves. This approach creates sparse spike
representations that capture the essential temporal dynamics while being computationally
efficient for neuromorphic processing.

Key Features:
============

**Delta Modulation Encoding**: The encoder converts continuous flux measurements into
discrete spike events by detecting significant changes in brightness. This approach
naturally emphasizes the transient events (like planetary transits) that are most
important for exoplanet detection.

**Two-Channel Spike Representation**: The encoder produces a dual-channel spike train:
- Channel 1: Positive flux changes (brightness increases)
- Channel 2: Negative flux changes (brightness decreases)

This representation preserves the directional information of flux changes while maintaining
the sparse, event-driven nature required for efficient neuromorphic computation.

**Biological Inspiration**: The encoding scheme mimics how biological visual systems
respond to temporal changes rather than absolute brightness levels, making it naturally
suited for detecting transient astronomical events.

Technical Implementation:
========================

The encoder processes polars DataFrames containing 'time' and 'flux' columns from the
ExoplanetGenerator. It calculates temporal derivatives and converts significant changes
into binary spike events based on a configurable sensitivity threshold.

Mathematical Foundation:
- Delta calculation: δ(t) = flux(t) - flux(t-1)
- Positive spikes: spike₁(t) = 1 if δ(t) > threshold, else 0
- Negative spikes: spike₂(t) = 1 if δ(t) < -threshold, else 0

Integration with GIF Framework:
==============================

The LightCurveEncoder seamlessly integrates with the GIF orchestrator through the
EncoderInterface contract, enabling plug-and-play compatibility with any SNN architecture
in the DU Core. The encoded spike trains are directly compatible with the neuromorphic
simulator for realistic hardware-constrained experiments.

Example Usage:
=============

    from applications.poc_exoplanet.encoders.light_curve_encoder import LightCurveEncoder
    from data_generators.exoplanet_generator import RealisticExoplanetGenerator
    
    # Generate synthetic light curve data
    generator = RealisticExoplanetGenerator(seed=42)
    light_curve_data = generator.generate()
    
    # Create and configure encoder
    encoder = LightCurveEncoder(threshold=1e-4)
    
    # Convert light curve to spike train
    spike_train = encoder.encode(light_curve_data)
    print(f"Encoded {len(light_curve_data)} time points to {spike_train.shape} spike train")
    
    # Integration with GIF framework
    from gif_framework.orchestrator import GIF
    gif_model = GIF(du_core)
    gif_model.attach_encoder(encoder)

This encoder enables the GIF framework to process astronomical data with the efficiency
and biological plausibility of spiking neural networks, providing the foundation for
neuromorphic exoplanet detection systems.
"""

import torch
import polars as pl
import numpy as np
from typing import Any, Dict, Union, Optional
import warnings

from gif_framework.interfaces.base_interfaces import EncoderInterface, SpikeTrain


class LightCurveEncoder(EncoderInterface):
    """
    Delta modulation encoder for astronomical light curve data.
    
    This encoder converts continuous light curve measurements into sparse spike trains
    using delta modulation, an encoding scheme that emphasizes temporal changes rather
    than absolute values. This approach is particularly effective for astronomical
    time-series data where transient events (like planetary transits) are the primary
    signals of interest.
    
    The encoder produces a two-channel spike representation:
    - Channel 1: Positive flux changes (brightness increases)
    - Channel 2: Negative flux changes (brightness decreases)
    
    This dual-channel approach preserves directional information while maintaining
    the sparse, event-driven nature required for efficient neuromorphic computation.
    
    Key Advantages:
    - **Sparse Representation**: Only significant changes generate spikes
    - **Temporal Emphasis**: Naturally highlights transient events
    - **Noise Robustness**: Small fluctuations below threshold are ignored
    - **Neuromorphic Compatibility**: Event-driven processing matches hardware constraints
    
    Attributes:
        threshold (float): Sensitivity threshold for spike generation
        normalize_input (bool): Whether to normalize input flux values
        
    Example:
        # Create encoder with custom threshold
        encoder = LightCurveEncoder(threshold=5e-5, normalize_input=True)
        
        # Encode light curve data
        spike_train = encoder.encode(light_curve_dataframe)
        
        # Check encoding statistics
        config = encoder.get_config()
        print(f"Threshold: {config['threshold']}")
    """
    
    def __init__(
        self,
        threshold: float = 1e-4,
        normalize_input: bool = True,
        device: str = "cpu"
    ) -> None:
        """
        Initialize the light curve encoder.
        
        Args:
            threshold (float): Sensitivity threshold for spike generation. Flux changes
                             smaller than this value will not generate spikes. Default
                             1e-4 is suitable for typical exoplanet transit depths.
            normalize_input (bool): Whether to normalize input flux to zero mean and
                                  unit variance before encoding. Recommended for
                                  consistent behavior across different data sources.
            device (str): Device to create tensors on ("cpu", "cuda", "mps"). Default: "cpu".
                                  
        Raises:
            ValueError: If threshold is not positive.
            
        Example:
            # Standard encoder for exoplanet detection
            encoder = LightCurveEncoder(threshold=1e-4)
            
            # High-sensitivity encoder for small signals
            encoder = LightCurveEncoder(threshold=5e-5, normalize_input=True)
        """
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            raise ValueError(f"threshold must be a positive number, got {threshold}")
            
        self.threshold = float(threshold)
        self.normalize_input = bool(normalize_input)
        self.device = torch.device(device)
        self.device = torch.device(device)
        
        # Statistics for monitoring and debugging
        self._last_encoding_stats = {
            'input_length': 0,
            'total_spikes': 0,
            'positive_spikes': 0,
            'negative_spikes': 0,
            'spike_rate': 0.0
        }
    
    def encode(self, raw_data: Any) -> SpikeTrain:
        """
        Convert light curve data into a two-channel spike train using delta modulation.
        
        This method implements the core delta modulation algorithm, converting continuous
        flux measurements into discrete spike events based on temporal changes. The
        resulting spike train emphasizes transient events while maintaining sparse
        representation suitable for neuromorphic processing.
        
        Args:
            raw_data (Any): Input light curve data. Expected to be a polars.DataFrame
                          with 'time' and 'flux' columns, as generated by the
                          ExoplanetGenerator. Can also accept numpy arrays or other
                          formats that can be converted to the expected structure.
                          
        Returns:
            SpikeTrain: A PyTorch tensor with shape [num_time_steps, 2] containing
                       the encoded spike train. Channel 0 represents positive flux
                       changes, Channel 1 represents negative flux changes.
                       
        Raises:
            ValueError: If input data format is invalid or missing required columns.
            RuntimeError: If encoding process fails.
            
        Example:
            # Encode exoplanet light curve
            light_curve = generator.generate()
            spike_train = encoder.encode(light_curve)
            
            # Analyze encoding results
            print(f"Input: {len(light_curve)} time points")
            print(f"Output: {spike_train.shape} spike train")
            print(f"Total spikes: {spike_train.sum().item()}")
        """
        # Validate and extract flux data
        flux_array = self._extract_flux_data(raw_data)
        
        # Normalize input if requested
        if self.normalize_input:
            flux_array = self._normalize_flux(flux_array)
        
        # Calculate temporal differences (delta modulation)
        flux_deltas = self._calculate_flux_deltas(flux_array)
        
        # Generate two-channel spike train
        spike_train = self._generate_spike_train(flux_deltas)
        
        # Update encoding statistics
        self._update_encoding_stats(flux_array, spike_train)
        
        return spike_train

    def _extract_flux_data(self, raw_data: Any) -> np.ndarray:
        """
        Extract flux array from input data.

        Args:
            raw_data: Input data in various formats.

        Returns:
            np.ndarray: Flux values as numpy array.

        Raises:
            ValueError: If data format is invalid or missing required columns.
        """
        if isinstance(raw_data, pl.DataFrame):
            # Expected format from ExoplanetGenerator
            if 'flux' not in raw_data.columns:
                raise ValueError("Input DataFrame must contain 'flux' column")
            return raw_data['flux'].to_numpy()

        elif isinstance(raw_data, np.ndarray):
            # Direct numpy array input
            if raw_data.ndim != 1:
                raise ValueError(f"Input array must be 1D, got shape {raw_data.shape}")
            return raw_data.copy()

        elif isinstance(raw_data, (list, tuple)):
            # Convert sequence to numpy array
            return np.array(raw_data, dtype=np.float64)

        else:
            raise ValueError(
                f"Unsupported input data type: {type(raw_data)}. "
                f"Expected polars.DataFrame, numpy.ndarray, list, or tuple."
            )

    def _normalize_flux(self, flux_array: np.ndarray) -> np.ndarray:
        """
        Normalize flux array to zero mean and unit variance.

        Args:
            flux_array: Input flux values.

        Returns:
            np.ndarray: Normalized flux values.
        """
        if len(flux_array) < 2:
            warnings.warn(
                "Cannot normalize flux array with less than 2 points",
                UserWarning,
                stacklevel=2
            )
            return flux_array

        mean_flux = np.mean(flux_array)
        std_flux = np.std(flux_array)

        if std_flux == 0:
            warnings.warn(
                "Flux array has zero variance, normalization skipped",
                UserWarning,
                stacklevel=2
            )
            return flux_array - mean_flux

        # Handle potential NaN/inf values in normalization
        with np.errstate(invalid='ignore'):
            normalized = (flux_array - mean_flux) / std_flux

        # Check for invalid values and handle them
        if np.any(~np.isfinite(normalized)):
            warnings.warn(
                "Invalid values encountered during normalization, returning mean-centered data",
                RuntimeWarning,
                stacklevel=2
            )
            return flux_array - mean_flux

        return normalized

    def _calculate_flux_deltas(self, flux_array: np.ndarray) -> np.ndarray:
        """
        Calculate temporal differences in flux values.

        Args:
            flux_array: Input flux values.

        Returns:
            np.ndarray: Flux differences with same length as input (first value is 0).
        """
        if len(flux_array) < 2:
            return np.zeros_like(flux_array)

        # Calculate differences: delta[t] = flux[t] - flux[t-1]
        deltas = np.zeros_like(flux_array)
        deltas[1:] = flux_array[1:] - flux_array[:-1]

        return deltas

    def _generate_spike_train(self, flux_deltas: np.ndarray) -> torch.Tensor:
        """
        Generate two-channel spike train from flux deltas.

        Args:
            flux_deltas: Temporal flux differences.

        Returns:
            torch.Tensor: Two-channel spike train [num_steps, 2].
        """
        num_steps = len(flux_deltas)
        spike_train = torch.zeros(num_steps, 2, dtype=torch.float32, device=self.device)

        # Channel 0: Positive changes (brightness increases)
        positive_spikes = flux_deltas > self.threshold
        spike_train[positive_spikes, 0] = 1.0

        # Channel 1: Negative changes (brightness decreases)
        negative_spikes = flux_deltas < -self.threshold
        spike_train[negative_spikes, 1] = 1.0

        return spike_train

    def _update_encoding_stats(self, flux_array: np.ndarray, spike_train: torch.Tensor) -> None:
        """Update internal encoding statistics."""
        total_spikes = spike_train.sum().item()
        positive_spikes = spike_train[:, 0].sum().item()
        negative_spikes = spike_train[:, 1].sum().item()

        self._last_encoding_stats = {
            'input_length': len(flux_array),
            'total_spikes': int(total_spikes),
            'positive_spikes': int(positive_spikes),
            'negative_spikes': int(negative_spikes),
            'spike_rate': float(total_spikes / len(flux_array)) if len(flux_array) > 0 else 0.0
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration parameters.

        Returns:
            Dict[str, Any]: Configuration dictionary containing all encoder parameters
                          and recent encoding statistics.

        Example:
            config = encoder.get_config()
            print(f"Threshold: {config['threshold']}")
            print(f"Last spike rate: {config['last_encoding_stats']['spike_rate']:.3f}")
        """
        return {
            'encoder_type': 'LightCurveEncoder',
            'encoding_scheme': 'delta_modulation',
            'threshold': self.threshold,
            'normalize_input': self.normalize_input,
            'output_channels': 2,
            'channel_0_meaning': 'positive_flux_changes',
            'channel_1_meaning': 'negative_flux_changes',
            'last_encoding_stats': self._last_encoding_stats.copy()
        }

    def calibrate(self, sample_data: Any) -> None:
        """
        Calibrate encoder parameters based on sample data.

        This method analyzes sample data to automatically adjust the encoding threshold
        for optimal spike generation. The calibration aims to achieve a target spike
        rate that balances information preservation with sparsity.

        Args:
            sample_data: Representative sample of data for calibration. Should be
                        in the same format as expected for encode().

        Example:
            # Calibrate encoder on sample data
            sample_light_curve = generator.generate()
            encoder.calibrate(sample_light_curve)

            # Check updated configuration
            config = encoder.get_config()
            print(f"Calibrated threshold: {config['threshold']}")
        """
        try:
            # Extract flux data for analysis
            flux_array = self._extract_flux_data(sample_data)

            if self.normalize_input:
                flux_array = self._normalize_flux(flux_array)

            # Calculate flux deltas for threshold estimation
            flux_deltas = self._calculate_flux_deltas(flux_array)

            if len(flux_deltas) < 10:
                warnings.warn("Insufficient data for calibration, keeping current threshold")
                return

            # Estimate optimal threshold based on flux delta statistics
            # Target: ~5-10% spike rate for good information/sparsity balance
            abs_deltas = np.abs(flux_deltas[flux_deltas != 0])  # Exclude zero deltas

            if len(abs_deltas) == 0:
                warnings.warn("No flux changes detected, keeping current threshold")
                return

            # Use 90th percentile as threshold to achieve ~10% spike rate
            new_threshold = np.percentile(abs_deltas, 90)

            # Ensure threshold is reasonable (not too small or too large)
            min_threshold = np.std(abs_deltas) * 0.1
            max_threshold = np.std(abs_deltas) * 2.0

            self.threshold = np.clip(new_threshold, min_threshold, max_threshold)

        except Exception as e:
            warnings.warn(f"Calibration failed: {e}. Keeping current threshold.")

    def get_encoding_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the most recent encoding operation.

        Returns:
            Dict[str, Any]: Statistics from last encode() call.
        """
        return self._last_encoding_stats.copy()

    def __repr__(self) -> str:
        """String representation of the encoder."""
        return (
            f"LightCurveEncoder(threshold={self.threshold:.2e}, "
            f"normalize_input={self.normalize_input})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Light Curve Encoder (Delta Modulation)\n"
            f"  Threshold: {self.threshold:.2e}\n"
            f"  Normalization: {'enabled' if self.normalize_input else 'disabled'}\n"
            f"  Output channels: 2 (positive/negative changes)\n"
            f"  Last encoding: {self._last_encoding_stats['total_spikes']} spikes"
        )
