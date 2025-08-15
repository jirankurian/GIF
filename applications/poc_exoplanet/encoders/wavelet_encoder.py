"""
Wavelet Transform Encoder for Transient Signal Detection
=======================================================

This module implements the WaveletEncoder class, which converts astronomical light curve
data into spike trains by analyzing time-frequency characteristics using wavelet transforms.
This encoder is specifically optimized for detecting transient, non-stationary signals
such as stellar flares, supernovae, gravitational microlensing events, and irregular
variable star behavior.

The encoder uses Continuous Wavelet Transform (CWT) to analyze signals in both time and
frequency domains simultaneously, making it ideal for detecting sudden bursts, irregular
patterns, and time-localized events that would be missed by purely frequency-domain
or time-domain analysis.

Key Features:
============

**Time-Frequency Analysis**: The encoder performs Continuous Wavelet Transform to capture
both temporal and spectral characteristics of signals. This dual representation is
essential for transient event detection.

**Multi-Scale Detection**: Uses multiple wavelet scales to detect events at different
time resolutions, from rapid flares (seconds) to extended outbursts (days).

**Transient Optimization**: The encoding scheme emphasizes sudden changes and localized
events, making it optimal for detecting irregular astronomical phenomena.

**Adaptive Thresholding**: Implements dynamic thresholding based on local signal
statistics to adapt to varying noise levels and baseline variations.

Technical Implementation:
========================

The encoder processes polars DataFrames containing 'time' and 'flux' columns from the
ExoplanetGenerator. It performs CWT analysis using Morlet wavelets, identifies
significant time-frequency features, and converts these into structured spike patterns
optimized for SNN processing.

Mathematical Foundation:
- CWT: W(a,b) = (1/√a) ∫ f(t) * ψ*((t-b)/a) dt
- Morlet wavelet: ψ(t) = π^(-1/4) * e^(iω₀t) * e^(-t²/2)
- Energy detection: E(a,b) = |W(a,b)|²
- Spike encoding: Convert energy peaks to temporal spike patterns

Integration with Meta-Cognitive Routing:
=======================================

The WaveletEncoder is designed for intelligent selection by the meta-cognitive routing
system. It provides metadata indicating its optimization for transient signals, enabling
the GIF framework to automatically select this encoder when processing tasks involving
sudden bursts, flares, or irregular patterns.

Example Usage:
=============

    from applications.poc_exoplanet.encoders.wavelet_encoder import WaveletEncoder
    from data_generators.exoplanet_generator import RealisticExoplanetGenerator
    
    # Generate light curve with transient events
    generator = RealisticExoplanetGenerator(seed=42)
    light_curve_data = generator.generate()
    
    # Create wavelet encoder optimized for transient detection
    encoder = WaveletEncoder(n_scales=8, n_time_channels=4)
    
    # Convert light curve to time-frequency spike train
    spike_train = encoder.encode(light_curve_data)
    print(f"Encoded to {spike_train.shape} time-frequency spike train")
    
    # Integration with meta-cognitive routing
    config = encoder.get_config()
    print(f"Signal type: {config['signal_type']}")  # 'transient'

This encoder enables the GIF framework to optimally detect and process transient
astronomical events with the temporal precision and biological plausibility of
spiking neural networks.
"""

import torch
import polars as pl
import numpy as np
from typing import Any, Dict, Union, Optional
import warnings
from scipy import signal
# Use PyWavelets for modern wavelet analysis (avoiding deprecated scipy functions)
try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False
    # Fallback implementation for environments without PyWavelets
    def cwt_fallback(data, wavelet_name, scales):
        """Simple CWT fallback implementation."""
        if wavelet_name == 'cmor':
            # Complex Morlet wavelet approximation
            results = []
            for scale in scales:
                # Simple convolution-based approximation
                kernel_size = min(len(data) // 4, int(scale * 10))
                if kernel_size < 3:
                    kernel_size = 3
                x = np.arange(kernel_size) - kernel_size // 2
                # Morlet-like kernel
                kernel = np.exp(1j * 2 * np.pi * x / scale) * np.exp(-x**2 / (2 * scale**2))
                kernel = kernel / np.sqrt(scale)
                # Convolve and extract same-size result
                conv_result = np.convolve(data, kernel, mode='same')
                results.append(conv_result)
            return np.array(results)
        else:
            # Ricker wavelet fallback
            return np.array([np.convolve(data, signal.ricker(len(data), scale), mode='same') for scale in scales])

from gif_framework.interfaces.base_interfaces import EncoderInterface, SpikeTrain


class WaveletEncoder(EncoderInterface):
    """
    Wavelet transform encoder optimized for transient signal detection.
    
    This encoder converts light curve data into time-frequency spike trains using
    Continuous Wavelet Transform analysis. It is specifically designed to detect
    and encode transient, non-stationary signals such as stellar flares, supernovae,
    microlensing events, and irregular variable star behavior.
    
    The encoder produces a multi-channel spike representation where each channel
    corresponds to different time-frequency characteristics:
    - Channels 0-3: Different wavelet scales (fine to coarse temporal resolution)
    - Each channel encodes energy at specific time-frequency locations
    
    Key Advantages:
    - **Transient Optimization**: Specifically tuned for sudden, localized events
    - **Time-Frequency Resolution**: Captures both temporal and spectral information
    - **Multi-Scale Analysis**: Detects events at multiple time resolutions
    - **Adaptive Sensitivity**: Adjusts to local signal characteristics
    
    Attributes:
        n_scales (int): Number of wavelet scales for multi-resolution analysis
        n_time_channels (int): Number of time channels for spike encoding
        energy_threshold (float): Minimum energy threshold for spike generation
        
    Example:
        # Create encoder optimized for transient signals
        encoder = WaveletEncoder(n_scales=8, n_time_channels=4)
        
        # Encode transient light curve
        spike_train = encoder.encode(light_curve_dataframe)
        
        # Check encoder configuration
        config = encoder.get_config()
        print(f"Optimized for: {config['signal_type']}")
    """
    
    def __init__(
        self, 
        n_scales: int = 8,
        n_time_channels: int = 4,
        energy_threshold: float = 0.2,
        wavelet_type: str = 'morlet',
        normalize_input: bool = True
    ) -> None:
        """
        Initialize the Wavelet encoder.
        
        Args:
            n_scales (int): Number of wavelet scales for analysis.
                          More scales provide finer frequency resolution.
            n_time_channels (int): Number of time channels for spike encoding.
                                 Default 4 provides good temporal resolution.
            energy_threshold (float): Minimum normalized energy threshold for spike
                                    generation. Higher values create sparser spikes.
            wavelet_type (str): Type of wavelet to use ('morlet' recommended).
            normalize_input (bool): Whether to normalize input flux before CWT.
                                  Recommended for consistent behavior.
                                  
        Raises:
            ValueError: If parameters are invalid.
            
        Example:
            # Standard encoder for flare detection
            encoder = WaveletEncoder(n_scales=8)
            
            # High-resolution encoder for detailed transient analysis
            encoder = WaveletEncoder(n_scales=16, energy_threshold=0.1)
        """
        if not isinstance(n_scales, int) or n_scales <= 0:
            raise ValueError(f"n_scales must be positive integer, got {n_scales}")
        if not isinstance(n_time_channels, int) or n_time_channels <= 0:
            raise ValueError(f"n_time_channels must be positive integer, got {n_time_channels}")
        if not isinstance(energy_threshold, (int, float)) or energy_threshold <= 0:
            raise ValueError(f"energy_threshold must be positive, got {energy_threshold}")
        if wavelet_type not in ['morlet', 'ricker']:
            raise ValueError(f"wavelet_type must be 'morlet' or 'ricker', got {wavelet_type}")
            
        self.n_scales = int(n_scales)
        self.n_time_channels = int(n_time_channels)
        self.energy_threshold = float(energy_threshold)
        self.wavelet_type = str(wavelet_type)
        self.normalize_input = bool(normalize_input)
        
        # Generate wavelet scales (logarithmic spacing)
        self.scales = np.logspace(0, np.log10(self.n_scales * 2), self.n_scales)
        
        # Statistics for monitoring and debugging
        self._last_encoding_stats = {
            'input_length': 0,
            'max_energy': 0.0,
            'total_energy': 0.0,
            'transient_events': 0,
            'total_spikes': 0,
            'spike_rate': 0.0,
            'energy_distribution': []
        }
    
    def encode(self, raw_data: Any) -> SpikeTrain:
        """
        Convert light curve data into time-frequency spike train using wavelet analysis.
        
        This method implements the CWT-based encoding algorithm, converting time-domain
        light curves into time-frequency spike patterns. The resulting spike train
        emphasizes transient events while maintaining temporal precision.
        
        Args:
            raw_data (Any): Input light curve data. Expected to be a polars.DataFrame
                          with 'time' and 'flux' columns.
                          
        Returns:
            SpikeTrain: A PyTorch tensor with shape [num_time_steps, n_time_channels]
                       containing the encoded time-frequency spike train.
                       
        Raises:
            ValueError: If input data format is invalid.
            RuntimeError: If wavelet processing fails.
            
        Example:
            # Encode transient stellar flare
            light_curve = generator.generate()
            spike_train = encoder.encode(light_curve)
            
            # Analyze transient content
            stats = encoder.get_encoding_stats()
            print(f"Transient events detected: {stats['transient_events']}")
        """
        # Validate and extract flux data
        flux_array = self._extract_flux_data(raw_data)
        
        # Normalize input if requested
        if self.normalize_input:
            flux_array = self._normalize_flux(flux_array)
        
        # Perform Continuous Wavelet Transform
        wavelet_coeffs = self._compute_cwt(flux_array)
        
        # Compute energy distribution
        energy_matrix = self._compute_energy_matrix(wavelet_coeffs)
        
        # Detect transient events
        transient_locations = self._detect_transient_events(energy_matrix)
        
        # Generate multi-channel spike train
        spike_train = self._generate_transient_spike_train(flux_array, energy_matrix, transient_locations)
        
        # Update encoding statistics
        self._update_encoding_stats(flux_array, energy_matrix, transient_locations, spike_train)
        
        return spike_train
    
    def _extract_flux_data(self, raw_data: Any) -> np.ndarray:
        """Extract flux array from input data."""
        if isinstance(raw_data, pl.DataFrame):
            if 'flux' not in raw_data.columns:
                raise ValueError("Input DataFrame must contain 'flux' column")
            return raw_data['flux'].to_numpy()
        elif isinstance(raw_data, np.ndarray):
            return raw_data.flatten()
        elif isinstance(raw_data, (list, tuple)):
            return np.array(raw_data)
        else:
            raise ValueError(f"Unsupported input type: {type(raw_data)}")
    
    def _normalize_flux(self, flux_array: np.ndarray) -> np.ndarray:
        """Normalize flux to zero mean and unit variance."""
        flux_mean = np.mean(flux_array)
        flux_std = np.std(flux_array)
        if flux_std > 0:
            return (flux_array - flux_mean) / flux_std
        else:
            return flux_array - flux_mean
    
    def _compute_cwt(self, flux_array: np.ndarray) -> np.ndarray:
        """Compute Continuous Wavelet Transform using PyWavelets or fallback."""
        try:
            if PYWAVELETS_AVAILABLE:
                if self.wavelet_type == 'morlet':
                    # Use complex Morlet wavelet (cmor) with PyWavelets
                    # cmor parameters: bandwidth=1.5, center_frequency=1.0
                    wavelet_coeffs, _ = pywt.cwt(flux_array, self.scales, 'cmor1.5-1.0')
                else:
                    # Use Mexican hat (Ricker) wavelet
                    wavelet_coeffs, _ = pywt.cwt(flux_array, self.scales, 'mexh')
            else:
                # Use fallback implementation
                if self.wavelet_type == 'morlet':
                    wavelet_coeffs = cwt_fallback(flux_array, 'cmor', self.scales)
                else:
                    wavelet_coeffs = cwt_fallback(flux_array, 'ricker', self.scales)

            return wavelet_coeffs

        except Exception as e:
            raise RuntimeError(f"Wavelet transform failed: {str(e)}")
    
    def _compute_energy_matrix(self, wavelet_coeffs: np.ndarray) -> np.ndarray:
        """Compute energy matrix from wavelet coefficients."""
        # Energy is the squared magnitude of complex coefficients
        energy_matrix = np.abs(wavelet_coeffs) ** 2
        
        # Normalize energy matrix
        max_energy = np.max(energy_matrix)
        if max_energy > 0:
            energy_matrix = energy_matrix / max_energy
        
        return energy_matrix
    
    def _detect_transient_events(self, energy_matrix: np.ndarray) -> list:
        """Detect transient events in the energy matrix."""
        transient_locations = []
        
        # For each scale, find energy peaks above threshold
        for scale_idx in range(energy_matrix.shape[0]):
            energy_trace = energy_matrix[scale_idx, :]
            
            # Find peaks above threshold
            peak_indices, _ = signal.find_peaks(energy_trace, height=self.energy_threshold)
            
            for peak_idx in peak_indices:
                transient_locations.append({
                    'time_idx': peak_idx,
                    'scale_idx': scale_idx,
                    'energy': energy_trace[peak_idx]
                })
        
        return transient_locations
    
    def _generate_transient_spike_train(self, flux_array: np.ndarray, energy_matrix: np.ndarray, 
                                      transient_locations: list) -> torch.Tensor:
        """Generate multi-channel spike train from transient analysis."""
        num_steps = len(flux_array)
        spike_train = torch.zeros(num_steps, self.n_time_channels, dtype=torch.float32)
        
        # Distribute transient events across time channels
        for event in transient_locations:
            time_idx = event['time_idx']
            energy = event['energy']
            
            # Assign to time channel based on scale
            channel = min(event['scale_idx'] % self.n_time_channels, self.n_time_channels - 1)
            
            # Set spike amplitude based on energy
            if time_idx < num_steps:
                spike_train[time_idx, channel] = max(spike_train[time_idx, channel].item(), energy)
        
        # Add background activity based on overall energy
        for channel in range(self.n_time_channels):
            # Sample energy from corresponding scales
            scale_start = channel * (self.n_scales // self.n_time_channels)
            scale_end = min(scale_start + (self.n_scales // self.n_time_channels), self.n_scales)
            
            if scale_end > scale_start:
                channel_energy = np.mean(energy_matrix[scale_start:scale_end, :], axis=0)
                
                # Add low-level background spikes
                background_mask = channel_energy > (self.energy_threshold * 0.5)
                background_indices = np.where(background_mask)[0]
                
                for idx in background_indices:
                    if idx < num_steps:
                        spike_train[idx, channel] = max(
                            spike_train[idx, channel].item(), 
                            channel_energy[idx] * 0.5
                        )
        
        return spike_train

    def calibrate(self, calibration_data: Any) -> None:
        """
        Calibrate the encoder using representative data.

        For the WaveletEncoder, calibration involves analyzing the energy
        characteristics of representative data to optimize energy thresholds
        and scale parameters for the specific dataset.

        Args:
            calibration_data (Any): Representative data for calibration
        """
        try:
            # Extract flux data for calibration
            flux_array = self._extract_flux_data(calibration_data)

            if self.normalize_input:
                flux_array = self._normalize_flux(flux_array)

            # Perform CWT analysis on calibration data
            wavelet_coeffs = self._compute_cwt(flux_array)
            energy_matrix = self._compute_energy_matrix(wavelet_coeffs)

            # Analyze energy characteristics
            if energy_matrix.size > 0:
                # Adaptive threshold based on energy statistics
                mean_energy = np.mean(energy_matrix)
                std_energy = np.std(energy_matrix)

                # Set threshold to capture significant transient events above noise
                adaptive_threshold = mean_energy + 2.5 * std_energy

                # Normalize by maximum energy
                max_energy = np.max(energy_matrix)
                if max_energy > 0:
                    normalized_threshold = adaptive_threshold / max_energy

                    # Update threshold if it's reasonable
                    if 0.01 <= normalized_threshold <= 0.8:
                        self.energy_threshold = normalized_threshold

                print(f"🔧 WaveletEncoder calibrated: threshold={self.energy_threshold:.3f}")

        except Exception as e:
            print(f"⚠️  WaveletEncoder calibration failed: {e}")
            # Continue with default parameters

    def _update_encoding_stats(self, flux_array: np.ndarray, energy_matrix: np.ndarray,
                             transient_locations: list, spike_train: torch.Tensor) -> None:
        """Update encoding statistics."""
        self._last_encoding_stats = {
            'input_length': len(flux_array),
            'max_energy': float(np.max(energy_matrix)),
            'total_energy': float(np.sum(energy_matrix)),
            'transient_events': len(transient_locations),
            'total_spikes': int(torch.sum(spike_train > 0).item()),
            'spike_rate': float(torch.sum(spike_train > 0).item()) / spike_train.numel(),
            'energy_distribution': np.mean(energy_matrix, axis=1).tolist()
        }
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration and metadata for meta-cognitive routing.
        
        Returns:
            Dict[str, Any]: Configuration dictionary with routing metadata.
        """
        return {
            'encoder_type': 'WaveletEncoder',
            'encoding_scheme': 'cwt_time_frequency',
            'signal_type': 'transient',
            'description': 'Best for detecting sudden bursts and transient events',
            'optimal_for': ['stellar_flares', 'supernovae', 'microlensing', 'irregular_variables'],
            'n_scales': self.n_scales,
            'n_time_channels': self.n_time_channels,
            'energy_threshold': self.energy_threshold,
            'wavelet_type': self.wavelet_type,
            'normalize_input': self.normalize_input,
            'output_channels': self.n_time_channels,
            'last_encoding_stats': self._last_encoding_stats.copy()
        }
    
    def get_encoding_stats(self) -> Dict[str, Any]:
        """Get statistics from the most recent encoding operation."""
        return self._last_encoding_stats.copy()
    
    def __repr__(self) -> str:
        """String representation of the encoder."""
        return (
            f"WaveletEncoder(n_scales={self.n_scales}, "
            f"n_channels={self.n_time_channels}, threshold={self.energy_threshold})"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Wavelet Transform Encoder (Transient Signal Optimization)\n"
            f"  Wavelet scales: {self.n_scales}\n"
            f"  Time channels: {self.n_time_channels}\n"
            f"  Energy threshold: {self.energy_threshold}\n"
            f"  Signal type: Transient\n"
            f"  Last encoding: {self._last_encoding_stats['transient_events']} events detected"
        )
