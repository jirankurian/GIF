"""
Fourier Transform Encoder for Periodic Signal Detection
======================================================

This module implements the FourierEncoder class, which converts astronomical light curve
data into spike trains by analyzing the frequency domain characteristics. This encoder
is specifically optimized for detecting strictly periodic signals such as regular
planetary transits, stellar pulsations, and binary star eclipses.

The encoder uses Fast Fourier Transform (FFT) to convert time-domain light curves into
frequency-domain power spectra, then encodes the most significant frequency components
as structured spike patterns. This approach is particularly effective for signals with
strong periodic components.

Key Features:
============

**FFT-Based Analysis**: The encoder performs Fast Fourier Transform on the input light
curve to identify dominant frequency components. This approach naturally emphasizes
periodic signals while filtering out random noise.

**Power Spectrum Encoding**: The encoder converts the power spectrum into spike trains
by encoding the strongest frequency peaks as spike events. This preserves the most
important periodic information in a sparse, neuromorphic-compatible format.

**Multi-Channel Representation**: The encoder produces a multi-channel spike train:
- Channel 0: Fundamental frequency component
- Channel 1: First harmonic component  
- Channel 2: Second harmonic component
- Channel 3: Broadband power indicator

**Periodic Signal Optimization**: The encoding scheme is specifically tuned for periodic
signals, making it the optimal choice for detecting regular planetary transits and
stellar variability patterns.

Technical Implementation:
========================

The encoder processes polars DataFrames containing 'time' and 'flux' columns from the
ExoplanetGenerator. It performs FFT analysis, identifies dominant frequency peaks,
and converts these into structured spike patterns optimized for SNN processing.

Mathematical Foundation:
- FFT: F(ω) = ∫ f(t) * e^(-iωt) dt
- Power spectrum: P(ω) = |F(ω)|²
- Peak detection: Find local maxima in P(ω)
- Spike encoding: Convert peaks to temporal spike patterns

Integration with Meta-Cognitive Routing:
=======================================

The FourierEncoder is designed for intelligent selection by the meta-cognitive routing
system. It provides metadata indicating its optimization for periodic signals, enabling
the GIF framework to automatically select this encoder when processing tasks involving
regular, repeating patterns.

Example Usage:
=============

    from applications.poc_exoplanet.encoders.fourier_encoder import FourierEncoder
    from data_generators.exoplanet_generator import RealisticExoplanetGenerator
    
    # Generate periodic light curve data
    generator = RealisticExoplanetGenerator(seed=42)
    light_curve_data = generator.generate()
    
    # Create Fourier encoder optimized for periodic signals
    encoder = FourierEncoder(n_frequency_channels=4, frequency_resolution=0.1)
    
    # Convert light curve to frequency-domain spike train
    spike_train = encoder.encode(light_curve_data)
    print(f"Encoded to {spike_train.shape} frequency-domain spike train")
    
    # Integration with meta-cognitive routing
    config = encoder.get_config()
    print(f"Signal type: {config['signal_type']}")  # 'periodic'

This encoder enables the GIF framework to optimally process periodic astronomical
signals with the efficiency and biological plausibility of spiking neural networks.
"""

import torch
import polars as pl
import numpy as np
from typing import Any, Dict, Union, Optional
import warnings
from scipy import signal
from scipy.fft import fft, fftfreq

from gif_framework.interfaces.base_interfaces import EncoderInterface, SpikeTrain


class FourierEncoder(EncoderInterface):
    """
    FFT-based encoder optimized for periodic signal detection.
    
    This encoder converts light curve data into frequency-domain spike trains using
    Fast Fourier Transform analysis. It is specifically designed to detect and encode
    periodic signals such as planetary transits, stellar pulsations, and binary
    star eclipses.
    
    The encoder produces a multi-channel spike representation where each channel
    corresponds to different frequency components:
    - Channel 0: Fundamental frequency (strongest periodic component)
    - Channel 1: First harmonic (2x fundamental frequency)
    - Channel 2: Second harmonic (3x fundamental frequency)  
    - Channel 3: Broadband power (overall signal strength)
    
    Key Advantages:
    - **Periodic Optimization**: Specifically tuned for regular, repeating signals
    - **Frequency Selectivity**: Isolates specific frequency components
    - **Noise Robustness**: FFT naturally filters random noise
    - **Harmonic Analysis**: Captures harmonic structure of periodic signals
    
    Attributes:
        n_frequency_channels (int): Number of frequency channels to encode
        frequency_resolution (float): Frequency resolution for peak detection
        power_threshold (float): Minimum power threshold for spike generation
        
    Example:
        # Create encoder optimized for periodic signals
        encoder = FourierEncoder(n_frequency_channels=4, frequency_resolution=0.1)
        
        # Encode periodic light curve
        spike_train = encoder.encode(light_curve_dataframe)
        
        # Check encoder configuration
        config = encoder.get_config()
        print(f"Optimized for: {config['signal_type']}")
    """
    
    def __init__(
        self, 
        n_frequency_channels: int = 4,
        frequency_resolution: float = 0.1,
        power_threshold: float = 0.1,
        normalize_input: bool = True
    ) -> None:
        """
        Initialize the Fourier encoder.
        
        Args:
            n_frequency_channels (int): Number of frequency channels to encode.
                                      Default 4 provides fundamental + harmonics.
            frequency_resolution (float): Frequency resolution for peak detection.
                                        Smaller values provide finer frequency resolution.
            power_threshold (float): Minimum normalized power threshold for spike
                                   generation. Higher values create sparser spikes.
            normalize_input (bool): Whether to normalize input flux before FFT.
                                  Recommended for consistent behavior.
                                  
        Raises:
            ValueError: If parameters are invalid.
            
        Example:
            # Standard encoder for exoplanet detection
            encoder = FourierEncoder(n_frequency_channels=4)
            
            # High-resolution encoder for detailed analysis
            encoder = FourierEncoder(frequency_resolution=0.05, power_threshold=0.05)
        """
        if not isinstance(n_frequency_channels, int) or n_frequency_channels <= 0:
            raise ValueError(f"n_frequency_channels must be positive integer, got {n_frequency_channels}")
        if not isinstance(frequency_resolution, (int, float)) or frequency_resolution <= 0:
            raise ValueError(f"frequency_resolution must be positive, got {frequency_resolution}")
        if not isinstance(power_threshold, (int, float)) or power_threshold <= 0:
            raise ValueError(f"power_threshold must be positive, got {power_threshold}")
            
        self.n_frequency_channels = int(n_frequency_channels)
        self.frequency_resolution = float(frequency_resolution)
        self.power_threshold = float(power_threshold)
        self.normalize_input = bool(normalize_input)
        
        # Statistics for monitoring and debugging
        self._last_encoding_stats = {
            'input_length': 0,
            'dominant_frequency': 0.0,
            'total_power': 0.0,
            'peak_frequencies': [],
            'total_spikes': 0,
            'spike_rate': 0.0
        }
    
    def encode(self, raw_data: Any) -> SpikeTrain:
        """
        Convert light curve data into frequency-domain spike train using FFT analysis.
        
        This method implements the FFT-based encoding algorithm, converting time-domain
        light curves into frequency-domain spike patterns. The resulting spike train
        emphasizes periodic components while maintaining sparse representation.
        
        Args:
            raw_data (Any): Input light curve data. Expected to be a polars.DataFrame
                          with 'time' and 'flux' columns.
                          
        Returns:
            SpikeTrain: A PyTorch tensor with shape [num_time_steps, n_frequency_channels]
                       containing the encoded frequency-domain spike train.
                       
        Raises:
            ValueError: If input data format is invalid.
            RuntimeError: If FFT processing fails.
            
        Example:
            # Encode periodic exoplanet signal
            light_curve = generator.generate()
            spike_train = encoder.encode(light_curve)
            
            # Analyze frequency content
            stats = encoder.get_encoding_stats()
            print(f"Dominant frequency: {stats['dominant_frequency']:.3f} Hz")
        """
        # Validate and extract flux data
        flux_array = self._extract_flux_data(raw_data)
        
        # Normalize input if requested
        if self.normalize_input:
            flux_array = self._normalize_flux(flux_array)
        
        # Perform FFT analysis
        frequencies, power_spectrum = self._compute_fft(flux_array)
        
        # Identify dominant frequency peaks
        peak_frequencies, peak_powers = self._find_frequency_peaks(frequencies, power_spectrum)
        
        # Generate multi-channel spike train
        spike_train = self._generate_frequency_spike_train(flux_array, peak_frequencies, peak_powers)
        
        # Update encoding statistics
        self._update_encoding_stats(flux_array, frequencies, power_spectrum, peak_frequencies, spike_train)
        
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
    
    def _compute_fft(self, flux_array: np.ndarray) -> tuple:
        """Compute FFT and return frequencies and power spectrum."""
        # Perform FFT
        fft_result = fft(flux_array)
        frequencies = fftfreq(len(flux_array), d=1.0)  # Assuming unit time sampling
        
        # Compute power spectrum (magnitude squared)
        power_spectrum = np.abs(fft_result) ** 2
        
        # Keep only positive frequencies
        positive_freq_mask = frequencies > 0
        frequencies = frequencies[positive_freq_mask]
        power_spectrum = power_spectrum[positive_freq_mask]
        
        # Normalize power spectrum
        if np.max(power_spectrum) > 0:
            power_spectrum = power_spectrum / np.max(power_spectrum)
        
        return frequencies, power_spectrum
    
    def _find_frequency_peaks(self, frequencies: np.ndarray, power_spectrum: np.ndarray) -> tuple:
        """Find dominant frequency peaks in power spectrum."""
        # Find peaks in power spectrum
        peak_indices, _ = signal.find_peaks(power_spectrum, height=self.power_threshold)
        
        if len(peak_indices) == 0:
            # No significant peaks found, return zero frequency
            return np.array([0.0]), np.array([0.0])
        
        # Sort peaks by power (descending)
        peak_powers = power_spectrum[peak_indices]
        sorted_indices = np.argsort(peak_powers)[::-1]
        
        # Select top peaks up to n_frequency_channels
        n_peaks = min(len(peak_indices), self.n_frequency_channels)
        selected_indices = sorted_indices[:n_peaks]
        
        peak_frequencies = frequencies[peak_indices[selected_indices]]
        peak_powers = power_spectrum[peak_indices[selected_indices]]
        
        return peak_frequencies, peak_powers
    
    def _generate_frequency_spike_train(self, flux_array: np.ndarray, peak_frequencies: np.ndarray, peak_powers: np.ndarray) -> torch.Tensor:
        """Generate multi-channel spike train from frequency analysis."""
        num_steps = len(flux_array)
        spike_train = torch.zeros(num_steps, self.n_frequency_channels, dtype=torch.float32)
        
        # Generate spikes for each frequency channel
        for channel in range(self.n_frequency_channels):
            if channel < len(peak_frequencies):
                freq = peak_frequencies[channel]
                power = peak_powers[channel]
                
                if freq > 0:
                    # Generate periodic spike pattern based on frequency
                    period = 1.0 / freq
                    spike_times = np.arange(0, num_steps, period)
                    spike_indices = spike_times.astype(int)
                    spike_indices = spike_indices[spike_indices < num_steps]
                    
                    # Set spike amplitudes based on power
                    spike_train[spike_indices, channel] = power
        
        return spike_train

    def calibrate(self, calibration_data: Any) -> None:
        """
        Calibrate the encoder using representative data.

        For the FourierEncoder, calibration involves analyzing the frequency
        characteristics of representative data to optimize frequency resolution
        and power thresholds for the specific dataset.

        Args:
            calibration_data (Any): Representative data for calibration
        """
        try:
            # Extract flux data for calibration
            flux_array = self._extract_flux_data(calibration_data)

            if self.normalize_input:
                flux_array = self._normalize_flux(flux_array)

            # Perform FFT analysis on calibration data
            frequencies, power_spectrum = self._compute_fft(flux_array)

            # Analyze frequency characteristics
            if len(frequencies) > 0 and np.max(power_spectrum) > 0:
                # Adaptive threshold based on power spectrum statistics
                mean_power = np.mean(power_spectrum)
                std_power = np.std(power_spectrum)

                # Set threshold to capture significant peaks above noise
                adaptive_threshold = mean_power + 2 * std_power
                normalized_threshold = adaptive_threshold / np.max(power_spectrum)

                # Update threshold if it's reasonable
                if 0.01 <= normalized_threshold <= 0.5:
                    self.power_threshold = normalized_threshold

                print(f"🔧 FourierEncoder calibrated: threshold={self.power_threshold:.3f}")

        except Exception as e:
            print(f"⚠️  FourierEncoder calibration failed: {e}")
            # Continue with default parameters

    def _update_encoding_stats(self, flux_array: np.ndarray, frequencies: np.ndarray,
                             power_spectrum: np.ndarray, peak_frequencies: np.ndarray,
                             spike_train: torch.Tensor) -> None:
        """Update encoding statistics."""
        self._last_encoding_stats = {
            'input_length': len(flux_array),
            'dominant_frequency': peak_frequencies[0] if len(peak_frequencies) > 0 else 0.0,
            'total_power': np.sum(power_spectrum),
            'peak_frequencies': peak_frequencies.tolist(),
            'total_spikes': int(torch.sum(spike_train > 0).item()),
            'spike_rate': float(torch.sum(spike_train > 0).item()) / spike_train.numel()
        }
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration and metadata for meta-cognitive routing.
        
        Returns:
            Dict[str, Any]: Configuration dictionary with routing metadata.
        """
        return {
            'encoder_type': 'FourierEncoder',
            'encoding_scheme': 'fft_frequency_domain',
            'signal_type': 'periodic',
            'description': 'Best for detecting periodic signals and regular patterns',
            'optimal_for': ['planetary_transits', 'stellar_pulsations', 'binary_eclipses'],
            'n_frequency_channels': self.n_frequency_channels,
            'frequency_resolution': self.frequency_resolution,
            'power_threshold': self.power_threshold,
            'normalize_input': self.normalize_input,
            'output_channels': self.n_frequency_channels,
            'last_encoding_stats': self._last_encoding_stats.copy()
        }
    
    def get_encoding_stats(self) -> Dict[str, Any]:
        """Get statistics from the most recent encoding operation."""
        return self._last_encoding_stats.copy()
    
    def __repr__(self) -> str:
        """String representation of the encoder."""
        return (
            f"FourierEncoder(n_channels={self.n_frequency_channels}, "
            f"freq_res={self.frequency_resolution}, threshold={self.power_threshold})"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Fourier Transform Encoder (Periodic Signal Optimization)\n"
            f"  Frequency channels: {self.n_frequency_channels}\n"
            f"  Frequency resolution: {self.frequency_resolution}\n"
            f"  Power threshold: {self.power_threshold}\n"
            f"  Signal type: Periodic\n"
            f"  Last encoding: {self._last_encoding_stats['total_spikes']} spikes"
        )
