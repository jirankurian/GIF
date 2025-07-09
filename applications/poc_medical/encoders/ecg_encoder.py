"""
ECG Encoder for Arrhythmia Detection
===================================

This module implements the ECG_Encoder class, which converts electrocardiogram (ECG) signals
into multi-channel spike trains suitable for processing by Spiking Neural Networks. This encoder
represents the medical domain implementation of the EncoderInterface, demonstrating the
plug-and-play architecture of the GIF framework for cross-domain applications.

The encoder uses sophisticated feature-based encoding that extracts clinically relevant
information from ECG signals and represents it as structured spike patterns. This approach
is more advanced than simple delta modulation and provides the SNN with pre-processed
features that are known to be diagnostically important.

Key Features:
============

**Multi-Channel Feature Encoding:**
The encoder generates a three-channel spike train where each channel represents a specific
cardiac feature:

1. **Channel 0 (R-Peak Timing)**: Fires spikes at the precise timing of detected R-peaks,
   providing the fundamental cardiac rhythm information.

2. **Channel 1 (Heart Rate Encoding)**: Uses rate-based coding to represent instantaneous
   heart rate, where the spike frequency in small time windows is proportional to the
   current heart rate.

3. **Channel 2 (QRS Duration)**: Employs latency-based coding to represent the width of
   QRS complexes, where the timing of spikes relative to R-peaks encodes the duration.

**Clinical Relevance:**
This encoding scheme captures the three most important features for arrhythmia detection:
- **Rhythm regularity** (from R-peak timing patterns)
- **Heart rate variability** (from instantaneous heart rate changes)  
- **Conduction abnormalities** (from QRS duration variations)

**Technical Implementation:**
- R-peak detection using scipy.signal.find_peaks with adaptive thresholding
- Robust heart rate calculation from R-R interval analysis
- QRS duration estimation using signal morphology analysis
- Efficient spike train generation with configurable temporal resolution

**Neuromorphic Advantages:**
The multi-channel approach allows the SNN to process different cardiac features in parallel,
mimicking the distributed processing of biological neural networks. The sparse spike
representation is highly efficient for neuromorphic hardware implementation.

Integration with GIF Framework:
==============================

The ECG_Encoder seamlessly integrates with the GIF orchestrator through the EncoderInterface
contract, enabling plug-and-play compatibility with any SNN architecture in the DU Core.
This demonstrates the framework's ability to adapt to completely different domains without
modifying the core processing components.

Example Usage:
=============

    from applications.poc_medical.encoders.ecg_encoder import ECG_Encoder
    from data_generators.ecg_generator import RealisticECGGenerator
    
    # Generate synthetic ECG data
    generator = RealisticECGGenerator(seed=42)
    noise_levels = {'baseline': 0.1, 'muscle': 0.02, 'powerline': 0.01}
    ecg_data = generator.generate(
        duration_seconds=10.0, sampling_rate=500,
        heart_rate_bpm=75.0, noise_levels=noise_levels
    )
    
    # Create and configure encoder
    encoder = ECG_Encoder(sampling_rate=500, num_channels=3)
    
    # Convert ECG to multi-channel spike train
    spike_train = encoder.encode(ecg_data)
    print(f"Encoded {len(ecg_data)} ECG samples to {spike_train.shape} spike train")
    
    # Integration with GIF framework
    from gif_framework.orchestrator import GIF
    gif_model = GIF(du_core)
    gif_model.attach_encoder(encoder)

This encoder enables the GIF framework to process medical diagnostic data with the same
efficiency and biological plausibility as astronomical data, proving the framework's
true cross-domain generalization capabilities.
"""

import torch
import polars as pl
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union
import warnings
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from gif_framework.interfaces.base_interfaces import EncoderInterface, SpikeTrain


class ECG_Encoder(EncoderInterface):
    """
    Multi-channel feature-based encoder for ECG signals.
    
    This encoder converts continuous ECG voltage signals into structured spike trains
    that capture clinically relevant cardiac features. Unlike simple threshold-based
    encoding, this approach extracts and encodes specific physiological features that
    are known to be important for arrhythmia detection.
    
    The encoder produces a three-channel spike representation:
    - Channel 0: R-peak timing events (fundamental rhythm)
    - Channel 1: Heart rate encoding (rate-based coding)
    - Channel 2: QRS duration encoding (latency-based coding)
    
    This multi-channel approach enables the SNN to process different aspects of cardiac
    function in parallel, leading to more robust and interpretable arrhythmia detection.
    
    Attributes:
        sampling_rate (int): ECG sampling rate in Hz (typically 250-1000)
        num_channels (int): Number of output spike channels (default 3)
        r_peak_threshold (float): Relative threshold for R-peak detection (0-1)
        heart_rate_window_ms (float): Time window for heart rate encoding in milliseconds
        qrs_search_window_ms (float): Search window around R-peaks for QRS duration
        min_rr_interval_ms (float): Minimum physiological R-R interval in milliseconds
        max_rr_interval_ms (float): Maximum physiological R-R interval in milliseconds
        
    Example:
        >>> encoder = ECG_Encoder(sampling_rate=500, num_channels=3)
        >>> spike_train = encoder.encode(ecg_dataframe)
        >>> print(f"Spike train shape: {spike_train.shape}")
    """
    
    def __init__(
        self,
        sampling_rate: int = 500,
        num_channels: int = 3,
        r_peak_threshold: float = 0.6,
        heart_rate_window_ms: float = 200.0,
        qrs_search_window_ms: float = 80.0,
        min_rr_interval_ms: float = 300.0,  # 200 BPM max
        max_rr_interval_ms: float = 2000.0   # 30 BPM min
    ) -> None:
        """
        Initialize the ECG encoder with physiologically appropriate parameters.
        
        Args:
            sampling_rate: ECG sampling frequency in Hz
            num_channels: Number of output spike channels
            r_peak_threshold: Relative threshold for R-peak detection (0-1)
            heart_rate_window_ms: Time window for heart rate encoding
            qrs_search_window_ms: Search window for QRS duration measurement
            min_rr_interval_ms: Minimum allowed R-R interval (physiological limit)
            max_rr_interval_ms: Maximum allowed R-R interval (physiological limit)
        """
        # Validate parameters
        if sampling_rate <= 0:
            raise ValueError(f"Sampling rate must be positive, got {sampling_rate}")
        if num_channels != 3:
            raise ValueError(f"Current implementation requires 3 channels, got {num_channels}")
        if not 0 < r_peak_threshold < 1:
            raise ValueError(f"R-peak threshold must be in (0,1), got {r_peak_threshold}")
            
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.r_peak_threshold = r_peak_threshold
        self.heart_rate_window_ms = heart_rate_window_ms
        self.qrs_search_window_ms = qrs_search_window_ms
        self.min_rr_interval_ms = min_rr_interval_ms
        self.max_rr_interval_ms = max_rr_interval_ms
        
        # Convert time parameters to samples
        self.heart_rate_window_samples = int(heart_rate_window_ms * sampling_rate / 1000)
        self.qrs_search_window_samples = int(qrs_search_window_ms * sampling_rate / 1000)
        self.min_rr_interval_samples = int(min_rr_interval_ms * sampling_rate / 1000)
        self.max_rr_interval_samples = int(max_rr_interval_ms * sampling_rate / 1000)
        
        # Encoding statistics for monitoring
        self.encoding_stats = {
            'total_samples_processed': 0,
            'total_r_peaks_detected': 0,
            'average_heart_rate_bpm': 0.0,
            'average_qrs_duration_ms': 0.0,
            'encoding_calls': 0
        }
    
    def encode(self, raw_data: Any) -> SpikeTrain:
        """
        Convert ECG signal data into a multi-channel spike train.
        
        This method implements the complete feature-based encoding pipeline:
        1. Extract and validate ECG voltage signal from input data
        2. Detect R-peaks using adaptive thresholding
        3. Calculate instantaneous heart rate from R-R intervals
        4. Estimate QRS duration for each detected complex
        5. Generate structured spike trains for each feature channel
        
        Args:
            raw_data: Input ECG data. Expected to be a polars.DataFrame with 'time' 
                     and 'voltage' columns, as generated by RealisticECGGenerator.
                     
        Returns:
            SpikeTrain: Multi-channel spike train tensor with shape 
                       [num_time_steps, num_channels]. Each channel represents:
                       - Channel 0: R-peak timing spikes
                       - Channel 1: Heart rate encoding spikes  
                       - Channel 2: QRS duration encoding spikes
                       
        Raises:
            ValueError: If input data format is invalid or contains insufficient data.
            RuntimeError: If encoding process fails due to signal quality issues.
            
        Example:
            >>> ecg_data = generator.generate(duration_seconds=10.0, sampling_rate=500)
            >>> spike_train = encoder.encode(ecg_data)
            >>> print(f"Encoded to shape: {spike_train.shape}")
        """
        # Validate and extract ECG signal
        voltage_signal, time_array = self._validate_and_extract_signal(raw_data)
        
        # Step 1: Detect R-peaks in the ECG signal
        r_peak_indices = self._detect_r_peaks(voltage_signal)
        
        # Step 2: Calculate instantaneous heart rate
        heart_rates = self._calculate_heart_rate(r_peak_indices, len(voltage_signal))
        
        # Step 3: Estimate QRS durations
        qrs_durations = self._estimate_qrs_durations(voltage_signal, r_peak_indices)
        
        # Step 4: Generate multi-channel spike train
        spike_train = self._generate_spike_train(
            signal_length=len(voltage_signal),
            r_peak_indices=r_peak_indices,
            heart_rates=heart_rates,
            qrs_durations=qrs_durations
        )
        
        # Update encoding statistics
        self._update_encoding_stats(r_peak_indices, heart_rates, qrs_durations)

        return spike_train

    def _validate_and_extract_signal(self, raw_data: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate input data format and extract ECG voltage signal.

        Args:
            raw_data: Input data, expected to be polars.DataFrame with ECG signal

        Returns:
            Tuple of (voltage_signal, time_array) as numpy arrays

        Raises:
            ValueError: If data format is invalid or required columns are missing
        """
        # Handle polars DataFrame (expected format)
        if isinstance(raw_data, pl.DataFrame):
            required_columns = ['time', 'voltage']
            missing_columns = [col for col in required_columns if col not in raw_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            voltage_signal = raw_data['voltage'].to_numpy()
            time_array = raw_data['time'].to_numpy()

        # Handle numpy array (assume it's voltage signal)
        elif isinstance(raw_data, np.ndarray):
            if raw_data.ndim != 1:
                raise ValueError(f"Expected 1D numpy array, got shape {raw_data.shape}")
            voltage_signal = raw_data
            time_array = np.arange(len(voltage_signal)) / self.sampling_rate

        # Handle torch tensor
        elif isinstance(raw_data, torch.Tensor):
            if raw_data.ndim != 1:
                raise ValueError(f"Expected 1D tensor, got shape {raw_data.shape}")
            voltage_signal = raw_data.detach().cpu().numpy()
            time_array = np.arange(len(voltage_signal)) / self.sampling_rate

        else:
            raise ValueError(f"Unsupported data type: {type(raw_data)}")

        # Validate signal properties
        if len(voltage_signal) < self.sampling_rate:
            raise ValueError(f"Signal too short: {len(voltage_signal)} samples < 1 second")

        if not np.all(np.isfinite(voltage_signal)):
            raise ValueError("Signal contains non-finite values (NaN or Inf)")

        return voltage_signal, time_array

    def _detect_r_peaks(self, voltage_signal: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks in ECG signal using adaptive thresholding.

        This method implements a robust R-peak detection algorithm that:
        1. Applies bandpass filtering to enhance QRS complexes
        2. Uses adaptive thresholding based on signal statistics
        3. Enforces physiological constraints on R-R intervals

        Args:
            voltage_signal: ECG voltage signal as numpy array

        Returns:
            Array of R-peak indices (sample positions)
        """
        # Step 1: Preprocess signal with bandpass filter (5-15 Hz for QRS enhancement)
        nyquist = self.sampling_rate / 2
        low_freq = 5.0 / nyquist
        high_freq = 15.0 / nyquist

        # Design Butterworth bandpass filter
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        filtered_signal = signal.filtfilt(b, a, voltage_signal)

        # Step 2: Calculate adaptive threshold
        signal_abs = np.abs(filtered_signal)
        signal_mean = np.mean(signal_abs)
        signal_std = np.std(signal_abs)
        adaptive_threshold = signal_mean + self.r_peak_threshold * signal_std

        # Step 3: Find peaks above threshold with minimum distance constraint
        min_distance = self.min_rr_interval_samples
        peak_indices, _ = signal.find_peaks(
            signal_abs,
            height=adaptive_threshold,
            distance=min_distance
        )

        # Step 4: Refine peak positions using original signal
        # Look for maximum in original signal within small window around detected peaks
        refined_peaks = []
        search_window = int(0.05 * self.sampling_rate)  # 50ms window

        for peak_idx in peak_indices:
            start_idx = max(0, peak_idx - search_window)
            end_idx = min(len(voltage_signal), peak_idx + search_window)

            # Find maximum in original signal within window
            local_signal = voltage_signal[start_idx:end_idx]
            local_max_idx = np.argmax(local_signal)
            refined_peak_idx = start_idx + local_max_idx

            refined_peaks.append(refined_peak_idx)

        return np.array(refined_peaks)

    def _calculate_heart_rate(self, r_peak_indices: np.ndarray, signal_length: int) -> np.ndarray:
        """
        Calculate instantaneous heart rate from R-peak intervals.

        Args:
            r_peak_indices: Array of R-peak sample positions
            signal_length: Total length of ECG signal in samples

        Returns:
            Array of heart rate values (BPM) for each sample
        """
        # Initialize heart rate array
        heart_rates = np.zeros(signal_length)

        if len(r_peak_indices) < 2:
            # Not enough peaks for heart rate calculation
            return heart_rates

        # Calculate R-R intervals in samples
        rr_intervals = np.diff(r_peak_indices)

        # Convert to heart rate in BPM
        rr_intervals_sec = rr_intervals / self.sampling_rate
        instantaneous_hr = 60.0 / rr_intervals_sec

        # Assign heart rate values between consecutive R-peaks
        for i in range(len(r_peak_indices) - 1):
            start_idx = r_peak_indices[i]
            end_idx = r_peak_indices[i + 1]
            heart_rates[start_idx:end_idx] = instantaneous_hr[i]

        # Handle edges: extend first and last values
        if len(r_peak_indices) > 0:
            heart_rates[:r_peak_indices[0]] = instantaneous_hr[0] if len(instantaneous_hr) > 0 else 75.0
            heart_rates[r_peak_indices[-1]:] = instantaneous_hr[-1] if len(instantaneous_hr) > 0 else 75.0

        return heart_rates

    def _estimate_qrs_durations(self, voltage_signal: np.ndarray, r_peak_indices: np.ndarray) -> np.ndarray:
        """
        Estimate QRS complex duration for each detected R-peak.

        Args:
            voltage_signal: ECG voltage signal
            r_peak_indices: Array of R-peak sample positions

        Returns:
            Array of QRS durations in milliseconds for each R-peak
        """
        qrs_durations = []

        for peak_idx in r_peak_indices:
            # Define search window around R-peak
            search_start = max(0, peak_idx - self.qrs_search_window_samples // 2)
            search_end = min(len(voltage_signal), peak_idx + self.qrs_search_window_samples // 2)

            # Extract QRS segment
            qrs_segment = voltage_signal[search_start:search_end]

            # Find QRS onset and offset using derivative-based method
            # Smooth the signal slightly to reduce noise
            smoothed_segment = gaussian_filter1d(qrs_segment, sigma=1.0)

            # Calculate first derivative
            derivative = np.gradient(smoothed_segment)

            # Find significant deflections (QRS boundaries)
            derivative_threshold = 0.3 * np.std(derivative)

            # QRS onset: first significant negative deflection before peak
            peak_in_segment = peak_idx - search_start
            onset_candidates = np.where(derivative[:peak_in_segment] < -derivative_threshold)[0]
            qrs_onset = onset_candidates[-1] if len(onset_candidates) > 0 else 0

            # QRS offset: first return to baseline after peak
            offset_candidates = np.where(np.abs(derivative[peak_in_segment:]) < derivative_threshold)[0]
            qrs_offset = (peak_in_segment + offset_candidates[0]) if len(offset_candidates) > 0 else len(qrs_segment) - 1

            # Calculate duration in milliseconds
            qrs_duration_samples = qrs_offset - qrs_onset
            qrs_duration_ms = (qrs_duration_samples / self.sampling_rate) * 1000

            # Apply physiological constraints (normal QRS: 80-120ms)
            qrs_duration_ms = np.clip(qrs_duration_ms, 60.0, 200.0)
            qrs_durations.append(qrs_duration_ms)

        return np.array(qrs_durations)

    def _generate_spike_train(
        self,
        signal_length: int,
        r_peak_indices: np.ndarray,
        heart_rates: np.ndarray,
        qrs_durations: np.ndarray
    ) -> SpikeTrain:
        """
        Generate multi-channel spike train from extracted ECG features.

        Args:
            signal_length: Length of original ECG signal in samples
            r_peak_indices: Array of R-peak positions
            heart_rates: Array of heart rate values for each sample
            qrs_durations: Array of QRS durations for each R-peak

        Returns:
            Multi-channel spike train tensor [signal_length, num_channels]
        """
        # Initialize spike train tensor
        spike_train = torch.zeros(signal_length, self.num_channels, dtype=torch.float32)

        # Channel 0: R-peak timing spikes
        for peak_idx in r_peak_indices:
            if 0 <= peak_idx < signal_length:
                spike_train[peak_idx, 0] = 1.0

        # Channel 1: Heart rate encoding (rate-based coding)
        # Generate spikes with frequency proportional to heart rate
        for i in range(0, signal_length, self.heart_rate_window_samples):
            window_end = min(i + self.heart_rate_window_samples, signal_length)
            window_hr = np.mean(heart_rates[i:window_end])

            # Convert heart rate to spike probability
            # Normal heart rate range: 60-100 BPM -> spike probability 0.1-0.5
            normalized_hr = (window_hr - 60.0) / 40.0  # Normalize to [0,1] for 60-100 BPM
            spike_prob = 0.1 + 0.4 * np.clip(normalized_hr, 0.0, 1.0)

            # Generate spikes based on probability
            window_spikes = torch.rand(window_end - i) < spike_prob
            spike_train[i:window_end, 1] = window_spikes.float()

        # Channel 2: QRS duration encoding (latency-based coding)
        # Spike timing relative to R-peak encodes QRS duration
        for i, peak_idx in enumerate(r_peak_indices):
            if i < len(qrs_durations) and 0 <= peak_idx < signal_length:
                qrs_duration = qrs_durations[i]

                # Convert QRS duration to latency offset (0-40 samples for 60-200ms)
                normalized_duration = (qrs_duration - 60.0) / 140.0  # Normalize to [0,1]
                latency_offset = int(40 * np.clip(normalized_duration, 0.0, 1.0))

                # Place spike at offset position
                spike_position = peak_idx + latency_offset
                if spike_position < signal_length:
                    spike_train[spike_position, 2] = 1.0

        return spike_train

    def _update_encoding_stats(
        self,
        r_peak_indices: np.ndarray,
        heart_rates: np.ndarray,
        qrs_durations: np.ndarray
    ) -> None:
        """Update encoding statistics for monitoring and debugging."""
        self.encoding_stats['encoding_calls'] += 1
        self.encoding_stats['total_samples_processed'] += len(heart_rates)
        self.encoding_stats['total_r_peaks_detected'] += len(r_peak_indices)

        if len(heart_rates) > 0:
            self.encoding_stats['average_heart_rate_bpm'] = float(np.mean(heart_rates))

        if len(qrs_durations) > 0:
            self.encoding_stats['average_qrs_duration_ms'] = float(np.mean(qrs_durations))

    def get_config(self) -> Dict[str, Any]:
        """
        Return encoder configuration for reproducibility and debugging.

        Returns:
            Dictionary containing all encoder parameters and current statistics
        """
        return {
            'encoder_type': 'ECG_Encoder',
            'sampling_rate': self.sampling_rate,
            'num_channels': self.num_channels,
            'r_peak_threshold': self.r_peak_threshold,
            'heart_rate_window_ms': self.heart_rate_window_ms,
            'qrs_search_window_ms': self.qrs_search_window_ms,
            'min_rr_interval_ms': self.min_rr_interval_ms,
            'max_rr_interval_ms': self.max_rr_interval_ms,
            'encoding_statistics': self.encoding_stats.copy()
        }

    def calibrate(self, sample_data: Any) -> None:
        """
        Calibrate encoder parameters based on sample ECG data.

        This method can be used to automatically adjust encoding parameters
        based on the characteristics of the input data, such as signal amplitude,
        noise level, and heart rate variability.

        Args:
            sample_data: Sample ECG data for calibration
        """
        try:
            # Extract signal for analysis
            voltage_signal, _ = self._validate_and_extract_signal(sample_data)

            # Detect R-peaks for calibration
            r_peak_indices = self._detect_r_peaks(voltage_signal)

            if len(r_peak_indices) >= 2:
                # Calculate average R-R interval
                rr_intervals = np.diff(r_peak_indices) / self.sampling_rate
                avg_rr_interval = np.mean(rr_intervals)

                # Adjust minimum distance based on observed heart rate
                # Use 60% of average R-R interval as minimum distance
                self.min_rr_interval_samples = int(0.6 * avg_rr_interval * self.sampling_rate)

                # Adjust R-peak threshold based on signal characteristics
                signal_std = np.std(voltage_signal)
                if signal_std > 0:
                    # Lower threshold for high-amplitude signals, higher for low-amplitude
                    amplitude_factor = np.clip(1.0 / signal_std, 0.3, 2.0)
                    self.r_peak_threshold = np.clip(0.6 * amplitude_factor, 0.2, 0.9)

                print(f"ECG_Encoder calibrated: min_rr_interval={self.min_rr_interval_samples} samples, "
                      f"r_peak_threshold={self.r_peak_threshold:.3f}")
            else:
                warnings.warn("Insufficient R-peaks detected for calibration")

        except Exception as e:
            warnings.warn(f"Calibration failed: {e}")
