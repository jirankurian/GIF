"""
Test Suite for ECG Clinical Data Generator
==========================================

Comprehensive tests for the complete ECG clinical data pipeline:
- ECGPhysicsEngine initialization and parameter validation
- ODE system implementation and mathematical correctness
- ECG signal generation and morphology validation
- ClinicalNoiseModel noise generation and characteristics
- RealisticECGGenerator complete pipeline integration
- Heart rate accuracy and timing verification
- Clinical noise validation and signal quality metrics
- Edge case handling and error conditions

These tests ensure the complete pipeline produces mathematically correct,
biophysically plausible ECG signals with realistic clinical noise.
"""

import pytest
import numpy as np
from dataclasses import FrozenInstanceError

# Import our complete ECG clinical data generator
from data_generators.ecg_generator import (
    ECGPhysicsEngine,
    ClinicalNoiseModel,
    RealisticECGGenerator
)


class TestECGPhysicsEngine:
    """Test the ECGPhysicsEngine class."""
    
    def test_engine_initialization(self):
        """Test that ECG engine initializes correctly with valid parameters."""
        engine = ECGPhysicsEngine(heart_rate_bpm=60.0)
        
        assert engine.heart_rate_bpm == 60.0
        assert abs(engine.omega - (2 * np.pi)) < 1e-10  # ω = 2π for 60 BPM
        assert engine.alpha == 1.0
        assert len(engine.pqrst_params) == 5  # P, Q, R, S, T waves
        
        # Check PQRST parameters are properly set
        assert 'P' in engine.pqrst_params
        assert 'Q' in engine.pqrst_params
        assert 'R' in engine.pqrst_params
        assert 'S' in engine.pqrst_params
        assert 'T' in engine.pqrst_params
    
    def test_heart_rate_validation(self):
        """Test that invalid heart rates are rejected."""
        # Test too low heart rate
        with pytest.raises(ValueError, match="outside physiological range"):
            ECGPhysicsEngine(heart_rate_bpm=20.0)
        
        # Test too high heart rate
        with pytest.raises(ValueError, match="outside physiological range"):
            ECGPhysicsEngine(heart_rate_bpm=250.0)
        
        # Test boundary values
        ECGPhysicsEngine(heart_rate_bpm=30.0)  # Should work
        ECGPhysicsEngine(heart_rate_bpm=200.0)  # Should work
    
    def test_omega_calculation(self):
        """Test that angular frequency is calculated correctly."""
        # Test various heart rates
        test_cases = [
            (60.0, 2 * np.pi),           # 60 BPM -> 2π rad/s
            (120.0, 4 * np.pi),          # 120 BPM -> 4π rad/s
            (30.0, np.pi),               # 30 BPM -> π rad/s
        ]
        
        for heart_rate, expected_omega in test_cases:
            engine = ECGPhysicsEngine(heart_rate_bpm=heart_rate)
            assert abs(engine.omega - expected_omega) < 1e-10
    
    def test_ode_system_structure(self):
        """Test that the ODE system returns correct structure."""
        engine = ECGPhysicsEngine(heart_rate_bpm=75.0)
        
        # Test with a sample state
        test_state = [1.0, 0.0, 0.0]
        derivatives = engine._ecg_model_odes(0.0, test_state)
        
        # Should return list of 3 derivatives
        assert isinstance(derivatives, list)
        assert len(derivatives) == 3
        
        # All derivatives should be finite numbers
        assert all(np.isfinite(d) for d in derivatives)
    
    def test_ode_system_equations(self):
        """Test that ODE system implements correct equations."""
        engine = ECGPhysicsEngine(heart_rate_bpm=60.0)
        
        # Test at initial state [1, 0, 0]
        state = [1.0, 0.0, 0.0]
        derivatives = engine._ecg_model_odes(0.0, state)
        
        x, y, z = state
        dx_dt, dy_dt, dz_dt = derivatives
        
        # Check first equation: dx/dt = y
        assert abs(dx_dt - y) < 1e-10
        
        # Check second equation: dy/dt = z
        assert abs(dy_dt - z) < 1e-10
        
        # Third equation: dz/dt = αz - ω²y - force(t)
        # At [1, 0, 0], y=0 and z=0, so dz/dt = -force(t)
        # The force should be non-zero due to R wave at θ=0
        assert abs(dz_dt) > 0  # Should have force contribution
    
    def test_ecg_generation_basic(self):
        """Test basic ECG signal generation."""
        engine = ECGPhysicsEngine(heart_rate_bpm=60.0)
        
        ecg_data = engine.generate_ecg_segment(
            duration_seconds=5.0,
            sampling_rate=500
        )
        
        # Check output format
        assert len(ecg_data) == 2500  # 5 seconds × 500 Hz
        assert 'time' in ecg_data.columns
        assert 'voltage' in ecg_data.columns
        
        # Check time array
        time_values = ecg_data['time'].to_numpy()
        assert time_values[0] == 0.0
        assert abs(time_values[-1] - 5.0) < 1e-6
        
        # Check voltage values are finite
        voltage_values = ecg_data['voltage'].to_numpy()
        assert np.all(np.isfinite(voltage_values))
    
    def test_ecg_morphology(self):
        """Test that generated ECG has proper morphology."""
        engine = ECGPhysicsEngine(heart_rate_bpm=60.0)
        
        # Generate longer segment for better analysis
        ecg_data = engine.generate_ecg_segment(
            duration_seconds=10.0,
            sampling_rate=500
        )
        
        voltage_values = ecg_data['voltage'].to_numpy()
        
        # ECG should have both positive and negative deflections
        assert voltage_values.max() > 0
        assert voltage_values.min() < 0
        
        # R wave should be the dominant positive peak
        max_voltage = voltage_values.max()
        assert max_voltage > 10.0  # R wave should be prominent
        
        # Signal should have reasonable dynamic range
        voltage_range = max_voltage - voltage_values.min()
        assert voltage_range > 20.0  # Should span significant range
    
    def test_heart_rate_accuracy(self):
        """Test that generated ECG matches specified heart rate."""
        heart_rate = 75.0
        engine = ECGPhysicsEngine(heart_rate_bpm=heart_rate)
        
        # Generate longer segment for accurate heart rate estimation
        ecg_data = engine.generate_ecg_segment(
            duration_seconds=20.0,
            sampling_rate=500
        )
        
        time_values = ecg_data['time'].to_numpy()
        voltage_values = ecg_data['voltage'].to_numpy()
        
        # Simple peak detection for R waves
        threshold = 0.5 * voltage_values.max()
        peaks = []
        for i in range(1, len(voltage_values) - 1):
            if (voltage_values[i] > voltage_values[i-1] and 
                voltage_values[i] > voltage_values[i+1] and 
                voltage_values[i] > threshold):
                peaks.append(i)
        
        if len(peaks) > 1:
            # Calculate heart rate from RR intervals
            peak_times = time_values[peaks]
            rr_intervals = np.diff(peak_times)
            avg_rr_interval = np.mean(rr_intervals)
            estimated_heart_rate = 60.0 / avg_rr_interval
            
            # Should be within 5% of target heart rate
            relative_error = abs(estimated_heart_rate - heart_rate) / heart_rate
            assert relative_error < 0.05
    
    def test_different_heart_rates(self):
        """Test ECG generation with different heart rates."""
        test_heart_rates = [45.0, 60.0, 80.0, 120.0]
        
        for heart_rate in test_heart_rates:
            engine = ECGPhysicsEngine(heart_rate_bpm=heart_rate)
            
            ecg_data = engine.generate_ecg_segment(
                duration_seconds=5.0,
                sampling_rate=500
            )
            
            # All should generate valid signals
            assert len(ecg_data) == 2500
            voltage_values = ecg_data['voltage'].to_numpy()
            assert np.all(np.isfinite(voltage_values))
            
            # Higher heart rates should have more frequent peaks
            # (This is a basic check - more sophisticated analysis could be added)
            assert voltage_values.max() > 5.0  # Should have prominent R waves
    
    def test_invalid_generation_parameters(self):
        """Test that invalid generation parameters raise errors."""
        engine = ECGPhysicsEngine(heart_rate_bpm=70.0)
        
        # Test negative duration
        with pytest.raises(ValueError, match="Duration must be positive"):
            engine.generate_ecg_segment(duration_seconds=-1.0)
        
        # Test zero duration
        with pytest.raises(ValueError, match="Duration must be positive"):
            engine.generate_ecg_segment(duration_seconds=0.0)
        
        # Test negative sampling rate
        with pytest.raises(ValueError, match="Sampling rate must be positive"):
            engine.generate_ecg_segment(duration_seconds=5.0, sampling_rate=-100)
        
        # Test too low sampling rate
        with pytest.raises(ValueError, match="too low for ECG"):
            engine.generate_ecg_segment(duration_seconds=5.0, sampling_rate=50)
        
        # Test too long duration
        with pytest.raises(ValueError, match="too long"):
            engine.generate_ecg_segment(duration_seconds=500.0)
    
    def test_reproducibility(self):
        """Test that ECG generation is reproducible."""
        # Note: The ODE solver may have some numerical variation,
        # but results should be very close for identical parameters
        engine1 = ECGPhysicsEngine(heart_rate_bpm=60.0)
        engine2 = ECGPhysicsEngine(heart_rate_bpm=60.0)
        
        ecg1 = engine1.generate_ecg_segment(duration_seconds=2.0, sampling_rate=500)
        ecg2 = engine2.generate_ecg_segment(duration_seconds=2.0, sampling_rate=500)
        
        voltage1 = ecg1['voltage'].to_numpy()
        voltage2 = ecg2['voltage'].to_numpy()
        
        # Should be very close (within numerical precision)
        max_diff = np.max(np.abs(voltage1 - voltage2))
        assert max_diff < 1e-10
    
    def test_pqrst_parameters(self):
        """Test that PQRST parameters are correctly configured."""
        engine = ECGPhysicsEngine(heart_rate_bpm=60.0)
        
        # Check that all required waves are present
        required_waves = ['P', 'Q', 'R', 'S', 'T']
        for wave in required_waves:
            assert wave in engine.pqrst_params
            
            theta_i, a_i, b_i = engine.pqrst_params[wave]
            
            # All parameters should be finite
            assert np.isfinite(theta_i)
            assert np.isfinite(a_i)
            assert np.isfinite(b_i)
            
            # Width parameter should be positive
            assert b_i > 0
        
        # R wave should have largest positive amplitude
        r_amplitude = engine.pqrst_params['R'][1]
        for wave, (_, a_i, _) in engine.pqrst_params.items():
            if wave != 'R':
                assert a_i <= r_amplitude

    def test_heart_rate_parameter_is_accurate(self):
        """Test that heart rate parameter produces correct fundamental frequency (Task 1.5 requirement).

        This validates the core timing of the biophysical model by analyzing the fundamental frequency.
        The McSharry model generates a continuous oscillation at the heart rate frequency.
        """
        from scipy.fft import fft, fftfreq

        # Test with 60 BPM (1 Hz fundamental frequency)
        heart_rate = 60.0
        expected_frequency = heart_rate / 60.0  # Convert BPM to Hz

        engine = ECGPhysicsEngine(heart_rate_bpm=heart_rate)

        # Generate ECG signal for frequency analysis
        duration = 20.0  # 20 seconds for good frequency resolution
        sampling_rate = 500
        ecg_data = engine.generate_ecg_segment(
            duration_seconds=duration,
            sampling_rate=sampling_rate
        )

        # Extract voltage signal
        voltage = ecg_data['voltage'].to_numpy()

        # Perform FFT to find dominant frequency
        fft_values = fft(voltage)
        frequencies = fftfreq(len(voltage), 1.0 / sampling_rate)

        # Calculate power spectrum (only positive frequencies)
        positive_freq_mask = frequencies > 0
        positive_frequencies = frequencies[positive_freq_mask]
        power_spectrum = np.abs(fft_values[positive_freq_mask]) ** 2

        # Find peak frequency in physiological range (0.5-3 Hz for 30-180 BPM)
        freq_mask = (positive_frequencies >= 0.5) & (positive_frequencies <= 3.0)
        if np.any(freq_mask):
            freq_subset = positive_frequencies[freq_mask]
            power_subset = power_spectrum[freq_mask]

            peak_idx = np.argmax(power_subset)
            dominant_frequency = freq_subset[peak_idx]

            # Assert fundamental frequency matches heart rate
            assert abs(dominant_frequency - expected_frequency) < 0.1, \
                f"Expected {expected_frequency:.2f} Hz, found {dominant_frequency:.2f} Hz"

        # Test with different heart rate
        heart_rate_2 = 75.0  # 1.25 Hz
        expected_frequency_2 = heart_rate_2 / 60.0

        engine_2 = ECGPhysicsEngine(heart_rate_bpm=heart_rate_2)
        ecg_data_2 = engine_2.generate_ecg_segment(
            duration_seconds=duration,
            sampling_rate=sampling_rate
        )

        voltage_2 = ecg_data_2['voltage'].to_numpy()
        fft_values_2 = fft(voltage_2)
        power_spectrum_2 = np.abs(fft_values_2[positive_freq_mask]) ** 2

        freq_mask_2 = (positive_frequencies >= 0.5) & (positive_frequencies <= 3.0)
        if np.any(freq_mask_2):
            freq_subset_2 = positive_frequencies[freq_mask_2]
            power_subset_2 = power_spectrum_2[freq_mask_2]

            peak_idx_2 = np.argmax(power_subset_2)
            dominant_frequency_2 = freq_subset_2[peak_idx_2]

            assert abs(dominant_frequency_2 - expected_frequency_2) < 0.1, \
                f"Expected {expected_frequency_2:.2f} Hz, found {dominant_frequency_2:.2f} Hz"


class TestClinicalNoiseModel:
    """Test the ClinicalNoiseModel class."""

    def test_noise_model_initialization(self):
        """Test that clinical noise model initializes correctly."""
        model = ClinicalNoiseModel(seed=42)
        assert hasattr(model, 'rng')

    def test_baseline_wander_generation(self):
        """Test baseline wander generation."""
        model = ClinicalNoiseModel(seed=123)
        time_array = np.linspace(0, 10, 5000)  # 10 seconds at 500 Hz

        baseline = model._generate_baseline_wander(time_array, amplitude=0.1)

        # Check output format
        assert len(baseline) == 5000
        assert isinstance(baseline, np.ndarray)

        # Check amplitude is reasonable (should be close to specified amplitude)
        rms_amplitude = np.std(baseline)
        assert 0.05 < rms_amplitude < 0.15  # Should be in reasonable range

        # Check frequency content (should be low frequency)
        # Simple check: signal should be relatively smooth
        diff_signal = np.diff(baseline)
        assert np.std(diff_signal) < 0.1  # Derivative should be small for low freq

    def test_muscle_artifact_generation(self):
        """Test muscle artifact generation."""
        model = ClinicalNoiseModel(seed=456)
        time_array = np.linspace(0, 5, 2500)  # 5 seconds at 500 Hz

        muscle = model._generate_muscle_artifact(time_array, amplitude=0.02)

        # Check output format
        assert len(muscle) == 2500
        assert isinstance(muscle, np.ndarray)

        # Check amplitude is reasonable
        rms_amplitude = np.std(muscle)
        assert 0.01 < rms_amplitude < 0.04  # Should be close to specified amplitude

        # Check high-frequency content (should be more variable than baseline)
        diff_signal = np.diff(muscle)
        assert np.std(diff_signal) > 0.001  # Should have high-frequency content

    def test_powerline_interference_generation(self):
        """Test powerline interference generation."""
        model = ClinicalNoiseModel(seed=789)
        time_array = np.linspace(0, 2, 1000)  # 2 seconds at 500 Hz

        # Test 50 Hz interference
        powerline_50 = model._generate_powerline_interference(
            time_array, 500, amplitude=0.01, frequency=50.0
        )

        # Test 60 Hz interference
        powerline_60 = model._generate_powerline_interference(
            time_array, 500, amplitude=0.01, frequency=60.0
        )

        # Check output format
        assert len(powerline_50) == 1000
        assert len(powerline_60) == 1000

        # Check amplitude (should be very close to specified for pure sine wave)
        rms_50 = np.std(powerline_50)
        rms_60 = np.std(powerline_60)

        # For sine wave, RMS = amplitude / sqrt(2) ≈ 0.707 * amplitude
        expected_rms = 0.01 / np.sqrt(2)
        assert abs(rms_50 - expected_rms) < 0.001
        assert abs(rms_60 - expected_rms) < 0.001

    def test_powerline_frequency_validation(self):
        """Test that powerline frequency validation works."""
        model = ClinicalNoiseModel(seed=101)
        time_array = np.linspace(0, 1, 500)  # 1 second at 500 Hz

        # Test frequency above Nyquist limit (250 Hz for 500 Hz sampling)
        with pytest.raises(ValueError, match="exceeds Nyquist limit"):
            model._generate_powerline_interference(time_array, 500, 0.01, 300.0)

    def test_composite_noise_generation(self):
        """Test composite noise generation combining all sources."""
        model = ClinicalNoiseModel(seed=202)
        time_array = np.linspace(0, 5, 2500)

        noise_levels = {
            'baseline': 0.1,
            'muscle': 0.02,
            'powerline': 0.01,
            'powerline_freq': 50.0
        }

        composite = model.generate_composite_noise(time_array, 500, noise_levels)

        # Check output format
        assert len(composite) == 2500
        assert isinstance(composite, np.ndarray)

        # Check that composite has reasonable amplitude
        # Should be roughly the sum of individual components
        composite_rms = np.std(composite)
        assert 0.05 < composite_rms < 0.2  # Should be reasonable combination

    def test_composite_noise_validation(self):
        """Test that composite noise generation validates inputs."""
        model = ClinicalNoiseModel(seed=303)
        time_array = np.linspace(0, 5, 2500)

        # Test missing required keys
        incomplete_noise = {'baseline': 0.1}  # Missing muscle and powerline
        with pytest.raises(ValueError, match="Missing required noise level keys"):
            model.generate_composite_noise(time_array, 500, incomplete_noise)

        # Test negative noise levels
        negative_noise = {'baseline': -0.1, 'muscle': 0.02, 'powerline': 0.01}
        with pytest.raises(ValueError, match="must be non-negative"):
            model.generate_composite_noise(time_array, 500, negative_noise)

        # Test empty time array
        valid_noise = {'baseline': 0.1, 'muscle': 0.02, 'powerline': 0.01}
        with pytest.raises(ValueError, match="time_array cannot be empty"):
            model.generate_composite_noise(np.array([]), 500, valid_noise)

    def test_reproducible_noise_generation(self):
        """Test that noise generation is reproducible with same seed."""
        model1 = ClinicalNoiseModel(seed=404)
        model2 = ClinicalNoiseModel(seed=404)

        time_array = np.linspace(0, 3, 1500)
        noise_levels = {'baseline': 0.08, 'muscle': 0.015, 'powerline': 0.005}

        noise1 = model1.generate_composite_noise(time_array, 500, noise_levels)
        noise2 = model2.generate_composite_noise(time_array, 500, noise_levels)

        # Should be identical with same seed
        np.testing.assert_array_almost_equal(noise1, noise2, decimal=10)

    def test_powerline_interference_adds_correct_frequency(self):
        """Test that powerline interference adds correct frequency component (Task 1.5 requirement).

        This proves our noise model is adding the correct type of artifact.
        """
        from scipy.fft import fft, fftfreq

        # Generate clean ECG signal
        engine = ECGPhysicsEngine(heart_rate_bpm=60.0)
        duration = 10.0
        sampling_rate = 500
        clean_ecg = engine.generate_ecg_segment(
            duration_seconds=duration,
            sampling_rate=sampling_rate
        )

        # Add powerline interference using ClinicalNoiseModel
        noise_model = ClinicalNoiseModel(seed=42)
        time_array = clean_ecg['time'].to_numpy()

        # Configure for 50 Hz powerline interference
        noise_levels = {
            'baseline': 0.0,      # No baseline wander
            'muscle': 0.0,        # No muscle artifact
            'powerline': 0.05,    # Strong powerline interference
            'powerline_freq': 50.0  # 50 Hz frequency
        }

        composite_noise = noise_model.generate_composite_noise(
            time_array, sampling_rate, noise_levels
        )

        # Combine clean signal with noise
        clean_voltage = clean_ecg['voltage'].to_numpy()
        noisy_voltage = clean_voltage + composite_noise

        # Perform FFT on the noisy signal
        fft_values = fft(noisy_voltage)
        frequencies = fftfreq(len(noisy_voltage), 1.0 / sampling_rate)

        # Calculate power spectrum (magnitude squared)
        power_spectrum = np.abs(fft_values) ** 2

        # Find frequency with maximum power in the range around 50 Hz
        freq_range_mask = (frequencies >= 45.0) & (frequencies <= 55.0)
        if np.any(freq_range_mask):
            freq_subset = frequencies[freq_range_mask]
            power_subset = power_spectrum[freq_range_mask]

            # Find peak frequency in the 45-55 Hz range
            peak_idx = np.argmax(power_subset)
            peak_frequency = freq_subset[peak_idx]

            # Assert that peak frequency is approximately 50 Hz
            assert abs(peak_frequency - 50.0) < 1.0, f"Peak frequency {peak_frequency} Hz not close to 50 Hz"

        # Additional validation: Test with 60 Hz
        noise_levels_60hz = {
            'baseline': 0.0,
            'muscle': 0.0,
            'powerline': 0.05,
            'powerline_freq': 60.0  # 60 Hz frequency
        }

        composite_noise_60 = noise_model.generate_composite_noise(
            time_array, sampling_rate, noise_levels_60hz
        )

        noisy_voltage_60 = clean_voltage + composite_noise_60
        fft_values_60 = fft(noisy_voltage_60)
        power_spectrum_60 = np.abs(fft_values_60) ** 2

        # Find peak in 55-65 Hz range
        freq_range_mask_60 = (frequencies >= 55.0) & (frequencies <= 65.0)
        if np.any(freq_range_mask_60):
            freq_subset_60 = frequencies[freq_range_mask_60]
            power_subset_60 = power_spectrum_60[freq_range_mask_60]

            peak_idx_60 = np.argmax(power_subset_60)
            peak_frequency_60 = freq_subset_60[peak_idx_60]

            assert abs(peak_frequency_60 - 60.0) < 1.0, f"Peak frequency {peak_frequency_60} Hz not close to 60 Hz"


class TestRealisticECGGenerator:
    """Test the RealisticECGGenerator class."""

    def test_generator_initialization(self):
        """Test that realistic ECG generator initializes correctly."""
        generator = RealisticECGGenerator(seed=42)

        assert generator.seed == 42
        assert hasattr(generator, 'noise_model')
        assert isinstance(generator.noise_model, ClinicalNoiseModel)

    def test_realistic_ecg_generation(self):
        """Test complete realistic ECG generation."""
        generator = RealisticECGGenerator(seed=123)

        noise_levels = {
            'baseline': 0.1,
            'muscle': 0.02,
            'powerline': 0.01,
            'powerline_freq': 50.0
        }

        ecg_data = generator.generate(
            duration_seconds=5.0,
            sampling_rate=500,
            heart_rate_bpm=75.0,
            noise_levels=noise_levels,
            rhythm_class="Normal Sinus Rhythm"
        )

        # Check output format
        expected_samples = 5.0 * 500  # 2500 samples
        assert len(ecg_data) == expected_samples

        # Check required columns
        required_columns = [
            'time', 'voltage', 'clean_voltage', 'noise',
            'heart_rate_bpm', 'rhythm_class',
            'baseline_noise_level', 'muscle_noise_level',
            'powerline_noise_level', 'powerline_frequency'
        ]
        for col in required_columns:
            assert col in ecg_data.columns

        # Check data ranges and consistency
        time_values = ecg_data['time'].to_numpy()
        voltage_values = ecg_data['voltage'].to_numpy()
        clean_values = ecg_data['clean_voltage'].to_numpy()
        noise_values = ecg_data['noise'].to_numpy()

        assert time_values.min() >= 0
        assert time_values.max() <= 5.0
        assert np.all(np.isfinite(voltage_values))
        assert np.all(np.isfinite(clean_values))
        assert np.all(np.isfinite(noise_values))

        # Check that voltage = clean + noise
        reconstructed = clean_values + noise_values
        reconstruction_error = np.mean(np.abs(voltage_values - reconstructed))
        assert reconstruction_error < 1e-10  # Should be numerically exact

    def test_signal_quality_metrics(self):
        """Test that generated signals have appropriate quality metrics."""
        generator = RealisticECGGenerator(seed=456)

        # Test with moderate noise levels
        noise_levels = {
            'baseline': 0.08,
            'muscle': 0.015,
            'powerline': 0.005
        }

        ecg_data = generator.generate(
            duration_seconds=10.0,
            sampling_rate=500,
            heart_rate_bpm=72.0,
            noise_levels=noise_levels
        )

        clean_signal = ecg_data['clean_voltage'].to_numpy()
        noise_signal = ecg_data['noise'].to_numpy()
        final_signal = ecg_data['voltage'].to_numpy()

        # Calculate signal quality metrics
        clean_rms = np.std(clean_signal)
        noise_rms = np.std(noise_signal)
        snr = clean_rms / noise_rms if noise_rms > 0 else float('inf')

        # ECG should have reasonable signal-to-noise ratio
        assert snr > 5.0  # Should be detectable
        assert snr < 100.0  # Should have realistic noise

        # Clean signal should dominate
        assert clean_rms > noise_rms

        # Final signal should have ECG characteristics
        assert final_signal.max() > 10.0  # Should have prominent R waves
        assert final_signal.min() < -5.0  # Should have negative deflections

    def test_different_noise_scenarios(self):
        """Test generation with different noise level scenarios."""
        generator = RealisticECGGenerator(seed=789)

        # Test quiet environment
        quiet_noise = {
            'baseline': 0.03,
            'muscle': 0.005,
            'powerline': 0.002
        }

        # Test noisy environment
        noisy_noise = {
            'baseline': 0.15,
            'muscle': 0.04,
            'powerline': 0.02
        }

        quiet_ecg = generator.generate(5.0, 500, 70.0, quiet_noise)
        noisy_ecg = generator.generate(5.0, 500, 70.0, noisy_noise)

        # Noisy environment should have higher noise levels
        quiet_noise_rms = np.std(quiet_ecg['noise'].to_numpy())
        noisy_noise_rms = np.std(noisy_ecg['noise'].to_numpy())

        assert noisy_noise_rms > quiet_noise_rms

        # Both should have valid ECG signals
        assert len(quiet_ecg) == len(noisy_ecg) == 2500
        assert np.all(np.isfinite(quiet_ecg['voltage'].to_numpy()))
        assert np.all(np.isfinite(noisy_ecg['voltage'].to_numpy()))

    def test_ground_truth_metadata(self):
        """Test that ground truth metadata is correctly included."""
        generator = RealisticECGGenerator(seed=101)

        noise_levels = {
            'baseline': 0.12,
            'muscle': 0.025,
            'powerline': 0.008,
            'powerline_freq': 60.0
        }

        ecg_data = generator.generate(
            duration_seconds=3.0,
            sampling_rate=250,
            heart_rate_bpm=85.0,
            noise_levels=noise_levels,
            rhythm_class="Test Rhythm"
        )

        # Check that metadata columns have consistent values
        heart_rate_col = ecg_data['heart_rate_bpm'].to_numpy()
        rhythm_col = ecg_data['rhythm_class'].to_numpy()
        baseline_col = ecg_data['baseline_noise_level'].to_numpy()

        # All values should be constant
        assert np.all(heart_rate_col == 85.0)
        assert np.all(rhythm_col == "Test Rhythm")
        assert np.all(baseline_col == 0.12)

        # Check other noise level metadata
        assert np.all(ecg_data['muscle_noise_level'].to_numpy() == 0.025)
        assert np.all(ecg_data['powerline_noise_level'].to_numpy() == 0.008)
        assert np.all(ecg_data['powerline_frequency'].to_numpy() == 60.0)

    def test_parameter_validation(self):
        """Test that invalid parameters are rejected."""
        generator = RealisticECGGenerator(seed=202)

        valid_noise = {'baseline': 0.1, 'muscle': 0.02, 'powerline': 0.01}

        # Test invalid duration
        with pytest.raises(ValueError, match="Duration must be positive"):
            generator.generate(-5.0, 500, 75.0, valid_noise)

        # Test invalid sampling rate
        with pytest.raises(ValueError, match="Sampling rate must be positive"):
            generator.generate(5.0, -500, 75.0, valid_noise)

        # Test invalid heart rate
        with pytest.raises(ValueError, match="outside physiological range"):
            generator.generate(5.0, 500, 300.0, valid_noise)

        # Test missing noise levels
        incomplete_noise = {'baseline': 0.1}
        with pytest.raises(ValueError, match="Missing required noise level keys"):
            generator.generate(5.0, 500, 75.0, incomplete_noise)

    def test_reproducible_generation(self):
        """Test that generation is reproducible with same seed."""
        generator1 = RealisticECGGenerator(seed=303)
        generator2 = RealisticECGGenerator(seed=303)

        noise_levels = {'baseline': 0.08, 'muscle': 0.015, 'powerline': 0.005}

        ecg1 = generator1.generate(3.0, 500, 68.0, noise_levels)
        ecg2 = generator2.generate(3.0, 500, 68.0, noise_levels)

        # Should be identical with same seed
        voltage1 = ecg1['voltage'].to_numpy()
        voltage2 = ecg2['voltage'].to_numpy()

        np.testing.assert_array_almost_equal(voltage1, voltage2, decimal=10)
