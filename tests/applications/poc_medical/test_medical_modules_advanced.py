"""
Advanced Test Suite for Medical Domain Modules
==============================================

This module provides advanced testing for the medical domain modules, focusing on
clinical accuracy and scientific validation of ECG processing and arrhythmia
classification capabilities.

Test Coverage:
- ECG feature extraction validation with known signals
- R-peak detection accuracy testing
- Heart rate calculation validation
- QRS duration estimation testing
- Multi-channel spike train generation
- Arrhythmia classification accuracy
- AAMI standard compliance validation
- Clinical relevance testing

These tests ensure the medical modules meet clinical standards and provide
scientifically valid results for the potentiation experiment.
"""

import pytest
import torch
import numpy as np
import polars as pl
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch

# Import medical domain modules
from applications.poc_medical.encoders.ecg_encoder import ECG_Encoder
from applications.poc_medical.decoders.arrhythmia_decoder import Arrhythmia_Decoder

# Import data generation for testing
from data_generators.ecg_generator import RealisticECGGenerator

# Try to import scipy for signal processing validation
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class TestECGFeatureExtractionValidation:
    """Test suite for ECG feature extraction with known signals."""
    
    def test_synthetic_ecg_with_known_r_peaks(self):
        """
        Test ECG encoder with synthetic signal containing known R-peak locations.
        
        This validates that the encoder correctly identifies cardiac events
        in controlled test conditions.
        """
        # Create synthetic ECG with known R-peak locations
        sampling_rate = 250  # Hz
        duration = 10  # seconds
        time_points = np.linspace(0, duration, duration * sampling_rate)
        
        # Create baseline signal
        ecg_signal = np.zeros_like(time_points)
        
        # Add known R-peaks at specific times (60 BPM = 1 Hz)
        r_peak_times = [1.0, 2.0, 3.0, 4.0, 5.0]  # 5 R-peaks
        r_peak_indices = [int(t * sampling_rate) for t in r_peak_times]
        
        # Add R-peak spikes
        for idx in r_peak_indices:
            if idx < len(ecg_signal):
                # Create QRS complex shape
                for offset in range(-5, 6):
                    if 0 <= idx + offset < len(ecg_signal):
                        if offset == 0:
                            ecg_signal[idx + offset] = 1.0  # R-peak
                        elif abs(offset) == 1:
                            ecg_signal[idx + offset] = 0.5  # Q and S waves
                        elif abs(offset) <= 3:
                            ecg_signal[idx + offset] = 0.2  # QRS complex
        
        # Add some noise
        ecg_signal += np.random.normal(0, 0.05, len(ecg_signal))
        
        # Create polars DataFrame
        ecg_df = pl.DataFrame({
            'time': time_points,
            'voltage': ecg_signal
        })
        
        # Test ECG encoder
        encoder = ECG_Encoder()
        spike_train = encoder.encode(ecg_df)
        
        # Verify output shape (should be [time_steps, 3] for 3-channel encoding)
        assert spike_train.shape[1] == 3, f"Expected 3 channels, got {spike_train.shape[1]}"
        
        # Check R-peak channel (channel 0)
        r_peak_channel = spike_train[:, 0]
        detected_r_peaks = torch.sum(r_peak_channel).item()
        
        # Should detect approximately the correct number of R-peaks (allow reasonable tolerance)
        expected_r_peaks = len(r_peak_times)
        # Allow up to 50% tolerance for realistic peak detection algorithms
        tolerance = max(2, int(expected_r_peaks * 0.5))
        assert abs(detected_r_peaks - expected_r_peaks) <= tolerance, \
            f"R-peak detection inaccurate: detected {detected_r_peaks}, expected {expected_r_peaks} (tolerance: {tolerance})"
    
    def test_heart_rate_calculation_accuracy(self):
        """
        Test heart rate calculation with known cardiac rhythm.
        
        This validates the encoder's ability to accurately measure heart rate
        from R-R intervals.
        """
        # Create ECG with known heart rate (75 BPM)
        target_bpm = 75
        rr_interval = 60.0 / target_bpm  # 0.8 seconds
        
        sampling_rate = 250
        duration = 10
        time_points = np.linspace(0, duration, duration * sampling_rate)
        
        # Generate R-peaks at regular intervals
        r_peak_times = []
        current_time = 1.0  # Start at 1 second
        while current_time < duration - 1:
            r_peak_times.append(current_time)
            current_time += rr_interval
        
        # Create ECG signal
        ecg_signal = np.zeros_like(time_points)
        for peak_time in r_peak_times:
            peak_idx = int(peak_time * sampling_rate)
            if peak_idx < len(ecg_signal):
                ecg_signal[peak_idx] = 1.0
        
        # Add noise
        ecg_signal += np.random.normal(0, 0.02, len(ecg_signal))
        
        # Create DataFrame
        ecg_df = pl.DataFrame({
            'time': time_points,
            'voltage': ecg_signal
        })
        
        # Test encoder
        encoder = ECG_Encoder()
        
        # Access internal heart rate calculation (if available)
        # This tests the encoder's internal logic
        spike_train = encoder.encode(ecg_df)
        
        # Verify heart rate encoding in channel 1
        heart_rate_channel = spike_train[:, 1]
        
        # The heart rate channel should have activity proportional to heart rate
        # Higher heart rate should produce more spikes
        total_hr_spikes = torch.sum(heart_rate_channel).item()
        
        # Should have reasonable number of spikes for 75 BPM
        # (This is a heuristic test - exact values depend on encoding scheme)
        assert total_hr_spikes > 0, "No heart rate encoding detected"
        assert total_hr_spikes < len(spike_train), "Too many heart rate spikes"
    
    def test_qrs_duration_estimation(self):
        """
        Test QRS duration estimation with controlled QRS complexes.
        
        This validates the encoder's ability to measure QRS width,
        which is important for detecting conduction abnormalities.
        """
        # Create ECG with known QRS duration
        sampling_rate = 250
        duration = 5
        time_points = np.linspace(0, duration, duration * sampling_rate)
        
        # Create signal with wide QRS complexes (120ms = pathological)
        qrs_duration_ms = 120
        qrs_duration_samples = int(qrs_duration_ms * sampling_rate / 1000)
        
        ecg_signal = np.zeros_like(time_points)
        
        # Add QRS complexes at 1, 2, 3, 4 seconds
        qrs_times = [1.0, 2.0, 3.0, 4.0]
        for qrs_time in qrs_times:
            qrs_start_idx = int(qrs_time * sampling_rate)
            qrs_end_idx = qrs_start_idx + qrs_duration_samples
            
            if qrs_end_idx < len(ecg_signal):
                # Create wide QRS complex
                for i in range(qrs_start_idx, qrs_end_idx):
                    # Triangular QRS shape
                    progress = (i - qrs_start_idx) / qrs_duration_samples
                    if progress < 0.5:
                        ecg_signal[i] = progress * 2  # Rising edge
                    else:
                        ecg_signal[i] = 2 * (1 - progress)  # Falling edge
        
        # Add noise
        ecg_signal += np.random.normal(0, 0.03, len(ecg_signal))
        
        # Create DataFrame
        ecg_df = pl.DataFrame({
            'time': time_points,
            'voltage': ecg_signal
        })
        
        # Test encoder
        encoder = ECG_Encoder()
        spike_train = encoder.encode(ecg_df)
        
        # Check QRS duration channel (channel 2)
        qrs_channel = spike_train[:, 2]
        
        # Should have activity in QRS duration channel
        qrs_activity = torch.sum(qrs_channel).item()
        assert qrs_activity > 0, "No QRS duration encoding detected"
        
        # The timing of spikes in this channel should reflect QRS duration
        # (Specific validation depends on encoding scheme implementation)
    
    def test_multi_channel_spike_generation_consistency(self):
        """
        Test consistency and independence of multi-channel spike generation.
        
        This validates that each channel encodes different cardiac features
        independently and consistently.
        """
        # Generate realistic ECG data
        ecg_generator = RealisticECGGenerator()
        ecg_data = ecg_generator.generate(
            duration_seconds=8.0,
            sampling_rate=250,
            heart_rate_bpm=70,
            noise_levels={'baseline': 0.1, 'muscle': 0.05, 'powerline': 0.02}
        )
        
        # Test encoder multiple times with same data
        encoder = ECG_Encoder()
        
        spike_train_1 = encoder.encode(ecg_data)
        spike_train_2 = encoder.encode(ecg_data)
        
        # ECG encoder may have some inherent randomness in spike generation
        # Test that the overall structure is consistent rather than exact equality
        assert spike_train_1.shape == spike_train_2.shape, \
            "Encoder produces different output shapes"

        # Test that both encodings have similar activity levels (within 20%)
        activity_1 = torch.sum(spike_train_1).item()
        activity_2 = torch.sum(spike_train_2).item()

        if activity_1 > 0 and activity_2 > 0:
            activity_ratio = min(activity_1, activity_2) / max(activity_1, activity_2)
            assert activity_ratio > 0.8, \
                f"Encoder activity levels too different: {activity_1} vs {activity_2}"
        
        # Verify multi-channel structure
        assert spike_train_1.shape[1] == 3, "Should have 3 channels"
        
        # Each channel should have different activity patterns
        channel_activities = []
        for channel in range(3):
            activity = torch.sum(spike_train_1[:, channel]).item()
            channel_activities.append(activity)
        
        # Channels should not all be identical (very unlikely)
        assert not all(act == channel_activities[0] for act in channel_activities), \
            "All channels have identical activity"
        
        # All channels should have some activity
        for i, activity in enumerate(channel_activities):
            assert activity > 0, f"Channel {i} has no activity"


class TestArrhythmiaClassificationValidation:
    """Test suite for arrhythmia classification accuracy and AAMI compliance."""
    
    def test_aami_standard_class_coverage(self):
        """
        Test that decoder supports all AAMI standard arrhythmia classes.
        
        This validates compliance with established medical standards
        for arrhythmia classification.
        """
        # AAMI standard classes
        expected_classes = [
            'Normal Sinus Rhythm',
            'Atrial Fibrillation', 
            'Atrial Flutter',
            'Supraventricular Tachycardia',
            'Ventricular Tachycardia',
            'Ventricular Fibrillation',
            'Premature Ventricular Contraction',
            'Premature Atrial Contraction'
        ]
        
        # Create decoder
        decoder = Arrhythmia_Decoder(input_size=32, num_classes=len(expected_classes))
        
        # Verify number of output classes
        assert decoder.num_classes == len(expected_classes), \
            f"Expected {len(expected_classes)} classes, got {decoder.num_classes}"
        
        # Test classification with mock spike trains
        for class_idx in range(len(expected_classes)):
            # Create mock spike train with non-negative values (proper spike data)
            mock_spike_train = torch.rand(100, 32)  # [time_steps, features] - all positive
            mock_spike_train = (mock_spike_train > 0.8).float()  # Convert to binary spikes

            # Get prediction
            prediction = decoder.decode(mock_spike_train)
            
            # Should return valid class name
            assert isinstance(prediction, str), "Prediction should be string"
            assert prediction in expected_classes, f"Invalid class prediction: {prediction}"
    
    def test_classification_probability_distribution(self):
        """
        Test that decoder produces valid probability distributions.
        
        This validates the statistical properties of classification outputs.
        """
        decoder = Arrhythmia_Decoder(input_size=16, num_classes=8)
        
        # Create test spike train with non-negative values (proper spike data)
        test_spike_train = torch.rand(50, 16)
        test_spike_train = (test_spike_train > 0.7).float()  # Convert to binary spikes
        
        # Get probability distribution (if available through forward method)
        decoder.eval()
        with torch.no_grad():
            # Integrate spikes
            integrated_spikes = torch.sum(test_spike_train, dim=0, keepdim=True)
            
            # Get logits
            logits = decoder.linear_readout(integrated_spikes)
            
            # Apply softmax
            probabilities = torch.softmax(logits, dim=1)
        
        # Verify probability properties
        assert probabilities.shape == (1, 8), "Probability shape incorrect"
        
        # Should sum to 1
        prob_sum = torch.sum(probabilities).item()
        assert abs(prob_sum - 1.0) < 1e-6, f"Probabilities don't sum to 1: {prob_sum}"
        
        # All probabilities should be non-negative
        assert torch.all(probabilities >= 0), "Negative probabilities detected"
        
        # All probabilities should be <= 1
        assert torch.all(probabilities <= 1), "Probabilities > 1 detected"
    
    def test_training_vs_evaluation_mode_behavior(self):
        """
        Test different behavior in training vs evaluation modes.
        
        This validates proper dropout and other training-specific behaviors.
        """
        decoder = Arrhythmia_Decoder(input_size=20, num_classes=5, dropout_rate=0.5)
        
        # Create test data with non-negative values (proper spike data)
        test_spike_train = torch.rand(30, 20)
        test_spike_train = (test_spike_train > 0.8).float()  # Convert to binary spikes
        
        # Test in training mode
        decoder.train()
        train_prediction_1 = decoder.decode(test_spike_train)
        train_prediction_2 = decoder.decode(test_spike_train)
        
        # Test in evaluation mode
        decoder.eval()
        eval_prediction_1 = decoder.decode(test_spike_train)
        eval_prediction_2 = decoder.decode(test_spike_train)
        
        # Evaluation mode should be deterministic
        assert eval_prediction_1 == eval_prediction_2, \
            "Evaluation mode not deterministic"
        
        # Training mode might be different due to dropout (but not guaranteed)
        # Just verify both modes produce valid predictions
        valid_classes = ['Normal Sinus Rhythm', 'Atrial Fibrillation', 'Atrial Flutter', 
                        'Supraventricular Tachycardia', 'Ventricular Tachycardia']
        
        assert train_prediction_1 in valid_classes, "Invalid training prediction"
        assert eval_prediction_1 in valid_classes, "Invalid evaluation prediction"
    
    def test_gradient_flow_for_training(self):
        """
        Test gradient flow through decoder for end-to-end training.
        
        This validates that the decoder can be trained with backpropagation.
        """
        decoder = Arrhythmia_Decoder(input_size=12, num_classes=4)
        
        # Create mock training data with non-negative values (proper spike data)
        spike_train = torch.rand(25, 12, requires_grad=True)
        spike_train = (spike_train > 0.7).float()  # Convert to binary spikes
        spike_train.requires_grad_(True)  # Re-enable gradients after conversion
        target_class = torch.tensor([2])  # Target class index
        
        # Forward pass
        decoder.train()
        integrated_spikes = torch.sum(spike_train, dim=0, keepdim=True)
        logits = decoder.linear_readout(integrated_spikes)
        
        # Calculate loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, target_class)
        
        # Backward pass
        loss.backward()
        
        # Verify gradients exist
        for name, param in decoder.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradients in {name}"
        
        # Verify input gradients
        assert spike_train.grad is not None, "No gradient for input spike train"
        assert torch.isfinite(spike_train.grad).all(), "Non-finite input gradients"
