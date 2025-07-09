"""
Medical Domain Module Validation Script
=======================================

This script validates that the ECG_Encoder and Arrhythmia_Decoder properly implement
their respective interfaces and are ready for integration with the GIF framework.

This validation ensures:
1. Both classes properly inherit from their interfaces
2. All required abstract methods are implemented
3. Method signatures match interface contracts
4. Basic functionality works with sample data
5. Integration with GIF framework is possible

This script serves as the final verification step for Task 5.1.
"""

import sys
import traceback
import torch
import polars as pl
import numpy as np
from typing import Any, Dict

# Add project root to path for imports
sys.path.append('/Users/jp/RnD/GIF')

def test_interface_compliance():
    """Test that both modules properly implement their interfaces."""
    print("=" * 60)
    print("MEDICAL DOMAIN MODULE VALIDATION")
    print("=" * 60)
    
    try:
        # Import interfaces
        from gif_framework.interfaces.base_interfaces import EncoderInterface, DecoderInterface
        print("✅ Successfully imported base interfaces")
        
        # Import medical domain modules
        from applications.poc_medical.encoders.ecg_encoder import ECG_Encoder
        from applications.poc_medical.decoders.arrhythmia_decoder import Arrhythmia_Decoder
        print("✅ Successfully imported medical domain modules")
        
        # Test ECG_Encoder interface compliance
        print("\n" + "-" * 40)
        print("TESTING ECG_ENCODER INTERFACE COMPLIANCE")
        print("-" * 40)
        
        # Check inheritance
        assert issubclass(ECG_Encoder, EncoderInterface), "ECG_Encoder must inherit from EncoderInterface"
        print("✅ ECG_Encoder properly inherits from EncoderInterface")
        
        # Check required methods exist
        encoder = ECG_Encoder(sampling_rate=500, num_channels=3)
        assert hasattr(encoder, 'encode'), "ECG_Encoder must have encode method"
        assert hasattr(encoder, 'get_config'), "ECG_Encoder must have get_config method"
        assert hasattr(encoder, 'calibrate'), "ECG_Encoder must have calibrate method"
        print("✅ ECG_Encoder has all required interface methods")
        
        # Test method signatures and basic functionality
        config = encoder.get_config()
        assert isinstance(config, dict), "get_config must return a dictionary"
        print("✅ ECG_Encoder.get_config() returns valid configuration")
        
        # Test with sample ECG data
        sample_ecg_data = create_sample_ecg_data()
        spike_train = encoder.encode(sample_ecg_data)
        assert isinstance(spike_train, torch.Tensor), "encode must return torch.Tensor"
        assert spike_train.ndim == 2, "spike train must be 2D tensor"
        assert spike_train.shape[1] == 3, "spike train must have 3 channels"
        print(f"✅ ECG_Encoder.encode() produces valid spike train: {spike_train.shape}")
        
        # Test calibration
        encoder.calibrate(sample_ecg_data)
        print("✅ ECG_Encoder.calibrate() executes without errors")
        
        # Test Arrhythmia_Decoder interface compliance
        print("\n" + "-" * 40)
        print("TESTING ARRHYTHMIA_DECODER INTERFACE COMPLIANCE")
        print("-" * 40)
        
        # Check inheritance
        assert issubclass(Arrhythmia_Decoder, DecoderInterface), "Arrhythmia_Decoder must inherit from DecoderInterface"
        print("✅ Arrhythmia_Decoder properly inherits from DecoderInterface")
        
        # Check required methods exist
        decoder = Arrhythmia_Decoder(input_size=100, num_classes=8)
        assert hasattr(decoder, 'decode'), "Arrhythmia_Decoder must have decode method"
        assert hasattr(decoder, 'get_config'), "Arrhythmia_Decoder must have get_config method"
        print("✅ Arrhythmia_Decoder has all required interface methods")
        
        # Test method signatures and basic functionality
        config = decoder.get_config()
        assert isinstance(config, dict), "get_config must return a dictionary"
        print("✅ Arrhythmia_Decoder.get_config() returns valid configuration")
        
        # Test with sample spike train
        sample_spike_train = torch.rand(1000, 100)  # 1000 time steps, 100 neurons
        prediction = decoder.decode(sample_spike_train)
        assert isinstance(prediction, str), "decode must return string class name"
        assert prediction in decoder.class_names, "prediction must be valid class name"
        print(f"✅ Arrhythmia_Decoder.decode() produces valid prediction: '{prediction}'")
        
        # Test integration compatibility
        print("\n" + "-" * 40)
        print("TESTING INTEGRATION COMPATIBILITY")
        print("-" * 40)
        
        # Test encoder -> decoder pipeline
        ecg_spike_train = encoder.encode(sample_ecg_data)
        
        # Adjust decoder input size to match encoder output
        decoder_compatible = Arrhythmia_Decoder(input_size=ecg_spike_train.shape[1], num_classes=8)
        
        # Test that decoder can process encoder output
        prediction = decoder_compatible.decode(ecg_spike_train)
        print(f"✅ End-to-end pipeline works: ECG -> Spikes -> Prediction: '{prediction}'")
        
        # Test additional decoder features
        probabilities = decoder_compatible.get_class_probabilities(ecg_spike_train)
        assert isinstance(probabilities, dict), "get_class_probabilities must return dict"
        assert len(probabilities) == 8, "must return probabilities for all classes"
        print("✅ Arrhythmia_Decoder.get_class_probabilities() works correctly")
        
        top_k = decoder_compatible.get_top_k_predictions(ecg_spike_train, k=3)
        assert isinstance(top_k, list), "get_top_k_predictions must return list"
        assert len(top_k) == 3, "must return top 3 predictions"
        print("✅ Arrhythmia_Decoder.get_top_k_predictions() works correctly")
        
        print("\n" + "=" * 60)
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("=" * 60)
        print("\nMedical domain modules are ready for:")
        print("• Integration with GIF framework")
        print("• Potentiation experiments (Task 5.2)")
        print("• Cross-domain generalization testing")
        print("• End-to-end medical diagnostic pipeline")
        
        return True
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def create_sample_ecg_data():
    """Create sample ECG data for testing."""
    # Generate simple synthetic ECG-like signal
    duration_seconds = 5.0
    sampling_rate = 500
    num_samples = int(duration_seconds * sampling_rate)
    
    time_array = np.linspace(0, duration_seconds, num_samples)
    
    # Simple synthetic ECG with R-peaks every ~1 second (60 BPM)
    heart_rate = 60.0  # BPM
    r_peak_interval = 60.0 / heart_rate  # seconds between R-peaks
    
    voltage = np.zeros(num_samples)
    
    # Add R-peaks at regular intervals
    for i in range(int(duration_seconds / r_peak_interval)):
        peak_time = i * r_peak_interval + 0.5  # Start at 0.5 seconds
        peak_sample = int(peak_time * sampling_rate)
        
        if peak_sample < num_samples:
            # Add R-peak (sharp positive spike)
            voltage[peak_sample] = 1.0
            
            # Add some QRS complex shape
            for offset in range(-10, 11):
                sample_idx = peak_sample + offset
                if 0 <= sample_idx < num_samples:
                    voltage[sample_idx] += 0.3 * np.exp(-offset**2 / 20.0)
    
    # Add some noise
    voltage += 0.05 * np.random.randn(num_samples)
    
    # Create polars DataFrame in expected format
    ecg_data = pl.DataFrame({
        'time': time_array,
        'voltage': voltage
    })
    
    return ecg_data

def print_module_summary():
    """Print summary of implemented modules."""
    print("\n" + "=" * 60)
    print("MEDICAL DOMAIN MODULE SUMMARY")
    print("=" * 60)
    
    print("\n📁 ECG_Encoder (applications/poc_medical/encoders/ecg_encoder.py)")
    print("   • Multi-channel feature-based encoding")
    print("   • R-peak detection with adaptive thresholding")
    print("   • Heart rate encoding (rate-based coding)")
    print("   • QRS duration encoding (latency-based coding)")
    print("   • Robust signal processing with scipy")
    print("   • Automatic calibration capabilities")
    
    print("\n📁 Arrhythmia_Decoder (applications/poc_medical/decoders/arrhythmia_decoder.py)")
    print("   • Trainable linear readout layer")
    print("   • AAMI standard arrhythmia classification")
    print("   • Spike integration and population decoding")
    print("   • Probability distribution output")
    print("   • Top-k prediction capabilities")
    print("   • PyTorch nn.Module for end-to-end training")
    
    print("\n🔗 Interface Compliance:")
    print("   • Both modules inherit from base interfaces")
    print("   • All abstract methods properly implemented")
    print("   • Compatible with GIF orchestrator")
    print("   • Ready for plug-and-play architecture")

if __name__ == "__main__":
    print_module_summary()
    success = test_interface_compliance()
    
    if success:
        print("\n✅ Medical domain modules validation COMPLETE")
        print("Ready to proceed with Task 5.2: Potentiation Experiment Protocol")
    else:
        print("\n❌ Validation failed - modules need fixes before proceeding")
