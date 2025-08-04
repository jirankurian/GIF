"""
Comprehensive Test and Validation for Task 5.2 - Potentiation Experiment Protocol
=================================================================================

This script performs thorough testing and validation of all components required
for the System Potentiation experiment, identifying and reporting any issues
that need to be fixed.

Test Categories:
1. Configuration validation
2. Pre-trained model availability
3. Medical modules functionality
4. Weight-reset protocol
5. Experiment functions
6. Data generation pipeline
7. Integration testing
8. Error handling
"""

import os
import sys
import traceback
import torch
import logging
from typing import Dict, Any, List, Tuple
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def test_configuration():
    """Test configuration loading and validation."""
    print("=== Testing Configuration ===")
    issues = []
    
    try:
        from applications.poc_medical.config_med import get_config, PRE_TRAINED_EXO_MODEL_PATH
        config = get_config()
        print("✅ Configuration loaded successfully")
        
        # Check if pre-trained model path exists
        if not os.path.exists(PRE_TRAINED_EXO_MODEL_PATH):
            issues.append(f"❌ Pre-trained model not found: {PRE_TRAINED_EXO_MODEL_PATH}")
        else:
            print(f"✅ Pre-trained model found: {PRE_TRAINED_EXO_MODEL_PATH}")
            
    except Exception as e:
        issues.append(f"❌ Configuration error: {str(e)}")
        
    return issues

def test_medical_modules():
    """Test medical domain modules."""
    print("\n=== Testing Medical Modules ===")
    issues = []
    
    try:
        # Test ECG Encoder
        from applications.poc_medical.encoders.ecg_encoder import ECG_Encoder
        from data_generators.ecg_generator import RealisticECGGenerator
        
        # Generate test ECG data
        ecg_gen = RealisticECGGenerator()
        ecg_data = ecg_gen.generate_ecg_segment(duration=5.0, sampling_rate=250)
        
        # Test encoder
        encoder = ECG_Encoder()
        spike_train = encoder.encode(ecg_data)
        print(f"✅ ECG Encoder working - output shape: {spike_train.shape}")
        
        # Test Arrhythmia Decoder
        from applications.poc_medical.decoders.arrhythmia_decoder import Arrhythmia_Decoder
        
        decoder = Arrhythmia_Decoder(input_size=spike_train.shape[1])
        prediction = decoder.decode(spike_train)
        print(f"✅ Arrhythmia Decoder working - prediction: {prediction}")
        
    except Exception as e:
        issues.append(f"❌ Medical modules error: {str(e)}")
        traceback.print_exc()
        
    return issues

def test_weight_reset_protocol():
    """Test the weight-reset protocol."""
    print("\n=== Testing Weight-Reset Protocol ===")
    issues = []
    
    try:
        from gif_framework.core.du_core import DU_Core_V1
        
        # Create DU_Core instance
        du_core = DU_Core_V1(input_size=3, hidden_sizes=[32, 16], output_size=8)
        
        # Store original weights
        original_weights = []
        for layer in du_core.linear_layers:
            original_weights.append(layer.weight.data.clone())
            
        # Reset weights
        du_core.reset_weights()
        
        # Check if weights changed
        weights_changed = False
        for i, layer in enumerate(du_core.linear_layers):
            if not torch.equal(original_weights[i], layer.weight.data):
                weights_changed = True
                break
                
        if weights_changed:
            print("✅ Weight reset protocol working correctly")
        else:
            issues.append("❌ Weight reset protocol failed - weights unchanged")
            
    except Exception as e:
        issues.append(f"❌ Weight reset error: {str(e)}")
        traceback.print_exc()
        
    return issues

def test_experiment_functions():
    """Test experiment function availability."""
    print("\n=== Testing Experiment Functions ===")
    issues = []
    
    try:
        from applications.poc_medical.main_med import (
            run_sota_baseline, 
            run_naive_gif_du, 
            run_pre_exposed_gif_du
        )
        print("✅ All experiment functions imported successfully")
        
        # Test if functions are callable
        if not callable(run_sota_baseline):
            issues.append("❌ run_sota_baseline is not callable")
        if not callable(run_naive_gif_du):
            issues.append("❌ run_naive_gif_du is not callable")
        if not callable(run_pre_exposed_gif_du):
            issues.append("❌ run_pre_exposed_gif_du is not callable")
            
        if not issues:
            print("✅ All experiment functions are callable")
            
    except Exception as e:
        issues.append(f"❌ Experiment functions error: {str(e)}")
        traceback.print_exc()
        
    return issues

def test_data_generation():
    """Test data generation pipeline."""
    print("\n=== Testing Data Generation ===")
    issues = []
    
    try:
        from data_generators.ecg_generator import RealisticECGGenerator
        
        generator = RealisticECGGenerator()
        
        # Test single ECG generation
        ecg_data = generator.generate_ecg_segment(duration=5.0, sampling_rate=250)
        print(f"✅ Single ECG generation working - shape: {ecg_data.shape}")
        
        # Test batch generation
        batch_data = generator.generate_batch(
            num_samples=5,
            duration=5.0,
            sampling_rate=250
        )
        print(f"✅ Batch ECG generation working - {len(batch_data)} samples")
        
    except Exception as e:
        issues.append(f"❌ Data generation error: {str(e)}")
        traceback.print_exc()
        
    return issues

def test_integration():
    """Test end-to-end integration."""
    print("\n=== Testing Integration ===")
    issues = []
    
    try:
        # Test complete pipeline
        from data_generators.ecg_generator import RealisticECGGenerator
        from applications.poc_medical.encoders.ecg_encoder import ECG_Encoder
        from applications.poc_medical.decoders.arrhythmia_decoder import Arrhythmia_Decoder
        from gif_framework.orchestrator import GIF
        from gif_framework.core.du_core import DU_Core_V1
        
        # Generate data
        generator = RealisticECGGenerator()
        ecg_data = generator.generate_ecg_segment(duration=5.0, sampling_rate=250)
        
        # Create components
        encoder = ECG_Encoder()
        decoder = Arrhythmia_Decoder(input_size=3)
        du_core = DU_Core_V1(input_size=3, hidden_sizes=[32, 16], output_size=8)
        
        # Create GIF framework
        gif = GIF(du_core=du_core)
        gif.attach_encoder(encoder)
        gif.attach_decoder(decoder)
        
        # Test processing
        result = gif.process(ecg_data)
        print(f"✅ End-to-end integration working - result: {result}")
        
    except Exception as e:
        issues.append(f"❌ Integration error: {str(e)}")
        traceback.print_exc()
        
    return issues

def main():
    """Run comprehensive testing and validation."""
    print("🧪 COMPREHENSIVE TASK 5.2 TESTING AND VALIDATION")
    print("=" * 60)
    
    all_issues = []
    
    # Run all tests
    all_issues.extend(test_configuration())
    all_issues.extend(test_medical_modules())
    all_issues.extend(test_weight_reset_protocol())
    all_issues.extend(test_experiment_functions())
    all_issues.extend(test_data_generation())
    all_issues.extend(test_integration())
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TESTING SUMMARY")
    print("=" * 60)
    
    if all_issues:
        print(f"❌ FOUND {len(all_issues)} ISSUES:")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
        print("\n🔧 ISSUES NEED TO BE FIXED")
        return False
    else:
        print("✅ ALL TESTS PASSED - NO ISSUES FOUND")
        print("🎉 Task 5.2 is fully validated and ready")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
