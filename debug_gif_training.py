#!/usr/bin/env python3
"""
GIF-DU Training Diagnostic Script
=================================

This script systematically diagnoses the training issues with the GIF-DU model
by testing each component individually and identifying the root cause of the
learning failure.

Issues to investigate:
1. Spike generation from encoder
2. Gradient flow through SNN
3. Learning rate and optimizer settings
4. Continual learning (GEM) implementation
5. Loss function and target formatting
"""

import sys
import os
sys.path.append('/Users/jp/RnD/GIF')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import polars as pl
from typing import Dict, Any

# Import GIF framework components
from applications.poc_exoplanet.config_exo import EXP_CONFIG
from applications.poc_exoplanet.encoders.light_curve_encoder import LightCurveEncoder
from applications.poc_exoplanet.decoders.exoplanet_decoder import ExoplanetDecoder
from gif_framework.core.du_core import DU_Core_V1
from gif_framework.orchestrator import GIF
from gif_framework.training.trainer import Continual_Trainer
from data_generators.exoplanet_generator import RealisticExoplanetGenerator
from gif_framework.core.memory_systems import EpisodicMemory

def test_data_generation():
    """Test 1: Verify data generation works correctly."""
    print("=" * 60)
    print("TEST 1: Data Generation")
    print("=" * 60)
    
    try:
        generator = RealisticExoplanetGenerator(seed=42)
        light_curve = generator.generate()
        
        print(f"✓ Generated light curve with {len(light_curve)} time points")
        print(f"✓ Columns: {light_curve.columns}")
        print(f"✓ Flux range: {light_curve['flux'].min():.6f} to {light_curve['flux'].max():.6f}")
        print(f"✓ Flux std: {light_curve['flux'].std():.6f}")
        
        # Check for planet signal from the data
        has_planet = light_curve['planet_radius_ratio'][0] > 0
        print(f"✓ Planet present: {has_planet}")
        if has_planet:
            print(f"✓ Planet radius ratio: {light_curve['planet_radius_ratio'][0]:.6f}")
        
        return light_curve, has_planet
        
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return None, None

def test_encoder(light_curve):
    """Test 2: Verify encoder generates meaningful spikes."""
    print("\n" + "=" * 60)
    print("TEST 2: Encoder Spike Generation")
    print("=" * 60)
    
    try:
        config = EXP_CONFIG["encoder"]
        encoder = LightCurveEncoder(
            threshold=config["threshold"],
            normalize_input=config["normalize_input"]
        )
        
        # Encode the light curve
        spike_train = encoder.encode(light_curve)
        
        print(f"✓ Encoder created with threshold: {config['threshold']:.2e}")
        print(f"✓ Spike train shape: {spike_train.shape}")
        print(f"✓ Spike train dtype: {spike_train.dtype}")
        
        # Analyze spike statistics
        total_spikes = spike_train.sum().item()
        positive_spikes = spike_train[:, 0].sum().item()
        negative_spikes = spike_train[:, 1].sum().item()
        spike_rate = total_spikes / spike_train.shape[0]
        
        print(f"✓ Total spikes: {total_spikes}")
        print(f"✓ Positive spikes: {positive_spikes}")
        print(f"✓ Negative spikes: {negative_spikes}")
        print(f"✓ Spike rate: {spike_rate:.4f}")
        
        # Check if spikes are being generated
        if total_spikes == 0:
            print("⚠️  WARNING: No spikes generated! This will prevent learning.")
            print("   Possible causes:")
            print("   - Threshold too high")
            print("   - Input data too smooth")
            print("   - Normalization issues")
            
            # Try with lower threshold
            print("\n   Testing with lower threshold...")
            encoder_low = LightCurveEncoder(threshold=config["threshold"] * 0.1)
            spike_train_low = encoder_low.encode(light_curve)
            total_spikes_low = spike_train_low.sum().item()
            print(f"   Lower threshold spikes: {total_spikes_low}")
            
        return spike_train, encoder
        
    except Exception as e:
        print(f"✗ Encoder test failed: {e}")
        return None, None

def test_du_core(spike_train):
    """Test 3: Verify DU Core processes spikes correctly."""
    print("\n" + "=" * 60)
    print("TEST 3: DU Core Processing")
    print("=" * 60)
    
    try:
        config = EXP_CONFIG["architecture"]
        
        # Create DU Core with memory system for continual learning
        memory_system = EpisodicMemory(capacity=1000)
        
        du_core = DU_Core_V1(
            input_size=config["input_size"],
            hidden_sizes=config["hidden_sizes"],
            output_size=config["output_size"],
            beta=config["neuron_beta"],
            threshold=config["neuron_threshold"],
            memory_system=memory_system
        )
        
        print(f"✓ DU Core created: {config['input_size']} → {config['hidden_sizes']} → {config['output_size']}")
        print(f"✓ Neuron beta: {config['neuron_beta']}")
        print(f"✓ Neuron threshold: {config['neuron_threshold']}")
        
        # Add batch dimension if needed
        if spike_train.dim() == 2:
            spike_train_batch = spike_train.unsqueeze(1)  # [time, 1, features]
        else:
            spike_train_batch = spike_train
            
        print(f"✓ Input spike train shape: {spike_train_batch.shape}")
        
        # Process through DU Core
        output_spikes = du_core.process(spike_train_batch)
        
        print(f"✓ Output spike train shape: {output_spikes.shape}")
        print(f"✓ Output spike train dtype: {output_spikes.dtype}")
        
        # Analyze output statistics
        total_output_spikes = output_spikes.sum().item()
        output_spike_rate = total_output_spikes / output_spikes.numel()
        
        print(f"✓ Total output spikes: {total_output_spikes}")
        print(f"✓ Output spike rate: {output_spike_rate:.6f}")
        
        if total_output_spikes == 0:
            print("⚠️  WARNING: No output spikes! This indicates a problem.")
            print("   Possible causes:")
            print("   - Input spikes too weak")
            print("   - Neuron threshold too high")
            print("   - Network weights too small")
            
            # Check network weights
            total_params = sum(p.numel() for p in du_core.parameters())
            weight_stats = []
            for name, param in du_core.named_parameters():
                weight_stats.append(f"   {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")
            
            print(f"   Total parameters: {total_params}")
            print("   Weight statistics:")
            for stat in weight_stats:
                print(stat)
        
        return output_spikes, du_core
        
    except Exception as e:
        print(f"✗ DU Core test failed: {e}")
        return None, None

def test_decoder(output_spikes):
    """Test 4: Verify decoder processes output spikes correctly."""
    print("\n" + "=" * 60)
    print("TEST 4: Decoder Processing")
    print("=" * 60)

    try:
        # Get the actual output size from the DU Core (should be 2)
        input_size = output_spikes.shape[-1]  # Last dimension is the feature dimension
        output_size = 1  # For regression mode

        decoder = ExoplanetDecoder(
            input_size=input_size,
            output_size=output_size
        )
        
        print(f"✓ Decoder created: {input_size} → {output_size}")
        
        # Test classification mode
        prediction = decoder.decode(output_spikes, mode='classification')
        print(f"✓ Classification prediction: {prediction}")
        print(f"✓ Prediction type: {type(prediction)}")
        
        # Test forward method for gradients
        spike_counts = output_spikes.sum(dim=0)  # Sum across time
        if spike_counts.dim() == 1:
            spike_counts = spike_counts.unsqueeze(0)  # Add batch dimension
            
        logits = decoder.forward_classification(spike_counts)
        print(f"✓ Logits shape: {logits.shape}")
        print(f"✓ Logits values: {logits}")
        print(f"✓ Logits require grad: {logits.requires_grad}")
        
        return logits, decoder

    except Exception as e:
        print(f"✗ Decoder test failed: {e}")
        return None, None

def test_gradient_flow(du_core, decoder, spike_train, target_label):
    """Test 5: Verify gradients flow through the network."""
    print("\n" + "=" * 60)
    print("TEST 5: Gradient Flow Analysis")
    print("=" * 60)

    try:
        # Add batch dimension if needed
        if spike_train.dim() == 2:
            spike_train_batch = spike_train.unsqueeze(1)
        else:
            spike_train_batch = spike_train

        # Forward pass
        output_spikes = du_core.process(spike_train_batch)
        spike_counts = output_spikes.sum(dim=0)
        if spike_counts.dim() == 1:
            spike_counts = spike_counts.unsqueeze(0)

        logits = decoder.forward_classification(spike_counts)

        # Create target tensor
        target = torch.tensor([target_label], dtype=torch.long)

        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, target)

        print(f"✓ Forward pass completed")
        print(f"✓ Loss value: {loss.item():.6f}")
        print(f"✓ Loss requires grad: {loss.requires_grad}")

        # Backward pass
        loss.backward()

        # Check gradients
        gradient_stats = []
        total_grad_norm = 0.0
        params_with_grad = 0
        params_without_grad = 0

        for name, param in du_core.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_stats.append(f"   {name}: norm={grad_norm:.6f}")
                total_grad_norm += grad_norm
                params_with_grad += 1
            else:
                gradient_stats.append(f"   {name}: NO GRADIENT")
                params_without_grad += 1

        for name, param in decoder.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_stats.append(f"   decoder.{name}: norm={grad_norm:.6f}")
                total_grad_norm += grad_norm
                params_with_grad += 1
            else:
                gradient_stats.append(f"   decoder.{name}: NO GRADIENT")
                params_without_grad += 1

        print(f"✓ Parameters with gradients: {params_with_grad}")
        print(f"✓ Parameters without gradients: {params_without_grad}")
        print(f"✓ Total gradient norm: {total_grad_norm:.6f}")

        if params_without_grad > 0:
            print("⚠️  WARNING: Some parameters have no gradients!")

        if total_grad_norm < 1e-8:
            print("⚠️  WARNING: Gradients are extremely small!")
            print("   This may indicate:")
            print("   - Vanishing gradient problem")
            print("   - Dead neurons")
            print("   - Learning rate too small")

        print("\nGradient details:")
        for stat in gradient_stats:
            print(stat)

        return loss.item(), total_grad_norm

    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        return None, None

if __name__ == "__main__":
    print("GIF-DU Training Diagnostic Script")
    print("=" * 60)
    
    # Test 1: Data Generation
    light_curve, has_planet = test_data_generation()
    if light_curve is None:
        exit(1)
    
    # Test 2: Encoder
    spike_train, encoder = test_encoder(light_curve)
    if spike_train is None:
        exit(1)
    
    # Test 3: DU Core
    output_spikes, du_core = test_du_core(spike_train)
    if output_spikes is None:
        exit(1)
    
    # Test 4: Decoder
    logits, decoder = test_decoder(output_spikes)
    if logits is None:
        exit(1)
    
    # Test 5: Gradient Flow
    target_label = 1 if has_planet else 0
    loss, grad_norm = test_gradient_flow(du_core, decoder, spike_train, target_label)

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    total_input_spikes = spike_train.sum().item()
    total_output_spikes = output_spikes.sum().item()

    print(f"Input spikes: {total_input_spikes}")
    print(f"Output spikes: {total_output_spikes}")
    print(f"Loss: {loss:.6f}")
    print(f"Gradient norm: {grad_norm:.6f}")

    # Identify likely issues
    issues = []
    if total_input_spikes == 0:
        issues.append("No input spikes generated - encoder threshold too high")
    if total_output_spikes == 0:
        issues.append("No output spikes generated - network not activating")
    if loss is not None and abs(loss - 0.693) < 0.001:
        issues.append("Loss stuck at log(2) - model not learning")
    if grad_norm is not None and grad_norm < 1e-6:
        issues.append("Gradients too small - vanishing gradient problem")

    if issues:
        print("\nIDENTIFIED ISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("\nNo obvious issues detected. Problem may be in training loop or hyperparameters.")
