#!/usr/bin/env python3
"""
GIF-DU Training Fix Script
==========================

This script implements fixes for the identified training issues:
1. Adjust neuron threshold to enable spike generation
2. Fix network initialization to prevent exploding gradients
3. Optimize hyperparameters for stable learning
4. Test the fixes with a simple training loop
"""

import sys
import os
sys.path.append('/Users/jp/RnD/GIF')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any

# Import GIF framework components
from applications.poc_exoplanet.config_exo import EXP_CONFIG
from applications.poc_exoplanet.encoders.light_curve_encoder import LightCurveEncoder
from applications.poc_exoplanet.decoders.exoplanet_decoder import ExoplanetDecoder
from gif_framework.core.du_core import DU_Core_V1
from gif_framework.orchestrator import GIF
from data_generators.exoplanet_generator import RealisticExoplanetGenerator
from gif_framework.core.memory_systems import EpisodicMemory

def create_fixed_config():
    """Create a fixed configuration with optimized parameters."""
    fixed_config = EXP_CONFIG.copy()
    
    # Fix 1: Lower neuron threshold to enable spike generation
    fixed_config["architecture"]["neuron_threshold"] = 0.5  # Was 1.0
    
    # Fix 2: Adjust neuron beta for better dynamics
    fixed_config["architecture"]["neuron_beta"] = 0.9  # Was 0.95
    
    # Fix 3: Reduce learning rate to prevent exploding gradients
    fixed_config["training"]["learning_rate"] = 1e-4  # Was 1e-3
    
    # Fix 4: Add gradient clipping
    fixed_config["training"]["gradient_clip_norm"] = 1.0
    
    # Fix 5: Smaller batch size for more stable training
    fixed_config["training"]["batch_size"] = 16  # Was 32
    
    print("Applied fixes:")
    print(f"  - Neuron threshold: 1.0 → 0.5")
    print(f"  - Neuron beta: 0.95 → 0.9") 
    print(f"  - Learning rate: 1e-3 → 1e-4")
    print(f"  - Added gradient clipping: 1.0")
    print(f"  - Batch size: 32 → 16")
    
    return fixed_config

def test_fixed_components():
    """Test the fixed components to ensure they work correctly."""
    print("=" * 60)
    print("TESTING FIXED COMPONENTS")
    print("=" * 60)
    
    # Use fixed configuration
    config = create_fixed_config()
    
    # Generate test data
    generator = RealisticExoplanetGenerator(seed=42)
    light_curve = generator.generate()
    has_planet = light_curve['planet_radius_ratio'][0] > 0
    
    # Test encoder (should work fine)
    encoder = LightCurveEncoder(
        threshold=config["encoder"]["threshold"],
        normalize_input=config["encoder"]["normalize_input"]
    )
    spike_train = encoder.encode(light_curve)
    print(f"✓ Encoder: {spike_train.sum().item()} spikes generated")
    
    # Test fixed DU Core
    memory_system = EpisodicMemory(capacity=1000)
    du_core = DU_Core_V1(
        input_size=config["architecture"]["input_size"],
        hidden_sizes=config["architecture"]["hidden_sizes"],
        output_size=config["architecture"]["output_size"],
        beta=config["architecture"]["neuron_beta"],
        threshold=config["architecture"]["neuron_threshold"],
        memory_system=memory_system
    )
    
    # Add batch dimension and process
    if spike_train.dim() == 2:
        spike_train_batch = spike_train.unsqueeze(1)
    else:
        spike_train_batch = spike_train
        
    output_spikes = du_core.process(spike_train_batch)
    total_output_spikes = output_spikes.sum().item()
    print(f"✓ DU Core: {total_output_spikes} output spikes generated")
    
    if total_output_spikes == 0:
        print("⚠️  Still no output spikes! Trying even lower threshold...")
        
        # Try with even lower threshold
        du_core_lower = DU_Core_V1(
            input_size=config["architecture"]["input_size"],
            hidden_sizes=config["architecture"]["hidden_sizes"],
            output_size=config["architecture"]["output_size"],
            beta=config["architecture"]["neuron_beta"],
            threshold=0.1,  # Much lower threshold
            memory_system=memory_system
        )
        
        output_spikes_lower = du_core_lower.process(spike_train_batch)
        total_output_spikes_lower = output_spikes_lower.sum().item()
        print(f"   Lower threshold (0.1): {total_output_spikes_lower} spikes")
        
        if total_output_spikes_lower > 0:
            print("   → Using threshold 0.1")
            du_core = du_core_lower
            output_spikes = output_spikes_lower
            config["architecture"]["neuron_threshold"] = 0.1
    
    # Test decoder
    decoder = ExoplanetDecoder(
        input_size=output_spikes.shape[-1],
        output_size=1
    )
    
    spike_counts = output_spikes.sum(dim=0)
    if spike_counts.dim() == 1:
        spike_counts = spike_counts.unsqueeze(0)
        
    logits = decoder.forward_classification(spike_counts)
    print(f"✓ Decoder: logits = {logits}")
    
    return config, du_core, decoder, spike_train, output_spikes, has_planet

def test_training_step(config, du_core, decoder, spike_train, target_label):
    """Test a single training step with the fixed configuration."""
    print("\n" + "=" * 60)
    print("TESTING TRAINING STEP")
    print("=" * 60)
    
    # Create optimizer with fixed learning rate
    optimizer = optim.Adam(
        list(du_core.parameters()) + list(decoder.parameters()),
        lr=config["training"]["learning_rate"]
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Forward pass
    if spike_train.dim() == 2:
        spike_train_batch = spike_train.unsqueeze(1)
    else:
        spike_train_batch = spike_train
        
    output_spikes = du_core.process(spike_train_batch)
    spike_counts = output_spikes.sum(dim=0)
    if spike_counts.dim() == 1:
        spike_counts = spike_counts.unsqueeze(0)
        
    logits = decoder.forward_classification(spike_counts)
    target = torch.tensor([target_label], dtype=torch.long)
    loss = criterion(logits, target)
    
    print(f"✓ Forward pass: loss = {loss.item():.6f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Apply gradient clipping
    if "gradient_clip_norm" in config["training"]:
        total_norm = torch.nn.utils.clip_grad_norm_(
            list(du_core.parameters()) + list(decoder.parameters()),
            config["training"]["gradient_clip_norm"]
        )
        print(f"✓ Gradient clipping: norm before = {total_norm:.6f}")
    
    # Check gradients
    total_grad_norm = 0.0
    for param in list(du_core.parameters()) + list(decoder.parameters()):
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item()
    
    print(f"✓ Total gradient norm: {total_grad_norm:.6f}")
    
    # Update parameters
    optimizer.step()
    
    # Test prediction after update
    with torch.no_grad():
        output_spikes_new = du_core.process(spike_train_batch)
        spike_counts_new = output_spikes_new.sum(dim=0)
        if spike_counts_new.dim() == 1:
            spike_counts_new = spike_counts_new.unsqueeze(0)
        logits_new = decoder.forward_classification(spike_counts_new)
        loss_new = criterion(logits_new, target)
        
    print(f"✓ After update: loss = {loss_new.item():.6f}")
    print(f"✓ Loss change: {loss_new.item() - loss.item():.6f}")
    
    return loss.item(), loss_new.item()

def main():
    """Run the complete fix validation."""
    print("GIF-DU Training Fix Script")
    print("=" * 60)
    
    # Test fixed components
    config, du_core, decoder, spike_train, output_spikes, has_planet = test_fixed_components()
    
    # Test training step
    target_label = 1 if has_planet else 0
    loss_before, loss_after = test_training_step(config, du_core, decoder, spike_train, target_label)
    
    # Summary
    print("\n" + "=" * 60)
    print("FIX VALIDATION SUMMARY")
    print("=" * 60)
    
    total_output_spikes = output_spikes.sum().item()
    
    print(f"Output spikes generated: {total_output_spikes}")
    print(f"Loss before training: {loss_before:.6f}")
    print(f"Loss after training: {loss_after:.6f}")
    print(f"Loss improvement: {loss_before - loss_after:.6f}")
    
    if total_output_spikes > 0:
        print("✅ SUCCESS: Network is now generating output spikes!")
    else:
        print("❌ FAILURE: Network still not generating spikes")
        
    if abs(loss_before - 0.693) > 0.01:
        print("✅ SUCCESS: Loss is no longer stuck at log(2)")
    else:
        print("❌ FAILURE: Loss still stuck at random chance")
        
    if abs(loss_after - loss_before) > 1e-6:
        print("✅ SUCCESS: Training is causing loss changes")
    else:
        print("❌ FAILURE: Training not affecting loss")

if __name__ == "__main__":
    main()
