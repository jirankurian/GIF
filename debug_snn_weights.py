#!/usr/bin/env python3
"""
Debug SNN Weights
=================

This script investigates the SNN weight initialization and membrane potential dynamics.
"""

import sys
import os
sys.path.append('/Users/jp/RnD/GIF')

import torch
import torch.nn as nn
import numpy as np

# Import GIF framework components
from applications.poc_exoplanet.config_exo import EXP_CONFIG
from applications.poc_exoplanet.encoders.light_curve_encoder import LightCurveEncoder
from gif_framework.core.du_core import DU_Core_V1
from data_generators.exoplanet_generator import RealisticExoplanetGenerator
from gif_framework.core.memory_systems import EpisodicMemory

def analyze_snn_weights():
    """Analyze the SNN weight initialization and dynamics."""
    print("=" * 60)
    print("ANALYZING SNN WEIGHTS AND DYNAMICS")
    print("=" * 60)
    
    config = EXP_CONFIG
    
    # Create DU Core
    memory_system = EpisodicMemory(capacity=1000)
    du_core = DU_Core_V1(
        input_size=config["architecture"]["input_size"],
        hidden_sizes=config["architecture"]["hidden_sizes"],
        output_size=config["architecture"]["output_size"],
        beta=config["architecture"]["neuron_beta"],
        threshold=config["architecture"]["neuron_threshold"],
        memory_system=memory_system
    )
    
    print(f"DU Core Architecture:")
    print(f"  Input size: {config['architecture']['input_size']}")
    print(f"  Hidden sizes: {config['architecture']['hidden_sizes']}")
    print(f"  Output size: {config['architecture']['output_size']}")
    print(f"  Beta: {config['architecture']['neuron_beta']}")
    print(f"  Threshold: {config['architecture']['neuron_threshold']}")
    
    # Analyze weights
    print(f"\nWeight Analysis:")
    for name, param in du_core.named_parameters():
        weight_mean = param.data.mean().item()
        weight_std = param.data.std().item()
        weight_min = param.data.min().item()
        weight_max = param.data.max().item()
        print(f"  {name}:")
        print(f"    Shape: {param.shape}")
        print(f"    Mean: {weight_mean:.6f}")
        print(f"    Std: {weight_std:.6f}")
        print(f"    Range: [{weight_min:.6f}, {weight_max:.6f}]")
    
    # Generate test input
    generator = RealisticExoplanetGenerator(seed=42)
    light_curve = generator.generate()
    
    encoder = LightCurveEncoder(
        threshold=config["encoder"]["threshold"],
        normalize_input=config["encoder"]["normalize_input"]
    )
    
    spike_train = encoder.encode(light_curve)
    print(f"\nInput Analysis:")
    print(f"  Spike train shape: {spike_train.shape}")
    print(f"  Total spikes: {spike_train.sum().item()}")
    print(f"  Spike rate: {spike_train.sum().item() / spike_train.numel():.6f}")
    print(f"  Max spike value: {spike_train.max().item()}")
    print(f"  Min spike value: {spike_train.min().item()}")
    
    # Add batch dimension
    if spike_train.dim() == 2:
        spike_train_batch = spike_train.unsqueeze(1)
    else:
        spike_train_batch = spike_train
    
    print(f"  Batch shape: {spike_train_batch.shape}")
    
    # Test with different thresholds
    print(f"\nTesting Different Thresholds:")
    thresholds = [1.0, 0.5, 0.1, 0.01, 0.001]
    
    for threshold in thresholds:
        # Create new DU Core with this threshold
        du_core_test = DU_Core_V1(
            input_size=config["architecture"]["input_size"],
            hidden_sizes=config["architecture"]["hidden_sizes"],
            output_size=config["architecture"]["output_size"],
            beta=config["architecture"]["neuron_beta"],
            threshold=threshold,
            memory_system=EpisodicMemory(capacity=1000)
        )
        
        # Copy weights from original
        du_core_test.load_state_dict(du_core.state_dict())
        
        # Test forward pass
        try:
            output_spikes = du_core_test.process(spike_train_batch)
            total_output_spikes = output_spikes.sum().item()
            output_rate = total_output_spikes / output_spikes.numel()
            print(f"  Threshold {threshold:6.3f}: {total_output_spikes:6.0f} spikes (rate: {output_rate:.6f})")
        except Exception as e:
            print(f"  Threshold {threshold:6.3f}: ERROR - {e}")
    
    # Test with scaled weights
    print(f"\nTesting Weight Scaling:")
    scales = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for scale in scales:
        # Create new DU Core
        du_core_scaled = DU_Core_V1(
            input_size=config["architecture"]["input_size"],
            hidden_sizes=config["architecture"]["hidden_sizes"],
            output_size=config["architecture"]["output_size"],
            beta=config["architecture"]["neuron_beta"],
            threshold=config["architecture"]["neuron_threshold"],
            memory_system=EpisodicMemory(capacity=1000)
        )
        
        # Scale the weights
        with torch.no_grad():
            for param in du_core_scaled.parameters():
                param.data *= scale
        
        # Test forward pass
        try:
            output_spikes = du_core_scaled.process(spike_train_batch)
            total_output_spikes = output_spikes.sum().item()
            output_rate = total_output_spikes / output_spikes.numel()
            print(f"  Scale {scale:4.1f}x: {total_output_spikes:6.0f} spikes (rate: {output_rate:.6f})")
        except Exception as e:
            print(f"  Scale {scale:4.1f}x: ERROR - {e}")
    
    # Test manual membrane potential calculation
    print(f"\nManual Membrane Potential Analysis:")
    
    # Get first layer weights and bias
    first_layer = du_core.linear_layers[0]
    weights = first_layer.weight.data  # Shape: [hidden_size, input_size]
    bias = first_layer.bias.data       # Shape: [hidden_size]
    
    print(f"  First layer weights shape: {weights.shape}")
    print(f"  First layer bias shape: {bias.shape}")
    
    # Take first few time steps of input
    input_sample = spike_train_batch[:10, 0, :]  # [10, 2]
    print(f"  Input sample shape: {input_sample.shape}")
    print(f"  Input sample spikes: {input_sample.sum().item()}")
    
    # Manual calculation
    membrane_potential = torch.zeros(weights.shape[0])  # [hidden_size]
    beta = config["architecture"]["neuron_beta"]
    threshold = config["architecture"]["neuron_threshold"]
    
    spike_count = 0
    for t in range(input_sample.shape[0]):
        # Linear transformation
        current_input = torch.matmul(weights, input_sample[t])  # [hidden_size]
        
        # Update membrane potential
        membrane_potential = beta * membrane_potential + current_input + bias
        
        # Check for spikes
        spikes = (membrane_potential >= threshold).float()
        spike_count += spikes.sum().item()
        
        # Reset spiked neurons
        membrane_potential = membrane_potential * (1 - spikes)
        
        if t < 5:  # Print first few steps
            print(f"    Step {t}: input_sum={input_sample[t].sum().item():.3f}, "
                  f"mem_max={membrane_potential.max().item():.6f}, "
                  f"spikes={spikes.sum().item()}")
    
    print(f"  Manual calculation: {spike_count} spikes in 10 steps")
    print(f"  Max membrane potential reached: {membrane_potential.max().item():.6f}")
    print(f"  Threshold: {threshold}")
    
    if membrane_potential.max().item() < threshold:
        print(f"  ⚠️  Membrane potential never reaches threshold!")
        print(f"  ⚠️  Need to increase weights or decrease threshold")

def main():
    """Run the analysis."""
    print("SNN Weight Analysis Script")
    print("=" * 60)
    
    analyze_snn_weights()

if __name__ == "__main__":
    main()
