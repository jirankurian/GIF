#!/usr/bin/env python3
"""
Performance Analysis Script
===========================

This script analyzes the performance bottlenecks in the GIF-DU training pipeline.
"""

import sys
import os
sys.path.append('/Users/jp/RnD/GIF')

import torch
import time
import numpy as np
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

def time_component(func, *args, **kwargs):
    """Time a function call and return result and duration."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def analyze_training_performance():
    """Analyze performance of each training component."""
    print("=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    config = EXP_CONFIG
    
    # Initialize components
    print("Initializing components...")
    
    encoder = LightCurveEncoder(
        threshold=config["encoder"]["threshold"],
        normalize_input=config["encoder"]["normalize_input"]
    )
    
    decoder = ExoplanetDecoder(
        input_size=config["architecture"]["output_size"],
        output_size=1
    )
    
    memory_system = EpisodicMemory(capacity=10000)
    
    du_core = DU_Core_V1(
        input_size=config["architecture"]["input_size"],
        hidden_sizes=config["architecture"]["hidden_sizes"],
        output_size=config["architecture"]["output_size"],
        beta=config["architecture"]["neuron_beta"],
        threshold=config["architecture"]["neuron_threshold"],
        memory_system=memory_system
    )
    
    gif_model = GIF(du_core)
    gif_model.attach_encoder(encoder)
    gif_model.attach_decoder(decoder)
    
    # Generate test data
    print("Generating test data...")
    generator = RealisticExoplanetGenerator(seed=42)
    
    # Time data generation
    light_curve, gen_time = time_component(generator.generate)
    print(f"✓ Data generation: {gen_time:.4f}s")
    
    # Time encoding
    spike_train, enc_time = time_component(encoder.encode, light_curve)
    print(f"✓ Encoding: {enc_time:.4f}s")
    print(f"  - Input spikes: {spike_train.sum().item()}")
    print(f"  - Spike train shape: {spike_train.shape}")
    
    # Time DU Core processing
    if spike_train.dim() == 2:
        spike_train_batch = spike_train.unsqueeze(1)
    else:
        spike_train_batch = spike_train
        
    output_spikes, du_time = time_component(du_core.process, spike_train_batch)
    print(f"✓ DU Core processing: {du_time:.4f}s")
    print(f"  - Output spikes: {output_spikes.sum().item()}")
    print(f"  - Processing rate: {spike_train.shape[0] / du_time:.0f} timesteps/sec")
    
    # Time decoding
    spike_counts = output_spikes.sum(dim=0)
    if spike_counts.dim() == 1:
        spike_counts = spike_counts.unsqueeze(0)
        
    logits, dec_time = time_component(decoder.forward_classification, spike_counts)
    print(f"✓ Decoding: {dec_time:.4f}s")
    
    # Time full forward pass
    def full_forward():
        spike_train = encoder.encode(light_curve)
        if spike_train.dim() == 2:
            spike_train_batch = spike_train.unsqueeze(1)
        else:
            spike_train_batch = spike_train
        output_spikes = du_core.process(spike_train_batch)
        spike_counts = output_spikes.sum(dim=0)
        if spike_counts.dim() == 1:
            spike_counts = spike_counts.unsqueeze(0)
        return decoder.forward_classification(spike_counts)
    
    _, forward_time = time_component(full_forward)
    print(f"✓ Full forward pass: {forward_time:.4f}s")
    
    # Time trainer step
    import torch.nn as nn
    import torch.optim as optim
    
    optimizer = optim.Adam(gif_model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    trainer = Continual_Trainer(
        gif_model=gif_model,
        optimizer=optimizer,
        criterion=criterion,
        memory_batch_size=config["training"]["memory_batch_size"],
        gradient_clip_norm=config["training"].get("gradient_clip_norm", None)
    )
    
    target_tensor = torch.tensor([1], dtype=torch.long)
    
    # Time first training step (no memory)
    _, train_time_1 = time_component(trainer.train_step, light_curve, target_tensor, "exoplanet_detection")
    print(f"✓ Training step 1 (no memory): {train_time_1:.4f}s")
    
    # Add some experiences to memory
    for i in range(5):
        trainer.train_step(light_curve, target_tensor, "exoplanet_detection")
    
    # Time training step with memory
    _, train_time_mem = time_component(trainer.train_step, light_curve, target_tensor, "exoplanet_detection")
    print(f"✓ Training step with memory: {train_time_mem:.4f}s")
    
    # Analyze bottlenecks
    print(f"\nBottleneck Analysis:")
    total_time = gen_time + forward_time + train_time_mem
    
    components = [
        ("Data Generation", gen_time),
        ("Forward Pass", forward_time),
        ("Training Overhead", train_time_mem - forward_time),
    ]
    
    for name, time_val in components:
        percentage = (time_val / total_time) * 100
        print(f"  {name}: {time_val:.4f}s ({percentage:.1f}%)")
    
    # Estimate epoch time
    batch_size = config["training"]["batch_size"]
    num_samples = 1000  # From the experiment
    batches_per_epoch = num_samples // batch_size
    
    estimated_epoch_time = batches_per_epoch * train_time_mem
    print(f"\nEstimated Performance:")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Time per batch: {train_time_mem:.4f}s")
    print(f"  Estimated epoch time: {estimated_epoch_time:.1f}s ({estimated_epoch_time/60:.1f} minutes)")
    
    # Identify specific bottlenecks
    print(f"\nSpecific Bottlenecks:")
    
    if du_time > 0.1:
        print(f"  ⚠️  DU Core processing is slow ({du_time:.4f}s)")
        print(f"      - Processing {spike_train.shape[0]} timesteps")
        print(f"      - Consider reducing sequence length or optimizing SNN")
    
    if train_time_mem - forward_time > 0.5:
        print(f"  ⚠️  Training overhead is high ({train_time_mem - forward_time:.4f}s)")
        print(f"      - GEM algorithm may be expensive")
        print(f"      - Memory operations may be slow")
    
    if gen_time > 0.1:
        print(f"  ⚠️  Data generation is slow ({gen_time:.4f}s)")
        print(f"      - Consider pre-generating data or caching")
    
    # Test sequence length impact
    print(f"\nSequence Length Impact:")
    sequence_lengths = [1000, 2000, 4800]
    
    for seq_len in sequence_lengths:
        # Create shorter spike train
        short_spike_train = spike_train[:seq_len]
        if short_spike_train.dim() == 2:
            short_spike_train = short_spike_train.unsqueeze(1)
        
        _, short_time = time_component(du_core.process, short_spike_train)
        rate = seq_len / short_time if short_time > 0 else 0
        print(f"  Length {seq_len}: {short_time:.4f}s ({rate:.0f} timesteps/sec)")

def main():
    """Run performance analysis."""
    print("GIF-DU Performance Analysis")
    print("=" * 60)
    
    analyze_training_performance()

if __name__ == "__main__":
    main()
