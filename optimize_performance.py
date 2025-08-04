#!/usr/bin/env python3
"""
Performance Optimization Script
===============================

This script implements several optimizations to speed up GIF-DU training:
1. Reduce sequence length through downsampling
2. Optimize SNN processing
3. Batch processing improvements
"""

import sys
import os
sys.path.append('/Users/jp/RnD/GIF')

import torch
import time
import numpy as np
import polars as pl
from typing import Dict, Any

# Import GIF framework components
from applications.poc_exoplanet.config_exo import EXP_CONFIG
from applications.poc_exoplanet.encoders.light_curve_encoder import LightCurveEncoder
from applications.poc_exoplanet.decoders.exoplanet_decoder import ExoplanetDecoder
from gif_framework.core.du_core import DU_Core_V1
from data_generators.exoplanet_generator import RealisticExoplanetGenerator
from gif_framework.core.memory_systems import EpisodicMemory

def downsample_light_curve(light_curve: pl.DataFrame, factor: int = 4) -> pl.DataFrame:
    """
    Downsample light curve by taking every Nth point.
    
    Args:
        light_curve: Original light curve DataFrame
        factor: Downsampling factor (e.g., 4 means take every 4th point)
    
    Returns:
        Downsampled light curve DataFrame
    """
    # Take every Nth row
    downsampled = light_curve[::factor]
    
    # Recalculate time to maintain proper spacing
    original_time_span = light_curve['time'][-1] - light_curve['time'][0]
    new_length = len(downsampled)
    
    # Create new time array with same span but fewer points
    new_time = np.linspace(
        light_curve['time'][0], 
        light_curve['time'][-1], 
        new_length
    )
    
    # Update the time column
    downsampled = downsampled.with_columns(
        pl.Series("time", new_time)
    )
    
    return downsampled

def test_downsampling_impact():
    """Test the impact of downsampling on performance and accuracy."""
    print("=" * 60)
    print("TESTING DOWNSAMPLING IMPACT")
    print("=" * 60)
    
    config = EXP_CONFIG
    
    # Generate test data
    generator = RealisticExoplanetGenerator(seed=42)
    original_light_curve = generator.generate()
    has_planet = original_light_curve['planet_radius_ratio'][0] > 0
    
    print(f"Original light curve: {len(original_light_curve)} points")
    print(f"Planet present: {has_planet}")
    
    # Initialize components
    encoder = LightCurveEncoder(
        threshold=config["encoder"]["threshold"],
        normalize_input=config["encoder"]["normalize_input"]
    )
    
    memory_system = EpisodicMemory(capacity=1000)
    du_core = DU_Core_V1(
        input_size=config["architecture"]["input_size"],
        hidden_sizes=config["architecture"]["hidden_sizes"],
        output_size=config["architecture"]["output_size"],
        beta=config["architecture"]["neuron_beta"],
        threshold=config["architecture"]["neuron_threshold"],
        memory_system=memory_system
    )
    
    decoder = ExoplanetDecoder(
        input_size=config["architecture"]["output_size"],
        output_size=1
    )
    
    # Test different downsampling factors
    factors = [1, 2, 4, 8, 16]
    results = []
    
    for factor in factors:
        print(f"\nTesting downsampling factor {factor}:")
        
        # Downsample if needed
        if factor == 1:
            test_light_curve = original_light_curve
        else:
            test_light_curve = downsample_light_curve(original_light_curve, factor)
        
        print(f"  Length: {len(test_light_curve)} points")
        
        # Time encoding
        start_time = time.time()
        spike_train = encoder.encode(test_light_curve)
        encoding_time = time.time() - start_time
        
        print(f"  Encoding time: {encoding_time:.4f}s")
        print(f"  Input spikes: {spike_train.sum().item()}")
        
        # Time DU Core processing
        if spike_train.dim() == 2:
            spike_train_batch = spike_train.unsqueeze(1)
        else:
            spike_train_batch = spike_train
        
        start_time = time.time()
        output_spikes = du_core.process(spike_train_batch)
        processing_time = time.time() - start_time
        
        print(f"  DU Core time: {processing_time:.4f}s")
        print(f"  Output spikes: {output_spikes.sum().item()}")
        
        # Test prediction
        spike_counts = output_spikes.sum(dim=0)
        if spike_counts.dim() == 1:
            spike_counts = spike_counts.unsqueeze(0)
        
        logits = decoder.forward_classification(spike_counts)
        prediction = torch.argmax(logits, dim=1).item()
        
        # Calculate prediction confidence
        probs = torch.softmax(logits, dim=1)
        confidence = probs.max().item()
        
        print(f"  Prediction: {prediction} (confidence: {confidence:.4f})")
        print(f"  Correct: {'✓' if prediction == (1 if has_planet else 0) else '✗'}")
        
        total_time = encoding_time + processing_time
        speedup = results[0]['total_time'] / total_time if results else 1.0
        
        print(f"  Total time: {total_time:.4f}s (speedup: {speedup:.1f}x)")
        
        results.append({
            'factor': factor,
            'length': len(test_light_curve),
            'encoding_time': encoding_time,
            'processing_time': processing_time,
            'total_time': total_time,
            'prediction': prediction,
            'confidence': confidence,
            'correct': prediction == (1 if has_planet else 0),
            'speedup': speedup
        })
    
    # Summary
    print(f"\nSummary:")
    print(f"{'Factor':<8} {'Length':<8} {'Time':<8} {'Speedup':<8} {'Correct':<8} {'Confidence':<10}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['factor']:<8} {r['length']:<8} {r['total_time']:<8.3f} "
              f"{r['speedup']:<8.1f} {'✓' if r['correct'] else '✗':<8} {r['confidence']:<10.3f}")
    
    # Find optimal factor
    correct_results = [r for r in results if r['correct']]
    if correct_results:
        optimal = min(correct_results, key=lambda x: x['total_time'])
        print(f"\nOptimal downsampling factor: {optimal['factor']}")
        print(f"  Speedup: {optimal['speedup']:.1f}x")
        print(f"  Accuracy maintained: ✓")
        return optimal['factor']
    else:
        print(f"\nWarning: No downsampling factor maintains accuracy!")
        return 1

def create_optimized_config(downsample_factor: int) -> Dict[str, Any]:
    """Create optimized configuration with performance improvements."""
    config = EXP_CONFIG.copy()
    
    # Add downsampling configuration
    config["data_processing"] = {
        "downsample_factor": downsample_factor,
        "enable_downsampling": downsample_factor > 1
    }
    
    # Optimize batch size for faster training
    config["training"]["batch_size"] = 32  # Increase from 16
    
    # Reduce memory batch size to speed up GEM
    config["training"]["memory_batch_size"] = 100  # Reduce from 1000
    
    print(f"Optimized Configuration:")
    print(f"  Downsample factor: {downsample_factor}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Memory batch size: {config['training']['memory_batch_size']}")
    
    return config

def estimate_training_time(config: Dict[str, Any]):
    """Estimate total training time with optimizations."""
    
    # Simulate one training step with optimizations
    generator = RealisticExoplanetGenerator(seed=42)
    light_curve = generator.generate()
    
    # Apply downsampling if enabled
    if config["data_processing"]["enable_downsampling"]:
        light_curve = downsample_light_curve(
            light_curve, 
            config["data_processing"]["downsample_factor"]
        )
    
    # Time one forward pass
    encoder = LightCurveEncoder(
        threshold=config["encoder"]["threshold"],
        normalize_input=config["encoder"]["normalize_input"]
    )
    
    memory_system = EpisodicMemory(capacity=1000)
    du_core = DU_Core_V1(
        input_size=config["architecture"]["input_size"],
        hidden_sizes=config["architecture"]["hidden_sizes"],
        output_size=config["architecture"]["output_size"],
        beta=config["architecture"]["neuron_beta"],
        threshold=config["architecture"]["neuron_threshold"],
        memory_system=memory_system
    )
    
    start_time = time.time()
    spike_train = encoder.encode(light_curve)
    if spike_train.dim() == 2:
        spike_train = spike_train.unsqueeze(1)
    output_spikes = du_core.process(spike_train)
    forward_time = time.time() - start_time
    
    # Estimate training parameters
    num_samples = 1000
    batch_size = config["training"]["batch_size"]
    num_epochs = 20
    
    batches_per_epoch = num_samples // batch_size
    time_per_batch = forward_time * batch_size  # Approximate
    time_per_epoch = batches_per_epoch * time_per_batch
    total_time = time_per_epoch * num_epochs
    
    print(f"\nTraining Time Estimation:")
    print(f"  Forward pass time: {forward_time:.4f}s")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Time per epoch: {time_per_epoch:.1f}s ({time_per_epoch/60:.1f} min)")
    print(f"  Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    return total_time

def main():
    """Run performance optimization analysis."""
    print("GIF-DU Performance Optimization")
    print("=" * 60)
    
    # Test downsampling impact
    optimal_factor = test_downsampling_impact()
    
    # Create optimized configuration
    optimized_config = create_optimized_config(optimal_factor)
    
    # Estimate training time
    estimated_time = estimate_training_time(optimized_config)
    
    print(f"\n" + "=" * 60)
    print(f"OPTIMIZATION SUMMARY")
    print(f"=" * 60)
    print(f"Recommended optimizations:")
    print(f"  1. Downsample by factor {optimal_factor}")
    print(f"  2. Increase batch size to 32")
    print(f"  3. Reduce memory batch size to 100")
    print(f"  4. Estimated training time: {estimated_time/60:.1f} minutes")

if __name__ == "__main__":
    main()
