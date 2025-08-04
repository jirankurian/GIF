#!/usr/bin/env python3
"""
Simple Training Test
====================

This script tests training without the GEM algorithm to isolate the issue.
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
from data_generators.exoplanet_generator import RealisticExoplanetGenerator
from gif_framework.core.memory_systems import EpisodicMemory

def simple_training_test():
    """Test simple training without GEM algorithm."""
    print("=" * 60)
    print("SIMPLE TRAINING TEST (NO GEM)")
    print("=" * 60)
    
    config = EXP_CONFIG
    
    # Generate test data
    generator = RealisticExoplanetGenerator(seed=42)
    
    # Create multiple samples for training
    training_samples = []
    for i in range(10):
        light_curve = generator.generate()
        has_planet = light_curve['planet_radius_ratio'][0] > 0
        target = 1 if has_planet else 0
        training_samples.append((light_curve, target))
        print(f"Sample {i+1}: Planet = {has_planet}")
    
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
    
    # Create optimizer and loss function
    optimizer = optim.Adam(
        list(du_core.parameters()) + list(decoder.parameters()),
        lr=config["training"]["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining with {len(training_samples)} samples...")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Neuron threshold: {config['architecture']['neuron_threshold']}")
    
    # Simple training loop
    losses = []
    for epoch in range(5):
        epoch_losses = []
        correct = 0
        total = 0
        
        for i, (light_curve, target) in enumerate(training_samples):
            # Forward pass
            spike_train = encoder.encode(light_curve)
            
            if spike_train.dim() == 2:
                spike_train = spike_train.unsqueeze(1)
                
            output_spikes = du_core.process(spike_train)
            spike_counts = output_spikes.sum(dim=0)
            if spike_counts.dim() == 1:
                spike_counts = spike_counts.unsqueeze(0)
                
            logits = decoder.forward_classification(spike_counts)
            target_tensor = torch.tensor([target], dtype=torch.long)
            loss = criterion(logits, target_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(du_core.parameters()) + list(decoder.parameters()),
                1.0
            )
            
            optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            prediction = torch.argmax(logits, dim=1).item()
            if prediction == target:
                correct += 1
            total += 1
            
            if i == 0:  # Print first sample details
                print(f"  Sample 1 - Target: {target}, Pred: {prediction}, Loss: {loss.item():.4f}")
        
        avg_loss = np.mean(epoch_losses)
        accuracy = correct / total
        losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
    
    # Check if learning occurred
    print(f"\nLearning Analysis:")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss change: {losses[0] - losses[-1]:.4f}")
    
    if abs(losses[0] - losses[-1]) > 0.01:
        print("✅ SUCCESS: Model is learning!")
    else:
        print("❌ FAILURE: Model not learning")
        
    if losses[-1] < 0.69:
        print("✅ SUCCESS: Loss below random chance")
    else:
        print("❌ FAILURE: Loss still at random chance")

def test_gem_interference():
    """Test if GEM algorithm is causing the issue."""
    print("\n" + "=" * 60)
    print("TESTING GEM INTERFERENCE")
    print("=" * 60)
    
    from gif_framework.training.trainer import Continual_Trainer
    from gif_framework.orchestrator import GIF
    
    config = EXP_CONFIG
    
    # Generate test data
    generator = RealisticExoplanetGenerator(seed=42)
    light_curve = generator.generate()
    has_planet = light_curve['planet_radius_ratio'][0] > 0
    target = 1 if has_planet else 0
    
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
    
    # Create GIF model
    gif_model = GIF(du_core)
    gif_model.attach_encoder(encoder)
    gif_model.attach_decoder(decoder)
    
    # Create optimizer and trainer
    optimizer = optim.Adam(gif_model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    trainer = Continual_Trainer(
        gif_model=gif_model,
        optimizer=optimizer,
        criterion=criterion,
        memory_batch_size=config["training"]["memory_batch_size"],
        gradient_clip_norm=config["training"].get("gradient_clip_norm", None)
    )
    
    print(f"Testing GEM trainer...")
    print(f"Memory batch size: {config['training']['memory_batch_size']}")
    
    # Test multiple training steps
    losses = []
    for step in range(5):
        target_tensor = torch.tensor([target], dtype=torch.long)
        loss = trainer.train_step(light_curve, target_tensor, "exoplanet_detection")
        losses.append(loss)
        print(f"Step {step+1}: Loss = {loss:.4f}")
    
    print(f"\nGEM Training Analysis:")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss change: {losses[0] - losses[-1]:.4f}")
    
    if abs(losses[0] - losses[-1]) > 0.01:
        print("✅ SUCCESS: GEM trainer is working!")
    else:
        print("❌ FAILURE: GEM trainer not learning")

def main():
    """Run all tests."""
    print("Simple Training Test Script")
    print("=" * 60)
    
    # Test 1: Simple training without GEM
    simple_training_test()
    
    # Test 2: GEM training
    test_gem_interference()

if __name__ == "__main__":
    main()
