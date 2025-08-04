#!/usr/bin/env python3
"""
Debug GEM Trainer
=================

This script debugs the GEM trainer step by step to find the exact issue.
"""

import sys
import os
sys.path.append('/Users/jp/RnD/GIF')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import GIF framework components
from applications.poc_exoplanet.config_exo import EXP_CONFIG
from applications.poc_exoplanet.encoders.light_curve_encoder import LightCurveEncoder
from applications.poc_exoplanet.decoders.exoplanet_decoder import ExoplanetDecoder
from gif_framework.core.du_core import DU_Core_V1
from gif_framework.orchestrator import GIF
from gif_framework.training.trainer import Continual_Trainer
from data_generators.exoplanet_generator import RealisticExoplanetGenerator
from gif_framework.core.memory_systems import EpisodicMemory

def debug_trainer_step():
    """Debug a single training step in detail."""
    print("=" * 60)
    print("DEBUGGING GEM TRAINER STEP")
    print("=" * 60)
    
    config = EXP_CONFIG
    
    # Generate test data
    generator = RealisticExoplanetGenerator(seed=42)
    light_curve = generator.generate()
    has_planet = light_curve['planet_radius_ratio'][0] > 0
    target = 1 if has_planet else 0
    
    print(f"Test data: Planet = {has_planet}, Target = {target}")
    
    # Initialize components exactly as in main_exo.py
    encoder = LightCurveEncoder(
        threshold=config["encoder"]["threshold"],
        normalize_input=config["encoder"]["normalize_input"]
    )
    
    decoder = ExoplanetDecoder(
        input_size=config["architecture"]["output_size"],
        output_size=1  # For regression mode
    )
    
    memory_system = EpisodicMemory(capacity=10000)  # Use default capacity
    
    du_core = DU_Core_V1(
        input_size=config["architecture"]["input_size"],
        hidden_sizes=config["architecture"]["hidden_sizes"],
        output_size=config["architecture"]["output_size"],
        beta=config["architecture"]["neuron_beta"],
        threshold=config["architecture"]["neuron_threshold"],
        memory_system=memory_system
    )
    
    # Create GIF model
    gif_model = GIF(du_core)
    gif_model.attach_encoder(encoder)
    gif_model.attach_decoder(decoder)
    
    print(f"✓ Components initialized")
    print(f"  - Encoder threshold: {config['encoder']['threshold']}")
    print(f"  - DU Core: {config['architecture']['input_size']} → {config['architecture']['hidden_sizes']} → {config['architecture']['output_size']}")
    print(f"  - Neuron threshold: {config['architecture']['neuron_threshold']}")
    print(f"  - Decoder: {config['architecture']['output_size']} → 1")
    
    # Test manual forward pass first
    print(f"\n--- Manual Forward Pass ---")
    spike_train = encoder.encode(light_curve)
    print(f"✓ Encoded spikes: {spike_train.sum().item()}")
    
    processed_spikes = du_core.process(spike_train)
    print(f"✓ Processed spikes: {processed_spikes.sum().item()}")
    
    spike_counts = processed_spikes.sum(dim=0)
    if spike_counts.dim() == 1:
        spike_counts = spike_counts.unsqueeze(0)
    print(f"✓ Spike counts shape: {spike_counts.shape}")
    print(f"✓ Spike counts: {spike_counts}")
    
    # Test decoder forward_classification
    try:
        logits = decoder.forward_classification(spike_counts)
        print(f"✓ Decoder logits: {logits}")
        print(f"✓ Logits shape: {logits.shape}")
        print(f"✓ Logits require grad: {logits.requires_grad}")
    except Exception as e:
        print(f"✗ Decoder failed: {e}")
        return
    
    # Test loss calculation
    target_tensor = torch.tensor([target], dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    
    try:
        loss = criterion(logits, target_tensor)
        print(f"✓ Loss: {loss.item():.6f}")
        print(f"✓ Loss requires grad: {loss.requires_grad}")
    except Exception as e:
        print(f"✗ Loss calculation failed: {e}")
        return
    
    # Test backward pass
    try:
        loss.backward()
        print(f"✓ Backward pass completed")
        
        # Check gradients
        total_grad_norm = 0.0
        for name, param in du_core.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                print(f"  {name}: grad_norm = {grad_norm:.6f}")
            else:
                print(f"  {name}: NO GRADIENT")
                
        for name, param in decoder.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                print(f"  decoder.{name}: grad_norm = {grad_norm:.6f}")
            else:
                print(f"  decoder.{name}: NO GRADIENT")
                
        print(f"✓ Total gradient norm: {total_grad_norm:.6f}")
        
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        return
    
    # Now test with trainer
    print(f"\n--- Testing with Trainer ---")
    
    # Reset gradients
    for param in du_core.parameters():
        if param.grad is not None:
            param.grad.zero_()
    for param in decoder.parameters():
        if param.grad is not None:
            param.grad.zero_()
    
    # Create optimizer and trainer
    optimizer = optim.Adam(gif_model.parameters(), lr=config["training"]["learning_rate"])
    
    trainer = Continual_Trainer(
        gif_model=gif_model,
        optimizer=optimizer,
        criterion=criterion,
        memory_batch_size=config["training"]["memory_batch_size"],
        gradient_clip_norm=config["training"].get("gradient_clip_norm", None)
    )
    
    print(f"✓ Trainer created")
    print(f"  - Learning rate: {config['training']['learning_rate']}")
    print(f"  - Memory batch size: {config['training']['memory_batch_size']}")
    print(f"  - Memory capacity: 10000")
    
    # Test trainer step
    try:
        target_tensor = torch.tensor([target], dtype=torch.long)
        loss_trainer = trainer.train_step(light_curve, target_tensor, "exoplanet_detection")
        print(f"✓ Trainer step completed")
        print(f"✓ Trainer loss: {loss_trainer:.6f}")
        
        # Check if loss is different from manual calculation
        if abs(loss_trainer - loss.item()) > 1e-6:
            print(f"⚠️  Loss difference: manual={loss.item():.6f}, trainer={loss_trainer:.6f}")
        else:
            print(f"✓ Loss matches manual calculation")
            
    except Exception as e:
        print(f"✗ Trainer step failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test multiple steps
    print(f"\n--- Testing Multiple Steps ---")
    losses = [loss_trainer]
    
    for step in range(4):
        try:
            loss_step = trainer.train_step(light_curve, target_tensor, "exoplanet_detection")
            losses.append(loss_step)
            print(f"Step {step+2}: Loss = {loss_step:.6f}")
        except Exception as e:
            print(f"✗ Step {step+2} failed: {e}")
            break
    
    print(f"\nLoss progression: {[f'{l:.6f}' for l in losses]}")
    
    if len(losses) > 1:
        loss_change = losses[0] - losses[-1]
        print(f"Total loss change: {loss_change:.6f}")
        
        if abs(loss_change) > 1e-6:
            print("✅ SUCCESS: Loss is changing!")
        else:
            print("❌ FAILURE: Loss not changing")

def main():
    """Run the debug."""
    print("GEM Trainer Debug Script")
    print("=" * 60)
    
    debug_trainer_step()

if __name__ == "__main__":
    main()
