#!/usr/bin/env python3
"""
Test Balanced Training
======================

This script tests training with both planet and non-planet samples to ensure
the model can learn to distinguish between them.
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

def test_balanced_training():
    """Test training with balanced planet/non-planet samples."""
    print("=" * 60)
    print("TESTING BALANCED TRAINING")
    print("=" * 60)
    
    config = EXP_CONFIG
    
    # Generate balanced dataset
    generator = RealisticExoplanetGenerator(seed=42)
    
    planet_samples = []
    non_planet_samples = []
    
    print("Generating balanced dataset...")
    attempts = 0
    while len(planet_samples) < 5 or len(non_planet_samples) < 5:
        attempts += 1
        if attempts > 100:
            print("Warning: Could not generate balanced dataset")
            break
            
        light_curve = generator.generate()
        has_planet = light_curve['planet_radius_ratio'][0] > 0
        
        if has_planet and len(planet_samples) < 5:
            planet_samples.append((light_curve, 1))
        elif not has_planet and len(non_planet_samples) < 5:
            non_planet_samples.append((light_curve, 0))
    
    print(f"Generated {len(planet_samples)} planet samples and {len(non_planet_samples)} non-planet samples")
    
    # Combine and shuffle
    all_samples = planet_samples + non_planet_samples
    np.random.shuffle(all_samples)
    
    # Initialize components
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
    
    print(f"\nTraining Configuration:")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Neuron threshold: {config['architecture']['neuron_threshold']}")
    print(f"  Memory batch size: {config['training']['memory_batch_size']}")
    
    # Test each sample individually first
    print(f"\nTesting Individual Samples:")
    for i, (light_curve, target) in enumerate(all_samples):
        spike_train = encoder.encode(light_curve)
        processed_spikes = du_core.process(spike_train)
        spike_counts = processed_spikes.sum(dim=0)
        
        print(f"  Sample {i+1} (target={target}):")
        print(f"    Input spikes: {spike_train.sum().item()}")
        print(f"    Output spikes: {processed_spikes.sum().item()}")
        print(f"    Spike distribution: [{spike_counts[0].item():.0f}, {spike_counts[1].item():.0f}]")
        
        # Test prediction
        if spike_counts.dim() == 1:
            spike_counts = spike_counts.unsqueeze(0)
        logits = decoder.forward_classification(spike_counts)
        prediction = torch.argmax(logits, dim=1).item()
        
        print(f"    Logits: [{logits[0,0].item():.2f}, {logits[0,1].item():.2f}]")
        print(f"    Prediction: {prediction}")
        
        # Calculate loss
        target_tensor = torch.tensor([target], dtype=torch.long)
        loss = criterion(logits, target_tensor)
        print(f"    Loss: {loss.item():.6f}")
    
    # Training loop
    print(f"\nTraining Loop:")
    losses = []
    accuracies = []
    
    for epoch in range(5):
        epoch_losses = []
        correct = 0
        total = 0
        
        for i, (light_curve, target) in enumerate(all_samples):
            target_tensor = torch.tensor([target], dtype=torch.long)
            
            try:
                loss = trainer.train_step(light_curve, target_tensor, "exoplanet_detection")
                epoch_losses.append(loss)
                
                # Test prediction after training step
                spike_train = encoder.encode(light_curve)
                processed_spikes = du_core.process(spike_train)
                spike_counts = processed_spikes.sum(dim=0)
                if spike_counts.dim() == 1:
                    spike_counts = spike_counts.unsqueeze(0)
                
                with torch.no_grad():
                    logits = decoder.forward_classification(spike_counts)
                    prediction = torch.argmax(logits, dim=1).item()
                    if prediction == target:
                        correct += 1
                    total += 1
                
            except Exception as e:
                print(f"    Training step {i+1} failed: {e}")
                continue
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        accuracy = correct / total if total > 0 else 0.0
        
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}, Accuracy = {accuracy:.4f}")
    
    # Final analysis
    print(f"\nTraining Analysis:")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Loss change: {losses[0] - losses[-1]:.6f}")
    print(f"  Initial accuracy: {accuracies[0]:.4f}")
    print(f"  Final accuracy: {accuracies[-1]:.4f}")
    print(f"  Accuracy change: {accuracies[-1] - accuracies[0]:.4f}")
    
    # Success criteria
    learning_occurred = abs(losses[0] - losses[-1]) > 0.01
    accuracy_improved = accuracies[-1] > accuracies[0]
    final_accuracy_good = accuracies[-1] > 0.6
    
    print(f"\nSuccess Criteria:")
    print(f"  Learning occurred (loss change > 0.01): {'✅' if learning_occurred else '❌'}")
    print(f"  Accuracy improved: {'✅' if accuracy_improved else '❌'}")
    print(f"  Final accuracy > 60%: {'✅' if final_accuracy_good else '❌'}")
    
    if learning_occurred and accuracy_improved:
        print(f"\n🎉 SUCCESS: Model is learning to distinguish planets from non-planets!")
        return True
    else:
        print(f"\n❌ FAILURE: Model not learning effectively")
        return False

def main():
    """Run the balanced training test."""
    print("Balanced Training Test Script")
    print("=" * 60)
    
    success = test_balanced_training()
    
    if success:
        print(f"\n✅ READY TO RUN FULL EXPERIMENT")
        print(f"The GIF-DU model is now properly configured and learning!")
    else:
        print(f"\n❌ NEED FURTHER DEBUGGING")
        print(f"The model still has issues that need to be resolved.")

if __name__ == "__main__":
    main()
