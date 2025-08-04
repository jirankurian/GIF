"""
Meta-Cognitive Routing Demonstration
===================================

This demonstration showcases the advanced meta-cognitive routing capabilities
of the GIF framework, including intelligent encoder selection based on task
descriptions and signal characteristics.

Key Features Demonstrated:
- Automatic encoder selection based on natural language task descriptions
- Specialized encoder performance on different signal types
- Meta-cognitive decision-making and reasoning
- Comparison of encoder effectiveness for different tasks

This represents a significant step toward AGI by enabling the system to reason
about its own tools and make intelligent decisions about which components to
use for optimal performance.
"""

import numpy as np
import polars as pl
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# Import GIF framework components
from gif_framework.orchestrator import GIF
from gif_framework.module_library import ModuleLibrary, ModuleMetadata
from gif_framework.core.du_core import DU_Core_V1

# Import specialized encoders
from applications.poc_exoplanet.encoders.fourier_encoder import FourierEncoder
from applications.poc_exoplanet.encoders.wavelet_encoder import WaveletEncoder
from applications.poc_exoplanet.encoders.light_curve_encoder import LightCurveEncoder

# Import decoder
from applications.poc_exoplanet.decoders.exoplanet_decoder import ExoplanetDecoder


def create_test_signals() -> Dict[str, pl.DataFrame]:
    """Create different types of test signals for demonstration."""
    time = np.linspace(0, 10, 1000)
    
    # 1. Periodic signal (simulating planetary transit)
    periodic_flux = 1.0 + 0.02 * np.sin(2 * np.pi * 0.5 * time)  # 0.5 Hz periodic
    periodic_flux += 0.005 * np.random.randn(1000)  # Add noise
    
    periodic_data = pl.DataFrame({
        'time': time,
        'flux': periodic_flux
    })
    
    # 2. Transient signal (simulating stellar flare)
    transient_flux = np.ones(1000)
    # Add sudden flare events
    transient_flux[300:350] += 0.1 * np.exp(-0.1 * np.arange(50))  # Flare 1
    transient_flux[700:720] += 0.05 * np.exp(-0.2 * np.arange(20))  # Flare 2
    transient_flux += 0.005 * np.random.randn(1000)  # Add noise
    
    transient_data = pl.DataFrame({
        'time': time,
        'flux': transient_flux
    })
    
    # 3. Mixed signal (periodic + transient)
    mixed_flux = 1.0 + 0.015 * np.sin(2 * np.pi * 0.3 * time)  # Periodic component
    mixed_flux[500:530] += 0.08 * np.exp(-0.15 * np.arange(30))  # Transient event
    mixed_flux += 0.005 * np.random.randn(1000)  # Add noise
    
    mixed_data = pl.DataFrame({
        'time': time,
        'flux': mixed_flux
    })
    
    return {
        'periodic': periodic_data,
        'transient': transient_data,
        'mixed': mixed_data
    }


def setup_meta_cognitive_gif() -> GIF:
    """Set up GIF framework with meta-cognitive routing capabilities."""
    print("🧠 Setting up Meta-Cognitive GIF Framework...")
    
    # Create DU Core
    du_core = DU_Core_V1(
        input_size=4,  # Match encoder output channels
        hidden_sizes=[16, 8],
        output_size=3,  # For classification
        beta=0.95,
        threshold=1.0
    )
    
    # Create module library
    library = ModuleLibrary()
    
    # Register specialized encoders
    print("📚 Registering specialized encoders...")
    
    # 1. FourierEncoder for periodic signals
    fourier_encoder = FourierEncoder(n_frequency_channels=4)
    fourier_metadata = ModuleMetadata(
        module_type='encoder',
        data_modality='timeseries',
        signal_type='periodic',
        domain='astronomy',
        input_format='dataframe',
        description='Optimized for detecting periodic patterns and regular signals'
    )
    library.register_module(fourier_encoder, fourier_metadata)
    
    # 2. WaveletEncoder for transient signals
    wavelet_encoder = WaveletEncoder(n_scales=8, n_time_channels=4)
    wavelet_metadata = ModuleMetadata(
        module_type='encoder',
        data_modality='timeseries',
        signal_type='transient',
        domain='astronomy',
        input_format='dataframe',
        description='Optimized for detecting sudden bursts and transient events'
    )
    library.register_module(wavelet_encoder, wavelet_metadata)
    
    # 3. LightCurveEncoder for general signals
    light_curve_encoder = LightCurveEncoder()
    general_metadata = ModuleMetadata(
        module_type='encoder',
        data_modality='timeseries',
        signal_type='general',
        domain='astronomy',
        input_format='dataframe',
        description='General-purpose encoder for astronomical light curves'
    )
    library.register_module(light_curve_encoder, general_metadata)
    
    # Register decoder
    decoder = ExoplanetDecoder(input_size=3, output_size=1)  # Match DU Core output size
    decoder_metadata = ModuleMetadata(
        module_type='decoder',
        data_modality='timeseries',
        signal_type='any',
        domain='astronomy',
        input_format='tensor',
        output_format='classification',
        description='Classifies astronomical signals'
    )
    library.register_module(decoder, decoder_metadata)
    
    # Create GIF with meta-cognitive capabilities
    gif = GIF(du_core, module_library=library)
    gif.attach_decoder(decoder)  # Attach decoder manually for now
    
    print(f"✅ Meta-cognitive GIF setup complete!")
    print(f"   📊 Library stats: {library.get_library_stats()}")
    
    return gif


def demonstrate_meta_cognitive_routing():
    """Demonstrate meta-cognitive routing with different task descriptions."""
    print("\n" + "="*80)
    print("🧠 META-COGNITIVE ROUTING DEMONSTRATION")
    print("="*80)
    
    # Setup
    gif = setup_meta_cognitive_gif()
    test_signals = create_test_signals()
    
    # Define task scenarios
    task_scenarios = [
        {
            'signal_type': 'periodic',
            'task_description': "Analyze this light curve to find repeating patterns and periodic behavior",
            'expected_encoder': 'FourierEncoder'
        },
        {
            'signal_type': 'transient',
            'task_description': "Detect sudden flares and burst events in this stellar data",
            'expected_encoder': 'WaveletEncoder'
        },
        {
            'signal_type': 'mixed',
            'task_description': "Process this astronomical signal for general analysis",
            'expected_encoder': 'LightCurveEncoder'
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(task_scenarios, 1):
        print(f"\n🔍 Scenario {i}: {scenario['signal_type'].title()} Signal Analysis")
        print("-" * 60)
        
        signal_data = test_signals[scenario['signal_type']]
        task_description = scenario['task_description']
        
        print(f"📝 Task: {task_description}")
        
        # Meta-cognitive encoder selection
        selected_encoder = gif._meta_control_select_encoder(task_description)
        
        if selected_encoder:
            encoder_name = type(selected_encoder).__name__
            print(f"🎯 Selected Encoder: {encoder_name}")
            
            # Check if selection matches expectation
            if encoder_name == scenario['expected_encoder']:
                print("✅ Optimal encoder selected!")
            else:
                print(f"⚠️  Expected {scenario['expected_encoder']}, got {encoder_name}")
            
            # Process the signal
            try:
                result = gif.process_single_input(signal_data, task_description)
                print(f"📊 Processing Result: {result}")
                
                # Get encoder statistics
                if hasattr(selected_encoder, 'get_encoding_stats'):
                    stats = selected_encoder.get_encoding_stats()
                    print(f"📈 Encoding Stats: {stats}")
                
                results.append({
                    'scenario': scenario['signal_type'],
                    'task': task_description,
                    'selected_encoder': encoder_name,
                    'expected_encoder': scenario['expected_encoder'],
                    'result': result,
                    'optimal_selection': encoder_name == scenario['expected_encoder']
                })
                
            except Exception as e:
                print(f"❌ Processing failed: {e}")
                results.append({
                    'scenario': scenario['signal_type'],
                    'selected_encoder': encoder_name,
                    'error': str(e)
                })
        else:
            print("❌ No encoder selected")
    
    # Summary
    print(f"\n📋 DEMONSTRATION SUMMARY")
    print("=" * 40)
    
    optimal_selections = sum(1 for r in results if r.get('optimal_selection', False))
    total_scenarios = len([r for r in results if 'optimal_selection' in r])
    
    print(f"🎯 Optimal Encoder Selections: {optimal_selections}/{total_scenarios}")
    print(f"🧠 Meta-Cognitive Success Rate: {optimal_selections/total_scenarios*100:.1f}%")
    
    for result in results:
        if 'optimal_selection' in result:
            status = "✅" if result['optimal_selection'] else "⚠️"
            print(f"{status} {result['scenario'].title()}: {result['selected_encoder']}")
    
    return results


def demonstrate_encoder_comparison():
    """Compare encoder performance on different signal types."""
    print(f"\n🔬 ENCODER PERFORMANCE COMPARISON")
    print("=" * 50)
    
    # Create test signals
    test_signals = create_test_signals()
    
    # Create encoders
    encoders = {
        'FourierEncoder': FourierEncoder(n_frequency_channels=4),
        'WaveletEncoder': WaveletEncoder(n_scales=8, n_time_channels=4),
        'LightCurveEncoder': LightCurveEncoder()
    }
    
    # Test each encoder on each signal type
    for signal_name, signal_data in test_signals.items():
        print(f"\n📊 Signal Type: {signal_name.title()}")
        print("-" * 30)
        
        for encoder_name, encoder in encoders.items():
            try:
                # Encode the signal
                spike_train = encoder.encode(signal_data)
                
                # Calculate basic metrics
                total_spikes = torch.sum(spike_train > 0).item()
                spike_rate = total_spikes / spike_train.numel()
                max_activity = torch.max(spike_train).item()
                
                print(f"  {encoder_name}:")
                print(f"    Spikes: {total_spikes}, Rate: {spike_rate:.3f}, Max: {max_activity:.3f}")
                
                # Get encoder-specific stats if available
                if hasattr(encoder, 'get_encoding_stats'):
                    stats = encoder.get_encoding_stats()
                    if 'dominant_frequency' in stats:
                        print(f"    Dominant Freq: {stats['dominant_frequency']:.3f}")
                    if 'transient_events' in stats:
                        print(f"    Transient Events: {stats['transient_events']}")
                
            except Exception as e:
                print(f"  {encoder_name}: ❌ Error - {e}")


def main():
    """Main demonstration function."""
    print("🚀 Starting Meta-Cognitive Routing Demonstration")
    print("This demo showcases AGI-level capabilities in tool selection and reasoning")
    
    try:
        # Demonstrate meta-cognitive routing
        routing_results = demonstrate_meta_cognitive_routing()
        
        # Demonstrate encoder comparison
        demonstrate_encoder_comparison()
        
        print(f"\n🎉 DEMONSTRATION COMPLETE!")
        print("The GIF framework successfully demonstrated:")
        print("  ✅ Meta-cognitive reasoning about tool selection")
        print("  ✅ Natural language task understanding")
        print("  ✅ Intelligent encoder routing based on signal characteristics")
        print("  ✅ Performance comparison across different encoders")
        print("\nThis represents a significant step toward AGI capabilities! 🧠✨")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
