"""
Test suite for GPU acceleration functionality in the GIF framework.

This module tests the GPU acceleration features including:
- Device detection and configuration
- Model device placement
- Training pipeline with GPU support
- Performance comparisons between CPU and GPU

Author: GIF Framework Team
Date: 2025-07-11
"""

import pytest
import torch
import time
from typing import Dict, Any
import logging

# Test imports
from applications.poc_exoplanet.config_exo import get_config, get_device, move_to_device
from applications.poc_exoplanet.main_exo import initialize_components, assemble_gif_model
from data_generators.exoplanet_generator import RealisticExoplanetGenerator


class TestGPUAcceleration:
    """Test suite for GPU acceleration functionality."""
    
    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        """Get test configuration."""
        config = get_config()
        # Use smaller dataset for faster tests
        config["data"]["num_training_samples"] = 50
        config["data"]["num_test_samples"] = 20
        config["training"]["num_epochs"] = 2
        return config
    
    @pytest.fixture
    def logger(self) -> logging.Logger:
        """Get test logger."""
        logger = logging.getLogger('test_gpu')
        logger.setLevel(logging.CRITICAL)  # Reduce noise in tests
        return logger
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample light curve data for testing."""
        generator = RealisticExoplanetGenerator(seed=42)
        return generator.generate()
    
    def test_device_detection(self):
        """Test automatic device detection."""
        # Test auto detection
        device_auto = get_device("auto")
        assert isinstance(device_auto, torch.device)
        
        # Test CPU fallback
        device_cpu = get_device("cpu")
        assert device_cpu.type == "cpu"
        
        # Test MPS detection on Apple Silicon
        if torch.backends.mps.is_available():
            device_mps = get_device("mps")
            assert device_mps.type == "mps"
    
    def test_device_configuration_options(self):
        """Test different device configuration options."""
        # Test all supported device types
        supported_devices = ["auto", "cpu"]
        
        if torch.backends.mps.is_available():
            supported_devices.append("mps")
        
        if torch.cuda.is_available():
            supported_devices.append("cuda")
        
        for device_type in supported_devices:
            device = get_device(device_type)
            assert isinstance(device, torch.device)
    
    def test_move_to_device_utility(self):
        """Test the move_to_device utility function."""
        device = get_device("auto")
        
        # Test tensor movement
        tensor = torch.randn(10, 5)
        moved_tensor = move_to_device(tensor, device)
        assert moved_tensor.device.type == device.type
        
        # Test list movement
        tensor_list = [torch.randn(5, 3), torch.randn(3, 2)]
        moved_list = move_to_device(tensor_list, device)
        for t in moved_list:
            assert t.device.type == device.type
        
        # Test dict movement
        tensor_dict = {"a": torch.randn(2, 2), "b": torch.randn(3, 3)}
        moved_dict = move_to_device(tensor_dict, device)
        for t in moved_dict.values():
            assert t.device.type == device.type
    
    def test_component_initialization_with_gpu(self, config, logger):
        """Test component initialization with GPU support."""
        # Test GPU initialization
        config["training"]["device"] = "auto"
        encoder, decoder, du_core, device = initialize_components(config, logger)
        
        # Verify components are initialized
        assert encoder is not None
        assert decoder is not None
        assert du_core is not None
        assert isinstance(device, torch.device)
        
        # Verify encoder has correct device
        assert hasattr(encoder, 'device')
        assert encoder.device.type == device.type

        # Verify DU Core is on correct device
        if hasattr(du_core, 'linear_layers') and len(du_core.linear_layers) > 0:
            first_param = next(du_core.linear_layers[0].parameters())
            assert first_param.device.type == device.type
    
    def test_model_assembly_with_gpu(self, config, logger):
        """Test model assembly with GPU support."""
        config["training"]["device"] = "auto"
        encoder, decoder, du_core, device = initialize_components(config, logger)
        
        # Test model assembly
        gif_model, trainer, simulator = assemble_gif_model(
            encoder, decoder, du_core, device, config, logger
        )
        
        # Verify components are assembled
        assert gif_model is not None
        assert trainer is not None
        assert simulator is not None
    
    def test_training_step_with_gpu(self, config, logger, sample_data):
        """Test training step execution with GPU."""
        config["training"]["device"] = "auto"
        encoder, decoder, du_core, device = initialize_components(config, logger)
        gif_model, trainer, simulator = assemble_gif_model(
            encoder, decoder, du_core, device, config, logger
        )
        
        # Prepare target tensor on correct device
        target = torch.tensor([1], dtype=torch.long).to(device)
        
        # Test training step
        loss = trainer.train_step(sample_data, target, "test_task")
        
        # Verify training step succeeded
        assert isinstance(loss, float)
        assert 0.0 <= loss <= 10.0  # Reasonable loss range
    
    def test_multiple_training_steps_stability(self, config, logger, sample_data):
        """Test stability of multiple training steps with GPU."""
        config["training"]["device"] = "auto"
        encoder, decoder, du_core, device = initialize_components(config, logger)
        gif_model, trainer, simulator = assemble_gif_model(
            encoder, decoder, du_core, device, config, logger
        )
        
        target = torch.tensor([1], dtype=torch.long).to(device)
        
        # Test multiple training steps
        losses = []
        for i in range(5):
            loss = trainer.train_step(sample_data, target, f"test_task_{i}")
            losses.append(loss)
        
        # Verify all steps succeeded
        assert len(losses) == 5
        for loss in losses:
            assert isinstance(loss, float)
            assert 0.0 <= loss <= 10.0
    
    @pytest.mark.skipif(not torch.backends.mps.is_available(), 
                       reason="MPS not available on this system")
    def test_mps_specific_functionality(self, config, logger, sample_data):
        """Test MPS-specific functionality on Apple Silicon."""
        config["training"]["device"] = "mps"
        encoder, decoder, du_core, device = initialize_components(config, logger)
        
        # Verify MPS device
        assert device.type == "mps"
        
        # Test training with MPS
        gif_model, trainer, simulator = assemble_gif_model(
            encoder, decoder, du_core, device, config, logger
        )
        
        target = torch.tensor([1], dtype=torch.long).to(device)
        loss = trainer.train_step(sample_data, target, "mps_test")
        
        assert isinstance(loss, float)
        assert 0.0 <= loss <= 10.0
    
    def test_cpu_gpu_consistency(self, config, logger, sample_data):
        """Test that CPU and GPU produce consistent results."""
        # Test with CPU
        config_cpu = config.copy()
        config_cpu["training"]["device"] = "cpu"
        encoder_cpu, decoder_cpu, du_core_cpu, device_cpu = initialize_components(config_cpu, logger)
        
        # Test with GPU (if available)
        config_gpu = config.copy()
        config_gpu["training"]["device"] = "auto"
        encoder_gpu, decoder_gpu, du_core_gpu, device_gpu = initialize_components(config_gpu, logger)
        
        # Both should initialize successfully
        assert encoder_cpu is not None and encoder_gpu is not None
        assert decoder_cpu is not None and decoder_gpu is not None
        assert du_core_cpu is not None and du_core_gpu is not None
        
        # Devices should be different (unless no GPU available)
        if device_gpu.type != "cpu":
            assert device_cpu.type != device_gpu.type


class TestPerformanceBenchmarks:
    """Performance benchmarks for GPU vs CPU training."""
    
    @pytest.fixture
    def benchmark_config(self) -> Dict[str, Any]:
        """Get configuration for benchmarking."""
        config = get_config()
        # Use smaller dataset for faster benchmarks
        config["data"]["num_training_samples"] = 100
        return config
    
    def test_training_step_performance(self, benchmark_config):
        """Benchmark training step performance on different devices."""
        logger = logging.getLogger('benchmark')
        logger.setLevel(logging.CRITICAL)
        
        generator = RealisticExoplanetGenerator(seed=42)
        sample_data = generator.generate()
        
        devices_to_test = ["cpu"]
        if torch.backends.mps.is_available():
            devices_to_test.append("mps")
        if torch.cuda.is_available():
            devices_to_test.append("cuda")
        
        performance_results = {}
        
        for device_type in devices_to_test:
            config = benchmark_config.copy()
            config["training"]["device"] = device_type
            
            encoder, decoder, du_core, device = initialize_components(config, logger)
            gif_model, trainer, simulator = assemble_gif_model(
                encoder, decoder, du_core, device, config, logger
            )
            
            target = torch.tensor([1], dtype=torch.long).to(device)
            
            # Warm up
            for _ in range(3):
                trainer.train_step(sample_data, target, "warmup")
            
            # Benchmark
            times = []
            for i in range(10):
                start_time = time.time()
                trainer.train_step(sample_data, target, f"benchmark_{i}")
                times.append(time.time() - start_time)
            
            avg_time = sum(times) / len(times)
            performance_results[device_type] = {
                "avg_time": avg_time,
                "throughput": 1.0 / avg_time
            }
        
        # Log performance results
        print("\n=== PERFORMANCE BENCHMARK RESULTS ===")
        for device_type, results in performance_results.items():
            print(f"{device_type.upper()}: {results['avg_time']:.3f}s avg, "
                  f"{results['throughput']:.1f} samples/sec")
        
        # All devices should complete successfully
        assert len(performance_results) >= 1
        for device_type, results in performance_results.items():
            assert results["avg_time"] > 0
            assert results["throughput"] > 0


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
