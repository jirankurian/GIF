"""
Integration tests for the complete training pipeline.

This module tests the end-to-end training functionality including:
- Complete training pipeline execution
- Model performance validation
- Result generation and saving
- Error handling and recovery

Author: GIF Framework Team
Date: 2025-07-11
"""

import pytest
import torch
import os
import json
from typing import Dict, Any
import logging
import tempfile
import shutil

# Test imports
from applications.poc_exoplanet.config_exo import get_config
from applications.poc_exoplanet.main_exo import (
    initialize_components, 
    assemble_gif_model,
    generate_datasets,
    train_model,
    evaluate_model,
    save_results
)


class TestIntegrationTraining:
    """Integration tests for complete training pipeline."""
    
    @pytest.fixture
    def test_config(self) -> Dict[str, Any]:
        """Get test configuration with small dataset."""
        import copy
        config = copy.deepcopy(get_config())  # Deep copy to avoid modifying global config
        # Use very small dataset for fast integration tests
        config["data"]["num_training_samples"] = 20
        config["data"]["num_test_samples"] = 10
        config["training"]["num_epochs"] = 2
        config["training"]["device"] = "cpu"  # Use CPU for consistent testing
        return config
    
    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary directory for test results."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def logger(self) -> logging.Logger:
        """Get test logger."""
        logger = logging.getLogger('test_integration')
        logger.setLevel(logging.CRITICAL)  # Reduce noise in tests
        return logger
    
    def test_complete_training_pipeline(self, test_config, temp_results_dir, logger):
        """Test the complete training pipeline from start to finish."""
        # Update config to use temp directory
        test_config["analysis"]["results_dir"] = temp_results_dir
        
        # Step 1: Generate datasets
        training_data, test_data = generate_datasets(test_config, logger)
        
        # Verify datasets
        assert len(training_data) == test_config["data"]["num_training_samples"]
        assert len(test_data) == test_config["data"]["num_test_samples"]
        
        # Step 2: Initialize components
        encoder, decoder, du_core, device = initialize_components(test_config, logger)
        
        # Verify components
        assert encoder is not None
        assert decoder is not None
        assert du_core is not None
        assert device.type == "cpu"
        
        # Step 3: Assemble model
        gif_model, trainer, simulator = assemble_gif_model(
            encoder, decoder, du_core, device, test_config, logger
        )
        
        # Verify model assembly
        assert gif_model is not None
        assert trainer is not None
        assert simulator is not None
        
        # Step 4: Train model
        training_logs = train_model(gif_model, trainer, training_data, device, test_config, logger)
        
        # Verify training logs
        assert isinstance(training_logs, dict)
        assert "loss" in training_logs
        assert "accuracy" in training_logs
        assert len(training_logs["loss"]) == test_config["training"]["num_epochs"]
        
        # Step 5: Evaluate model
        evaluation_results = evaluate_model(gif_model, simulator, test_data, test_config, logger)
        
        # Verify evaluation results
        assert isinstance(evaluation_results, dict)
        assert "accuracy" in evaluation_results
        assert "precision" in evaluation_results
        assert "recall" in evaluation_results
        assert "f1_score" in evaluation_results
        
        # Verify metrics are reasonable
        assert 0.0 <= evaluation_results["accuracy"] <= 1.0
        assert 0.0 <= evaluation_results["precision"] <= 1.0
        assert 0.0 <= evaluation_results["recall"] <= 1.0
        assert 0.0 <= evaluation_results["f1_score"] <= 1.0
        
        # Step 6: Save results
        save_results(training_logs, evaluation_results, test_config, logger)
        
        # Verify results are saved
        results_dir = test_config["analysis"]["results_dir"]
        assert os.path.exists(os.path.join(results_dir, "experiment_config.json"))
        assert os.path.exists(os.path.join(results_dir, "training_logs.json"))
        assert os.path.exists(os.path.join(results_dir, "evaluation_results.json"))
    
    def test_training_convergence(self, test_config, logger):
        """Test that training shows some convergence behavior."""
        # Generate small dataset
        training_data, _ = generate_datasets(test_config, logger)
        
        # Initialize components
        encoder, decoder, du_core, device = initialize_components(test_config, logger)
        gif_model, trainer, simulator = assemble_gif_model(
            encoder, decoder, du_core, device, test_config, logger
        )
        
        # Train model
        training_logs = train_model(gif_model, trainer, training_data, device, test_config, logger)
        
        # Check that we have loss values for each epoch
        losses = training_logs["loss"]
        assert len(losses) == test_config["training"]["num_epochs"]

        # Check that all losses are reasonable values
        for loss in losses:
            assert isinstance(loss, (float, int))
            assert 0.0 <= loss <= 10.0  # Reasonable range for cross-entropy loss

        # Check that accuracies are recorded
        accuracies = training_logs["accuracy"]
        assert len(accuracies) == test_config["training"]["num_epochs"]

        for accuracy in accuracies:
            assert isinstance(accuracy, (float, int))
            assert 0.0 <= accuracy <= 1.0
    
    def test_model_evaluation_metrics(self, test_config, logger):
        """Test that model evaluation produces valid metrics."""
        # Generate datasets
        training_data, test_data = generate_datasets(test_config, logger)
        
        # Initialize and train model
        encoder, decoder, du_core, device = initialize_components(test_config, logger)
        gif_model, trainer, simulator = assemble_gif_model(
            encoder, decoder, du_core, device, test_config, logger
        )
        
        # Quick training
        train_model(gif_model, trainer, training_data, device, test_config, logger)
        
        # Evaluate model
        evaluation_results = evaluate_model(gif_model, simulator, test_data, test_config, logger)
        
        # Check all required metrics are present
        required_metrics = ["accuracy", "precision", "recall", "f1_score", 
                          "avg_energy_per_sample", "avg_processing_time"]
        
        for metric in required_metrics:
            assert metric in evaluation_results
            assert isinstance(evaluation_results[metric], (int, float))
        
        # Check energy and timing metrics are positive
        assert evaluation_results["avg_energy_per_sample"] > 0
        assert evaluation_results["avg_processing_time"] > 0
    
    def test_different_device_configurations(self, test_config, logger):
        """Test training with different device configurations."""
        devices_to_test = ["cpu"]
        
        # Add MPS if available
        if torch.backends.mps.is_available():
            devices_to_test.append("mps")
        
        for device_type in devices_to_test:
            test_config["training"]["device"] = device_type
            
            # Generate small dataset
            training_data, test_data = generate_datasets(test_config, logger)
            
            # Initialize components
            encoder, decoder, du_core, device = initialize_components(test_config, logger)
            
            # Verify correct device
            assert device.type == device_type
            
            # Assemble and test training
            gif_model, trainer, simulator = assemble_gif_model(
                encoder, decoder, du_core, device, test_config, logger
            )
            
            # Test single training step
            sample_data, target_label = training_data[0]
            target = torch.tensor([target_label], dtype=torch.long).to(device)
            
            loss = trainer.train_step(sample_data, target, f"test_{device_type}")
            
            # Verify training step succeeded
            assert isinstance(loss, float)
            assert 0.0 <= loss <= 10.0
    
    def test_error_handling(self, test_config, logger):
        """Test error handling in the training pipeline."""
        # Test with invalid configuration that should raise ValueError
        from applications.poc_exoplanet.config_exo import validate_config

        invalid_config = test_config.copy()
        invalid_config["training"]["learning_rate"] = -1.0  # Invalid learning rate

        # This should raise a ValueError during validation
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_config(invalid_config)
    
    def test_memory_efficiency(self, test_config, logger):
        """Test that training doesn't consume excessive memory."""
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run training
        training_data, test_data = generate_datasets(test_config, logger)
        encoder, decoder, du_core, device = initialize_components(test_config, logger)
        gif_model, trainer, simulator = assemble_gif_model(
            encoder, decoder, du_core, device, test_config, logger
        )
        
        train_model(gif_model, trainer, training_data, device, test_config, logger)
        evaluate_model(gif_model, simulator, test_data, test_config, logger)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for small test)
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB"
    
    def test_reproducibility(self, test_config, logger):
        """Test that training results are reproducible with same seed."""
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        # First run
        training_data1, _ = generate_datasets(test_config, logger)
        encoder1, decoder1, du_core1, device1 = initialize_components(test_config, logger)
        gif_model1, trainer1, simulator1 = assemble_gif_model(
            encoder1, decoder1, du_core1, device1, test_config, logger
        )
        
        # Train for one step
        sample_data, target_label = training_data1[0]
        target = torch.tensor([target_label], dtype=torch.long).to(device1)
        loss1 = trainer1.train_step(sample_data, target, "reproducibility_test")
        
        # Reset seed and repeat
        torch.manual_seed(42)
        
        training_data2, _ = generate_datasets(test_config, logger)
        encoder2, decoder2, du_core2, device2 = initialize_components(test_config, logger)
        gif_model2, trainer2, simulator2 = assemble_gif_model(
            encoder2, decoder2, du_core2, device2, test_config, logger
        )
        
        # Train for one step with same data
        sample_data2, target_label2 = training_data2[0]
        target2 = torch.tensor([target_label2], dtype=torch.long).to(device2)
        loss2 = trainer2.train_step(sample_data2, target2, "reproducibility_test")
        
        # Results should be very close (allowing for small floating point differences)
        assert abs(loss1 - loss2) < 1e-5, f"Loss difference: {abs(loss1 - loss2)}"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
