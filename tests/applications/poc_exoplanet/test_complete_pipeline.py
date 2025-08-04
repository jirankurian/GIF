"""
Comprehensive Test Suite for Complete Exoplanet POC Pipeline
===========================================================

This module provides comprehensive testing for the complete exoplanet detection
pipeline including configuration management, component initialization, and
end-to-end execution validation.

Test Categories:
- Configuration loading and validation
- Component initialization and assembly
- End-to-end pipeline execution
- Main script functionality
- Error handling and robustness

The test suite ensures the complete Task 4.3 pipeline assembly works correctly
and all components are properly integrated.
"""

import pytest
import torch
import tempfile
import os
import subprocess
import time
from typing import Dict, Any
from unittest.mock import patch, MagicMock

# Import components to test
from applications.poc_exoplanet.config_exo import get_config, create_output_directories
from applications.poc_exoplanet.encoders.light_curve_encoder import LightCurveEncoder
from applications.poc_exoplanet.decoders.exoplanet_decoder import ExoplanetDecoder
from gif_framework.core.du_core import DU_Core_V1
from gif_framework.core.memory_systems import EpisodicMemory
from gif_framework.orchestrator import GIF
from gif_framework.training.trainer import Continual_Trainer
from data_generators.exoplanet_generator import RealisticExoplanetGenerator
from simulators.neuromorphic_sim import NeuromorphicSimulator


class TestConfigurationManagement:
    """Test suite for configuration loading and validation."""

    def test_config_loading(self):
        """Test that configuration loads successfully."""
        config = get_config()
        
        # Verify main sections exist
        assert "data" in config
        assert "architecture" in config
        assert "encoder" in config
        assert "decoder" in config
        assert "training" in config
        assert "simulation" in config
        assert "analysis" in config
        assert "metadata" in config

    def test_config_structure_validation(self):
        """Test that configuration has all required parameters."""
        config = get_config()
        
        # Data configuration
        assert config["data"]["num_training_samples"] > 0
        assert config["data"]["num_test_samples"] > 0
        
        # Architecture configuration
        assert config["architecture"]["input_size"] == 2  # Delta modulation channels
        assert len(config["architecture"]["hidden_sizes"]) > 0
        assert config["architecture"]["output_size"] > 0
        assert config["architecture"]["memory_capacity"] > 0
        
        # Training configuration
        assert config["training"]["learning_rate"] > 0
        assert config["training"]["batch_size"] > 0
        assert config["training"]["num_epochs"] > 0

    def test_output_directory_creation(self):
        """Test that output directories are created correctly."""
        config = get_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Modify config to use temporary directory
            config["analysis"]["results_dir"] = os.path.join(temp_dir, "results")
            config["analysis"]["logs_dir"] = os.path.join(temp_dir, "logs")
            config["analysis"]["plots_dir"] = os.path.join(temp_dir, "plots")
            
            create_output_directories(config)
            
            # Verify directories were created
            assert os.path.exists(config["analysis"]["results_dir"])
            assert os.path.exists(config["analysis"]["logs_dir"])
            assert os.path.exists(config["analysis"]["plots_dir"])


class TestComponentInitialization:
    """Test suite for individual component initialization."""

    @pytest.fixture
    def config(self):
        """Provide configuration for tests."""
        return get_config()

    def test_encoder_initialization(self, config):
        """Test encoder initialization with configuration."""
        encoder = LightCurveEncoder(threshold=config["encoder"]["threshold"])
        
        assert encoder.threshold == config["encoder"]["threshold"]
        assert hasattr(encoder, 'encode')
        assert hasattr(encoder, 'get_config')
        assert hasattr(encoder, 'calibrate')

    def test_decoder_initialization(self, config):
        """Test decoder initialization with configuration."""
        decoder = ExoplanetDecoder(
            input_size=config["architecture"]["output_size"],
            output_size=config["decoder"]["regression_output_size"]
        )
        
        assert decoder.input_size == config["architecture"]["output_size"]
        assert decoder.output_size == config["decoder"]["regression_output_size"]
        assert hasattr(decoder, 'decode')
        assert hasattr(decoder, 'get_config')

    def test_du_core_initialization_with_memory(self, config):
        """Test DU Core initialization with memory system."""
        memory = EpisodicMemory(capacity=config["architecture"]["memory_capacity"])

        du_core = DU_Core_V1(
            input_size=config["architecture"]["input_size"],
            hidden_sizes=config["architecture"]["hidden_sizes"],
            output_size=config["architecture"]["output_size"],
            memory_system=memory
        )

        assert du_core.input_size == config["architecture"]["input_size"]
        assert du_core.hidden_sizes == config["architecture"]["hidden_sizes"]
        assert du_core.output_size == config["architecture"]["output_size"]
        assert du_core._memory_system is memory  # Memory system is stored as private attribute

    def test_gif_orchestrator_assembly(self, config):
        """Test GIF orchestrator assembly with components."""
        # Initialize components
        memory = EpisodicMemory(capacity=config["architecture"]["memory_capacity"])
        encoder = LightCurveEncoder(threshold=config["encoder"]["threshold"])
        decoder = ExoplanetDecoder(
            input_size=config["architecture"]["output_size"],
            output_size=config["decoder"]["regression_output_size"]
        )
        du_core = DU_Core_V1(
            input_size=config["architecture"]["input_size"],
            hidden_sizes=config["architecture"]["hidden_sizes"],
            output_size=config["architecture"]["output_size"],
            memory_system=memory
        )
        
        # Assemble GIF model
        gif_model = GIF(du_core=du_core)
        gif_model.attach_encoder(encoder)
        gif_model.attach_decoder(decoder)
        
        assert gif_model._du_core is du_core  # DU Core is stored as private attribute
        assert gif_model._encoder is encoder  # Encoder is stored as private attribute
        assert gif_model._decoder is decoder  # Decoder is stored as private attribute

    def test_trainer_initialization(self, config):
        """Test trainer initialization with GIF model."""
        # Initialize components
        memory = EpisodicMemory(capacity=config["architecture"]["memory_capacity"])
        encoder = LightCurveEncoder(threshold=config["encoder"]["threshold"])
        decoder = ExoplanetDecoder(
            input_size=config["architecture"]["output_size"],
            output_size=config["decoder"]["regression_output_size"]
        )
        du_core = DU_Core_V1(
            input_size=config["architecture"]["input_size"],
            hidden_sizes=config["architecture"]["hidden_sizes"],
            output_size=config["architecture"]["output_size"],
            memory_system=memory
        )
        gif_model = GIF(du_core=du_core)
        gif_model.attach_encoder(encoder)
        gif_model.attach_decoder(decoder)
        
        # Initialize trainer
        optimizer = torch.optim.Adam(gif_model.parameters(), lr=config["training"]["learning_rate"])
        loss_fn = torch.nn.CrossEntropyLoss()
        trainer = Continual_Trainer(
            gif_model, 
            optimizer, 
            loss_fn, 
            memory_batch_size=config["training"]["memory_batch_size"]
        )
        
        assert trainer.gif_model is gif_model  # Trainer stores gif_model, not model
        assert trainer.optimizer is optimizer
        assert trainer.criterion is loss_fn  # Trainer stores criterion, not loss_fn

    def test_simulator_initialization(self, config):
        """Test neuromorphic simulator initialization."""
        memory = EpisodicMemory(capacity=config["architecture"]["memory_capacity"])
        du_core = DU_Core_V1(
            input_size=config["architecture"]["input_size"],
            hidden_sizes=config["architecture"]["hidden_sizes"],
            output_size=config["architecture"]["output_size"],
            memory_system=memory
        )
        
        simulator = NeuromorphicSimulator(
            snn_model=du_core,
            energy_per_synop=config["simulation"]["energy_per_synop"]
        )
        
        assert simulator.snn_model is du_core
        assert simulator.energy_per_synop == config["simulation"]["energy_per_synop"]


class TestEndToEndPipeline:
    """Test suite for end-to-end pipeline execution."""

    @pytest.fixture
    def complete_pipeline(self):
        """Set up complete pipeline for testing."""
        config = get_config()

        # Initialize components
        memory = EpisodicMemory(capacity=config["architecture"]["memory_capacity"])
        generator = RealisticExoplanetGenerator(seed=42)
        encoder = LightCurveEncoder(threshold=config["encoder"]["threshold"])
        decoder = ExoplanetDecoder(
            input_size=config["architecture"]["output_size"],
            output_size=config["decoder"]["regression_output_size"]
        )
        du_core = DU_Core_V1(
            input_size=config["architecture"]["input_size"],
            hidden_sizes=config["architecture"]["hidden_sizes"],
            output_size=config["architecture"]["output_size"],
            memory_system=memory
        )

        # Assemble GIF model
        gif_model = GIF(du_core=du_core)
        gif_model.attach_encoder(encoder)
        gif_model.attach_decoder(decoder)

        # Initialize trainer and simulator
        optimizer = torch.optim.Adam(gif_model.parameters(), lr=config["training"]["learning_rate"])
        loss_fn = torch.nn.CrossEntropyLoss()
        trainer = Continual_Trainer(
            gif_model,
            optimizer,
            loss_fn,
            memory_batch_size=config["training"]["memory_batch_size"]
        )
        simulator = NeuromorphicSimulator(
            snn_model=du_core,
            energy_per_synop=config["simulation"]["energy_per_synop"]
        )

        return {
            'config': config,
            'generator': generator,
            'encoder': encoder,
            'decoder': decoder,
            'du_core': du_core,
            'gif_model': gif_model,
            'trainer': trainer,
            'simulator': simulator
        }

    def test_single_training_step(self, complete_pipeline):
        """Test single training step execution."""
        components = complete_pipeline

        # Generate sample data
        sample_data = components['generator'].generate()
        target = torch.tensor([1])  # Planet detected
        task_id = 'test_task'

        # Execute training step
        loss = components['trainer'].train_step(sample_data, target, task_id)

        # Verify training step completed successfully
        assert isinstance(loss, float)
        assert loss >= 0  # Loss should be non-negative

    def test_neuromorphic_simulation(self, complete_pipeline):
        """Test neuromorphic simulation execution."""
        components = complete_pipeline

        # Generate and encode test data
        test_data = components['generator'].generate()
        encoded_data = components['encoder'].encode(test_data)

        # Run simulation
        output_spikes, sim_stats = components['simulator'].run(encoded_data.unsqueeze(1))

        # Verify simulation results
        assert isinstance(output_spikes, torch.Tensor)
        assert output_spikes.shape[0] == encoded_data.shape[0]  # Same time steps
        assert output_spikes.shape[2] == components['config']['architecture']['output_size']

        # Verify statistics
        assert 'total_spikes' in sim_stats
        assert 'total_synops' in sim_stats
        assert 'estimated_energy_joules' in sim_stats
        assert sim_stats['total_spikes'] >= 0
        assert sim_stats['total_synops'] >= 0
        assert sim_stats['estimated_energy_joules'] >= 0

    def test_complete_data_flow(self, complete_pipeline):
        """Test complete data flow through entire pipeline."""
        components = complete_pipeline

        # Step 1: Generate data
        light_curve_data = components['generator'].generate()
        assert len(light_curve_data) > 0

        # Step 2: Encode to spikes
        spike_train = components['encoder'].encode(light_curve_data)
        assert spike_train.shape[1] == 2  # Two channels

        # Step 3: Process through DU Core
        du_output = components['du_core'].forward(
            spike_train.unsqueeze(1),
            num_steps=spike_train.shape[0]
        )
        assert du_output.shape[-1] == components['config']['architecture']['output_size']

        # Step 4: Decode results
        classification = components['decoder'].decode(du_output.squeeze(1), mode='classification')
        regression = components['decoder'].decode(du_output.squeeze(1), mode='regression')

        assert classification in [0, 1]  # Binary classification
        assert isinstance(regression, float)  # Regression returns float for single output

    def test_pipeline_performance_metrics(self, complete_pipeline):
        """Test pipeline performance and timing."""
        components = complete_pipeline

        # Measure encoding performance
        test_data = components['generator'].generate()
        start_time = time.time()
        spike_train = components['encoder'].encode(test_data)
        encoding_time = time.time() - start_time

        # Measure simulation performance
        start_time = time.time()
        output_spikes, sim_stats = components['simulator'].run(spike_train.unsqueeze(1))
        simulation_time = time.time() - start_time

        # Verify reasonable performance
        assert encoding_time < 1.0  # Should complete in under 1 second
        assert simulation_time < 5.0  # Should complete in under 5 seconds

        # Verify energy efficiency
        energy_per_spike = sim_stats['estimated_energy_joules'] / max(sim_stats['total_spikes'], 1)
        assert energy_per_spike < 1e-6  # Should be very energy efficient

    def test_error_handling_robustness(self, complete_pipeline):
        """Test error handling and robustness."""
        components = complete_pipeline

        # Test with invalid data
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            components['encoder'].encode(None)

        # Test with empty data - encoder handles this gracefully with expected warning
        import polars as pl
        import warnings

        empty_data = pl.DataFrame({'time': [], 'flux': []})
        # Suppress the expected warning for cleaner test output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = components['encoder'].encode(empty_data)
        assert result.shape[0] == 0  # Should return empty spike train

        # Test decoder with invalid input
        with pytest.raises((ValueError, TypeError)):
            components['decoder'].decode(torch.tensor([]), mode='classification')


class TestMainScriptFunctionality:
    """Test suite for main script functionality."""

    def test_main_script_imports(self):
        """Test that main script imports work correctly."""
        try:
            import applications.poc_exoplanet.main_exo
            assert True  # Import successful
        except ImportError as e:
            pytest.fail(f"Main script import failed: {e}")

    def test_main_script_startup(self):
        """Test that main script can start without errors."""
        # Test script startup for a few seconds
        proc = subprocess.Popen(
            ['python3', 'applications/poc_exoplanet/main_exo.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Let it run for a few seconds
        time.sleep(3)

        # Terminate and check for errors
        proc.terminate()
        stdout, stderr = proc.communicate()

        # Should start without immediate errors
        assert "Traceback" not in stderr
        assert "Error" not in stderr
        assert "EXOPLANET DETECTION PROOF-OF-CONCEPT EXPERIMENT" in stdout
