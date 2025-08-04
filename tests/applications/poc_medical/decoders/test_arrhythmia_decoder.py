"""
Comprehensive Test Suite for Arrhythmia_Decoder
===============================================

This module provides exhaustive testing for the Arrhythmia_Decoder class, validating
all aspects of spike train decoding, classification, and neural network functionality.

Test Categories:
- Interface compliance and inheritance
- Neural network architecture validation
- Spike integration and population decoding
- Classification accuracy and probability output
- Training mode functionality
- Error handling and edge cases
- Configuration and statistics
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

from applications.poc_medical.decoders.arrhythmia_decoder import Arrhythmia_Decoder
from gif_framework.interfaces.base_interfaces import DecoderInterface


class TestArrhythmiaDecoderInterface:
    """Test Arrhythmia_Decoder interface compliance and basic functionality."""
    
    def test_inheritance(self):
        """Test that Arrhythmia_Decoder properly inherits from required interfaces."""
        decoder = Arrhythmia_Decoder(input_size=10)
        assert isinstance(decoder, DecoderInterface)
        assert isinstance(decoder, nn.Module)
        assert hasattr(decoder, 'decode')
        assert hasattr(decoder, 'get_config')
        assert hasattr(decoder, 'forward')
    
    def test_initialization_default_params(self):
        """Test decoder initialization with default parameters."""
        decoder = Arrhythmia_Decoder(input_size=100)
        assert decoder.input_size == 100
        assert decoder.num_classes == 8
        assert decoder.dropout_rate == 0.1
        assert len(decoder.class_names) == 8
        assert decoder.class_names[0] == "Normal Sinus Rhythm"
        assert decoder.class_names[1] == "Atrial Fibrillation"
    
    def test_initialization_custom_params(self):
        """Test decoder initialization with custom parameters."""
        custom_classes = ["Class1", "Class2", "Class3"]
        decoder = Arrhythmia_Decoder(
            input_size=50,
            num_classes=3,
            class_names=custom_classes,
            dropout_rate=0.2
        )
        assert decoder.input_size == 50
        assert decoder.num_classes == 3
        assert decoder.class_names == custom_classes
        assert decoder.dropout_rate == 0.2
    
    def test_initialization_validation(self):
        """Test parameter validation during initialization."""
        with pytest.raises(ValueError, match="Input size must be positive"):
            Arrhythmia_Decoder(input_size=0)
        
        with pytest.raises(ValueError, match="Number of classes must be > 1"):
            Arrhythmia_Decoder(input_size=10, num_classes=1)
        
        with pytest.raises(ValueError, match="Dropout rate must be in"):
            Arrhythmia_Decoder(input_size=10, dropout_rate=1.5)
        
        with pytest.raises(ValueError, match="Number of class names"):
            Arrhythmia_Decoder(input_size=10, num_classes=3, class_names=["A", "B"])


class TestArrhythmiaDecoderArchitecture:
    """Test neural network architecture and components."""
    
    @pytest.fixture
    def decoder(self):
        """Create decoder for testing."""
        return Arrhythmia_Decoder(input_size=20, num_classes=4)
    
    def test_linear_layer_architecture(self, decoder):
        """Test linear readout layer architecture."""
        assert hasattr(decoder, 'linear_readout')
        assert isinstance(decoder.linear_readout, nn.Linear)
        assert decoder.linear_readout.in_features == 20
        assert decoder.linear_readout.out_features == 4
        assert decoder.linear_readout.bias is not None
    
    def test_dropout_layer(self, decoder):
        """Test dropout layer configuration."""
        assert hasattr(decoder, 'dropout')
        assert isinstance(decoder.dropout, nn.Dropout)
        assert decoder.dropout.p == 0.1
    
    def test_weight_initialization(self, decoder):
        """Test that weights are properly initialized."""
        # Check that weights are not all zeros (indicating proper initialization)
        weights = decoder.linear_readout.weight.data
        assert not torch.allclose(weights, torch.zeros_like(weights))
        
        # Check bias initialization
        bias = decoder.linear_readout.bias.data
        assert torch.allclose(bias, torch.zeros_like(bias))
    
    def test_parameter_count(self, decoder):
        """Test parameter counting functionality."""
        config = decoder.get_config()
        total_params = config['model_parameters']['total_parameters']
        trainable_params = config['model_parameters']['trainable_parameters']
        
        # Should have weights (20*4) + bias (4) = 84 parameters
        expected_params = 20 * 4 + 4
        assert total_params == expected_params
        assert trainable_params == expected_params


class TestArrhythmiaDecoderSpikeIntegration:
    """Test spike integration and validation functionality."""
    
    @pytest.fixture
    def decoder(self):
        return Arrhythmia_Decoder(input_size=10, num_classes=3)
    
    def test_spike_integration_2d(self, decoder):
        """Test spike integration for 2D input (single sample)."""
        # Create test spike train: [time_steps, neurons]
        spike_train = torch.rand(100, 10)  # 100 time steps, 10 neurons
        spike_counts = decoder._integrate_spikes(spike_train)
        
        assert spike_counts.shape == (10,)  # Should sum over time dimension
        assert torch.allclose(spike_counts, torch.sum(spike_train, dim=0))
    
    def test_spike_integration_3d(self, decoder):
        """Test spike integration for 3D input (batch)."""
        # Create test spike train: [time_steps, batch_size, neurons]
        spike_train = torch.rand(50, 5, 10)  # 50 time steps, 5 batch, 10 neurons
        spike_counts = decoder._integrate_spikes(spike_train)
        
        assert spike_counts.shape == (5, 10)  # Should sum over time, keep batch
        assert torch.allclose(spike_counts, torch.sum(spike_train, dim=0))
    
    def test_spike_validation_numpy(self, decoder):
        """Test spike train validation with numpy input."""
        spike_train_np = np.random.rand(50, 10)
        validated = decoder._validate_spike_train(spike_train_np)
        
        assert isinstance(validated, torch.Tensor)
        assert validated.shape == (50, 10)
        assert validated.dtype == torch.float32
    
    def test_spike_validation_list(self, decoder):
        """Test spike train validation with list input."""
        spike_train_list = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        validated = decoder._validate_spike_train(spike_train_list)
        
        assert isinstance(validated, torch.Tensor)
        assert validated.shape == (3, 2)
    
    def test_spike_validation_errors(self, decoder):
        """Test spike train validation error handling."""
        # Invalid type
        with pytest.raises(ValueError, match="Unsupported spike train type"):
            decoder._validate_spike_train("invalid")
        
        # Invalid dimensions
        with pytest.raises(ValueError, match="Spike train must be 2D or 3D"):
            decoder._validate_spike_train(torch.rand(10))
        
        # Non-finite values
        spike_train = torch.tensor([[1.0, float('nan')], [0.0, 1.0]])
        with pytest.raises(ValueError, match="non-finite values"):
            decoder._validate_spike_train(spike_train)
    
    def test_negative_spike_handling(self, decoder):
        """Test handling of negative spike values."""
        spike_train = torch.tensor([[-1.0, 1.0], [0.5, -0.5]])
        
        with pytest.warns(UserWarning, match="negative values"):
            validated = decoder._validate_spike_train(spike_train)
        
        # Should clamp negative values to zero
        assert torch.all(validated >= 0)
    
    def test_feature_size_mismatch(self, decoder):
        """Test handling of feature size mismatch."""
        # Decoder expects 10 features, but provide 5
        spike_train = torch.rand(50, 5)
        
        with pytest.raises(ValueError, match="Expected 10 input features, got 5"):
            decoder._integrate_spikes(spike_train)


class TestArrhythmiaDecoderClassification:
    """Test classification functionality and output."""
    
    @pytest.fixture
    def decoder(self):
        return Arrhythmia_Decoder(input_size=10, num_classes=4)
    
    def test_decode_single_sample(self, decoder):
        """Test decoding single spike train sample."""
        spike_train = torch.rand(50, 10)  # 50 time steps, 10 neurons
        prediction = decoder.decode(spike_train)
        
        assert isinstance(prediction, str)
        assert prediction in decoder.class_names
    
    def test_decode_deterministic(self, decoder):
        """Test that decoding is deterministic for same input."""
        spike_train = torch.rand(50, 10)
        
        prediction1 = decoder.decode(spike_train)
        prediction2 = decoder.decode(spike_train)
        
        assert prediction1 == prediction2
    
    def test_forward_training_mode(self, decoder):
        """Test forward pass in training mode."""
        decoder.train()
        spike_train = torch.rand(20, 5, 10)  # Batch of 5 samples
        
        logits = decoder.forward(spike_train)
        
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (5, 4)  # Batch size 5, 4 classes
        assert logits.requires_grad  # Should support gradients
    
    def test_forward_eval_mode(self, decoder):
        """Test forward pass in evaluation mode."""
        decoder.eval()
        spike_train = torch.rand(20, 5, 10)
        
        with torch.no_grad():
            logits = decoder.forward(spike_train)
        
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (5, 4)
    
    def test_class_probabilities(self, decoder):
        """Test class probability output."""
        spike_train = torch.rand(50, 10)
        prob_dict = decoder.get_class_probabilities(spike_train)
        
        assert isinstance(prob_dict, dict)
        assert len(prob_dict) == 4
        assert all(name in prob_dict for name in decoder.class_names)
        
        # Probabilities should sum to 1
        total_prob = sum(prob_dict.values())
        assert abs(total_prob - 1.0) < 1e-6
        
        # All probabilities should be non-negative
        assert all(prob >= 0 for prob in prob_dict.values())
    
    def test_top_k_predictions(self, decoder):
        """Test top-k prediction functionality."""
        spike_train = torch.rand(50, 10)
        top_3 = decoder.get_top_k_predictions(spike_train, k=3)
        
        assert isinstance(top_3, list)
        assert len(top_3) == 3
        
        # Should be sorted by probability (descending)
        probs = [prob for _, prob in top_3]
        assert probs == sorted(probs, reverse=True)
        
        # All should be valid class names
        for class_name, prob in top_3:
            assert class_name in decoder.class_names
            assert 0 <= prob <= 1


class TestArrhythmiaDecoderStatistics:
    """Test statistics tracking and configuration."""
    
    @pytest.fixture
    def decoder(self):
        return Arrhythmia_Decoder(input_size=10, num_classes=3)
    
    def test_statistics_initialization(self, decoder):
        """Test that statistics are properly initialized."""
        stats = decoder.decoding_stats
        assert stats['total_decodings'] == 0
        assert stats['average_confidence'] == 0.0
        assert stats['average_spike_count'] == 0.0
        assert len(stats['class_predictions']) == 3
    
    def test_statistics_update(self, decoder):
        """Test that statistics are updated during decoding."""
        spike_train = torch.rand(50, 10)
        
        initial_decodings = decoder.decoding_stats['total_decodings']
        prediction = decoder.decode(spike_train)
        
        assert decoder.decoding_stats['total_decodings'] == initial_decodings + 1
        assert decoder.decoding_stats['class_predictions'][prediction] == 1
        assert decoder.decoding_stats['average_confidence'] > 0
        assert decoder.decoding_stats['average_spike_count'] >= 0
    
    def test_statistics_reset(self, decoder):
        """Test statistics reset functionality."""
        # Generate some statistics
        spike_train = torch.rand(50, 10)
        decoder.decode(spike_train)
        
        # Reset and verify
        decoder.reset_stats()
        stats = decoder.decoding_stats
        
        assert stats['total_decodings'] == 0
        assert stats['average_confidence'] == 0.0
        assert stats['average_spike_count'] == 0.0
        assert all(count == 0 for count in stats['class_predictions'].values())
    
    def test_get_config(self, decoder):
        """Test configuration retrieval."""
        config = decoder.get_config()
        
        assert isinstance(config, dict)
        assert config['decoder_type'] == 'Arrhythmia_Decoder'
        assert config['input_size'] == 10
        assert config['num_classes'] == 3
        assert config['dropout_rate'] == 0.1
        assert 'class_names' in config
        assert 'decoding_statistics' in config
        assert 'model_parameters' in config


class TestArrhythmiaDecoderTraining:
    """Test training-related functionality."""
    
    @pytest.fixture
    def decoder(self):
        return Arrhythmia_Decoder(input_size=10, num_classes=3)
    
    def test_gradient_flow(self, decoder):
        """Test that gradients flow properly through the decoder."""
        decoder.train()
        spike_train = torch.rand(20, 5, 10, requires_grad=True)
        target = torch.randint(0, 3, (5,))
        
        logits = decoder.forward(spike_train)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        
        # Check that gradients exist
        assert decoder.linear_readout.weight.grad is not None
        assert decoder.linear_readout.bias.grad is not None
        
        # Gradients should not be all zeros
        assert not torch.allclose(decoder.linear_readout.weight.grad, 
                                torch.zeros_like(decoder.linear_readout.weight.grad))
    
    def test_dropout_behavior(self, decoder):
        """Test dropout behavior in training vs evaluation mode."""
        spike_train = torch.rand(20, 10)
        
        # In training mode, dropout should be active
        decoder.train()
        output1 = decoder.forward(spike_train)
        output2 = decoder.forward(spike_train)
        
        # Outputs might be different due to dropout (though not guaranteed)
        # At minimum, dropout layer should be in training mode
        assert decoder.dropout.training
        
        # In eval mode, dropout should be inactive
        decoder.eval()
        output3 = decoder.forward(spike_train)
        output4 = decoder.forward(spike_train)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output3, output4)
        assert not decoder.dropout.training
    
    def test_parameter_updates(self, decoder):
        """Test that parameters can be updated during training."""
        original_weight = decoder.linear_readout.weight.data.clone()
        
        # Simulate training step
        decoder.train()
        spike_train = torch.rand(20, 5, 10)
        target = torch.randint(0, 3, (5,))
        
        optimizer = torch.optim.SGD(decoder.parameters(), lr=0.1)
        
        logits = decoder.forward(spike_train)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        optimizer.step()
        
        # Weights should have changed
        assert not torch.allclose(original_weight, decoder.linear_readout.weight.data)
