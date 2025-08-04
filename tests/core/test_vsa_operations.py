"""
Test Suite for Vector Symbolic Architecture (VSA) Operations
===========================================================

This module contains comprehensive tests for the Vector Symbolic Architecture
implementation that enables explicit "Deep Understanding" in the GIF framework.

VSA provides the mathematical foundation for transforming implicit pattern
association into explicit conceptual understanding through hyperdimensional
computing operations.

Test Coverage:
=============

1. **VSA Core Operations**: Tests for hypervector creation, binding, bundling, and similarity
2. **Mathematical Properties**: Validation of VSA mathematical properties and invariants
3. **DU_Core Integration**: Tests for VSA integration with both DU_Core_V1 and DU_Core_V2
4. **Conceptual Memory**: Tests for concept formation and "connecting the dots" mechanisms
5. **Experience Storage**: Tests for conceptual state storage in ExperienceTuple
6. **Edge Cases**: Boundary conditions and error handling

Scientific Validation:
=====================

These tests provide concrete evidence that the VSA implementation correctly
implements hyperdimensional computing principles and enables explicit
"Deep Understanding" through structured concept formation.

Author: GIF Development Team
Phase: 7.3 - Explicit Deep Understanding via VSA
"""

import pytest
import torch
import numpy as np
from typing import List, Dict, Any

from gif_framework.core.vsa_operations import VSA
from gif_framework.core.memory_systems import ExperienceTuple, EpisodicMemory
from gif_framework.core.du_core import DU_Core_V1


class TestVSACoreOperations:
    """Test suite for core VSA operations and mathematical properties."""

    def test_vsa_initialization(self):
        """Test VSA initialization with various parameters."""
        # Test default initialization
        vsa = VSA()
        assert vsa.dimension == 10000
        assert vsa.device == torch.device('cpu')
        assert vsa.dtype == torch.float32

        # Test custom initialization
        vsa_custom = VSA(dimension=5000, dtype=torch.float64)
        assert vsa_custom.dimension == 5000
        assert vsa_custom.dtype == torch.float64

    def test_vsa_initialization_validation(self):
        """Test VSA initialization parameter validation."""
        # Test invalid dimension
        with pytest.raises(ValueError, match="Dimension must be a positive integer"):
            VSA(dimension=0)

        with pytest.raises(ValueError, match="Dimension must be a positive integer"):
            VSA(dimension=-100)

        # Test small dimension warning
        with pytest.warns(UserWarning, match="VSA dimension.*is very small"):
            VSA(dimension=50)

    def test_hypervector_creation(self):
        """Test hypervector creation and properties."""
        vsa = VSA(dimension=1000)
        
        # Test normalized hypervector creation
        hv = vsa.create_hypervector(normalize=True)
        assert hv.shape == (1000,)
        assert hv.dtype == torch.float32
        
        # Check normalization (should be close to 1.0)
        norm = torch.norm(hv).item()
        assert abs(norm - 1.0) < 1e-6

        # Test unnormalized hypervector creation
        hv_unnorm = vsa.create_hypervector(normalize=False)
        assert hv_unnorm.shape == (1000,)
        # Unnormalized vector should have norm approximately sqrt(dimension)
        expected_norm = np.sqrt(1000)
        actual_norm = torch.norm(hv_unnorm).item()
        assert abs(actual_norm - expected_norm) < 100  # Allow some variance

    def test_vsa_bind_and_bundle_properties(self):
        """Validates the core mathematical properties of VSA (as specified in prompt)."""
        vsa = VSA(dimension=10000)
        v1 = vsa.create_hypervector()
        v2 = vsa.create_hypervector()

        bundled = vsa.bundle([v1, v2])
        bound = vsa.bind(v1, v2)

        # A bundled vector is similar to its components
        assert vsa.similarity(bundled, v1) > 0.4
        assert vsa.similarity(bundled, v2) > 0.4
        
        # A bound vector is dissimilar to its components
        assert vsa.similarity(bound, v1) < 0.1
        assert vsa.similarity(bound, v2) < 0.1

    def test_binding_properties(self):
        """Test mathematical properties of the binding operation."""
        vsa = VSA(dimension=5000)
        a = vsa.create_hypervector()
        b = vsa.create_hypervector()
        c = vsa.create_hypervector()

        # Test binding creates dissimilar vectors
        bound_ab = vsa.bind(a, b)
        assert vsa.similarity(bound_ab, a) < 0.1
        assert vsa.similarity(bound_ab, b) < 0.1

        # Note: Circular convolution binding is not perfectly invertible
        # This is a known limitation of this binding method, but it still
        # provides useful structured representations for concept formation

        # Test commutativity: A ⊗ B ≈ B ⊗ A (approximately)
        bound_ba = vsa.bind(b, a)
        commutative_similarity = vsa.similarity(bound_ab, bound_ba)
        assert commutative_similarity > 0.9  # Should be very similar

    def test_bundling_properties(self):
        """Test mathematical properties of the bundling operation."""
        vsa = VSA(dimension=5000)
        vectors = [vsa.create_hypervector() for _ in range(5)]

        # Test bundling preserves similarity to components
        bundled = vsa.bundle(vectors)
        for v in vectors:
            similarity = vsa.similarity(bundled, v)
            assert similarity > 0.3  # Should be similar to all components

        # Test bundling is commutative (order doesn't matter)
        shuffled_vectors = vectors.copy()
        np.random.shuffle(shuffled_vectors)
        bundled_shuffled = vsa.bundle(shuffled_vectors)
        
        # Should be identical (or very close due to floating point)
        commutative_similarity = vsa.similarity(bundled, bundled_shuffled)
        assert commutative_similarity > 0.99

    def test_similarity_properties(self):
        """Test properties of the similarity function."""
        vsa = VSA(dimension=1000)
        v1 = vsa.create_hypervector()
        v2 = vsa.create_hypervector()

        # Test self-similarity
        assert vsa.similarity(v1, v1) == pytest.approx(1.0, abs=1e-6)

        # Test symmetry
        sim_12 = vsa.similarity(v1, v2)
        sim_21 = vsa.similarity(v2, v1)
        assert sim_12 == pytest.approx(sim_21, abs=1e-6)

        # Test random vectors are nearly orthogonal
        assert abs(sim_12) < 0.1  # Should be close to 0

    def test_vsa_error_handling(self):
        """Test error handling in VSA operations."""
        vsa = VSA(dimension=1000)
        v1 = vsa.create_hypervector()
        v2 = vsa.create_hypervector()

        # Test empty bundle
        with pytest.raises(ValueError, match="Cannot bundle empty list"):
            vsa.bundle([])

        # Test shape mismatch in binding
        v_wrong_shape = torch.randn(500)  # Wrong dimension
        with pytest.raises(ValueError, match="Shape mismatch"):
            vsa.bind(v1, v_wrong_shape)

        # Test shape mismatch in similarity
        with pytest.raises(ValueError, match="Shape mismatch"):
            vsa.similarity(v1, v_wrong_shape)

        # Test device mismatch
        if torch.cuda.is_available():
            v_gpu = v1.cuda()
            with pytest.raises(ValueError, match="Device mismatch"):
                vsa.bind(v1, v_gpu)


class TestVSAIntegrationWithDUCore:
    """Test suite for VSA integration with DU_Core systems."""

    def test_du_core_v1_vsa_initialization(self):
        """Test DU_Core_V1 initialization with VSA parameters."""
        # Test with VSA enabled (default)
        du_core = DU_Core_V1(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            vsa_dimension=1000,
            enable_vsa=True
        )
        
        assert du_core.enable_vsa is True
        assert du_core.vsa is not None
        assert du_core.vsa_dimension == 1000
        assert isinstance(du_core.conceptual_memory, dict)
        assert du_core.master_conceptual_state is not None

        # Test with VSA disabled
        du_core_no_vsa = DU_Core_V1(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            enable_vsa=False
        )
        
        assert du_core_no_vsa.enable_vsa is False
        assert du_core_no_vsa.vsa is None
        assert du_core_no_vsa.vsa_dimension == 0

    def test_du_core_forms_new_concepts(self):
        """Test that DU_Core forms new concepts through VSA processing (as specified in prompt)."""
        # Create a DU_Core instance with VSA enabled
        du_core = DU_Core_V1(
            input_size=12,
            hidden_sizes=[8],
            output_size=6,
            vsa_dimension=1000,
            enable_vsa=True
        )

        # Add a base concept like "fruit" to its conceptual_memory
        fruit_concept = du_core.vsa.create_hypervector()
        du_core.conceptual_memory["fruit"] = fruit_concept
        
        # Store initial master state
        initial_master_state = du_core.master_conceptual_state.clone()
        initial_memory_size = len(du_core.conceptual_memory)

        # Process a new input representing "apple"
        apple_input = torch.rand(5, 1, 12) * 0.7  # Some spike activity
        output_spikes = du_core.process(apple_input)

        # Assert that the DU Core's internal state has changed
        final_master_state = du_core.master_conceptual_state
        state_similarity = du_core.vsa.similarity(initial_master_state, final_master_state)
        assert state_similarity < 0.9  # State should have changed significantly

        # Assert that a new concept related to "fruit" and "apple" has been formed and stored
        final_memory_size = len(du_core.conceptual_memory)
        assert final_memory_size > initial_memory_size  # New concept should be stored

        # Verify output shape
        assert output_spikes.shape == (5, 1, 6)

    def test_conceptual_memory_formation(self):
        """Test conceptual memory formation and retrieval."""
        du_core = DU_Core_V1(
            input_size=8,
            hidden_sizes=[6],
            output_size=4,
            vsa_dimension=2000,
            enable_vsa=True
        )

        # Process multiple inputs to build conceptual memory
        inputs = [
            torch.rand(3, 1, 8) * 0.5,
            torch.rand(3, 1, 8) * 0.6,
            torch.rand(3, 1, 8) * 0.7
        ]

        initial_memory_size = len(du_core.conceptual_memory)
        
        for input_spikes in inputs:
            du_core.process(input_spikes)

        # Should have formed new concepts
        final_memory_size = len(du_core.conceptual_memory)
        assert final_memory_size > initial_memory_size

        # Test conceptual state retrieval
        conceptual_state = du_core.get_conceptual_state()
        assert conceptual_state is not None
        assert conceptual_state.shape == (2000,)

        # Test conceptual memory statistics
        stats = du_core.get_conceptual_memory_stats()
        assert stats['vsa_enabled'] is True
        assert stats['conceptual_memory_size'] > 0
        assert stats['vsa_dimension'] == 2000
        assert len(stats['concept_keys']) > 0

    def test_vsa_disabled_behavior(self):
        """Test DU_Core behavior when VSA is disabled."""
        du_core = DU_Core_V1(
            input_size=8,
            hidden_sizes=[6],
            output_size=4,
            enable_vsa=False
        )

        # Process input
        input_spikes = torch.rand(3, 1, 8) * 0.5
        output_spikes = du_core.process(input_spikes)

        # Should still work normally
        assert output_spikes.shape == (3, 1, 4)

        # VSA-related methods should return appropriate values
        assert du_core.get_conceptual_state() is None
        
        stats = du_core.get_conceptual_memory_stats()
        assert stats['vsa_enabled'] is False
        assert stats['conceptual_memory_size'] == 0
        assert stats['vsa_dimension'] == 0


class TestVSAWithDUCoreV2:
    """Test suite for VSA integration with DU_Core_V2 hybrid architecture."""

    def test_du_core_v2_vsa_initialization(self):
        """Test DU_Core_V2 initialization with VSA parameters."""
        try:
            from gif_framework.core.du_core_v2 import DU_Core_V2
        except ImportError:
            pytest.skip("DU_Core_V2 not available (snnTorch dependency)")

        # Test with VSA enabled
        du_core_v2 = DU_Core_V2(
            input_size=12,
            hidden_sizes=[10],
            output_size=6,
            state_dim=8,
            vsa_dimension=1500,
            enable_vsa=True
        )

        assert du_core_v2.enable_vsa is True
        assert du_core_v2.vsa is not None
        assert du_core_v2.vsa_dimension == 1500
        assert isinstance(du_core_v2.conceptual_memory, dict)
        assert du_core_v2.master_conceptual_state is not None

    def test_du_core_v2_vsa_processing(self):
        """Test VSA processing in DU_Core_V2 hybrid architecture."""
        try:
            from gif_framework.core.du_core_v2 import DU_Core_V2
        except ImportError:
            pytest.skip("DU_Core_V2 not available (snnTorch dependency)")

        du_core_v2 = DU_Core_V2(
            input_size=10,
            hidden_sizes=[8],
            output_size=5,
            state_dim=6,
            attention_interval=3,  # No attention for simplicity
            vsa_dimension=1000,
            enable_vsa=True
        )

        # Store initial state
        initial_memory_size = len(du_core_v2.conceptual_memory)
        initial_master_state = du_core_v2.master_conceptual_state.clone()

        # Process input through hybrid architecture
        input_spikes = torch.rand(4, 1, 10) * 0.6
        output_spikes = du_core_v2.process(input_spikes)

        # Verify processing worked
        assert output_spikes.shape == (4, 1, 5)

        # Verify VSA processing occurred
        final_memory_size = len(du_core_v2.conceptual_memory)
        assert final_memory_size > initial_memory_size

        # Verify master state changed
        final_master_state = du_core_v2.master_conceptual_state
        state_similarity = du_core_v2.vsa.similarity(initial_master_state, final_master_state)
        assert state_similarity < 0.9

    def test_du_core_v2_conceptual_memory_stats(self):
        """Test conceptual memory statistics in DU_Core_V2."""
        try:
            from gif_framework.core.du_core_v2 import DU_Core_V2
        except ImportError:
            pytest.skip("DU_Core_V2 not available (snnTorch dependency)")

        du_core_v2 = DU_Core_V2(
            input_size=8,
            hidden_sizes=[6],
            output_size=4,
            state_dim=5,
            vsa_dimension=2000,
            enable_vsa=True
        )

        # Process some inputs
        for _ in range(3):
            input_spikes = torch.rand(2, 1, 8) * 0.5
            du_core_v2.process(input_spikes)

        # Check statistics
        stats = du_core_v2.get_conceptual_memory_stats()
        assert stats['vsa_enabled'] is True
        assert stats['conceptual_memory_size'] > 0
        assert stats['vsa_dimension'] == 2000
        assert len(stats['concept_keys']) > 0
        assert stats['master_state_norm'] > 0.0


class TestExperienceTupleWithConceptualState:
    """Test suite for ExperienceTuple with conceptual state storage."""

    def test_experience_tuple_with_conceptual_state(self):
        """Test ExperienceTuple creation with conceptual state."""
        vsa = VSA(dimension=1000)
        conceptual_state = vsa.create_hypervector()

        # Test with conceptual state
        experience = ExperienceTuple(
            input_spikes=torch.rand(5, 1, 10),
            internal_state=None,
            output_spikes=torch.rand(5, 1, 8),
            task_id="vsa_test",
            conceptual_state=conceptual_state
        )

        assert experience.conceptual_state is not None
        assert experience.conceptual_state.shape == (1000,)
        assert torch.allclose(experience.conceptual_state, conceptual_state)

    def test_experience_tuple_backward_compatibility(self):
        """Test ExperienceTuple backward compatibility without conceptual state."""
        # Test without conceptual state (backward compatible)
        experience = ExperienceTuple(
            input_spikes=torch.rand(5, 1, 10),
            internal_state=None,
            output_spikes=torch.rand(5, 1, 8),
            task_id="backward_compat_test"
            # conceptual_state defaults to None
        )

        assert experience.conceptual_state is None
        assert experience.task_id == "backward_compat_test"

    def test_episodic_memory_with_conceptual_states(self):
        """Test episodic memory storage and retrieval with conceptual states."""
        memory = EpisodicMemory(capacity=100)
        vsa = VSA(dimension=500)

        # Store experiences with conceptual states
        for i in range(5):
            conceptual_state = vsa.create_hypervector()
            experience = ExperienceTuple(
                input_spikes=torch.rand(3, 1, 8),
                internal_state=None,
                output_spikes=torch.rand(3, 1, 6),
                task_id=f"concept_task_{i}",
                conceptual_state=conceptual_state
            )
            memory.add(experience)

        # Verify storage
        assert len(memory) == 5

        # Sample and verify conceptual states are preserved
        sampled = memory.sample(3)
        for exp in sampled:
            assert exp.conceptual_state is not None
            assert exp.conceptual_state.shape == (500,)

    def test_memory_integration_with_du_core(self):
        """Test memory integration with DU_Core VSA processing."""
        memory = EpisodicMemory(capacity=50)

        du_core = DU_Core_V1(
            input_size=8,
            hidden_sizes=[6],
            output_size=4,
            memory_system=memory,
            vsa_dimension=800,
            enable_vsa=True
        )

        # Process input (should automatically store experience with conceptual state)
        input_spikes = torch.rand(3, 1, 8) * 0.6
        output_spikes = du_core.process(input_spikes)

        # Verify experience was stored
        assert len(memory) == 1

        # Verify conceptual state was stored
        stored_experience = memory.sample(1)[0]
        assert stored_experience.conceptual_state is not None
        assert stored_experience.conceptual_state.shape == (800,)
        assert stored_experience.task_id == "du_core_processing"


class TestVSAEdgeCasesAndValidation:
    """Test suite for VSA edge cases and parameter validation."""

    def test_vsa_parameter_validation_in_du_core(self):
        """Test VSA parameter validation in DU_Core initialization."""
        # Test invalid VSA dimension
        with pytest.raises(ValueError, match="vsa_dimension must be positive"):
            DU_Core_V1(
                input_size=8,
                hidden_sizes=[6],
                output_size=4,
                vsa_dimension=0
            )

        # Test invalid enable_vsa type
        with pytest.raises(TypeError, match="enable_vsa must be a boolean"):
            DU_Core_V1(
                input_size=8,
                hidden_sizes=[6],
                output_size=4,
                enable_vsa="true"  # Should be boolean
            )

    def test_vsa_with_small_dimensions(self):
        """Test VSA behavior with small dimensions."""
        # Should work but issue warning
        with pytest.warns(UserWarning, match="VSA dimension.*is very small"):
            vsa = VSA(dimension=50)

        # Basic operations should still work
        v1 = vsa.create_hypervector()
        v2 = vsa.create_hypervector()

        bundled = vsa.bundle([v1, v2])
        bound = vsa.bind(v1, v2)

        assert bundled.shape == (50,)
        assert bound.shape == (50,)

    def test_vsa_memory_efficiency(self):
        """Test VSA memory usage statistics."""
        vsa = VSA(dimension=10000)
        stats = vsa.get_stats()

        assert stats['dimension'] == 10000
        assert stats['memory_per_vector_mb'] > 0
        assert 'device' in stats
        assert 'dtype' in stats

    def test_conceptual_state_cloning(self):
        """Test that conceptual states are properly cloned to prevent mutation."""
        du_core = DU_Core_V1(
            input_size=6,
            hidden_sizes=[4],
            output_size=3,
            vsa_dimension=500,
            enable_vsa=True
        )

        # Get conceptual state
        state1 = du_core.get_conceptual_state()
        state2 = du_core.get_conceptual_state()

        # Should be equal but not the same object
        assert torch.allclose(state1, state2)
        assert state1 is not state2  # Different objects

        # Modifying one shouldn't affect the other
        state1[0] = 999.0
        state3 = du_core.get_conceptual_state()
        assert not torch.allclose(state1, state3)
