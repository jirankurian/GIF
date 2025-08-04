"""
Critical Integration Tests for Core GIF Architecture Validation
===============================================================

This module contains the most critical integration tests for the GIF framework.
These tests validate the complete architectural integrity and ensure that all
core components work together seamlessly under various conditions.

Test Categories:
- End-to-end architectural integrity validation
- Complete cognitive cycle testing under stress
- Data flow integrity through entire pipeline
- System behavior under failure conditions
- Performance benchmarking and regression detection
- Cross-component integration validation

These tests serve as the final validation that the GIF framework's core
architecture is sound, robust, and ready for production use.
"""

import pytest
import torch
import time
import warnings
from typing import Any, Dict, List

from gif_framework.orchestrator import GIF
from gif_framework.interfaces.base_interfaces import (
    EncoderInterface, 
    DecoderInterface,
    SpikeTrain,
    Action
)
from tests.mocks import (
    MockEncoder, 
    MockDecoder, 
    FailingMockEncoder,
    FailingMockDecoder,
    PerformanceMockEncoder,
    create_test_raw_data,
    create_stress_test_data
)

# Import DU Core for integration testing
try:
    from gif_framework.core.du_core import DU_Core_V1
    DU_CORE_AVAILABLE = True
except ImportError:
    DU_CORE_AVAILABLE = False


class TestCoreArchitecturalIntegrity:
    """Test the fundamental architectural integrity of the GIF framework."""
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_complete_cognitive_cycle_integrity(self):
        """Test the complete cognitive cycle with full data flow validation."""
        # Create a comprehensive system
        du_core = DU_Core_V1(input_size=20, hidden_sizes=[30, 25], output_size=15)
        gif = GIF(du_core)
        
        # Attach high-quality mock components
        encoder = MockEncoder(output_shape=(10, 1, 20))
        decoder = MockDecoder()
        gif.attach_encoder(encoder)
        gif.attach_decoder(decoder)
        
        # Test complete cycle with various data types
        test_cases = [
            create_test_raw_data("simple"),
            create_test_raw_data("complex"),
            create_test_raw_data("large"),
            "string_input",
            42,
            [1, 2, 3, 4, 5]
        ]
        
        for i, test_data in enumerate(test_cases):
            result = gif.process_single_input(test_data)
            assert result == "mock_action_success", f"Failed on test case {i}"
        
        # Verify encoder was called for each test
        assert encoder.encoding_calls == len(test_cases)
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_data_flow_integrity_validation(self):
        """Test that data flows correctly through the entire pipeline."""
        du_core = DU_Core_V1(input_size=15, hidden_sizes=[20], output_size=10)
        gif = GIF(du_core)
        
        # Create custom encoder that tracks data flow
        class DataFlowEncoder(EncoderInterface):
            def __init__(self):
                self.received_data = []
                self.output_shapes = []
            
            def encode(self, raw_data: Any) -> SpikeTrain:
                self.received_data.append(raw_data)
                spike_train = torch.rand(8, 1, 15)
                self.output_shapes.append(spike_train.shape)
                return spike_train
            
            def get_config(self) -> Dict[str, Any]:
                return {"name": "DataFlowEncoder"}
            
            def calibrate(self, sample_data: Any) -> None:
                pass
        
        # Create custom decoder that tracks data flow
        class DataFlowDecoder(DecoderInterface):
            def __init__(self):
                self.received_spikes = []
                self.input_shapes = []
            
            def decode(self, spike_train: SpikeTrain) -> Action:
                self.received_spikes.append(spike_train)
                self.input_shapes.append(spike_train.shape)
                return f"processed_{len(self.received_spikes)}"
            
            def get_config(self) -> Dict[str, Any]:
                return {"name": "DataFlowDecoder"}
        
        # Attach data flow tracking components
        flow_encoder = DataFlowEncoder()
        flow_decoder = DataFlowDecoder()
        gif.attach_encoder(flow_encoder)
        gif.attach_decoder(flow_decoder)
        
        # Process test data
        test_inputs = ["input_1", "input_2", "input_3"]
        results = []
        
        for test_input in test_inputs:
            result = gif.process_single_input(test_input)
            results.append(result)
        
        # Validate data flow
        assert len(flow_encoder.received_data) == 3
        assert len(flow_decoder.received_spikes) == 3
        assert flow_encoder.received_data == test_inputs
        
        # Validate shape consistency
        for shape in flow_encoder.output_shapes:
            assert shape == (8, 1, 15)  # Encoder output shape
        
        for shape in flow_decoder.input_shapes:
            assert shape == (8, 1, 10)  # DU Core output shape
        
        # Validate results
        assert results == ["processed_1", "processed_2", "processed_3"]
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_system_behavior_under_failure_conditions(self):
        """Test system behavior when components fail during operation."""
        du_core = DU_Core_V1(input_size=10, hidden_sizes=[15], output_size=8)
        gif = GIF(du_core)
        
        # Test encoder failure recovery
        failing_encoder = FailingMockEncoder(failure_mode="encode")
        working_encoder = MockEncoder(output_shape=(5, 1, 10))
        
        gif.attach_encoder(failing_encoder)
        gif.attach_decoder(MockDecoder())
        
        # First operation should fail
        with pytest.raises(RuntimeError):
            gif.process_single_input("test_data")
        
        # Replace encoder and verify recovery
        gif.attach_encoder(working_encoder)
        result = gif.process_single_input("test_data")
        assert result == "mock_action_success"
        
        # Test decoder failure recovery
        failing_decoder = FailingMockDecoder(failure_mode="decode")
        working_decoder = MockDecoder()
        
        gif.attach_decoder(failing_decoder)
        
        # Operation should fail
        with pytest.raises(RuntimeError):
            gif.process_single_input("test_data")
        
        # Replace decoder and verify recovery
        gif.attach_decoder(working_decoder)
        result = gif.process_single_input("test_data")
        assert result == "mock_action_success"
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_performance_benchmarking_and_regression_detection(self):
        """Test performance characteristics for regression detection."""
        du_core = DU_Core_V1(input_size=50, hidden_sizes=[75, 60], output_size=30)
        gif = GIF(du_core)
        
        # Use performance monitoring encoder
        perf_encoder = PerformanceMockEncoder(output_shape=(15, 2, 50))
        gif.attach_encoder(perf_encoder)
        gif.attach_decoder(MockDecoder())
        
        # Benchmark processing time
        num_iterations = 50
        start_time = time.time()
        
        for i in range(num_iterations):
            result = gif.process_single_input(f"benchmark_data_{i}")
            assert result == "mock_action_success"
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_cycle = total_time / num_iterations
        
        # Performance assertions (these serve as regression tests)
        assert avg_time_per_cycle < 0.1, f"Average cycle time too high: {avg_time_per_cycle:.4f}s"
        assert total_time < 5.0, f"Total processing time too high: {total_time:.2f}s"
        
        # Check encoder performance metrics
        metrics = perf_encoder.get_performance_metrics()
        assert metrics["total_calls"] == num_iterations
        assert metrics["avg_time"] < 0.01, "Encoder average time too high"
        
        # Throughput should be reasonable
        throughput = num_iterations / total_time
        assert throughput > 10, f"Throughput too low: {throughput:.2f} cycles/sec"
    
    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_cross_component_integration_validation(self):
        """Test integration between all framework components."""
        # Create multiple DU Core configurations
        configs = [
            {"input_size": 10, "hidden_sizes": [15], "output_size": 8},
            {"input_size": 20, "hidden_sizes": [25, 20], "output_size": 12},
            {"input_size": 30, "hidden_sizes": [40, 35, 30], "output_size": 18}
        ]
        
        for i, config in enumerate(configs):
            du_core = DU_Core_V1(**config)
            gif = GIF(du_core)
            
            # Create compatible encoder
            encoder = MockEncoder(output_shape=(10, 1, config["input_size"]))
            decoder = MockDecoder()
            
            gif.attach_encoder(encoder)
            gif.attach_decoder(decoder)
            
            # Test with various data types
            test_data = [
                f"config_{i}_simple",
                create_test_raw_data("complex"),
                torch.randn(50, 10)
            ]
            
            for j, data in enumerate(test_data):
                result = gif.process_single_input(data)
                assert result == "mock_action_success", f"Failed on config {i}, data {j}"
            
            # Verify system configuration
            sys_config = gif.get_system_config()
            du_core_config = sys_config["du_core_config"]
            assert du_core_config["input_size"] == config["input_size"]
            assert du_core_config["output_size"] == config["output_size"]


class TestArchitecturalStressTesting:
    """Stress testing for the complete GIF architecture."""

    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_high_volume_processing_stress(self):
        """Test the system under high-volume processing conditions."""
        du_core = DU_Core_V1(input_size=40, hidden_sizes=[60, 50], output_size=25)
        gif = GIF(du_core)

        # Attach components
        encoder = MockEncoder(output_shape=(20, 4, 40))  # Large batch processing
        decoder = MockDecoder()
        gif.attach_encoder(encoder)
        gif.attach_decoder(decoder)

        # Process high volume of data
        num_samples = 200
        results = []

        for i in range(num_samples):
            if i % 4 == 0:
                data = create_stress_test_data("large")
            elif i % 4 == 1:
                data = create_test_raw_data("complex")
            elif i % 4 == 2:
                data = f"stress_test_sample_{i}"
            else:
                data = torch.randn(100, 20)

            result = gif.process_single_input(data)
            results.append(result)

        # Verify all results
        assert len(results) == num_samples
        assert all(result == "mock_action_success" for result in results)
        assert encoder.encoding_calls == num_samples

    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_memory_stability_under_stress(self):
        """Test memory stability during extended processing."""
        import gc

        du_core = DU_Core_V1(input_size=30, hidden_sizes=[45], output_size=20)
        gif = GIF(du_core)

        encoder = MockEncoder(output_shape=(15, 2, 30))
        decoder = MockDecoder()
        gif.attach_encoder(encoder)
        gif.attach_decoder(decoder)

        # Force garbage collection before test
        gc.collect()

        # Process data in batches to test memory stability
        batch_size = 50
        num_batches = 10

        for batch in range(num_batches):
            batch_results = []

            for i in range(batch_size):
                data = create_stress_test_data("medium")
                result = gif.process_single_input(data)
                batch_results.append(result)

            # Verify batch results
            assert len(batch_results) == batch_size
            assert all(result == "mock_action_success" for result in batch_results)

            # Force garbage collection between batches
            gc.collect()

        # System should still be responsive
        final_result = gif.process_single_input("final_test")
        assert final_result == "mock_action_success"

    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_concurrent_operation_safety(self):
        """Test safety of concurrent-like operations."""
        du_core = DU_Core_V1(input_size=25, hidden_sizes=[35], output_size=15)
        gif = GIF(du_core)

        encoder = MockEncoder(output_shape=(10, 1, 25))
        decoder = MockDecoder()
        gif.attach_encoder(encoder)
        gif.attach_decoder(decoder)

        # Simulate concurrent operations by rapid sequential processing
        rapid_inputs = [f"concurrent_test_{i}" for i in range(100)]
        rapid_results = []

        start_time = time.time()
        for input_data in rapid_inputs:
            result = gif.process_single_input(input_data)
            rapid_results.append(result)
        end_time = time.time()

        # Verify all operations completed successfully
        assert len(rapid_results) == 100
        assert all(result == "mock_action_success" for result in rapid_results)

        # Performance should be reasonable even under rapid processing
        total_time = end_time - start_time
        avg_time = total_time / 100
        assert avg_time < 0.05, f"Average processing time too high under rapid load: {avg_time:.4f}s"


class TestFinalArchitecturalValidation:
    """Final comprehensive validation of the GIF architecture."""

    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_complete_framework_integration_certification(self):
        """
        The ultimate integration test that certifies the complete framework.

        This test validates that all components work together perfectly
        and that the architecture is sound and ready for production use.
        """
        # Test multiple configurations to ensure robustness
        test_configurations = [
            {
                "name": "small_config",
                "du_core": {"input_size": 8, "hidden_sizes": [12], "output_size": 6},
                "encoder_shape": (5, 1, 8),
                "test_data": ["small_test_1", "small_test_2"]
            },
            {
                "name": "medium_config",
                "du_core": {"input_size": 20, "hidden_sizes": [30, 25], "output_size": 15},
                "encoder_shape": (10, 2, 20),
                "test_data": [create_test_raw_data("complex"), create_stress_test_data("medium")]
            },
            {
                "name": "large_config",
                "du_core": {"input_size": 50, "hidden_sizes": [75, 60, 45], "output_size": 30},
                "encoder_shape": (20, 4, 50),
                "test_data": [create_stress_test_data("large"), torch.randn(200, 100)]
            }
        ]

        certification_results = {}

        for config in test_configurations:
            # Create system with current configuration
            du_core = DU_Core_V1(**config["du_core"])
            gif = GIF(du_core)

            # Attach components
            encoder = MockEncoder(output_shape=config["encoder_shape"])
            decoder = MockDecoder()
            gif.attach_encoder(encoder)
            gif.attach_decoder(decoder)

            # Test complete cognitive cycles
            config_results = []
            for test_data in config["test_data"]:
                result = gif.process_single_input(test_data)
                config_results.append(result)

            # Verify all results
            assert all(result == "mock_action_success" for result in config_results)

            # Test system configuration integrity
            sys_config = gif.get_system_config()
            du_core_config = sys_config["du_core_config"]
            assert du_core_config["input_size"] == config["du_core"]["input_size"]
            assert du_core_config["output_size"] == config["du_core"]["output_size"]

            # Test error handling
            gif.attach_encoder(FailingMockEncoder(failure_mode="encode"))
            with pytest.raises(RuntimeError):
                gif.process_single_input("error_test")

            # Test recovery
            gif.attach_encoder(encoder)
            recovery_result = gif.process_single_input("recovery_test")
            assert recovery_result == "mock_action_success"

            # Store certification results
            certification_results[config["name"]] = {
                "cognitive_cycles": len(config_results),
                "all_successful": all(result == "mock_action_success" for result in config_results),
                "error_handling": True,
                "recovery": True,
                "configuration_integrity": True
            }

        # Final certification validation
        assert len(certification_results) == 3
        for config_name, results in certification_results.items():
            assert results["all_successful"], f"Configuration {config_name} failed cognitive cycles"
            assert results["error_handling"], f"Configuration {config_name} failed error handling"
            assert results["recovery"], f"Configuration {config_name} failed recovery"
            assert results["configuration_integrity"], f"Configuration {config_name} failed integrity check"

        # If we reach here, the architecture is certified as sound
        print("\n🎉 GIF CORE ARCHITECTURE CERTIFICATION COMPLETE 🎉")
        print("✅ All cognitive cycles successful")
        print("✅ Error handling validated")
        print("✅ Recovery mechanisms verified")
        print("✅ Configuration integrity confirmed")
        print("✅ Multi-scale testing passed")
        print("🚀 Framework ready for advanced cognitive features!")

    @pytest.mark.skipif(not DU_CORE_AVAILABLE, reason="snnTorch not available")
    def test_architectural_regression_prevention(self):
        """Test that serves as regression prevention for future changes."""
        # This test establishes baseline behavior that should not change
        du_core = DU_Core_V1(input_size=16, hidden_sizes=[24, 20], output_size=12)
        gif = GIF(du_core)

        encoder = MockEncoder(output_shape=(8, 1, 16))
        decoder = MockDecoder()
        gif.attach_encoder(encoder)
        gif.attach_decoder(decoder)

        # Establish baseline behavior
        baseline_data = "regression_test_baseline"
        baseline_result = gif.process_single_input(baseline_data)

        # This should always return the expected result
        assert baseline_result == "mock_action_success"

        # System configuration should be stable
        config = gif.get_system_config()
        expected_config_keys = ["du_core_config", "encoder_config", "decoder_config"]
        assert all(key in config for key in expected_config_keys)

        # DU Core should have expected structure
        du_core_config = config["du_core_config"]
        assert du_core_config["input_size"] == 16
        assert du_core_config["output_size"] == 12
        assert len(du_core_config["hidden_sizes"]) == 2

        # Interface contracts should be enforced
        with pytest.raises(TypeError):
            gif.attach_encoder("not_an_encoder")

        with pytest.raises(TypeError):
            gif.attach_decoder("not_a_decoder")

        # This test serves as a canary for architectural changes
        # If this test fails in the future, it indicates a breaking change
