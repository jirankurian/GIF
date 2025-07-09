"""
Test Suite for Exoplanet Data Generator
======================================

Comprehensive tests for both the core physics engine and realism layer:
- SystemParameters dataclass validation
- ParameterSampler realistic parameter generation
- ExoplanetPhysicsEngine light curve generation
- StellarVariabilityModel stellar rotation signals
- InstrumentalNoiseModel colored noise generation
- RealisticExoplanetGenerator complete pipeline

These tests ensure the complete pipeline produces mathematically correct,
physically plausible, and realistically noisy synthetic data.
"""

import pytest
import numpy as np
from dataclasses import FrozenInstanceError

# Import our complete data generator components
from data_generators.exoplanet_generator import (
    SystemParameters,
    ParameterSampler,
    ExoplanetPhysicsEngine,
    StellarVariabilityModel,
    InstrumentalNoiseModel,
    RealisticExoplanetGenerator
)


class TestSystemParameters:
    """Test the SystemParameters dataclass."""
    
    def test_valid_parameters(self):
        """Test that valid parameters are accepted."""
        params = SystemParameters(
            t0=0.5, per=3.5, rp=0.1, a=10.0, inc=89.5,
            ecc=0.1, w=90.0, u=[0.1, 0.3], limb_dark="quadratic"
        )
        assert params.t0 == 0.5
        assert params.per == 3.5
        assert params.rp == 0.1
    
    def test_invalid_period(self):
        """Test that negative periods are rejected."""
        with pytest.raises(ValueError, match="Orbital period must be positive"):
            SystemParameters(
                t0=0.5, per=-1.0, rp=0.1, a=10.0, inc=89.5,
                ecc=0.1, w=90.0, u=[0.1, 0.3], limb_dark="quadratic"
            )
    
    def test_invalid_planet_radius(self):
        """Test that invalid planet radius ratios are rejected."""
        with pytest.raises(ValueError, match="Planet radius ratio must be"):
            SystemParameters(
                t0=0.5, per=3.5, rp=1.5, a=10.0, inc=89.5,
                ecc=0.1, w=90.0, u=[0.1, 0.3], limb_dark="quadratic"
            )
    
    def test_invalid_semi_major_axis(self):
        """Test that semi-major axis must be > 1."""
        with pytest.raises(ValueError, match="Semi-major axis must be"):
            SystemParameters(
                t0=0.5, per=3.5, rp=0.1, a=0.5, inc=89.5,
                ecc=0.1, w=90.0, u=[0.1, 0.3], limb_dark="quadratic"
            )
    
    def test_invalid_inclination(self):
        """Test that inclination must be 0-90 degrees."""
        with pytest.raises(ValueError, match="Inclination must be"):
            SystemParameters(
                t0=0.5, per=3.5, rp=0.1, a=10.0, inc=95.0,
                ecc=0.1, w=90.0, u=[0.1, 0.3], limb_dark="quadratic"
            )
    
    def test_invalid_eccentricity(self):
        """Test that eccentricity must be < 1."""
        with pytest.raises(ValueError, match="Eccentricity must be"):
            SystemParameters(
                t0=0.5, per=3.5, rp=0.1, a=10.0, inc=89.5,
                ecc=1.5, w=90.0, u=[0.1, 0.3], limb_dark="quadratic"
            )
    
    def test_invalid_limb_darkening_coefficients(self):
        """Test that quadratic limb darkening requires 2 coefficients."""
        with pytest.raises(ValueError, match="Quadratic limb darkening requires 2 coefficients"):
            SystemParameters(
                t0=0.5, per=3.5, rp=0.1, a=10.0, inc=89.5,
                ecc=0.1, w=90.0, u=[0.1], limb_dark="quadratic"
            )


class TestParameterSampler:
    """Test the ParameterSampler class."""
    
    def test_reproducible_sampling(self):
        """Test that sampling with same seed produces identical results."""
        sampler1 = ParameterSampler(seed=42)
        sampler2 = ParameterSampler(seed=42)
        
        params1 = sampler1.sample_plausible_system()
        params2 = sampler2.sample_plausible_system()
        
        assert params1.t0 == params2.t0
        assert params1.per == params2.per
        assert params1.rp == params2.rp
    
    def test_parameter_ranges(self):
        """Test that sampled parameters fall within expected ranges."""
        sampler = ParameterSampler(seed=123)
        
        # Sample multiple systems to test ranges
        for _ in range(100):
            params = sampler.sample_plausible_system()
            
            assert 0.0 <= params.t0 <= 1.0
            assert 1.0 <= params.per <= 100.0
            assert 0.01 <= params.rp <= 0.2
            assert 5.0 <= params.a <= 50.0
            assert 87.0 <= params.inc <= 90.0
            assert 0.0 <= params.ecc <= 0.4
            assert 0.0 <= params.w <= 360.0
            assert len(params.u) == 2
            assert all(0.0 <= coeff <= 1.0 for coeff in params.u)
            assert params.limb_dark == "quadratic"
    
    def test_parameter_diversity(self):
        """Test that sampler produces diverse parameters."""
        sampler = ParameterSampler(seed=456)
        
        periods = []
        radii = []
        
        for _ in range(50):
            params = sampler.sample_plausible_system()
            periods.append(params.per)
            radii.append(params.rp)
        
        # Check that we get reasonable diversity
        assert np.std(periods) > 10.0  # Should span significant range
        assert np.std(radii) > 0.02   # Should span significant range


class TestExoplanetPhysicsEngine:
    """Test the ExoplanetPhysicsEngine class."""
    
    @pytest.fixture
    def sample_params(self):
        """Provide a valid set of parameters for testing."""
        return SystemParameters(
            t0=0.0, per=3.0, rp=0.1, a=10.0, inc=90.0,
            ecc=0.0, w=90.0, u=[0.1, 0.3], limb_dark="quadratic"
        )
    
    @pytest.fixture
    def physics_engine(self, sample_params):
        """Provide a configured physics engine for testing."""
        return ExoplanetPhysicsEngine(sample_params)
    
    def test_engine_initialization(self, sample_params):
        """Test that engine initializes correctly."""
        engine = ExoplanetPhysicsEngine(sample_params)
        assert engine.params == sample_params
        assert hasattr(engine, 'batman_params')
    
    def test_light_curve_generation(self, physics_engine):
        """Test basic light curve generation."""
        time_array = np.linspace(-0.1, 0.1, 100)
        light_curve = physics_engine.generate_light_curve(time_array)
        
        # Check output format
        assert len(light_curve) == 100
        assert 'time' in light_curve.columns
        assert 'flux' in light_curve.columns
        
        # Check that we have a transit (flux should dip below 1.0)
        min_flux = light_curve['flux'].min()
        assert min_flux < 1.0
        
        # Check that baseline flux is approximately 1.0
        baseline_flux = light_curve['flux'].max()
        assert abs(baseline_flux - 1.0) < 1e-6
    
    def test_transit_depth(self, physics_engine):
        """Test that transit depth matches theoretical expectation."""
        time_array = np.linspace(-0.1, 0.1, 1000)
        light_curve = physics_engine.generate_light_curve(time_array)
        
        min_flux = light_curve['flux'].min()
        transit_depth = 1.0 - min_flux
        expected_depth = physics_engine.params.rp ** 2
        
        # Should be within 1% of theoretical depth for circular orbit
        relative_error = abs(transit_depth - expected_depth) / expected_depth
        assert relative_error < 0.01
    
    def test_empty_time_array(self, physics_engine):
        """Test that empty time array raises appropriate error."""
        with pytest.raises(ValueError, match="time_array cannot be empty"):
            physics_engine.generate_light_curve(np.array([]))
    
    def test_invalid_time_array(self, physics_engine):
        """Test that invalid time values raise appropriate error."""
        time_array = np.array([0.0, np.inf, 1.0])
        with pytest.raises(ValueError, match="time_array contains non-finite values"):
            physics_engine.generate_light_curve(time_array)


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_complete_workflow(self):
        """Test the complete workflow from sampling to light curve generation."""
        # 1. Sample parameters
        sampler = ParameterSampler(seed=789)
        params = sampler.sample_plausible_system()
        
        # 2. Create engine
        engine = ExoplanetPhysicsEngine(params)
        
        # 3. Generate light curve
        time_array = np.linspace(params.t0 - 0.1, params.t0 + 0.1, 500)
        light_curve = engine.generate_light_curve(time_array)
        
        # 4. Validate results
        assert len(light_curve) == 500
        assert light_curve['flux'].min() < 1.0  # Should have transit
        assert light_curve['flux'].max() <= 1.0  # Should not exceed baseline
        
        # 5. Check physical consistency
        transit_depth = 1.0 - light_curve['flux'].min()
        assert transit_depth > 0.0001  # Should be detectable
        assert transit_depth < 0.1     # Should be reasonable for our parameter ranges


class TestStellarVariabilityModel:
    """Test the StellarVariabilityModel class."""

    def test_model_initialization(self):
        """Test that stellar variability model initializes correctly."""
        model = StellarVariabilityModel(Q0=10.0, dQ=1.0, seed=42)
        assert model.Q0 == 10.0
        assert model.dQ == 1.0

    def test_variability_generation(self):
        """Test basic stellar variability generation."""
        model = StellarVariabilityModel(seed=123)
        time_array = np.linspace(0, 100, 1000)

        variability = model.generate_variability(
            time_array,
            stellar_rotation_period=25.0,
            amplitude=0.001
        )

        # Check output format
        assert len(variability) == 1000
        assert isinstance(variability, np.ndarray)

        # Check amplitude is reasonable
        rms_amplitude = np.std(variability)
        assert 0.0005 < rms_amplitude < 0.002  # Should be close to requested amplitude

    def test_reproducible_variability(self):
        """Test that variability generation is reproducible with same seed."""
        model1 = StellarVariabilityModel(seed=456)
        model2 = StellarVariabilityModel(seed=456)

        time_array = np.linspace(0, 50, 500)

        var1 = model1.generate_variability(time_array, 20.0, 0.001)
        var2 = model2.generate_variability(time_array, 20.0, 0.001)

        np.testing.assert_array_almost_equal(var1, var2, decimal=10)

    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        model = StellarVariabilityModel(seed=789)
        time_array = np.linspace(0, 10, 100)

        # Test negative rotation period
        with pytest.raises(ValueError, match="Rotation period must be positive"):
            model.generate_variability(time_array, -5.0, 0.001)

        # Test negative amplitude
        with pytest.raises(ValueError, match="Amplitude must be non-negative"):
            model.generate_variability(time_array, 25.0, -0.001)

        # Test unrealistic amplitude
        with pytest.raises(ValueError, match="unrealistically high"):
            model.generate_variability(time_array, 25.0, 0.2)

        # Test empty time array
        with pytest.raises(ValueError, match="time_array cannot be empty"):
            model.generate_variability(np.array([]), 25.0, 0.001)


class TestInstrumentalNoiseModel:
    """Test the InstrumentalNoiseModel class."""

    def test_model_initialization(self):
        """Test that instrumental noise model initializes correctly."""
        model = InstrumentalNoiseModel(seed=42)
        assert hasattr(model, 'rng')

    def test_white_noise_generation(self):
        """Test white noise generation (alpha=0)."""
        model = InstrumentalNoiseModel(seed=123)
        time_array = np.linspace(0, 10, 1000)

        noise = model.generate_noise(time_array, noise_level=0.001, color_alpha=0.0)

        # Check output format
        assert len(noise) == 1000
        assert isinstance(noise, np.ndarray)

        # Check noise level
        noise_rms = np.std(noise)
        assert 0.0008 < noise_rms < 0.0012  # Should be close to requested level

    def test_colored_noise_generation(self):
        """Test colored noise generation (alpha>0)."""
        model = InstrumentalNoiseModel(seed=456)
        time_array = np.linspace(0, 100, 2000)

        # Test pink noise (alpha=1)
        pink_noise = model.generate_noise(time_array, noise_level=0.0001, color_alpha=1.0)

        # Test brown noise (alpha=2)
        brown_noise = model.generate_noise(time_array, noise_level=0.0001, color_alpha=2.0)

        # Both should have correct length and amplitude
        assert len(pink_noise) == 2000
        assert len(brown_noise) == 2000

        pink_rms = np.std(pink_noise)
        brown_rms = np.std(brown_noise)

        assert 0.00008 < pink_rms < 0.00012
        assert 0.00008 < brown_rms < 0.00012

    def test_reproducible_noise(self):
        """Test that noise generation is reproducible with same seed."""
        model1 = InstrumentalNoiseModel(seed=789)
        model2 = InstrumentalNoiseModel(seed=789)

        time_array = np.linspace(0, 20, 500)

        noise1 = model1.generate_noise(time_array, 0.0001, 1.0)
        noise2 = model2.generate_noise(time_array, 0.0001, 1.0)

        np.testing.assert_array_almost_equal(noise1, noise2, decimal=10)

    def test_invalid_noise_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        model = InstrumentalNoiseModel(seed=101)
        time_array = np.linspace(0, 10, 100)

        # Test negative noise level
        with pytest.raises(ValueError, match="Noise level must be non-negative"):
            model.generate_noise(time_array, -0.001, 1.0)

        # Test unrealistic noise level
        with pytest.raises(ValueError, match="unrealistically high"):
            model.generate_noise(time_array, 0.02, 1.0)

        # Test invalid color alpha
        with pytest.raises(ValueError, match="Color alpha must be 0-3"):
            model.generate_noise(time_array, 0.001, -1.0)

        # Test empty time array
        with pytest.raises(ValueError, match="time_array cannot be empty"):
            model.generate_noise(np.array([]), 0.001, 1.0)


class TestRealisticExoplanetGenerator:
    """Test the RealisticExoplanetGenerator class."""

    def test_generator_initialization(self):
        """Test that realistic generator initializes correctly."""
        generator = RealisticExoplanetGenerator(
            seed=42,
            stellar_variability_amplitude=0.001,
            instrumental_noise_level=0.0001,
            noise_color_alpha=1.0
        )

        assert generator.seed == 42
        assert generator.stellar_variability_amplitude == 0.001
        assert generator.instrumental_noise_level == 0.0001
        assert generator.noise_color_alpha == 1.0
        assert hasattr(generator, 'parameter_sampler')
        assert hasattr(generator, 'stellar_model')
        assert hasattr(generator, 'noise_model')

    def test_realistic_data_generation(self):
        """Test complete realistic data generation."""
        generator = RealisticExoplanetGenerator(seed=123)

        data = generator.generate(
            observation_duration=10.0,  # Short duration for fast testing
            cadence_minutes=60.0        # 1-hour cadence
        )

        # Check output format
        expected_points = int(10.0 / (60.0 / (24 * 60)))  # ~240 points
        assert len(data) == expected_points

        # Check required columns
        required_columns = [
            'time', 'flux', 'perfect_flux', 'stellar_variability',
            'instrumental_noise', 'transit_period', 'planet_radius_ratio'
        ]
        for col in required_columns:
            assert col in data.columns

        # Check data ranges
        time_values = data['time'].to_numpy()
        flux_values = data['flux'].to_numpy()

        assert time_values.min() >= 0
        assert time_values.max() <= 10.0
        assert 0.99 < flux_values.mean() < 1.01  # Should be close to 1.0
        assert flux_values.min() < 1.0  # Should have transit dips

    def test_signal_component_combination(self):
        """Test that signal components are properly combined."""
        generator = RealisticExoplanetGenerator(seed=456)

        data = generator.generate(observation_duration=5.0, cadence_minutes=30.0)

        # Extract components
        perfect_flux = data['perfect_flux'].to_numpy()
        stellar_var = data['stellar_variability'].to_numpy()
        inst_noise = data['instrumental_noise'].to_numpy()
        final_flux = data['flux'].to_numpy()

        # Check that final flux = perfect + stellar + instrumental
        reconstructed = perfect_flux + stellar_var + inst_noise
        reconstruction_error = np.mean(np.abs(final_flux - reconstructed))

        assert reconstruction_error < 1e-10  # Should be numerically exact

    def test_reproducible_generation(self):
        """Test that generation is reproducible with same seed."""
        generator1 = RealisticExoplanetGenerator(seed=789)
        generator2 = RealisticExoplanetGenerator(seed=789)

        data1 = generator1.generate(observation_duration=5.0, cadence_minutes=60.0)
        data2 = generator2.generate(observation_duration=5.0, cadence_minutes=60.0)

        # Check that flux values are identical
        flux1 = data1['flux'].to_numpy()
        flux2 = data2['flux'].to_numpy()

        np.testing.assert_array_almost_equal(flux1, flux2, decimal=10)

    def test_noise_amplitude_scaling(self):
        """Test that noise amplitudes scale correctly."""
        # Generate data with different noise levels
        low_noise_gen = RealisticExoplanetGenerator(
            seed=101,
            stellar_variability_amplitude=0.0005,
            instrumental_noise_level=0.00005
        )

        high_noise_gen = RealisticExoplanetGenerator(
            seed=101,  # Same seed for fair comparison
            stellar_variability_amplitude=0.002,
            instrumental_noise_level=0.0002
        )

        low_data = low_noise_gen.generate(observation_duration=10.0)
        high_data = high_noise_gen.generate(observation_duration=10.0)

        # Check that higher noise settings produce higher variability
        low_stellar_rms = np.std(low_data['stellar_variability'].to_numpy())
        high_stellar_rms = np.std(high_data['stellar_variability'].to_numpy())

        low_inst_rms = np.std(low_data['instrumental_noise'].to_numpy())
        high_inst_rms = np.std(high_data['instrumental_noise'].to_numpy())

        assert high_stellar_rms > low_stellar_rms
        assert high_inst_rms > low_inst_rms

    def test_invalid_generation_parameters(self):
        """Test that invalid generation parameters raise errors."""
        generator = RealisticExoplanetGenerator(seed=202)

        # Test negative observation duration
        with pytest.raises(ValueError, match="Observation duration must be positive"):
            generator.generate(observation_duration=-10.0)

        # Test negative cadence
        with pytest.raises(ValueError, match="Cadence must be positive"):
            generator.generate(cadence_minutes=-30.0)

    def test_ground_truth_parameters(self):
        """Test that ground truth parameters are included correctly."""
        generator = RealisticExoplanetGenerator(seed=303)

        data = generator.generate(observation_duration=5.0)

        # Check that ground truth columns exist and have consistent values
        period = data['transit_period'].to_numpy()
        radius_ratio = data['planet_radius_ratio'].to_numpy()
        depth = data['transit_depth'].to_numpy()

        # All values in each column should be identical (constant parameters)
        assert np.all(period == period[0])
        assert np.all(radius_ratio == radius_ratio[0])
        assert np.all(depth == depth[0])

        # Check physical consistency
        expected_depth = radius_ratio[0] ** 2
        assert abs(depth[0] - expected_depth) < 1e-10
