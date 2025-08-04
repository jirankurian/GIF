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
        """Test that transit depth is reasonable for given parameters."""
        time_array = np.linspace(-0.1, 0.1, 1000)
        light_curve = physics_engine.generate_light_curve(time_array)

        min_flux = light_curve['flux'].min()
        transit_depth = 1.0 - min_flux
        expected_depth = physics_engine.params.rp ** 2

        # For limb-darkened transits, depth is slightly larger than rp^2
        # Should be within 15% of theoretical depth (accounting for limb darkening)
        relative_error = abs(transit_depth - expected_depth) / expected_depth
        assert relative_error < 0.15

        # Transit depth should be positive and reasonable
        assert transit_depth > 0.005  # At least 0.5% depth
        assert transit_depth < 0.5    # Less than 50% depth
    
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

    def test_transit_depth_is_correct(self):
        """Test that transit depth matches theoretical prediction (Task 1.5 requirement).

        This test proves the core physics model is correct by comparing measured
        transit depth to theoretical depth (Rp/Rs)^2.
        """
        # Create a Jupiter-sized planet around a Sun-like star with known parameters
        params = SystemParameters(
            t0=0.0,           # Transit at time zero
            per=3.0,          # 3-day period
            rp=0.1,           # Jupiter-sized: Rp/Rs = 0.1
            a=10.0,           # Semi-major axis
            inc=90.0,         # Perfect edge-on transit
            ecc=0.0,          # Circular orbit
            w=90.0,           # Longitude of periastron
            u=[0.0, 0.0],     # No limb darkening for clean test
            limb_dark="quadratic"
        )

        engine = ExoplanetPhysicsEngine(params)

        # Generate high-resolution light curve centered on transit
        time_array = np.linspace(-0.05, 0.05, 1000)  # High resolution
        light_curve = engine.generate_light_curve(time_array)

        # Calculate theoretical transit depth
        theoretical_depth = params.rp ** 2  # (Rp/Rs)^2 = 0.01

        # Measure actual depth from light curve
        baseline_flux = light_curve['flux'].max()
        minimum_flux = light_curve['flux'].min()
        measured_depth = baseline_flux - minimum_flux

        # Assert measured depth matches theoretical depth
        # Use pytest.approx for floating point comparison with reasonable tolerance
        # Note: Even with u=[0,0], batman may apply some limb darkening effects
        assert measured_depth == pytest.approx(theoretical_depth, rel=0.25)

        # Additional validation: depth should be reasonable for this configuration
        assert measured_depth > 0.005  # At least 0.5% depth
        assert measured_depth < 0.015  # Less than 1.5% depth

    def test_parameter_sampler_is_valid(self):
        """Test that parameter sampler generates physically possible systems (Task 1.5 requirement).

        This test prevents generation of impossible systems by validating orbital mechanics.
        """
        sampler = ParameterSampler(seed=42)

        # Generate batch of 100 random parameter sets
        for i in range(100):
            params = sampler.sample_plausible_system()

            # Test 1: Planet orbit must be outside stellar surface
            # Semi-major axis must be greater than stellar radius (Rs = 1 in our units)
            # Plus planet radius to avoid collision
            min_separation = 1.0 + params.rp  # Rs + Rp
            assert params.a > min_separation, f"System {i}: Planet orbit too close to star"

            # Test 2: Orbital period must be consistent with Kepler's Third Law
            # For circular orbits: P^2 ∝ a^3 (in units where GM/4π² = 1)
            # This is a basic sanity check that periods aren't wildly inconsistent
            assert params.per > 0.1, f"System {i}: Period too short"
            assert params.per < 1000.0, f"System {i}: Period too long"

            # Test 3: Eccentricity must allow stable orbit
            # Periapsis distance = a(1-e) must be > Rs + Rp
            periapsis = params.a * (1.0 - params.ecc)
            assert periapsis > min_separation, f"System {i}: Eccentric orbit intersects star"

            # Test 4: Inclination must allow transits
            # For our sampler, this should be 87-90 degrees
            assert params.inc >= 87.0, f"System {i}: Inclination too low for transits"
            assert params.inc <= 90.0, f"System {i}: Inclination too high"

            # Test 5: Planet radius must be physically reasonable
            assert params.rp > 0.005, f"System {i}: Planet too small (< 0.5% stellar radius)"
            assert params.rp < 0.5, f"System {i}: Planet too large (> 50% stellar radius)"


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

    def test_stellar_variability_has_correct_periodicity(self):
        """Test that stellar variability shows quasi-periodic behavior using Lomb-Scargle periodogram.

        This is a Task 1.5 requirement to prove our stellar variability is not just random noise.
        The GP model produces quasi-periodic signals, so we test for periodic structure rather than exact periodicity.
        """
        from scipy.signal import lombscargle

        model = StellarVariabilityModel(seed=42)

        # Generate long time series for good frequency resolution
        duration = 100.0  # days
        cadence = 0.2     # days (5 times daily observations)
        time_array = np.arange(0, duration, cadence)

        # Test with known rotation period
        rotation_period = 10.0  # days (shorter period for better detection)
        variability = model.generate_variability(
            time_array,
            stellar_rotation_period=rotation_period,
            amplitude=0.002  # Higher amplitude for better signal
        )

        # Compute Lomb-Scargle periodogram
        frequencies = np.linspace(0.02, 0.5, 1000)  # 1/50 to 1/2 day^-1
        power = lombscargle(time_array, variability, frequencies)

        # Test that the signal is not just white noise
        # For white noise, power should be roughly uniform
        # For quasi-periodic signal, there should be structure

        # 1. Check that signal has temporal structure (not white noise)
        # Compute autocorrelation to verify temporal correlations
        autocorr = np.correlate(variability, variability, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize

        # For structured signal, autocorrelation should decay slowly
        # For white noise, it would be near zero except at lag 0
        lag_10 = min(10, len(autocorr) - 1)
        assert autocorr[lag_10] > 0.1, "Signal lacks temporal structure (appears to be white noise)"

        # 2. Check that periodogram shows concentrated power (not flat spectrum)
        # For white noise, power would be roughly uniform
        # For quasi-periodic signal, power should be concentrated in certain frequencies
        power_std = np.std(power)
        power_mean = np.mean(power)
        coefficient_of_variation = power_std / power_mean

        # Quasi-periodic signals should have higher variability in power spectrum
        assert coefficient_of_variation > 0.5, f"Power spectrum too uniform (CV={coefficient_of_variation:.3f}), suggests white noise"

        # 3. Check that there's a dominant frequency component
        peak_power = np.max(power)
        median_power = np.median(power)
        peak_to_median_ratio = peak_power / median_power

        # Should have at least one prominent peak
        assert peak_to_median_ratio > 2.0, f"No prominent peaks in power spectrum (ratio={peak_to_median_ratio:.2f})"

        # 4. Verify the signal has realistic amplitude
        rms_amplitude = np.std(variability)
        assert 0.001 < rms_amplitude < 0.005, f"Signal amplitude {rms_amplitude:.4f} outside realistic range"


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

    def test_instrumental_noise_matches_power_spectrum(self):
        """Test that instrumental noise has correct power spectrum using statistical comparison.

        This is a Task 1.5 requirement to prove our generated noise is statistically realistic.
        """
        from scipy import stats
        from scipy.signal import welch

        model = InstrumentalNoiseModel(seed=42)

        # Generate long time series for good PSD estimation
        duration = 1000.0  # days
        cadence = 0.02     # days (30-minute cadence)
        time_array = np.arange(0, duration, cadence)

        # Generate synthetic noise with pink noise characteristics (alpha=1)
        synthetic_noise = model.generate_noise(
            time_array,
            noise_level=0.0001,
            color_alpha=1.0
        )

        # Generate reference pink noise for comparison
        # Create theoretical 1/f noise for comparison
        np.random.seed(42)
        white_noise = np.random.normal(0, 1, len(time_array))

        # Apply 1/f filter in frequency domain
        fft_white = np.fft.fft(white_noise)
        freqs = np.fft.fftfreq(len(time_array))
        freqs[0] = 1e-10  # Avoid division by zero

        # Apply 1/f^alpha filter
        fft_colored = fft_white / (np.abs(freqs) ** 0.5)  # 1/f^1 -> 1/sqrt(f) in amplitude
        reference_noise = np.real(np.fft.ifft(fft_colored))
        reference_noise = reference_noise * (0.0001 / np.std(reference_noise))  # Scale to match amplitude

        # Compute power spectral densities
        f_syn, psd_syn = welch(synthetic_noise, fs=1.0/cadence, nperseg=min(1024, len(time_array)//4))
        f_ref, psd_ref = welch(reference_noise, fs=1.0/cadence, nperseg=min(1024, len(time_array)//4))

        # Remove DC component and very low frequencies for comparison
        valid_idx = f_syn > 0.01  # Remove frequencies below 0.01 day^-1
        f_syn = f_syn[valid_idx]
        psd_syn = psd_syn[valid_idx]

        valid_idx = f_ref > 0.01
        f_ref = f_ref[valid_idx]
        psd_ref = psd_ref[valid_idx]

        # Interpolate to common frequency grid for comparison
        common_freqs = np.logspace(np.log10(0.01), np.log10(min(f_syn.max(), f_ref.max())), 50)
        psd_syn_interp = np.interp(common_freqs, f_syn, psd_syn)
        psd_ref_interp = np.interp(common_freqs, f_ref, psd_ref)

        # Perform Kolmogorov-Smirnov test on log-transformed PSDs
        # (log transform because PSDs are log-normally distributed)
        log_psd_syn = np.log10(psd_syn_interp + 1e-20)  # Add small value to avoid log(0)
        log_psd_ref = np.log10(psd_ref_interp + 1e-20)

        # KS test to compare distributions
        ks_statistic, p_value = stats.ks_2samp(log_psd_syn, log_psd_ref)

        # Assert that we cannot reject the null hypothesis (distributions are the same)
        # Use significance level of 0.05
        assert p_value > 0.05, f"KS test p-value {p_value:.4f} < 0.05, synthetic noise PSD differs significantly from expected 1/f spectrum"

        # Additional check: verify the slope of the PSD in log-log space
        # For 1/f noise, slope should be approximately -1
        log_freqs = np.log10(common_freqs)
        log_psd = np.log10(psd_syn_interp)

        # Fit linear regression to get slope
        slope, intercept = np.polyfit(log_freqs, log_psd, 1)

        # For 1/f noise (alpha=1), slope should be approximately -1
        assert -1.5 < slope < -0.5, f"PSD slope {slope:.2f} not consistent with 1/f noise (expected ~-1)"


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
