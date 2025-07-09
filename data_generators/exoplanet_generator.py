"""
Exoplanet Light Curve Generator - Complete Realistic Data Pipeline
=================================================================

High-fidelity synthetic data generator for exoplanet transit light curves.
This module implements both the foundational physics engine and the realism layer
for generating synthetic observations indistinguishable from real astronomical data.

The complete pipeline consists of six main components:

**Core Physics Engine (Task 1.1):**
1. SystemParameters: A structured dataclass for holding physical parameters
2. ParameterSampler: A class for generating realistic parameter combinations
3. ExoplanetPhysicsEngine: The main physics simulation engine using batman-package

**Realism Layer (Task 1.2):**
4. StellarVariabilityModel: Gaussian Process model for stellar rotation signals
5. InstrumentalNoiseModel: Colored noise generator for authentic instrumental effects
6. RealisticExoplanetGenerator: Complete orchestrator combining all components

Key Features:
- Physics-based transit models using batman-package (C-optimized)
- Realistic stellar variability using celerite2 Gaussian Processes
- Authentic instrumental noise with proper temporal structure
- Realistic parameter sampling from astrophysically plausible ranges
- Type-safe parameter structures with comprehensive validation
- Efficient polars DataFrame output for downstream processing
- Modular design enabling both perfect and realistic data generation

Usage Example:
    >>> # Generate realistic synthetic data
    >>> generator = RealisticExoplanetGenerator(seed=42)
    >>> realistic_data = generator.generate()
    >>> print(f"Generated {len(realistic_data)} data points")

    >>> # Or use individual components for perfect physics
    >>> sampler = ParameterSampler(seed=42)
    >>> params = sampler.sample_plausible_system()
    >>> engine = ExoplanetPhysicsEngine(params)
    >>> time_array = np.linspace(0, 10, 1000)
    >>> perfect_curve = engine.generate_light_curve(time_array)

Author: GIF Development Team
Phase: 1.1 - Core Physics Engine, 1.2 - Realism Layer
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import polars as pl
import scipy.fft

try:
    import batman
except ImportError:
    raise ImportError(
        "batman-package is required for exoplanet physics simulation. "
        "Install with: pip install batman-package"
    )

try:
    import celerite2
    from celerite2 import terms
except ImportError:
    raise ImportError(
        "celerite2 is required for stellar variability modeling. "
        "Install with: pip install celerite2"
    )


@dataclass
class SystemParameters:
    """
    Physical parameters defining a star-planet system for transit modeling.

    This dataclass provides a type-safe, structured way to hold all the physical
    parameters required for generating exoplanet transit light curves. All parameters
    follow standard astronomical conventions and units.

    Attributes:
        t0 (float): Time of the first transit center in days. Defines the phase
            reference point for the transit timing.
        per (float): Orbital period of the planet in days. Must be positive.
        rp (float): Planet radius in units of stellar radius (Rp/R*).
            Dimensionless ratio, typically 0.01-0.2 for known exoplanets.
        a (float): Semi-major axis of the orbit in units of stellar radius (a/R*).
            Dimensionless ratio, must be > 1 for stable orbits.
        inc (float): Orbital inclination in degrees. 90° corresponds to a perfectly
            edge-on orbit (maximum transit depth). Values 87-90° ensure transits occur.
        ecc (float): Orbital eccentricity. 0 = circular orbit, values < 1 for bound orbits.
            Most exoplanets have ecc < 0.4.
        w (float): Longitude of periastron in degrees (0-360°). Defines the orientation
            of the elliptical orbit. Only relevant for eccentric orbits (ecc > 0).
        u (List[float]): Limb-darkening coefficients. For quadratic limb darkening,
            this should be a list of two values [u1, u2] where 0 ≤ u1, u2 ≤ 1.
        limb_dark (str): Limb-darkening model to use. Supported values include
            "quadratic", "linear", "uniform". "quadratic" is most commonly used.

    Example:
        >>> params = SystemParameters(
        ...     t0=0.5, per=3.5, rp=0.1, a=10.0, inc=89.5,
        ...     ecc=0.1, w=90.0, u=[0.1, 0.3], limb_dark="quadratic"
        ... )
    """
    t0: float
    per: float
    rp: float
    a: float
    inc: float
    ecc: float
    w: float
    u: List[float]
    limb_dark: str

    def __post_init__(self):
        """Validate physical parameters after initialization."""
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """
        Validate that all parameters are physically plausible.

        Raises:
            ValueError: If any parameter is outside physically reasonable bounds.
        """
        if self.per <= 0:
            raise ValueError(f"Orbital period must be positive, got {self.per}")

        if not (0 < self.rp < 1):
            raise ValueError(f"Planet radius ratio must be 0 < rp < 1, got {self.rp}")

        if self.a <= 1:
            raise ValueError(f"Semi-major axis must be > 1 stellar radius, got {self.a}")

        if not (0 <= self.inc <= 90):
            raise ValueError(f"Inclination must be 0-90 degrees, got {self.inc}")

        if not (0 <= self.ecc < 1):
            raise ValueError(f"Eccentricity must be 0 ≤ ecc < 1, got {self.ecc}")

        if not (0 <= self.w <= 360):
            raise ValueError(f"Longitude of periastron must be 0-360 degrees, got {self.w}")

        if self.limb_dark == "quadratic" and len(self.u) != 2:
            raise ValueError(f"Quadratic limb darkening requires 2 coefficients, got {len(self.u)}")

        if any(coeff < 0 or coeff > 1 for coeff in self.u):
            raise ValueError(f"Limb darkening coefficients must be 0-1, got {self.u}")


class ParameterSampler:
    """
    Generates realistic combinations of physical parameters for exoplanet systems.

    This class encapsulates the logic for creating physically plausible star-planet
    systems by sampling parameters from realistic ranges based on observed exoplanet
    populations. The sampling ranges are designed to cover the diversity of known
    exoplanets while ensuring all generated systems are physically stable.

    The sampler uses uniform distributions for simplicity, but could be extended
    to use more sophisticated distributions based on exoplanet occurrence rates.

    Attributes:
        rng (np.random.Generator): Random number generator for reproducible sampling.

    Example:
        >>> sampler = ParameterSampler(seed=42)
        >>> params = sampler.sample_plausible_system()
        >>> print(f"Generated system with period {params.per:.2f} days")
    """

    def __init__(self, seed: int = None):
        """
        Initialize the parameter sampler.

        Args:
            seed (int, optional): Random seed for reproducible parameter generation.
                If None, uses system entropy for random initialization.
        """
        self.rng = np.random.default_rng(seed)

    def sample_plausible_system(self) -> SystemParameters:
        """
        Sample a physically plausible exoplanet system.

        Generates random parameters from realistic ranges based on the observed
        population of exoplanets. All parameters are sampled to ensure the
        resulting system is physically stable and produces observable transits.

        Sampling Ranges:
            - t0: [0, 1] days (phase reference)
            - per: [1, 100] days (short to moderate periods)
            - rp: [0.01, 0.2] (Earth-size to Super-Jupiter)
            - a: [5, 50] stellar radii (stable orbits)
            - inc: [87, 90] degrees (ensures transits)
            - ecc: [0, 0.4] (low to moderate eccentricity)
            - w: [0, 360] degrees (random orientation)
            - u: [0, 1] for each coefficient (realistic limb darkening)

        Returns:
            SystemParameters: A validated set of physical parameters defining
                a complete star-planet system ready for transit modeling.

        Raises:
            ValueError: If the sampled parameters fail validation (should be rare
                given the carefully chosen ranges).
        """
        # Sample basic orbital parameters
        t0 = self.rng.uniform(0.0, 1.0)
        per = self.rng.uniform(1.0, 100.0)
        rp = self.rng.uniform(0.01, 0.2)
        a = self.rng.uniform(5.0, 50.0)

        # Sample inclination to ensure transits occur
        inc = self.rng.uniform(87.0, 90.0)

        # Sample eccentricity and orientation
        ecc = self.rng.uniform(0.0, 0.4)
        w = self.rng.uniform(0.0, 360.0)

        # Sample quadratic limb darkening coefficients
        u1 = self.rng.uniform(0.0, 1.0)
        u2 = self.rng.uniform(0.0, 1.0)
        u = [u1, u2]

        # Use quadratic limb darkening model
        limb_dark = "quadratic"

        return SystemParameters(
            t0=t0, per=per, rp=rp, a=a, inc=inc,
            ecc=ecc, w=w, u=u, limb_dark=limb_dark
        )


class ExoplanetPhysicsEngine:
    """
    Core physics engine for generating mathematically perfect transit light curves.

    This class uses the batman-package library to perform high-precision transit
    modeling based on the physical parameters of a star-planet system. The engine
    generates noise-free, "ground truth" light curves that serve as the foundation
    for more realistic synthetic data generation.

    The batman-package is chosen for its C-optimized performance and wide adoption
    in the astrophysics community. It implements the analytical transit model
    of Mandel & Agol (2002) with support for various limb-darkening laws.

    Attributes:
        params (SystemParameters): The physical parameters defining the star-planet system.
        batman_params (batman.TransitParams): Batman-specific parameter object.

    Example:
        >>> params = SystemParameters(...)
        >>> engine = ExoplanetPhysicsEngine(params)
        >>> time = np.linspace(0, 10, 1000)
        >>> light_curve = engine.generate_light_curve(time)
        >>> transit_depth = 1.0 - light_curve['flux'].min()
    """

    def __init__(self, params: SystemParameters):
        """
        Initialize the physics engine with system parameters.

        Args:
            params (SystemParameters): Validated physical parameters defining
                the star-planet system to be modeled.
        """
        self.params = params
        self._setup_batman_params()

    def _setup_batman_params(self) -> None:
        """
        Configure batman TransitParams object from SystemParameters.

        This method translates our structured SystemParameters into the format
        expected by the batman-package library.
        """
        self.batman_params = batman.TransitParams()

        # Copy all parameters from our dataclass to batman format
        self.batman_params.t0 = self.params.t0
        self.batman_params.per = self.params.per
        self.batman_params.rp = self.params.rp
        self.batman_params.a = self.params.a
        self.batman_params.inc = self.params.inc
        self.batman_params.ecc = self.params.ecc
        self.batman_params.w = self.params.w
        self.batman_params.u = self.params.u
        self.batman_params.limb_dark = self.params.limb_dark

    def generate_light_curve(self, time_array: np.ndarray) -> pl.DataFrame:
        """
        Generate a mathematically perfect transit light curve.

        This method uses the batman-package to compute the theoretical light curve
        for the configured star-planet system over the specified time array.
        The resulting light curve is noise-free and represents the pure physical
        signal of the planetary transit.

        Args:
            time_array (np.ndarray): Array of time values (in days) at which to
                evaluate the light curve. Should be evenly spaced for best results.

        Returns:
            pl.DataFrame: A polars DataFrame with two columns:
                - 'time': The input time array
                - 'flux': Normalized flux values (1.0 = no transit, <1.0 = in transit)

        Raises:
            ValueError: If time_array is empty or contains invalid values.
            RuntimeError: If batman computation fails.

        Example:
            >>> time = np.linspace(-0.1, 0.1, 1000)  # 0.2 days around transit
            >>> lc = engine.generate_light_curve(time)
            >>> print(f"Transit depth: {1.0 - lc['flux'].min():.4f}")
        """
        # Validate input
        if len(time_array) == 0:
            raise ValueError("time_array cannot be empty")

        if not np.all(np.isfinite(time_array)):
            raise ValueError("time_array contains non-finite values")

        try:
            # Initialize batman model with our parameters and time array
            model = batman.TransitModel(self.batman_params, time_array)

            # Generate the light curve
            flux = model.light_curve(self.batman_params)

            # Create polars DataFrame for efficient data handling
            light_curve_df = pl.DataFrame({
                "time": time_array,
                "flux": flux
            })

            return light_curve_df

        except Exception as e:
            raise RuntimeError(f"Batman computation failed: {str(e)}") from e


class StellarVariabilityModel:
    """
    Gaussian Process model for generating realistic stellar variability signals.

    This class models the quasi-periodic brightness variations caused by stellar
    rotation and surface features like starspots. Real stars are not perfectly
    stable - they have dark spots on their surface that cause periodic dimming
    as the star rotates. This creates correlated noise that can hide or mimic
    planetary transits.

    The model uses Gaussian Processes (GPs) with a RotationTerm kernel from the
    celerite2 library, which is specifically designed for modeling stellar
    rotation signals. This approach is standard in astrophysics and provides
    realistic quasi-periodic variability with the proper temporal correlations.

    Key Features:
    - Physics-based modeling of stellar rotation and starspots
    - Quasi-periodic signals with realistic correlation structure
    - High-performance computation using celerite2's Rust backend
    - Configurable amplitude and rotation period

    Example:
        >>> model = StellarVariabilityModel()
        >>> time = np.linspace(0, 100, 10000)
        >>> variability = model.generate_variability(time, rotation_period=25.0, amplitude=0.001)
        >>> print(f"Stellar variability RMS: {np.std(variability):.6f}")
    """

    def __init__(self, Q0: float = 10.0, dQ: float = 1.0, seed: Optional[int] = None):
        """
        Initialize the stellar variability model.

        Args:
            Q0 (float): Quality factor of the primary oscillation mode. Controls
                the coherence of the rotation signal. Higher values = more coherent.
                Default 10.0 is realistic for solar-type stars.
            dQ (float): Difference in quality factors between modes. Controls the
                complexity of the rotation signal. Default 1.0 provides realistic
                multi-mode behavior.
            seed (int, optional): Random seed for reproducible variability generation.
                If None, uses system entropy.
        """
        self.Q0 = Q0
        self.dQ = dQ
        self.rng = np.random.default_rng(seed)

    def generate_variability(
        self,
        time_array: np.ndarray,
        stellar_rotation_period: float,
        amplitude: float
    ) -> np.ndarray:
        """
        Generate realistic stellar variability using Gaussian Processes.

        This method creates a quasi-periodic signal that mimics the brightness
        variations caused by stellar rotation and surface features. The signal
        has the proper temporal correlations and realistic amplitude scaling.

        Args:
            time_array (np.ndarray): Array of time values (in days) at which to
                evaluate the stellar variability. Should be evenly spaced.
            stellar_rotation_period (float): Rotation period of the star in days.
                Typical values: 10-40 days for solar-type stars.
            amplitude (float): RMS amplitude of the variability as a fraction
                of the stellar flux. Typical values: 0.0001-0.01 (0.01%-1%).

        Returns:
            np.ndarray: Array of stellar variability values with the same length
                as time_array. These are flux variations to be added to the
                perfect transit signal.

        Raises:
            ValueError: If inputs are invalid or out of realistic ranges.
            RuntimeError: If GP computation fails.

        Example:
            >>> time = np.linspace(0, 100, 5000)
            >>> variability = model.generate_variability(time, 25.0, 0.001)
            >>> # variability now contains realistic stellar rotation signal
        """
        # Validate inputs
        if len(time_array) == 0:
            raise ValueError("time_array cannot be empty")

        if not np.all(np.isfinite(time_array)):
            raise ValueError("time_array contains non-finite values")

        if stellar_rotation_period <= 0:
            raise ValueError(f"Rotation period must be positive, got {stellar_rotation_period}")

        if amplitude < 0:
            raise ValueError(f"Amplitude must be non-negative, got {amplitude}")

        if amplitude > 0.1:  # 10% variability is unrealistically high
            raise ValueError(f"Amplitude {amplitude} is unrealistically high (>10%)")

        try:
            # Create RotationTerm kernel for quasi-periodic stellar rotation
            # This kernel models the rotation signal with realistic damping
            kernel = terms.RotationTerm(
                amp=amplitude**2,  # Variance (amplitude squared)
                period=stellar_rotation_period,
                Q0=self.Q0,
                dQ=self.dQ
            )

            # Initialize Gaussian Process with the rotation kernel
            gp = celerite2.GaussianProcess(kernel, mean=0.0)

            # Compute the GP model for the given time array
            gp.compute(time_array)

            # Generate a single realization of the stellar variability
            # This creates a quasi-periodic signal with realistic correlations
            variability = gp.sample(size=1, random_state=self.rng)[0]

            return variability

        except Exception as e:
            raise RuntimeError(f"Stellar variability generation failed: {str(e)}") from e


class InstrumentalNoiseModel:
    """
    Generator for realistic instrumental noise with proper temporal structure.

    This class models the complex noise characteristics of space-based telescopes
    like Kepler and TESS. Unlike simple white noise, real instrumental noise has
    temporal correlations and a specific "texture" that follows a colored noise
    power spectrum (1/f^α). This creates the authentic noise signature that
    domain experts can recognize.

    The model generates colored noise by creating white noise in the frequency
    domain, applying a power law filter, and transforming back to the time domain.
    This approach captures the essential characteristics of real instrumental
    systematics without requiring training data.

    Key Features:
    - Colored noise with configurable power law spectrum (1/f^α)
    - Proper temporal correlations matching real instruments
    - Efficient FFT-based generation for large datasets
    - Support for different noise colors (white, pink, brown, etc.)

    Example:
        >>> model = InstrumentalNoiseModel(seed=42)
        >>> time = np.linspace(0, 100, 10000)
        >>> noise = model.generate_noise(time, noise_level=0.0001, color_alpha=1.0)
        >>> print(f"Instrumental noise RMS: {np.std(noise):.6f}")
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the instrumental noise model.

        Args:
            seed (int, optional): Random seed for reproducible noise generation.
                If None, uses system entropy for random initialization.
        """
        self.rng = np.random.default_rng(seed)

    def generate_noise(
        self,
        time_array: np.ndarray,
        noise_level: float,
        color_alpha: float = 1.0
    ) -> np.ndarray:
        """
        Generate realistic instrumental noise with colored power spectrum.

        This method creates noise with a 1/f^α power spectrum that mimics the
        temporal correlations found in real space telescope data. The noise
        has proper scaling and realistic statistical properties.

        Args:
            time_array (np.ndarray): Array of time values at which to generate
                noise. The length determines the noise array size.
            noise_level (float): RMS amplitude of the noise as a fraction of
                the stellar flux. Typical values: 1e-5 to 1e-3 (10-1000 ppm).
            color_alpha (float): Power law exponent for colored noise spectrum.
                - α = 0: White noise (flat spectrum)
                - α = 1: Pink noise (1/f spectrum) - realistic for most instruments
                - α = 2: Brown noise (1/f² spectrum) - very correlated
                Default 1.0 provides realistic pink noise.

        Returns:
            np.ndarray: Array of instrumental noise values with the same length
                as time_array. These are flux variations to be added to the
                perfect transit signal.

        Raises:
            ValueError: If inputs are invalid or out of realistic ranges.
            RuntimeError: If noise generation fails.

        Example:
            >>> time = np.linspace(0, 100, 5000)
            >>> noise = model.generate_noise(time, 0.0001, 1.0)
            >>> # noise now contains realistic instrumental systematics
        """
        # Validate inputs
        if len(time_array) == 0:
            raise ValueError("time_array cannot be empty")

        if noise_level < 0:
            raise ValueError(f"Noise level must be non-negative, got {noise_level}")

        if noise_level > 0.01:  # 1% noise is unrealistically high for space telescopes
            raise ValueError(f"Noise level {noise_level} is unrealistically high (>1%)")

        if color_alpha < 0 or color_alpha > 3:
            raise ValueError(f"Color alpha must be 0-3, got {color_alpha}")

        try:
            n_points = len(time_array)

            # Handle edge case of very small arrays
            if n_points < 4:
                # For very small arrays, just return white noise
                return self.rng.normal(0, noise_level, n_points)

            # Generate white noise in frequency domain
            # Use complex white noise for proper FFT handling
            white_noise_fft = self.rng.normal(0, 1, n_points) + 1j * self.rng.normal(0, 1, n_points)

            # Create frequency array for power law filtering
            freqs = scipy.fft.fftfreq(n_points)

            # Avoid division by zero at DC component
            freqs[0] = freqs[1]  # Set DC to first non-zero frequency

            # Apply 1/f^α power law filter
            # The power spectrum is proportional to 1/f^α
            # So the amplitude scaling is proportional to 1/f^(α/2)
            power_law_filter = np.abs(freqs) ** (-color_alpha / 2.0)

            # Apply the filter to create colored noise in frequency domain
            colored_noise_fft = white_noise_fft * power_law_filter

            # Transform back to time domain
            colored_noise = scipy.fft.ifft(colored_noise_fft).real

            # Normalize to desired noise level (RMS)
            current_rms = np.std(colored_noise)
            if current_rms > 0:
                colored_noise = colored_noise * (noise_level / current_rms)
            else:
                # Fallback to white noise if normalization fails
                colored_noise = self.rng.normal(0, noise_level, n_points)

            return colored_noise

        except Exception as e:
            raise RuntimeError(f"Instrumental noise generation failed: {str(e)}") from e


class RealisticExoplanetGenerator:
    """
    Complete orchestrator for generating realistic exoplanet transit data.

    This class combines all components of the exoplanet data generation pipeline
    to produce synthetic light curves that are indistinguishable from real
    astronomical observations. It integrates the perfect physics engine with
    realistic stellar variability and instrumental noise models.

    The generator follows the complete workflow:
    1. Sample realistic system parameters
    2. Generate perfect transit physics
    3. Add stellar variability (starspot rotation signals)
    4. Add instrumental noise (telescope systematics)
    5. Combine all components into final realistic data

    This provides a single, high-level API for the GIF framework to generate
    challenging, realistic training data that will push the DU core to develop
    robust understanding of exoplanet signals.

    Key Features:
    - Complete end-to-end realistic data generation
    - Configurable noise levels and characteristics
    - Ground truth parameter tracking for supervised learning
    - Efficient polars DataFrame output
    - Reproducible generation with seeding

    Example:
        >>> generator = RealisticExoplanetGenerator(seed=42)
        >>> data = generator.generate()
        >>> print(f"Generated realistic light curve with {len(data)} points")
        >>> print(f"Transit depth: {1.0 - data['flux'].min():.4f}")
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        stellar_variability_amplitude: float = 0.001,
        instrumental_noise_level: float = 0.0001,
        noise_color_alpha: float = 1.0
    ):
        """
        Initialize the realistic exoplanet generator.

        Args:
            seed (int, optional): Random seed for reproducible data generation.
                If None, uses system entropy.
            stellar_variability_amplitude (float): RMS amplitude of stellar
                variability as a fraction of flux. Default 0.001 (0.1%) is
                realistic for solar-type stars.
            instrumental_noise_level (float): RMS amplitude of instrumental
                noise as a fraction of flux. Default 0.0001 (100 ppm) is
                realistic for space telescopes like Kepler/TESS.
            noise_color_alpha (float): Power law exponent for instrumental
                noise spectrum. Default 1.0 (pink noise) is realistic.
        """
        self.seed = seed
        self.stellar_variability_amplitude = stellar_variability_amplitude
        self.instrumental_noise_level = instrumental_noise_level
        self.noise_color_alpha = noise_color_alpha

        # Initialize all components with consistent seeding
        self.parameter_sampler = ParameterSampler(seed=seed)
        self.stellar_model = StellarVariabilityModel(seed=seed)
        self.noise_model = InstrumentalNoiseModel(seed=seed)

    def generate(
        self,
        observation_duration: float = 100.0,
        cadence_minutes: float = 30.0
    ) -> pl.DataFrame:
        """
        Generate a complete realistic exoplanet light curve.

        This method executes the full pipeline to create synthetic data that
        combines perfect transit physics with realistic noise sources. The
        result is indistinguishable from real astronomical observations.

        Args:
            observation_duration (float): Total observation time in days.
                Default 100 days provides good coverage for most transit periods.
            cadence_minutes (float): Time between observations in minutes.
                Default 30 minutes matches typical space telescope cadence.

        Returns:
            pl.DataFrame: Complete realistic light curve with columns:
                - 'time': Time values in days
                - 'flux': Final realistic flux (perfect + stellar + instrumental)
                - 'perfect_flux': Perfect physics-only flux (for comparison)
                - 'stellar_variability': Stellar rotation signal component
                - 'instrumental_noise': Instrumental noise component
                - Plus ground truth parameters as metadata columns

        Raises:
            ValueError: If observation parameters are invalid.
            RuntimeError: If any component of the generation pipeline fails.

        Example:
            >>> generator = RealisticExoplanetGenerator(seed=123)
            >>> data = generator.generate(observation_duration=50.0, cadence_minutes=15.0)
            >>> # Analyze the realistic synthetic data
            >>> snr = (1.0 - data['perfect_flux'].min()) / data['instrumental_noise'].std()
            >>> print(f"Transit signal-to-noise ratio: {snr:.1f}")
        """
        # Validate inputs
        if observation_duration <= 0:
            raise ValueError(f"Observation duration must be positive, got {observation_duration}")

        if cadence_minutes <= 0:
            raise ValueError(f"Cadence must be positive, got {cadence_minutes}")

        try:
            # 1. Sample realistic system parameters
            params = self.parameter_sampler.sample_plausible_system()

            # 2. Create time array based on observation parameters
            cadence_days = cadence_minutes / (24 * 60)  # Convert minutes to days
            n_points = int(observation_duration / cadence_days)
            time_array = np.linspace(0, observation_duration, n_points)

            # 3. Generate perfect transit physics
            physics_engine = ExoplanetPhysicsEngine(params)
            perfect_light_curve = physics_engine.generate_light_curve(time_array)
            perfect_flux = perfect_light_curve['flux'].to_numpy()

            # 4. Generate stellar variability
            # Use orbital period as proxy for stellar rotation period
            stellar_rotation_period = params.per * (1.0 + 0.5 * np.random.random())  # Add some variation
            stellar_variability = self.stellar_model.generate_variability(
                time_array,
                stellar_rotation_period,
                self.stellar_variability_amplitude
            )

            # 5. Generate instrumental noise
            instrumental_noise = self.noise_model.generate_noise(
                time_array,
                self.instrumental_noise_level,
                self.noise_color_alpha
            )

            # 6. Combine all components
            final_flux = perfect_flux + stellar_variability + instrumental_noise

            # 7. Create comprehensive output DataFrame
            result_df = pl.DataFrame({
                "time": time_array,
                "flux": final_flux,
                "perfect_flux": perfect_flux,
                "stellar_variability": stellar_variability,
                "instrumental_noise": instrumental_noise,
                # Ground truth parameters for supervised learning
                "transit_period": [params.per] * len(time_array),
                "planet_radius_ratio": [params.rp] * len(time_array),
                "transit_depth": [params.rp**2] * len(time_array),
                "orbital_inclination": [params.inc] * len(time_array),
                "eccentricity": [params.ecc] * len(time_array),
                "stellar_rotation_period": [stellar_rotation_period] * len(time_array),
            })

            return result_df

        except Exception as e:
            raise RuntimeError(f"Realistic data generation failed: {str(e)}") from e


# Example usage and comprehensive validation
if __name__ == "__main__":
    """
    Example usage demonstrating the complete realistic data generation pipeline.

    This section provides working examples of both the perfect physics engine
    and the complete realistic generator, showing the difference between
    perfect and realistic synthetic data.
    """
    print("Exoplanet Realistic Data Generator - Complete Pipeline")
    print("=" * 60)

    # PART 1: Demonstrate Realistic Data Generation
    print("\nPART 1: REALISTIC DATA GENERATION")
    print("-" * 40)

    # 1. Create realistic generator
    print("1. Creating realistic exoplanet generator...")
    realistic_generator = RealisticExoplanetGenerator(
        seed=42,
        stellar_variability_amplitude=0.001,  # 0.1% stellar variability
        instrumental_noise_level=0.0001,      # 100 ppm instrumental noise
        noise_color_alpha=1.0                 # Pink noise (1/f)
    )

    # 2. Generate realistic data
    print("2. Generating realistic synthetic data...")
    realistic_data = realistic_generator.generate(
        observation_duration=50.0,  # 50 days of observations
        cadence_minutes=30.0        # 30-minute cadence
    )

    # 3. Analyze realistic data
    print("3. Analyzing realistic synthetic data...")
    n_points = len(realistic_data)
    time_span = realistic_data['time'].max() - realistic_data['time'].min()

    # Extract components
    perfect_flux = realistic_data['perfect_flux'].to_numpy()
    stellar_var = realistic_data['stellar_variability'].to_numpy()
    inst_noise = realistic_data['instrumental_noise'].to_numpy()
    final_flux = realistic_data['flux'].to_numpy()

    # Calculate statistics
    transit_depth_perfect = 1.0 - perfect_flux.min()
    transit_depth_realistic = 1.0 - final_flux.min()
    stellar_rms = np.std(stellar_var)
    noise_rms = np.std(inst_noise)

    print(f"   - Data points: {n_points}")
    print(f"   - Time span: {time_span:.1f} days")
    print(f"   - Perfect transit depth: {transit_depth_perfect:.4f} ({transit_depth_perfect*100:.2f}%)")
    print(f"   - Realistic transit depth: {transit_depth_realistic:.4f} ({transit_depth_realistic*100:.2f}%)")
    print(f"   - Stellar variability RMS: {stellar_rms:.6f} ({stellar_rms*1e6:.0f} ppm)")
    print(f"   - Instrumental noise RMS: {noise_rms:.6f} ({noise_rms*1e6:.0f} ppm)")

    # Calculate signal-to-noise ratio
    signal_amplitude = transit_depth_perfect
    total_noise_rms = np.sqrt(stellar_rms**2 + noise_rms**2)
    snr = signal_amplitude / total_noise_rms if total_noise_rms > 0 else float('inf')

    print(f"   - Transit signal-to-noise ratio: {snr:.1f}")

    # 4. Validation of realistic data
    print("4. Validating realistic data generation...")

    # Check that components are properly combined
    reconstructed_flux = perfect_flux + stellar_var + inst_noise
    reconstruction_error = np.mean(np.abs(final_flux - reconstructed_flux))

    if reconstruction_error < 1e-10:
        print("   ✓ Signal components properly combined")
    else:
        print(f"   ⚠ Signal reconstruction error: {reconstruction_error:.2e}")

    # Check noise characteristics
    if 0.5e-3 < stellar_rms < 2e-3:  # Expected range for stellar variability
        print("   ✓ Stellar variability amplitude is realistic")
    else:
        print(f"   ⚠ Stellar variability amplitude unusual: {stellar_rms:.6f}")

    if 0.5e-4 < noise_rms < 2e-4:  # Expected range for instrumental noise
        print("   ✓ Instrumental noise level is realistic")
    else:
        print(f"   ⚠ Instrumental noise level unusual: {noise_rms:.6f}")

    # PART 2: Compare with Perfect Physics
    print("\nPART 2: PERFECT PHYSICS COMPARISON")
    print("-" * 40)

    # 5. Generate perfect physics for comparison
    print("5. Generating perfect physics data for comparison...")
    sampler = ParameterSampler(seed=42)  # Same seed for fair comparison
    params = sampler.sample_plausible_system()

    # Create time array for single transit
    transit_duration = 0.2
    time_array = np.linspace(
        params.t0 - transit_duration/2,
        params.t0 + transit_duration/2,
        1000
    )

    engine = ExoplanetPhysicsEngine(params)
    perfect_light_curve = engine.generate_light_curve(time_array)

    # 6. Compare perfect vs realistic
    print("6. Comparing perfect vs realistic data...")
    perfect_depth = 1.0 - perfect_light_curve['flux'].min()
    perfect_baseline = perfect_light_curve['flux'].mean()

    print(f"   Perfect physics:")
    print(f"   - Transit depth: {perfect_depth:.4f} ({perfect_depth*100:.2f}%)")
    print(f"   - Baseline flux: {perfect_baseline:.6f}")
    print(f"   - Noise level: 0.000000 (perfect)")

    print(f"   Realistic data:")
    print(f"   - Transit depth: {transit_depth_realistic:.4f} ({transit_depth_realistic*100:.2f}%)")
    print(f"   - Effective noise: {total_noise_rms:.6f} ({total_noise_rms*1e6:.0f} ppm)")
    print(f"   - SNR degradation: {snr:.1f}x")

    # 7. Final validation
    print("\n7. Final validation...")

    if snr > 5.0:  # Transit should be detectable
        print("   ✓ Transit remains detectable in realistic data")
    else:
        print(f"   ⚠ Transit may be difficult to detect (SNR = {snr:.1f})")

    if 0.8 < (transit_depth_realistic / transit_depth_perfect) < 1.2:
        print("   ✓ Realistic transit depth is reasonable")
    else:
        ratio = transit_depth_realistic / transit_depth_perfect
        print(f"   ⚠ Realistic transit depth ratio unusual: {ratio:.2f}")

    # Optional: Save example data
    try:
        realistic_data.write_csv("realistic_exoplanet_data.csv")
        perfect_light_curve.write_csv("perfect_transit.csv")
        print("\n   Example data saved:")
        print("   - realistic_exoplanet_data.csv (complete realistic dataset)")
        print("   - perfect_transit.csv (perfect physics comparison)")
    except Exception as e:
        print(f"\n   Could not save example files: {e}")

    print("\n" + "=" * 60)
    print("REALISTIC DATA GENERATION COMPLETE!")
    print("Ready for integration with GIF framework encoder interfaces.")
    print("=" * 60)
