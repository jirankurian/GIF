**Your Current Task: Task 1.2 - The Exoplanet Generator: The Realism Layer (Generative Noise)**

**Protocol Reminder:** Before you begin, you must execute your full **Cognitive Cycle**. This involves reviewing the `/Rules`, the `\docs` directory, the `/Reference/`, reading the `.context/` logs (including the log from Task 1.1), and analyzing the current codebase (specifically the `exoplanet_generator.py` file we just created). After your analysis, formulate your micro-plan for this task and present it for approval.

---

### **Task Objective**

Your goal is to build upon the `ExoplanetPhysicsEngine` from the previous task. You will create the components necessary to layer realistic noise on top of the perfect transit signals. This will transform our generator from a simple physics simulator into a high-fidelity data source capable of producing synthetic light curves that are indistinguishable from real astronomical observations.

This task involves simulating two distinct, real-world phenomena: **Stellar Variability** and **Instrumental Noise**.

---

### **Domain & Technical Specifications**

You are to implement two new classes within the `data_generators/exoplanet_generator.py` file.

#### **1. The Stellar Variability Model**

* **Domain Context:** Real stars are not perfectly stable. They have features like starspots (cooler, darker patches) on their surface. As the star rotates, these spots move across the star's disk, causing its observed brightness to vary in a complex, semi-regular (quasi-periodic) way. This is a major source of "noise" that can hide or mimic planetary transits.
* **Technical Approach:** The standard, most powerful technique used by astrophysicists to model this type of correlated noise is **Gaussian Processes (GPs)**. We will use a high-performance GP library, **`celerite2`**, which is particularly well-suited for this task and has a Rust-based backend for speed.
* **Action:** Implement a class named `StellarVariabilityModel`.
* **Required Method:**
    * `generate_variability(self, time_array: np.ndarray, stellar_rotation_period: float, amplitude: float) -> np.ndarray`:
        1.  This method will initialize a `celerite2` GP model.
        2.  The kernel for the GP must be a **`celerite2.terms.RotationTerm`**. This specific kernel is designed to model quasi-periodic signals like stellar rotation. It requires several parameters: `amp` (the amplitude of the variability), `period` (the star's rotation period), `Q0` (the quality factor of the oscillations), and `dQ` (the difference in quality factors).
        3.  For this implementation, you will use fixed but realistic values for the quality factors (e.g., `Q0=10.0`, `dQ=1.0`). The `amp` and `period` will be inputs to the method.
        4.  You will then use the `gp.sample()` method, passing it the `time_array`, to generate a single realization of the stellar variability noise.
        5.  The method should return the resulting 1D NumPy array of noise values.

#### **2. The Instrumental Noise Model**

* **Domain Context:** Space telescopes like Kepler and TESS have their own unique noise signatures caused by electronics, thermal variations, and other systematic effects. This noise is not simple random (white) noise; it has a specific "texture" and temporal structure.
* **Technical Approach:** To create authentic-sounding instrumental noise without relying on real data, we will use a placeholder for a sophisticated generative model like a **TimeGAN** or a **Diffusion Model**. For this task, you will implement a placeholder that generates structured "colored" noise. This allows us to build the full pipeline now and replace this placeholder with a fully trained generative model later without changing the rest of the code.
* **Action:** Implement a class named `InstrumentalNoiseModel`.
* **Required Method:**
    * `generate_noise(self, time_array: np.ndarray, noise_level: float, color_alpha: float = 1.0) -> np.ndarray`:
        1.  This method will generate "colored noise," which has a power spectrum that follows a `1/f^alpha` distribution. This is a much better approximation of real instrumental noise than simple white noise.
        2.  You can generate this noise by creating a white noise signal in the frequency domain, multiplying it by the `1/f^alpha` power law, and then performing an inverse Fourier transform to get back to the time domain.
        3.  The `noise_level` parameter will control the overall amplitude (standard deviation) of the noise. The `color_alpha` parameter controls the "color" (e.g., alpha=0 is white noise, alpha=1 is pink noise, alpha=2 is brown noise).
        4.  The method should return the resulting 1D NumPy array of instrumental noise.

#### **3. The Final Integrated Generator**

* **Action:** Implement a final orchestrator class, `RealisticExoplanetGenerator`, that combines all the components.
* **Justification:** This class encapsulates the entire data generation process, providing a clean, high-level API for the rest of the framework to use.
* **Required Method:**
    * `generate(self) -> polars.DataFrame`:
        1.  Instantiate the `ParameterSampler` from Task 1.1 and get a `SystemParameters` object.
        2.  Create a `time_array` (e.g., 20,000 data points spanning 100 days).
        3.  Instantiate the `ExoplanetPhysicsEngine` with the sampled parameters and generate the `perfect_flux`.
        4.  Instantiate the `StellarVariabilityModel` and generate the `stellar_variability_noise`. Use the sampled orbital period as a proxy for the stellar rotation period for simplicity.
        5.  Instantiate the `InstrumentalNoiseModel` and generate the `instrumental_noise`.
        6.  Combine the signals: `final_flux = perfect_flux + stellar_variability_noise + instrumental_noise`.
        7.  Return a **`polars.DataFrame`** containing the `time` and the final `final_flux`. It should also include the ground truth parameters used for the generation as metadata or additional columns.

---

**Summary of your task:**

1.  Open the file `data_generators/exoplanet_generator.py`.
2.  Add the implementation for the `StellarVariabilityModel` class using `celerite2`.
3.  Add the implementation for the `InstrumentalNoiseModel` class that generates colored noise.
4.  Add the final `RealisticExoplanetGenerator` class that orchestrates the entire process and combines the perfect signal with the two noise layers.
5.  Ensure all new classes and methods have clear, professional docstrings.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**