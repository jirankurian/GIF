**Your Current Task: Task 1.1 - The Exoplanet Generator: Core Physics Engine**

**Protocol Reminder:** Before you begin, you must execute your full **Cognitive Cycle**. Read everything in the `/Rules` directory, the `/Reference/` directory, and the `.context/` directory. After your analysis, formulate your micro-plan for this task and present it for approval.

---

### **Task Objective**

Your goal is to build the foundational physics engine for our synthetic data generator. This module will be responsible for creating **mathematically perfect, noise-free transit light curves**. A transit is the event where an exoplanet passes in front of its star from our point of view, causing a temporary, periodic dip in the star's observed brightness. This engine will serve as the "ground truth" generator for all subsequent data realism tasks.

---

### **Domain & Technical Specifications**

You will implement this using the **`batman-package`** library in Python. This library is chosen because it is a highly optimized and widely trusted tool in the astrophysics community for modeling transits, with its core computations written in C for high performance.

#### **1. The Data Structure for Physical Parameters**

First, you must define a structured way to hold all the physical parameters of a star-planet system.

* **Action:** Create a Python `dataclass` named `SystemParameters`.
* **Justification:** Using a `dataclass` provides type safety, autogenerates useful methods (`__init__`, `__repr__`), and serves as a clear, self-documenting "schema" for our physical systems.
* **Required Fields:**
    * `t0`: `float` - Time of the first transit center (in days).
    * `per`: `float` - The planet's orbital period (in days).
    * `rp`: `float` - The planet's radius (in units of the star's radius).
    * `a`: `float` - The semi-major axis of the orbit (in units of the star's radius).
    * `inc`: `float` - The orbital inclination in degrees (90° is a perfect edge-on transit).
    * `ecc`: `float` - The orbital eccentricity (0 is a perfect circle).
    * `w`: `float` - The longitude of periastron in degrees (for eccentric orbits).
    * `u`: `List[float]` - A list of limb-darkening coefficients (e.g., `[0.1, 0.3]`). Limb darkening is the effect where a star appears darker at its edges. `batman` can handle this.
    * `limb_dark`: `str` - The limb-darkening model to use (e.g., "quadratic").

#### **2. The Parameter Sampling Engine**

The framework needs a way to generate a wide variety of physically plausible star-planet systems.

* **Action:** Implement a class named `ParameterSampler`.
* **Justification:** This class encapsulates the logic for creating realistic parameter sets, separating the "system creation" logic from the "light curve generation" logic.
* **Required Method:**
    * `sample_plausible_system() -> SystemParameters`: This method will randomly sample values for each parameter from realistic ranges and return a fully populated `SystemParameters` object.
    * **Sampling Ranges:**
        * `t0`: Uniformly between 0 and 1.
        * `per`: Uniformly between 1 and 100.
        * `rp`: Uniformly between 0.01 (Earth-sized) and 0.2 (Super-Jupiter).
        * `a`: This is constrained by Kepler's Third Law, but for simplicity, sample uniformly between 5 and 50.
        * `inc`: Uniformly between 87° and 90° (to ensure most planets actually transit).
        * `ecc`: Uniformly between 0 and 0.4 (most known exoplanets have low eccentricity).
        * `w`: Uniformly between 0 and 360.
        * `u`: For a "quadratic" model, sample two coefficients `u1` and `u2` uniformly between 0 and 1.
        * `limb_dark`: Set to "quadratic".

#### **3. The Core Physics Engine**

This is the main class that will use the `batman-package` to perform the simulation.

* **Action:** Implement a class named `ExoplanetPhysicsEngine`.
* **Justification:** This class will be the core component of this module. It will be initialized with a specific system's parameters and will be responsible for the single task of generating the corresponding light curve.
* **Required Methods:**
    * `__init__(self, params: SystemParameters)`: The initializer takes a `SystemParameters` object, preparing the engine for a specific system.
    * `generate_light_curve(self, time_array: np.ndarray) -> polars.DataFrame`: This is the workhorse method.
        1.  It initializes the `batman.TransitParams` object.
        2.  It populates this object with all the values from the `self.params` dataclass.
        3.  It initializes the `batman.TransitModel`, passing it the `TransitParams` object and the `time_array`.
        4.  It calls the `model.light_curve(params)` method to generate the flux values.
        5.  It returns the result as a **`polars.DataFrame`** with two columns: `"time"` and `"flux"`.

---
**Summary of your task:**

1.  Create the file `data_generators/exoplanet_generator.py`.
2.  Inside this file, implement the three components described above:
    * The `SystemParameters` dataclass.
    * The `ParameterSampler` class.
    * The `ExoplanetPhysicsEngine` class.
3.  Ensure all classes and methods have clear docstrings explaining their purpose, arguments, and return values.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**
