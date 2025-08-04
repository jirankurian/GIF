**Your Current Task: Task 1.3 - The ECG Generator: Core Biophysical Engine**

**Protocol Reminder:** Before you begin, you must execute your full **Cognitive Cycle**. This involves reviewing the `/Rules` directory, the `\docs` directory, the `/Reference/`, reading all logs in the `.context/` (including from the previous), and analyzing the current codebase. After your analysis, formulate your micro-plan for this task and present it for approval.

-----

### **Task Objective**

Your goal is to build the foundational physics engine for our electrocardiogram (ECG) data generator. This module will be responsible for creating **mathematically clean, "textbook" ECG signals**. It will not include noise or patient-specific artifacts; those will be added in the next task. This engine will simulate the fundamental biophysics of the heart's electrical cycle.

-----

### **Domain & Technical Specifications**

#### **1. The Biophysical Model**

  * **Domain Context:** An ECG signal represents the electrical activity of a heartbeat. A healthy heartbeat has a distinct and repeating pattern of waves: the P wave, the QRS complex, and the T wave. This entire pattern is not random; it is the result of a predictable, cyclical process in the heart's electrical system.

  * **Technical Approach:** We will simulate this process using a well-established **dynamical systems model**. Specifically, you will implement the model described by **McSharry et al. (2003)**. This model uses a set of three coupled Ordinary Differential Equations (ODEs) to describe the state of the heart's electrical signal as it moves through its cycle.

  * **The ODE System:** The state of the system is described by three variables, `(x, y, z)`. The final ECG signal we will use is the `x` variable. The equations that govern the system are:

    $$
    $$$$\\frac{dx}{dt} = y

    $$
    $$$$$$
    $$\\frac{dy}{dt} = z

    $$
    $$$$$$
    $$\\frac{dz}{dt} = \\alpha z - \\omega^2 y - \\text{force}(t)

    $$
    $$$$The `force(t)` term is what creates the distinct P, Q, R, S, and T waves. It is a sum of Gaussian functions, where each Gaussian corresponds to one of the waves.

#### **2. The Core Engine Implementation**

  * **Action:** You will create a new file, `data_generators/ecg_generator.py`, and implement a class named `ECGPhysicsEngine`.
  * **Justification:** This class will encapsulate the complex biophysical model, providing a simple and clean interface for generating ECG signals. This separates the complex physics from the rest of the application.
  * **Required Libraries:** You will need **`numpy`**, **`scipy.integrate.solve_ivp`** (for solving the ODEs), and **`polars`**.

#### **3. Implementation Details for `ECGPhysicsEngine`**

  * **`__init__(self, heart_rate_bpm: float = 60.0)`:**

      * The initializer will take a target heart rate in beats per minute (BPM).
      * It will store the angular frequency, `omega`, which is calculated from the heart rate: $$\omega = \frac{2 \pi \cdot \text{heart_rate_bpm}}{60}$$.
      * It will also store the hardcoded parameters for the Gaussian functions that define the P, Q, R, S, and T waves. You will use the following standard values from the McSharry paper:
        ```python
        # (theta_i, a_i, b_i) for P, Q, R, S, T waves
        self.pqrst_params = {
            'P': (-1/3 * np.pi, 1.2, 0.25),
            'Q': (-1/12 * np.pi, -5.0, 0.1),
            'R': (0.0, 30.0, 0.1),
            'S': (1/12 * np.pi, -7.5, 0.1),
            'T': (1/2 * np.pi, 0.75, 0.4),
        }
        ```

  * **`_ecg_model_odes(self, t: float, state: list) -> list`:**

      * This will be a private helper method that defines the system of three ODEs.
      * It takes the current time `t` and the current `state = [x, y, z]` as input.
      * It calculates the angle on the limit cycle, `theta = np.arctan2(y, x)`.
      * It calculates the `force` term by summing the Gaussian functions for each wave (P, Q, R, S, T) using the parameters stored in `self.pqrst_params`. The formula for each Gaussian is: $$a_i \cdot \exp\left(-\frac{(\theta - \theta_i)^2}{2b_i^2}\right)$$.
      * It returns the derivatives `[dx/dt, dy/dt, dz/dt]` as a list.

  * **`generate_ecg_segment(self, duration_seconds: float, sampling_rate: int) -> polars.DataFrame`:**

      * This is the main public method.
      * It creates the time array for the simulation using `np.linspace(0, duration_seconds, duration_seconds * sampling_rate)`.
      * It defines the initial state of the system, e.g., `initial_state = [1.0, 0.0, 0.0]`.
      * It calls **`scipy.integrate.solve_ivp`**, passing it the `_ecg_model_odes` method, the time span, the initial state, and the time array for evaluation (`t_eval`).
      * From the solution returned by `solve_ivp`, it extracts the `x` component (`solution.y[0]`).
      * It returns the final signal as a **`polars.DataFrame`** with two columns: `"time"` and `"voltage"`.

-----

**Summary of your task:**

1.  Create a new file named `data_generators/ecg_generator.py`.
2.  Inside this file, implement the `ECGPhysicsEngine` class.
3.  The class must contain an `__init__` method to set up parameters, a private `_ecg_model_odes` method to define the McSharry ODE system, and a public `generate_ecg_segment` method that uses `scipy.integrate.solve_ivp` to generate a clean ECG signal.
4.  Ensure the final output is a `polars.DataFrame`.
5.  Provide professional docstrings for the class and its methods.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**