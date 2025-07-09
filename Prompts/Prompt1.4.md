**Your Current Task: Task 1.4 - The ECG Generator: The Clinical Realism Layer**

**Protocol Reminder:** Before you begin, execute your full **Cognitive Cycle**. Review the `/Rules` directory, the `/Reference/`, the session logs in `.context/` (including the previous task), and the current project codebase, specifically the `ecg_generator.py` file. After your analysis, formulate your micro-plan for this task and present it for approval.

---

### **Task Objective**

Your goal is to build the "realism" layer for our ECG generator. You will implement a module that simulates the common sources of noise and artifact found in clinical ECG recordings. You will then create a final orchestrator class that integrates this noise layer with the clean, biophysically-modeled ECG signals generated in the previous task. The final output must be a high-fidelity signal that is challenging and realistic enough to train a robust medical diagnostic AI.

---

### **Domain & Technical Specifications**

You will continue working in the `data_generators/ecg_generator.py` file.

#### **1. The Clinical Noise Model**

* **Domain Context:** Real ECG signals are never perfectly clean. They are corrupted by various physiological and environmental factors. An AI trained only on clean data will fail in a real clinical setting. We must simulate the most common noise sources to ensure our DU Core learns to separate the true cardiac signal from these artifacts.
* **Action:** You will implement a new class named `ClinicalNoiseModel`.
* **Justification:** This class will encapsulate all the logic for generating different types of realistic noise, keeping the noise generation process modular and separate from the core signal generation.

#### **2. Implementation Details for `ClinicalNoiseModel`**

This class will have several methods, each responsible for generating a specific type of noise.

* **`_generate_baseline_wander(self, time_array: np.ndarray) -> np.ndarray`:**
    * **Domain Context:** This is a low-frequency artifact caused by the patient's breathing, which causes the baseline of the ECG to drift up and down.
    * **Technical Implementation:** You will model this using a low-frequency sine wave. The method should generate a sine wave with a realistic respiratory frequency (e.g., between 0.15 Hz and 0.3 Hz) and a small amplitude (e.g., 0.05 - 0.15 mV).
    * **Returns:** A 1D NumPy array of the baseline wander noise.

* **`_generate_muscle_artifact(self, time_array: np.ndarray) -> np.ndarray`:**
    * **Domain Context:** This is high-frequency noise caused by electrical activity from skeletal muscles (EMG artifact). It looks like sharp, random static layered on the signal.
    * **Technical Implementation:** You will model this by generating high-frequency "colored" noise. Generate a white noise signal, and then apply a high-pass filter to it (e.g., a Butterworth filter from `scipy.signal`) to keep only the high-frequency components that are characteristic of EMG noise.
    * **Returns:** A 1D NumPy array of the muscle artifact noise.

* **`_generate_powerline_interference(self, time_array: np.ndarray, sampling_rate: int) -> np.ndarray`:**
    * **Domain Context:** This is interference from the electrical grid, which manifests as a constant, high-frequency hum (typically 50 Hz or 60 Hz).
    * **Technical Implementation:** Model this as a high-amplitude sine wave with a frequency of exactly 50 Hz (a common standard). Its amplitude should be small but noticeable.
    * **Returns:** A 1D NumPy array of the powerline interference noise.

* **`generate_composite_noise(self, time_array: np.ndarray, sampling_rate: int, noise_levels: dict) -> np.ndarray`:**
    * **This is the main public method of the class.**
    * It will call the three private methods above to generate each type of noise.
    * The `noise_levels` dictionary will contain the amplitudes for each noise type (e.g., `{'baseline': 0.1, 'muscle': 0.02, 'powerline': 0.01}`).
    * It will sum these noise signals together to create a single, composite noise array.
    * **Returns:** The final, composite 1D NumPy noise array.

#### **3. The Final Integrated Generator**

* **Action:** Implement the final orchestrator class for this module, `RealisticECGGenerator`.
* **Justification:** This class provides a simple, high-level API to the rest of the framework. A user of this class doesn't need to know about ODEs or noise models; they just need to ask for realistic ECG data. This encapsulation is a core principle of good software design.
* **Required Method:**
    * `generate(self, duration_seconds: int, sampling_rate: int, heart_rate_bpm: float, noise_levels: dict) -> polars.DataFrame`:
        1.  Instantiate the `ECGPhysicsEngine` from Task 1.3 with the given `heart_rate_bpm`.
        2.  Call its `generate_ecg_segment` method to get the `clean_ecg_df`.
        3.  Extract the `time` and `voltage` arrays from the DataFrame.
        4.  Instantiate the `ClinicalNoiseModel`.
        5.  Call its `generate_composite_noise` method with the `time_array`, `sampling_rate`, and `noise_levels` to get the `composite_noise` array.
        6.  Add the noise to the clean signal: `final_voltage = clean_voltage + composite_noise`.
        7.  Create a new **`polars.DataFrame`** with columns for `"time"` and `"final_voltage"`. It should also include the ground truth parameters (`heart_rate_bpm`, noise levels, and the true arrhythmia class, which for now is "Normal Sinus Rhythm") as metadata or additional columns for later use in training.
        8.  Return the final DataFrame.

---

**Summary of your task:**

1.  Open the file `data_generators/ecg_generator.py`.
2.  Implement the `ClinicalNoiseModel` class with its three private methods for generating specific noise types and one public method for creating a composite noise signal.
3.  Implement the final orchestrator class, `RealisticECGGenerator`, which uses the `ECGPhysicsEngine` and the new `ClinicalNoiseModel` to produce a final, high-fidelity ECG signal.
4.  Ensure all new code is professionally documented with clear docstrings.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**