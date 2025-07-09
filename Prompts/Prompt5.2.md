**Your Current Task: Task 5.2 - Engineering the Potentiation Experiment Protocol**

**Protocol Reminder:** Before you begin, execute your full **Cognitive Cycle**. Review the `/Rules` directory, the `/Reference/` for Phase 5, read all logs in the `.context/` (confirming the creation of the medical domain modules), and analyze the current project structure. This task will create the main entry point script that brings all our work together to test the central AGI hypothesis of your research. After your analysis, formulate your micro-plan and present it for approval.

-----

### **Task Objective**

Your goal is to engineer the master script that will conduct the **System Potentiation experiment**. This script will orchestrate three distinct training and evaluation runs to generate the data needed to rigorously compare our framework's learning efficiency under different conditions. This is not just a training script; it is a carefully designed scientific experiment in code form.

-----

### **Domain & Technical Specifications**

#### **1. The Scientific Method of the Experiment**

  * **Domain Context:** To prove "system potentiation," we need to show that a GIF-DU system with *diverse prior experience* learns a new task more efficiently than an identical system with no prior experience. The key challenge is to prove that this is not just "knowledge transfer" (re-using old knowledge) but a fundamental improvement in the *ability to learn*.
  * **The Experimental Control (Weight Reset):** Our method for proving this is the **weight-reset protocol**. We will take the `DU_Core` that has been trained on exoplanets, but before training it on the new medical task, we will **completely re-randomize its synaptic weights**. By doing this, we wipe out its specific knowledge of exoplanets. If this "experienced but wiped" brain still learns the medical task faster than a "naive" brain that was never exposed to anything, it proves that the prior experience must have improved the learning mechanism *itself* (e.g., its meta-parameters or structural properties), not just the knowledge it held. This is the evidence for potentiation.

#### **2. Implementation Details**

  * **Action:** You will create the main experiment script located at: `applications/poc_medical/main_med.py`.
  * **Justification:** This script will be the single entry point for running our entire second proof-of-concept, encapsulating all the experimental logic and ensuring reproducibility.

#### **3. The Experiment Configuration File**

  * **Action:** First, create a new configuration file: `applications/poc_medical/config_med.py`.
  * **Implementation:** This file will define the parameters for all three experimental runs.
    ```python
    # applications/poc_medical/config_med.py

    # Path to the trained exoplanet model from Phase 4
    PRE_TRAINED_EXO_MODEL_PATH = "results/poc_exoplanet/best_model.pth"

    # Shared settings for GIF-DU runs
    GIF_CONFIG = {
        "snn_architecture": {
            "input_size": 3, # Example for ECG feature encoder
            "hidden_sizes": [128, 64],
            "output_size": 5, # e.g., 5 AAMI arrhythmia classes
        },
        "training": {
            "learning_rate": 1e-4,
            "num_epochs": 100,
        }
    }

    # Settings for the SOTA baseline model
    SOTA_BASELINE_CONFIG = {
        "model_type": "ArrhythmiNet_CNN",
        "learning_rate": 1e-3,
        "num_epochs": 100,
    }
    ```

#### **4. The `main_med.py` Script Structure**

The script should be organized with a `main()` function that calls three separate helper functions, one for each arm of the experiment.

  * **`run_sota_baseline(config: dict)`:**

      * **Purpose:** To establish a performance benchmark using a standard, non-GIF model.
      * **Implementation:** This function will instantiate a standard medical AI model (e.g., a simple CNN architecture for ECG classification), train it on our synthetic ECG data, evaluate it, and log its final performance metrics.

  * **`run_naive_gif_du(config: dict)`:**

      * **Purpose:** This is our **control group**. It establishes the baseline learning performance of our architecture on the medical task.
      * **Implementation:**
        1.  Instantiate all GIF components (`ECG_Encoder`, `Arrhythmia_Decoder`, etc.).
        2.  Instantiate a `DU_Core_V1` with **fresh, randomly initialized weights**.
        3.  Assemble the full `GIF` model.
        4.  Run the full training process using the `Continual_Trainer`.
        5.  **Critically, you must log the model's accuracy at every N samples (or every epoch) to a specific file (e.g., `results/logs/naive_run_log.csv`).** This creates the learning curve.

  * **`run_pre_exposed_gif_du(config: dict)`:**

      * **Purpose:** This is our **experimental group**. It tests if prior diverse experience improves the learning mechanism itself.
      * **Implementation:**
        1.  Instantiate a `DU_Core_V1` with the same architecture as the naive model.
        2.  **This is the crucial step:** Load the state dictionary from the saved, fully-trained exoplanet model (`PRE_TRAINED_EXO_MODEL_PATH`).
        3.  **IMMEDIATELY after loading, you must call a `reset_weights()` method on the `DU_Core_V1` instance.** (You will need to add this simple method to the `DU_Core_V1` class, which iterates through its linear layers and calls their `reset_parameters()` method). This wipes the specific knowledge of exoplanets.
        4.  Assemble the full `GIF` model using this "experienced but wiped" core.
        5.  Run the exact same training protocol as the naive run on the exact same data.
        6.  **Log its learning curve to a separate file (e.g., `results/logs/pre_exposed_run_log.csv`).**

-----

**Summary of your task:**

1.  Create the configuration file `applications/poc_medical/config_med.py`.
2.  Add a simple `reset_weights()` method to the `DU_Core_V1` class in `gif_framework/core/du_core.py`.
3.  Create the main experiment script `applications/poc_medical/main_med.py`.
4.  Implement the three distinct functions (`run_sota_baseline`, `run_naive_gif_du`, `run_pre_exposed_gif_du`) within the script.
5.  Ensure the logic for the "Pre-Exposed" run correctly loads the old model state and then immediately calls the `reset_weights()` method before training begins.
6.  Ensure both GIF-DU runs produce detailed log files of their learning curves.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**