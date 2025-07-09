**Your Current Task: Task 4.3 - Assembling the Full Exoplanet POC Pipeline**

**Protocol Reminder:** Before you begin, you must execute your full **Cognitive Cycle**. Review the `/Rules` directory, the `/Reference/` for Phase 4, read all logs in the `.context/` to confirm the completion of Tasks 4.1 and 4.2, and analyze the existing project structure. You will see that we have all the necessary components (`data_generator`, `encoder`, `decoder`, `du_core`, `trainer`, `simulator`) but they are not yet connected into a single, runnable application. This task builds that final connection. After your analysis, formulate your micro-plan and present it for approval.

-----

### **Task Objective**

Your goal is to create the main **entry point script** for our exoplanetary science experiment. This script will act as the master conductor, responsible for initializing all the different components we've built, configuring them for a specific experiment, and orchestrating the entire training and evaluation pipeline from start to finish.

-----

### **Domain & Technical Specifications**

#### **1. The Importance of Configuration Management**

  * **Domain Context:** In scientific computing and professional software development, it is critical to separate the **configuration** of an experiment from the **code** that runs it. We should never "hardcode" parameters like learning rates, layer sizes, or file paths directly into our main logic.
  * **Technical Approach:** We will create a dedicated Python configuration file (`config_exo.py`). This file will act as a central dashboard for all experiment parameters. The main script will import its settings from this file. This makes our experiments highly reproducible, easy to modify, and easy to track (as we can save the config file for each experiment run).

#### **2. The "Entry Point" Script**

  * **Domain Context:** A complex application needs a single, clear starting point. This is often called the "main" script or entry point. All the logic for setting up and running the application should be coordinated from here.
  * **Technical Approach:** You will create a `main_exo.py` file that contains the primary orchestration logic. This script will read the configuration, build the necessary objects, and execute the training and evaluation loop.

-----

### **Step-by-Step Implementation Plan**

#### **Step 1: Create the Experiment Configuration File**

  * **Action:** Create a new file: `applications/poc_exoplanet/config_exo.py`.
  * **Implementation:** Inside this file, define a Python dictionary or a `dataclass` to hold all the parameters for our experiment. This provides a single source of truth.
    ```python
    # applications/poc_exoplanet/config_exo.py
    from typing import List

    # Using a dictionary for simplicity
    EXP_CONFIG = {
        # Data Generator Settings
        "num_training_samples": 10000,
        "num_test_samples": 2000,
        "data_duration_s": 27.4, # Approx one TESS sector
        "data_sampling_rate": 120, # samples per minute

        # DU Core v1 SNN Architecture
        "input_size": 2, # From our Delta Modulation Encoder
        "hidden_sizes": [128, 64],
        "output_size": 2, # Planet vs. Not Planet
        "neuron_beta": 0.95,
        "neuron_threshold": 1.0,

        # RTL & Training Settings
        "learning_rate": 1e-3,
        "batch_size": 128,
        "num_epochs": 50,
        # Add STDP parameters if your rule needs them

        # Simulator Settings
        "sim_num_steps": 200, # Number of timesteps to simulate per sample
        "energy_per_synop": 2.5e-11, # Joules
    }
    ```

#### **Step 2: Create the Main Experiment Script**

  * **Action:** Create the main entry point file: `applications/poc_exoplanet/main_exo.py`.
  * **Implementation:** This script will contain a `main()` function that orchestrates the entire process.

#### **Step 3: Implement the Orchestration Logic in `main()`**

This is the core of the task. You will implement the following sequence of operations inside the `main()` function.

1.  **Import Everything:** Import all necessary classes: `GIF`, `DU_Core_V1`, `LightCurveEncoder`, `ExoplanetDecoder`, `Continual_Trainer`, `NeuromorphicSimulator`, `RealisticExoplanetGenerator`, and the `EXP_CONFIG` object.

2.  **Load Configuration:** `config = EXP_CONFIG`.

3.  **Instantiate All Components:** Create instances of every class, feeding them the parameters from the `config` object.

    ```python
    # 1. Instantiate Data Generator
    data_generator = RealisticExoplanetGenerator()

    # 2. Instantiate GIF components
    encoder = LightCurveEncoder(...)
    decoder = ExoplanetDecoder(...)
    du_core = DU_Core_V1(
        input_size=config["input_size"],
        hidden_sizes=config["hidden_sizes"],
        ... # etc.
    )

    # 3. Assemble the main GIF model
    gif_model = GIF(du_core=du_core)
    gif_model.attach_encoder(encoder)
    gif_model.attach_decoder(decoder)

    # 4. Instantiate Trainer and Simulator
    optimizer = torch.optim.Adam(gif_model.parameters(), lr=config["learning_rate"])
    loss_fn = torch.nn.CrossEntropyLoss()
    trainer = Continual_Trainer(gif_model, optimizer, loss_fn)
    simulator = NeuromorphicSimulator(gif_model, energy_per_synop=config["energy_per_synop"])
    ```

4.  **Implement the Training Loop:**

      * Create the main loop that iterates for `config["num_epochs"]`.
      * Inside the epoch loop, create a loop that gets data samples from your `data_generator`.
      * For each sample, you will call the `trainer.train_step(...)` method, passing the data and target labels.
      * You should collect and log the loss and accuracy from each step.

5.  **Implement the Evaluation Process:**

      * After the training loop is complete, you will run an evaluation loop on the test data (`num_test_samples`).
      * In this loop, you will pass the test samples through the `simulator.run(...)` method to get the model's predictions and the neuromorphic performance stats.
      * You will collect all predictions and stats.

6.  **Analyze and Save Results:**

      * At the end of the script, instantiate the `ContinualLearningAnalyzer` (from Task 3.4).
      * Feed it the logs from your training and evaluation runs.
      * Call its methods to calculate the final metrics and generate the summary plots.
      * Save these results (the final table data and the plots) to a `/results/poc_exoplanet/` directory.

-----

**Summary of your task:**

1.  Create the configuration file `applications/poc_exoplanet/config_exo.py` and populate it with the necessary experiment parameters.
2.  Create the main script `applications/poc_exoplanet/main_exo.py`.
3.  In the `main()` function of this script, implement the full application pipeline:
      * Load configuration.
      * Instantiate all necessary objects (generator, GIF components, trainer, simulator).
      * Assemble the `GIF` model by injecting the core and attaching the modules.
      * Run the main training loop.
      * Run the final evaluation loop.
      * Use the analysis suite to process and save the final results.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**