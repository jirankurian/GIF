### **Phase 4 Breakdown: Hardware Simulation & Proof of Concept I (Exoplanet)**

**Overall Objective:** To build a software-based simulator for neuromorphic hardware and use it to run the complete exoplanetary science proof-of-concept. This will validate the integrated GIF-DU system's ability to learn a complex scientific task from high-fidelity synthetic data and will generate the first set of key results for our research paper.

---

#### **Task 4.1: Developing the Neuromorphic Hardware Simulator**

* **Objective:** To create a simulation layer that mimics the key operational characteristics of neuromorphic hardware, specifically its event-driven nature and its potential for energy efficiency. This allows us to test our SNNs in a more realistic computational environment without needing physical hardware.
* **Key Activities:**
    1.  **Create the Simulator Module:** `simulators/neuromorphic_sim.py`.
    2.  **Implement an Event-Driven Execution Engine:** The standard `Trainer` loop processes data in fixed batches. This task involves creating a `NeuromorphicSimulator` class that operates differently. It will manage a timeline or event queue, processing spikes as they occur rather than iterating through a dense tensor. The simulation proceeds in discrete time steps, and computation only happens for neurons that are actively spiking.
    3.  **Implement an Energy Consumption Model:** The simulator will incorporate a simple but principled energy model. After each simulation run, it will calculate an estimated energy cost based on the total number of synaptic operations (SynOps) and spikes generated. We can define these costs based on figures from real hardware like Intel's Loihi 2 (e.g., X joules per SynOp).
    4.  **Integrate with the `Trainer`:** The `Continual_Trainer` from Phase 3 will be adapted to run *within* this new simulation environment, allowing us to orchestrate experiments under these neuromorphic constraints.
* **Justification:** Your research makes strong claims about the efficiency benefits of using SNNs and neuromorphic hardware. Since we lack physical hardware, creating a credible simulation is **non-negotiable** for backing up these claims. An event-driven simulator also forces us to design our algorithms in a way that is compatible with real neuromorphic chips, improving the real-world relevance of our work.
* **Required Libraries/Tools:** `numpy`, `torch`.

---

#### **Task 4.2: Implementing the Exoplanet Application Modules (Encoder/Decoder)**

* **Objective:** To build the concrete, domain-specific `Encoder` and `Decoder` for the exoplanet POC. These modules will be the first "plug-ins" for our GIF architecture.
* **Key Activities:**
    1.  **Implement the `LightCurveEncoder`:**
        * Create the file: `applications/poc_exoplanet/encoders/light_curve_encoder.py`.
        * The class `LightCurveEncoder` will inherit directly from `EncoderInterface` (defined in Phase 2).
        * It will implement the `encode()` method, which takes a 1D `Polars` Series or `NumPy` array (a light curve segment from our generator) and converts it into a `torch.Tensor` representing a spike train, using one of the chosen encoding schemes (e.g., delta modulation).
    2.  **Implement the `ExoplanetDecoder`:**
        * Create the file: `applications/poc_exoplanet/decoders/exoplanet_decoder.py`.
        * The class `ExoplanetDecoder` will inherit from `DecoderInterface`.
        * It will implement two decoding methods to fulfill the paper's claims:
            1.  `decode_classification()`: Takes the output spikes from the DU Core and uses a simple readout layer (e.g., population coding or a linear layer) to output a classification (Planet Candidate, False Positive).
            2.  `decode_regression()`: Takes the output spikes and maps them to continuous physical values (e.g., planetary radius, period) for the "Advanced Feature Prediction" task.
* **Justification:** This task is the practical realization of the GIF's modular philosophy. By creating concrete classes that inherit from our abstract interfaces, we demonstrate that the "plug-and-play" architecture works. This separation of concerns is clean, scalable, and makes the system easy to extend to new domains in the future.

---

#### **Task 4.3: Assembling the Full Exoplanet POC Pipeline**

* **Objective:** To integrate all the components developed so far (Data Generator, GIF Orchestrator, DU Core, RTL Engine, Hardware Simulator, and the new Exoplanet modules) into a single, cohesive, and runnable application.
* **Key Activities:**
    1.  **Create the Main Experiment Script:** `applications/poc_exoplanet/main_exo.py`.
    2.  **Configuration Management:** Use a simple configuration file (e.g., `config.yaml`) to manage all experiment parameters (SNN architecture, learning rates, data generator settings, number of epochs, etc.). This avoids hardcoding and makes experiments reproducible.
    3.  **Orchestration Script:** The `main_exo.py` script will:
        * Load the experiment configuration.
        * Instantiate the `ExoplanetGenerator` (from Phase 1).
        * Instantiate the `GIF` orchestrator and the `DU_Core` (from Phase 2).
        * Instantiate and attach the `LightCurveEncoder` and `ExoplanetDecoder` (from Task 4.2).
        * Instantiate the `NeuromorphicSimulator` (from Task 4.1).
        * Run the full training and evaluation process, managed by the `Continual_Trainer` within the simulator.
* **Justification:** This integration step is where the entire theoretical framework comes to life. It moves the project from a collection of disparate modules to a single, functional system. Using a configuration file for management is a professional practice that makes our complex experiments easy to run, modify, and reproduce, which is essential for publication.

---

#### **Task 4.4: Executing and Analyzing the Exoplanet Experiment**

* **Objective:** To run the full exoplanet POC experiment, collect performance data, and rigorously analyze the results to validate the initial claims of the framework and generate the data for Table IV in your paper.
* **Key Activities:**
    1.  **Execute the Experiment:** Run the `main_exo.py` script. This will be a computationally intensive step.
    2.  **Implement Logging:** Ensure that the training loop logs all necessary metrics at each step: loss, accuracy, spike counts, SynOp counts, etc. These logs should be saved to a structured file format (e.g., a `.csv` or `.json` file managed by `Polars`).
    3.  **Build the Analysis Notebook/Script:** Create a Jupyter Notebook or Python script, `applications/poc_exoplanet/analysis.ipynb`. This script will:
        * Load the results from the log files.
        * Calculate the final performance metrics (Accuracy, TPR, FPR, F1-Score, etc.).
        * Calculate the simulated neuromorphic performance (Energy Consumption, Latency).
        * Generate the final results table (replicating Table IV from your paper).
        * Create plots of the training process (e.g., accuracy/loss over time).
    4.  **Compare to Baselines:** Run equivalent experiments on standard baseline models (e.g., a simple CNN or Transformer) using the same synthetic data to provide a direct, fair comparison for your results table.
* **Justification:** This is the final step where we generate the scientific evidence. A dedicated analysis script ensures that our results are reproducible and that the process of generating tables and plots is automated. Comparing against well-chosen baselines on the *exact same data* is fundamental to proving that the GIF-DU framework provides a tangible advantage over existing methods, a key requirement for a high-impact publication.