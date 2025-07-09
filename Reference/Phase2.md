### **Phase 2 Breakdown: Core Framework Implementation**

**Overall Objective:** To engineer the foundational software skeleton of the General Intelligence Framework (GIF). This involves creating the core interfaces, the central orchestrator, and the initial version of the Deep Understanding (DU) core. We will prioritize modularity, Object-Oriented Programming (OOP) principles, and Dependency Injection (DI) to build a system that is as flexible and powerful as the one described in your research.

---

#### **Task 2.1: Defining the Core Architectural Contracts (Interfaces)**

* **Objective:** To establish the "rules of engagement" for all future modular components. We will create abstract interfaces that define what it *means* to be an `Encoder` or a `Decoder` within the GIF ecosystem, without specifying *how* they should be implemented.
* **Key Activities:**
    1.  Create the file: `gif_framework/interfaces/base_interfaces.py`.
    2.  Using Python's `abc` (Abstract Base Class) module, define the `EncoderInterface`. This class will specify abstract methods that every concrete encoder *must* implement, such as:
        * `encode(self, raw_data: Any) -> SpikeTrain`: The core method that converts raw data into a spike train. We will define a custom `SpikeTrain` type alias (e.g., `SpikeTrain = torch.Tensor`) for clarity.
        * `get_config(self) -> Dict`: A method to return the encoder's configuration for logging and reproducibility.
        * `calibrate(self, sample_data: Any) -> None`: An interface for the Real-Time Learning (RTL) driven auto-calibration. Initially, it can be a pass-through method, but the hook must exist.
    3.  Similarly, define the `DecoderInterface` in the same file. It will include methods like:
        * `decode(self, spike_train: SpikeTrain) -> Action`: Converts the DU core's output spikes into a final result. We'll define a custom `Action` type alias.
        * `get_config(self) -> Dict`.
* **Justification:** This is the cornerstone of a truly modular, "plug-and-play" architecture. By defining these contracts using ABCs, we enforce a design pattern where the central framework doesn't need to know the internal details of any specific module. It only needs to know that the module "honors the contract." This makes the framework infinitely extensible and directly implements the high-level OOP and DI goals.

---

#### **Task 2.2: Implementing the Central Orchestrator (The GIF Class)**

* **Objective:** To build the central nervous system of the framework—the `GIF` class that manages the lifecycle of the components and the flow of information.
* **Key Activities:**
    1.  Create the file: `gif_framework/orchestrator.py`.
    2.  Implement the `GIF` class.
    3.  The `__init__` method will accept a `DU_Core` instance as an argument. This is the **Dependency Injection** pattern in action. The GIF is not responsible for creating the "brain"; it is simply given one to work with.
    4.  Implement public methods for managing modules:
        * `attach_encoder(self, encoder: EncoderInterface)`: This method will use the interface from Task 2.1 as a type hint to ensure that only valid encoder objects can be attached.
        * `attach_decoder(self, decoder: DecoderInterface)`.
    5.  Implement the core operational loop: `process_single_input(self, raw_data: Any) -> Action`. This method will execute the full, single-pass cognitive cycle:
        * Check if an encoder is attached.
        * `spikes = self.encoder.encode(raw_data)`
        * `processed_spikes = self.du_core.process(spikes)`
        * `action = self.decoder.decode(processed_spikes)`
        * Return the `action`.
* **Justification:** This task builds the operational heart of the framework. Using DI makes the `GIF` class highly decoupled and testable. It doesn't care *which* DU core or encoder it's using, only that they conform to their respective interfaces. This design is clean, robust, and directly reflects the conceptual separation between the GIF "body" and DU "brain."

---

#### **Task 2.3: Building the DU Core v1 (The SNN Brain)**

* **Objective:** To create the first functional version of the Deep Understanding core. This initial version will focus on a robust and configurable Spiking Neural Network (SNN) architecture, laying the groundwork for the more complex learning and memory systems in the next phase.
* **Key Activities:**
    1.  Create the file: `gif_framework/core/du_core.py`.
    2.  Using the **`snnTorch`** library, implement the `DU_Core_V1` class.
    3.  The architecture will be a configurable, multi-layer network of **Leaky Integrate-and-Fire (LIF)** neurons. The class `__init__` method will be highly parameterized to allow for easy experimentation:
        * `input_size`, `hidden_sizes: List[int]`, `output_size`.
        * Neuron parameters (e.g., `beta` for leakage, `threshold`).
        * A boolean flag `recurrent: bool` to allow for adding recurrent connections within layers.
    4.  Implement the core processing method: `process(self, input_spikes: SpikeTrain, num_steps: int) -> SpikeTrain`. This method will:
        * Initialize the membrane potentials of all neurons.
        * Run the simulation loop for `num_steps`, feeding the input spikes into the network.
        * Record and return the output spike train from the final layer.
* **Justification:** This task creates the central intelligence of our framework. `snnTorch` is an excellent choice as it builds directly on PyTorch, which is highly performant on your M2 Pro hardware via the MPS backend. By making the SNN architecture highly configurable from the start, we create a powerful and flexible "brain" that we can easily modify and experiment with in later phases without needing to rewrite the entire class.

---

#### **Task 2.4: Establishing a Rigorous Testing Suite**

* **Objective:** To ensure the foundational components are correct, robust, and integrate seamlessly. This establishes a professional development practice from the outset.
* **Key Activities:**
    1.  **Setup `pytest`:** Configure the project to use `pytest` as the testing framework.
    2.  **Write Unit Tests:**
        * **Interfaces:** Create mock classes that correctly and incorrectly implement the `EncoderInterface` and `DecoderInterface`. Write tests for the `GIF` orchestrator to ensure it correctly accepts the valid mocks and raises `TypeError` for the invalid ones.
        * **DU Core:** Write tests for `DU_Core_V1` to verify that it can be instantiated with various configurations and that the `process` method returns output spike trains of the expected shape and data type.
    3.  **Write a Core Integration Test:**
        * Create a test that instantiates the entire framework in its simplest form.
        * It will create a `GIF` instance, inject a `DU_Core_V1`, and attach simple `MockEncoder` and `MockDecoder` instances.
        * The test will then call `process_single_input` with dummy data and assert that the output is what is expected. This validates that the entire architectural chain of command is working correctly.
* **Justification:** A project that aims to be a landmark contribution cannot have brittle code. A comprehensive testing suite is non-negotiable. It provides a safety net that prevents regressions as we add more complex features (like RTL and memory) in subsequent phases. The integration test, in particular, is crucial for proving that our core architectural philosophy—decoupled components communicating through well-defined interfaces—is sound and correctly implemented.
