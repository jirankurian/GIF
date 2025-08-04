**Your Current Task: Task 2.2 - Implementing the Central Orchestrator (The GIF Class)**

**Protocol Reminder:** Before you begin, execute your full **Cognitive Cycle**. Review the `/Rules` directory, the `/docs` directory, the `/Reference/` for Phase 2, read all logs in the `.context/` (including the previous outcomes), and analyze the current project structure. After this analysis, formulate your micro-plan for this task and present it for my approval.

---

### **Task Objective**

Your goal is to build the central nervous system of our framework: the `GIF` class. This class will act as the primary orchestrator, managing all the modular components (the "body parts") and directing the flow of information through the system. It will not contain any domain-specific logic itself; its sole purpose is to manage the `Encoder`, the `Decoder`, and the `DU_Core` (the "brain").

---

### **Domain & Technical Specifications**

#### **1. The "Orchestrator" Pattern and Dependency Injection**

* **Domain Context:** In advanced software architecture, it's crucial to separate "coordination" logic from "implementation" logic. The `GIF` class is a classic **Orchestrator**. It doesn't know *how* to encode data or *how* the brain works; it only knows the correct *order* of operations: `encode`, then `process`, then `decode`.
* **Technical Approach:** We will use a powerful software design pattern called **Dependency Injection (DI)**. Instead of the `GIF` creating its own "brain" (`DU_Core`), we will "inject" the brain into it when it's created. This makes our framework incredibly flexible—we can give the same GIF body a simple brain (our `DU_Core_V1`) or an advanced one (the future `DU_Core_V2`) without changing a single line of code in the `GIF` class itself.

#### **2. Implementation Details**

* **Action:** You will create a new file located at: `gif_framework/orchestrator.py`.
* **Justification:** This new file will house our central `GIF` class, keeping the high-level coordination logic separate from the low-level interface definitions and core processing modules.
* **Required Imports:**
    * From `gif_framework.interfaces.base_interfaces`, you must import `EncoderInterface` and `DecoderInterface`. These are the contracts we defined in the previous task.
    * You will also need to import the (not yet created) `DU_Core_V1` from `gif_framework.core.du_core`. Python allows you to import modules that don't exist yet for type-hinting purposes.
    * You will need `typing` for `Optional`.

#### **3. The `GIF` Class Implementation**

* **Action:** You will define the `GIF` class.
* **`__init__(self, du_core: 'DU_Core_V1')`:**
    * **This is the Dependency Injection point.** The initializer must accept one argument: an instance of our `DU_Core_V1` class (the "brain").
    * It should store this instance as a private attribute, e.g., `self._du_core = du_core`.
    * It should also initialize private attributes for the encoder and decoder to `None`, e.g., `self._encoder: Optional[EncoderInterface] = None`. Using `Optional` and the interface type hint is a professional practice that makes the code self-documenting.

* **`attach_encoder(self, encoder: EncoderInterface)` and `attach_decoder(self, decoder: DecoderInterface)`:**
    * These are the public methods for "plugging in" our modules.
    * They each take one argument, an `encoder` or `decoder` instance.
    * **Crucially, they must use the `EncoderInterface` and `DecoderInterface` as type hints.** This is how we enforce our architectural contract. Python's type checkers will now ensure that only valid modules can be attached to the GIF.
    * The methods will simply assign the provided instance to the corresponding private attribute (e.g., `self._encoder = encoder`).

* **`process_single_input(self, raw_data: Any) -> Action`:**
    * This is the core operational method that executes one full cognitive cycle.
    * It must perform a series of checks and operations in the following exact order:
        1.  **Check for Modules:** It must first check if both `self._encoder` and `self._decoder` have been attached. If either is `None`, it should raise a `RuntimeError` with a clear message like "Encoder or Decoder not attached. Please attach modules before processing."
        2.  **Encode:** Call the encoder to convert the raw data into spikes: `spike_train = self._encoder.encode(raw_data)`.
        3.  **Process (Think):** Pass the spikes to the DU Core for processing: `processed_spikes = self._du_core.process(spike_train)`.
        4.  **Decode:** Call the decoder to get the final result: `action = self._decoder.decode(processed_spikes)`.
        5.  **Return:** Return the final `action`.

---

**Summary of your task:**

1.  Create the new file `gif_framework/orchestrator.py`.
2.  Import the necessary interfaces and classes for type hinting.
3.  Implement the `GIF` class.
4.  The `__init__` method must use Dependency Injection to receive the `du_core`.
5.  The `attach_` methods must use our defined interfaces as type hints to enforce the modular contract.
6.  The `process_single_input` method must implement the full encode-process-decode cycle with proper error checking.
7.  Ensure all code is professionally formatted and documented with clear docstrings explaining the purpose of the orchestrator and its methods.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**