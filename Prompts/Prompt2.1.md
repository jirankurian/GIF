**Your Current Task: Task 2.1 - Defining the Core Architectural Contracts (Interfaces)**

**Protocol Reminder:** Before you begin, you must execute your full **Cognitive Cycle**. Review the `/Rules` directory, the `/docs` directory, the `/Reference/` for Phase 2, read all logs in the `.context/` to understand our progress, and analyze the current project structure. After your analysis, formulate a precise micro-plan for this task and present it for my approval.

---

### **Task Objective**

Your goal is to establish the foundational software contracts for the entire General Intelligence Framework. You will create the abstract "blueprints" for what it means to be an **Encoder** and a **Decoder**. This is a critical step in software engineering that ensures our framework is truly modular and extensible. You will be defining the rules that all future "plug-in" components must follow.

---

### **Domain & Technical Specifications**

#### **1. The Concept of Interfaces in Software Engineering**

* **Domain Context:** In professional Object-Oriented Programming (OOP), an "interface" (often implemented in Python using an Abstract Base Class, or ABC) is a contract. It declares a set of methods that a class *must* implement to be considered a certain "type." For our GIF framework, any component that wants to be an "Encoder" must have an `encode()` method. This allows the central GIF orchestrator to work with any encoder, regardless of its internal logic, because it is guaranteed to have the methods it needs. This is a core principle of creating decoupled, maintainable, and scalable systems.

#### **2. Implementation Details**

* **Action:** You will create a new file located at: `gif_framework/interfaces/base_interfaces.py`.
* **Justification:** We are centralizing our core architectural contracts in a single, dedicated location. This makes the design of the framework clear and easy for any new developer to understand.
* **Required Libraries:** You will need to import `abc` (for `ABC` and `abstractmethod`) and `typing` (for `Any`, `Dict`, and creating custom types).

#### **3. Defining Custom Type Aliases**

To improve code readability and maintainability, you will first define two custom type aliases at the top of the file.

* `SpikeTrain = torch.Tensor`: This alias signifies that throughout our framework, a "SpikeTrain" is represented as a PyTorch tensor.
* `Action = Any`: This alias represents the final output of a decoder. It is set to `Any` because a decoder's action could be a simple classification (a string or integer), a set of regression values (a dictionary or array), or a complex command for a robot. This provides maximum flexibility.

#### **4. The `EncoderInterface`**

* **Action:** You will define a class `EncoderInterface` that inherits from `abc.ABC`.
* **Methods to Implement:**
    * **`@abstractmethod def encode(self, raw_data: Any) -> SpikeTrain:`**
        * **Purpose:** This is the primary method of any encoder. Its job is to take raw data in any format (e.g., a NumPy array, a Polars DataFrame, a string) and convert it into the framework's standardized `SpikeTrain` format.
        * **Implementation:** As an abstract method, its body will simply be `pass`.
    * **`@abstractmethod def get_config(self) -> Dict[str, Any]:`**
        * **Purpose:** This method is for reproducibility and logging. It must return a dictionary containing all the configurable parameters of the specific encoder instance (e.g., encoding method, thresholds, etc.).
        * **Implementation:** The body will be `pass`.
    * **`@abstractmethod def calibrate(self, sample_data: Any) -> None:`**
        * **Purpose:** This method is the hook for the **RTL-driven auto-calibration** feature described in the research papers. The GIF orchestrator can call this method to allow an encoder to dynamically adjust its parameters based on a sample of new, unseen data.
        * **Implementation:** The body will be `pass`.

#### **5. The `DecoderInterface`**

* **Action:** You will define a class `DecoderInterface` that also inherits from `abc.ABC`.
* **Methods to Implement:**
    * **`@abstractmethod def decode(self, spike_train: SpikeTrain) -> Action:`**
        * **Purpose:** This is the core method of any decoder. It takes a `SpikeTrain` (the output from the DU Core) and translates it into a final, meaningful output or "Action".
        * **Implementation:** The body will be `pass`.
    * **`@abstractmethod def get_config(self) -> Dict[str, Any]:`**
        * **Purpose:** Same as the encoder's `get_config`, this method returns the decoder's configuration for logging and reproducibility.
        * **Implementation:** The body will be `pass`.

---

**Summary of your task:**

1.  Create the file `gif_framework/interfaces/base_interfaces.py`.
2.  Import the necessary libraries (`abc`, `typing`, `torch`).
3.  Define the `SpikeTrain` and `Action` type aliases.
4.  Implement the `EncoderInterface` ABC with its three abstract methods (`encode`, `get_config`, `calibrate`).
5.  Implement the `DecoderInterface` ABC with its two abstract methods (`decode`, `get_config`).
6.  Ensure the file is professionally formatted and includes clear docstrings for each class and method explaining its purpose in the GIF architecture.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**