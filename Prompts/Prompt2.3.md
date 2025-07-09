**Your Current Task: Task 2.3 - Building the DU Core v1 (The SNN Brain)**

**Protocol Reminder:** Before you begin, execute your full **Cognitive Cycle**. Review the `/Rules` directory, the `/Reference/` for Phase 2, read all logs in the `.context/` (including all previous logs), and analyze the existing codebase. The `GIF` orchestrator we just built now expects a `DU_Core` instance. Your task is to create the first version of this critical component. After your analysis, formulate your micro-plan and present it for approval.

---

### **Task Objective**

Your goal is to implement the first functional version of the **Deep Understanding Core (`DU_Core_V1`)**. This class will be the "brain" of our framework. It will be a configurable, multi-layer Spiking Neural Network (SNN) built using the **`snnTorch`** library. This initial version will focus on creating a robust and flexible architecture for processing spike trains, laying the foundation for the advanced learning and memory mechanisms we will add in Phase 3.

---

### **Domain & Technical Specifications**

#### **1. SNNs and the Leaky Integrate-and-Fire (LIF) Neuron Model**

* **Domain Context:** At the heart of our DU Core is the Spiking Neural Network. Unlike traditional neurons that output continuous values, spiking neurons communicate through discrete events or "spikes," which happen at specific points in time. This is more biologically plausible and computationally efficient. The most common model for a spiking neuron is the **Leaky Integrate-and-Fire (LIF)** model.
* **LIF Neuron Behavior:**
    1.  **Integrate:** The neuron's internal voltage (its "membrane potential") increases when it receives an input spike.
    2.  **Leaky:** Over time, if it doesn't receive input, its membrane potential slowly "leaks" away, decaying back to a resting state.
    3.  **Fire:** If the membrane potential reaches a specific `threshold`, the neuron fires a single output spike and its potential is reset.
* **Technical Approach:** We will use the `snnTorch` library, which provides a highly optimized `snn.Leaky` class that implements this LIF neuron model. It integrates seamlessly with PyTorch's automatic differentiation, which will be important for later tasks.

#### **2. Implementation Details**

* **Action:** You will create a new file located at: `gif_framework/core/du_core.py`.
* **Justification:** This file will house our core cognitive engine, keeping it separate from the high-level orchestrator and the specific interface definitions. This adheres to our modular design philosophy.
* **Required Libraries:** You will need to import `torch`, `torch.nn`, `snntorch as snn`, and `typing.List`.

#### **3. The `DU_Core_V1` Class Implementation**

* **Action:** You will define the class `DU_Core_V1`, ensuring it inherits from `torch.nn.Module`.
* **`__init__(self, input_size: int, hidden_sizes: List[int], output_size: int, beta: float = 0.95, threshold: float = 1.0, recurrent: bool = False)`:**
    * This initializer must be highly configurable to allow for easy experimentation.
    * `input_size`, `hidden_sizes`, `output_size`: Define the network's layer dimensions. `hidden_sizes` is a list to allow for an arbitrary number of hidden layers.
    * `beta`: The leakage rate for the LIF neurons (a value closer to 1 means less leakage).
    * `threshold`: The membrane potential threshold for firing a spike.
    * `recurrent`: A flag to enable recurrent connections (we can implement this logic in a future version, but the parameter should exist now).
    * **Inside the `__init__`:**
        1.  You will dynamically create the network layers and store them in a `torch.nn.ModuleList`.
        2.  The architecture will be a sequence of `Linear` layers followed by `snn.Leaky` neuron layers.
        3.  For example, the first layer would be a `torch.nn.Linear(input_size, hidden_sizes[0])` followed by an `snn.Leaky(beta=beta, threshold=threshold)`. You will loop through the `hidden_sizes` list to create all the hidden layers.

* **`forward(self, input_spikes: torch.Tensor, num_steps: int) -> torch.Tensor`:**
    * This is the main PyTorch forward pass method.
    * **Parameters:** It takes the `input_spikes` tensor (with shape `[num_steps, batch_size, input_size]`) and the total number of simulation `num_steps`.
    * **Implementation Steps:**
        1.  **Initialize Potentials & Spikes:** Before the loop, initialize the membrane potentials for all `snn.Leaky` layers to zero. You will also need lists to record the output spike train of each layer over time.
        2.  **Simulation Loop:** You will loop from `step = 0` to `num_steps - 1`.
        3.  **Layer-by-Layer Processing:** In each step of the loop, you will pass the spikes through the network sequentially:
            * Take the input spikes for the current step (`input_spikes[step]`).
            * Pass them through the first `Linear` layer.
            * Pass the result through the first `snn.Leaky` layer, which returns the output spikes and the updated membrane potential for that layer.
            * Repeat this for all subsequent layers, feeding the output spikes of one layer into the next.
        4.  **Record Output:** Store the output spikes from the final layer at each time step.
        5.  **Return Value:** After the loop finishes, stack the recorded output spikes into a single tensor of shape `[num_steps, batch_size, output_size]` and return it.

---

**Summary of your task:**

1.  Create the new file `gif_framework/core/du_core.py`.
2.  Implement the `DU_Core_V1` class, inheriting from `torch.nn.Module`.
3.  Create a highly configurable `__init__` method that dynamically builds a multi-layer SNN using `torch.nn.Linear` and `snn.Leaky` layers.
4.  Implement the `forward` method, which contains the core simulation loop for processing spike trains over a given number of time steps.
5.  Ensure all code is professionally documented with clear docstrings explaining the class's role and the parameters of its methods.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**