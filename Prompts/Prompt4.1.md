**Your Current Task: Task 4.1 - Developing the Neuromorphic Hardware Simulator**

**Protocol Reminder:** Before you begin, you must execute your full **Cognitive Cycle**. Review the `/Rules` directory, the `/Reference/` for Phase 4, read all logs in the `.context/` to confirm the completion of Phase 3, and analyze the current codebase. You will note that while we have built SNNs, our training and execution logic does not yet account for the specific way neuromorphic hardware operates. This task builds that simulation layer. After your analysis, formulate your micro-plan and present it for approval.

---

### **Task Objective**

Your goal is to build a software-based simulator that mimics the key operational principles of **neuromorphic hardware**. This is not about emulating a specific chip's architecture perfectly. Instead, it's about creating a Python-based environment that enforces two fundamental properties of neuromorphic computing:
1.  **Event-Driven Computation:** The simulation only performs work when and where "spikes" (events) occur.
2.  **Energy Efficiency Modeling:** The simulation estimates the energy cost of a computation based on neural activity, allowing us to quantitatively support our claims of efficiency.

---

### **Domain & Technical Specifications**

#### **1. The Concept of Event-Driven vs. Clock-Driven Computation**

* **Domain Context:**
    * **Traditional GPUs (Clock-Driven):** When you run a standard neural network on a GPU, the hardware performs massive matrix multiplications at every single step. Every neuron and synapse is involved in the calculation, whether its output is meaningful or not. This is powerful but energy-intensive.
    * **Neuromorphic Chips (Event-Driven):** Neuromorphic hardware operates more like a biological brain. A neuron consumes almost no power until it receives a spike or fires a spike. Computation is sparse and happens only where there is activity. This is the source of its incredible energy efficiency.
* **Our Technical Goal:** We will simulate this by creating a training loop that processes data one time-step at a time and explicitly counts the "events" (spikes) to model the computational load.

#### **2. The Concept of Energy Modeling via Synaptic Operations (SynOps)**

* **Domain Context:** The primary consumer of energy in a brain (and a neuromorphic chip) is the synapse. A **Synaptic Operation (SynOp)** occurs every time a spike is transmitted across a synapse, causing a change in the downstream neuron. To estimate the energy cost of running our SNN, we can simply count the total number of SynOps and multiply it by the known energy cost of a single SynOp for a real chip.
* **Our Technical Goal:** Our simulator must include a counter that increments every time a spike is processed by a synapse, providing a final `total_synops` count for any given run.

#### **3. Implementation Details**

* **Action:** You will create a new file located at: `simulators/neuromorphic_sim.py`.
* **Justification:** This new module cleanly separates the hardware simulation logic from the core framework (`gif_framework`) and the specific applications (`applications`), adhering to our modular design principles.
* **Required Libraries:** `torch`, `typing.Dict`.

#### **4. The `NeuromorphicSimulator` Class Implementation**

* **Action:** You will implement the `NeuromorphicSimulator` class.
* **`__init__(self, snn_model: torch.nn.Module, energy_per_synop: float = 2.5e-11)`:**
    * The initializer takes the SNN model to be simulated (e.g., our `DU_Core_V1`).
    * It also takes the `energy_per_synop` in Joules. The default value `2.5e-11` is a realistic estimate based on published figures for hardware like Intel's Loihi 2.

* **`run(self, input_spikes: torch.Tensor) -> (torch.Tensor, Dict[str, float])`:**
    * This is the main public method that executes the full, event-driven simulation.
    * **Parameters:** It takes a `torch.Tensor` of input spikes with shape `[num_steps, batch_size, input_size]`.
    * **Implementation Steps:**
        1.  **Initialization:** Initialize counters for `total_spikes` and `total_synops` to zero. Initialize lists to store the output spikes from the final layer. Initialize the SNN model's internal states (membrane potentials) by calling a (to-be-created) `model.reset()` method.
        2.  **Event-Driven Loop:** Loop from `step = 0` to `num_steps - 1`.
        3.  **Process One Time-Step:** In each loop, pass only the current time-slice of spikes (`input_spikes[step]`) to the SNN model's `forward` method.
        4.  **Count Activity:** After each step, you must inspect the state of the SNN model to count the activity:
            * Iterate through each `snn.Leaky` layer in your `snn_model`.
            * The number of spikes fired in that layer at the current step is `layer.spk1.sum()`. Add this to `total_spikes`.
            * The number of synaptic operations for a linear layer preceding a spiking layer is approximately the number of input spikes to that layer multiplied by the number of output neurons (fan-out). You can estimate this or, for a more accurate measure, count the non-zero elements in the spike tensor and multiply by the corresponding weights. For this task, a simple estimation is sufficient. Add this to `total_synops`.
        5.  **Record Output:** Append the output spikes from the final layer to your list of outputs.
        6.  **Calculate Final Stats:** After the loop, calculate `estimated_energy_joules = total_synops * self.energy_per_synop`.
        7.  **Return Values:** Return two values:
            * The final output spike train (a tensor stacked from your list of outputs).
            * A dictionary of statistics: `{'total_spikes': total_spikes, 'total_synops': total_synops, 'estimated_energy_joules': estimated_energy_joules}`.

---

**Summary of your task:**

1.  Create the new file `simulators/neuromorphic_sim.py`.
2.  Implement the `NeuromorphicSimulator` class.
3.  The `run` method must implement an **event-driven simulation loop** that processes data one time-step at a time.
4.  Inside the loop, it must **count the spikes and estimate the synaptic operations** to model computational load.
5.  The method must return both the final SNN output and a dictionary containing the calculated performance statistics (total spikes, total SynOps, estimated energy).
6.  Ensure all code is professionally documented.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**