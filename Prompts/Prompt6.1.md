**Your Current Task: Task 6.1 - Implementing the DU Core v2 (Hybrid SNN/SSM Architecture)**

**Protocol Reminder:** Before you begin, you must execute your full **Cognitive Cycle**. Review the `/Rules` directory, the `/Reference/` for Phase 6, read all logs in the `.context/` to confirm the successful completion of the prior phases, and analyze the current codebase, specifically the `du_core.py` containing our `DU_Core_V1`. This task is a major upgrade. After your analysis, formulate your micro-plan for this task and present it for my approval.

---

### **Task Objective**

Your goal is to engineer the **`DU_Core_V2`**, an advanced cognitive core that replaces our initial LIF-SNN model. This new version will implement the **hybrid Spiking Neural Network (SNN) / State Space Model (SSM) architecture** that is a central theoretical contribution of the research papers. This involves creating a novel SNN layer whose internal dynamics are governed by the mathematics of modern SSMs like Mamba.

---

### **Domain & Technical Specifications**

#### **1. The "Best of Both Worlds" Architecture**

* **Domain Context:** As our research synthesis concluded, the future of high-performance sequence modeling lies in **hybrid architectures**. Pure Transformers are powerful but computationally expensive for long sequences. Pure SSMs (like Mamba) are incredibly efficient but can lack the complex reasoning power of attention. Our `DU_Core_V2` will follow the blueprint of cutting-edge models like NVIDIA's Nemotron-H by building a heterogeneous architecture.
* **Technical Approach:** We will create a new SNN layer that is not a simple LIF neuron, but a more complex system whose internal state (membrane potential) evolves according to SSM equations. The final `DU_Core_V2` will then be built using a majority of these efficient new layers, with a few standard attention layers strategically placed for high-level reasoning.

#### **2. Implementation Step 1: The `HybridSNNSSMLayer`**

* **Action:** You will first create a new `torch.nn.Module` class named `HybridSNNSSMLayer`. This will be the fundamental building block of our new core.
* **Justification:** This novel layer is the core technical innovation of this task. It tightly couples the energy-efficient, event-driven nature of SNNs with the powerful, long-memory state-tracking of SSMs.
* **Implementation Details:**
    * **`__init__(self, input_dim, output_dim, state_dim)`:** The initializer will create the learnable SSM matrices: `A` (state transition), `B` (input projection), and `C` (output projection). Crucially, following the **Mamba architecture**, these will not be static. You will implement them as small linear networks that take the input spikes and generate the A, B, and C matrices dynamically for each time step. This is the **selective state** mechanism.
    * **`forward(self, input_spikes, hidden_state)`:**
        1.  It takes the spikes for the current time step and the previous `hidden_state` (the `h_{t-1}` from the SSM equation).
        2.  It calculates the dynamic A, B, C matrices based on the `input_spikes`.
        3.  It updates the hidden state using the core SSM equation: `hidden_state = A * hidden_state + B * input_spikes`.
        4.  It calculates a pre-synaptic potential using the output equation: `potential = C * hidden_state`.
        5.  **Spiking Mechanism:** This `potential` value is then fed into a standard `snn.Leaky` neuron instance. If the neuron's membrane potential crosses its threshold, it fires a spike.
        6.  The method returns the output spikes and the new `hidden_state`.

#### **3. Implementation Step 2: The `DU_Core_V2` Orchestrator**

* **Action:** You will create a new file, `gif_framework/core/du_core_v2.py`, and implement the `DU_Core_V2` class.
* **Justification:** This class assembles our new `HybridSNNSSMLayer` blocks into the final, heterogeneous architecture.
* **Implementation Details:**
    * **`__init__(self, ...)`:** The initializer will construct the network as a `torch.nn.ModuleList`.
    * **The Heterogeneous Structure:** The list of layers will not be uniform. It will consist mostly of our new `HybridSNNSSMLayer` instances. However, at strategic points (e.g., after every two hybrid layers), you will insert a standard `torch.nn.MultiheadAttention` layer.
    * **`forward(self, input_spikes, num_steps)`:**
        1.  The forward pass will be more complex than in V1. It will still loop through `num_steps`.
        2.  For steps involving a `HybridSNNSSMLayer`, it will process the data sequentially, passing the hidden state from one step to the next.
        3.  When it encounters an `Attention` layer, the logic must change. It will need to take the full sequence of hidden states generated so far, reshape it into the `[sequence, batch, features]` format expected by attention, and perform a global self-attention operation. The output of the attention layer is then fed back into the subsequent hybrid layers. This allows the model to periodically perform a "global check-in" on its entire history before returning to efficient, step-by-step processing.

---

**Summary of your task:**

1.  Create a new file `gif_framework/core/du_core_v2.py`.
2.  First, implement the `HybridSNNSSMLayer` class. This is a novel SNN layer where the neuron's state is updated via the selective state-space equations from the Mamba architecture.
3.  Second, implement the `DU_Core_V2` class. This class will construct a deep network using a sequence of your new `HybridSNNSSMLayer`s, strategically interspersed with standard `torch.nn.MultiheadAttention` layers.
4.  The `forward` method of `DU_Core_V2` must correctly handle the data flow through this mixed, heterogeneous architecture.
5.  Ensure all new code is documented to explain the advanced concepts being implemented.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**