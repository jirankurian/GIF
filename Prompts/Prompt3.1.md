**Your Current Task: Task 3.1 - Implementing the Synaptic Plasticity Engine (RTL Rules)**

**Protocol Reminder:** Before you begin, you must execute your full **Cognitive Cycle**. Review the `/Rules` directory, the `/Reference/` for Phase 3, read all logs in the `.context/` to confirm the completion of Phase 2, and analyze the existing codebase. The `DU_Core_V1` we built is currently static; this task will create the mechanisms to make it dynamic and adaptive. After your analysis, formulate your micro-plan for this task and present it for approval.

---

### **Task Objective**

Your goal is to build the engine that drives **Real-Time Learning (RTL)** within our framework. You will create a modular library of **synaptic plasticity rules**. These rules are mathematical formulas that govern how the connection strengths (synaptic weights) between neurons in our SNN change based on their activity. This is the fundamental mechanism of learning and memory formation in both biological brains and our DU Core.

---

### **Domain & Technical Specifications**

#### **1. The Concept of Synaptic Plasticity**

* **Domain Context:** A neural network "learns" by adjusting the weights of its synapses. In standard deep learning, this is done via backpropagation, a global process that happens during a separate "training phase." In our biologically-inspired framework, learning is **local and continuous**. A synapse strengthens or weakens based only on the activity of the two neurons it connects (and potentially a third "modulatory" signal). This online, continuous adjustment is what we call RTL.
* **Technical Approach:** We will implement these learning rules as interchangeable Python classes. This will allow the DU Core to be configured with different learning mechanisms so we can experiment to find the most effective ones. Each rule will be a "strategy" for updating weights.

#### **2. Implementation Details**

* **Action:** You will create a new file located at: `gif_framework/core/rtl_mechanisms.py`.
* **Justification:** This new file will contain our library of learning rules, cleanly separating the "how to learn" logic from the core SNN architecture itself. This is a key principle of modular design.
* **Required Libraries:** You will need `torch`, `torch.nn`, `abc` (for the interface), and `typing`.

#### **3. The `PlasticityRuleInterface`**

* **Action:** First, you will define an Abstract Base Class (ABC) named `PlasticityRuleInterface` that inherits from `abc.ABC`.
* **Justification:** This contract ensures that any learning rule we create in the future will be compatible with our DU Core. It forces all rules to have the same "shape."
* **Abstract Method:**
    * `@abstractmethod def apply(self, **kwargs) -> torch.Tensor:`
        * **Purpose:** This is the core method for any learning rule. It will contain the logic for calculating the weight changes (`delta_w`). It must return a tensor of weight updates that has the same shape as the weight matrix it applies to.
        * We use `**kwargs` to make the interface flexible; different rules might need different inputs (e.g., some need spike times, others need a modulatory signal).

#### **4. Concrete Rule 1: Spike-Timing-Dependent Plasticity (STDP)**

* **Domain Context:** STDP is a fundamental process in neuroscience. It states that the precise timing of spikes matters. If a presynaptic neuron fires *just before* a postsynaptic neuron, causing it to fire, the connection is strengthened (Long-Term Potentiation, or LTP). If it fires *just after* the postsynaptic neuron (i.e., it didn't contribute to the firing), the connection is weakened (Long-Term Depression, or LTD).
* **Action:** Implement a class `STDP_Rule(PlasticityRuleInterface)`.
* **Technical Implementation:**
    * **`__init__(self, learning_rate_ltp: float, learning_rate_ltd: float, tau_ltp: float, tau_ltd: float)`:** The initializer will take the parameters for the STDP rule.
    * **`apply(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> torch.Tensor:`** This method will implement the update rule.
        1.  It will require an efficient way to track the timing between `pre_spikes` and `post_spikes` for every synapse. A common technique is to maintain "trace" variables for each neuron that represent a running average of its recent spiking activity.
        2.  For LTP (pre-before-post), the weight update is `Δw = learning_rate_ltp * pre_trace`.
        3.  For LTD (post-before-pre), the weight update is `Δw = -learning_rate_ltd * post_trace`.
        4.  You will need to implement the logic to update these traces and apply the weight changes accordingly. The `snnTorch` documentation contains tutorials on implementing STDP that can serve as a technical reference.

#### **5. Concrete Rule 2: Three-Factor Hebbian Rule**

* **Domain Context:** This rule is a step closer to goal-directed learning. Basic Hebbian learning is "neurons that fire together, wire together." A three-factor rule adds a crucial third component: "...*when a third signal says it's important*." This third signal is a "neuromodulator" and can represent reward, surprise, or task-relevance.
* **Action:** Implement a class `ThreeFactor_Hebbian_Rule(PlasticityRuleInterface)`.
* **Technical Implementation:**
    * **`__init__(self, learning_rate: float)`:** Simple initializer for the learning rate.
    * **`apply(self, pre_activity: torch.Tensor, post_activity: torch.Tensor, modulatory_signal: float) -> torch.Tensor:`**
        1.  This method takes the activity (e.g., spike tensors) of the pre- and post-synaptic neuron populations.
        2.  It also takes a scalar `modulatory_signal`.
        3.  The weight update `Δw` for each synapse is calculated simply as: `Δw = learning_rate * pre_activity * post_activity * modulatory_signal`. This is a straightforward element-wise multiplication of the activity tensors, scaled by the learning rate and the global modulatory signal.
        4.  This rule is simpler to implement than STDP but is powerful for reinforcement-style learning scenarios.

---

**Summary of your task:**

1.  Create the new file `gif_framework/core/rtl_mechanisms.py`.
2.  Implement the `PlasticityRuleInterface` ABC.
3.  Implement the `STDP_Rule` class, including the logic for exponential traces and timing-dependent updates.
4.  Implement the `ThreeFactor_Hebbian_Rule` class, which includes the third modulatory signal in its calculation.
5.  Ensure all classes are professionally documented with clear docstrings explaining the learning rule they implement and their parameters.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**