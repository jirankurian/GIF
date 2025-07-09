**Your Current Task: Task 3.3 - Building the Continual Learning Engine (GEM Integration)**

**Protocol Reminder:** Before you begin, execute your full **Cognitive Cycle**. Review the `/Rules` directory, the `/Reference/` for Phase 3, read all logs in `.context/` (which confirm the creation of the RTL rules and the memory system), and analyze the existing codebase. You will see that the `DU_Core`, `rtl_mechanisms`, and `memory_systems` are all separate. This task will unite them into a functional learning system. After your analysis, formulate your micro-plan and present it for approval.

---

### **Task Objective**

Your goal is to implement the **continual learning process** for the DU Core. This involves two major actions:
1.  Upgrading the `DU_Core_V1` to make its learning rule and memory system "pluggable."
2.  Implementing a new `Continual_Trainer` class that orchestrates the learning process using the **Gradient Episodic Memory (GEM)** algorithm, as specified in our research. This algorithm is the primary defense against catastrophic forgetting.

---

### **Domain & Technical Specifications**

#### **1. The Concept of Gradient Episodic Memory (GEM)**

* **Domain Context:** A critical failure mode of neural networks is **catastrophic forgetting**. When you train a network on Task B after it has learned Task A, it often completely forgets how to do Task A. GEM prevents this.
* **The Analogy:** Think of it like a student studying for a final exam. Before learning a new chapter (Task B), the student first does a quick review of a few problems from an old chapter (Task A). When learning the new material, the student makes sure the new knowledge doesn't contradict the old knowledge.
* **Technical Implementation:** In GEM, before we update our network's weights for a new data sample, we first check if this update would increase the error on a few "memories" from past tasks. If it would, we slightly adjust (or "project") the update so that it helps with the new task while not harming performance on the old ones.

#### **2. Task 3.3a: Upgrading the `DU_Core_V1`**

* **Action:** You will modify the `DU_Core_V1` class in `gif_framework/core/du_core.py`.
* **Justification:** To make our framework truly modular, the "brain" should not have its learning rule or memory system hardcoded. We should be able to give it different rules or memory systems to experiment with. This is achieved through Dependency Injection.
* **Implementation Steps:**
    1.  Import `PlasticityRuleInterface` from `rtl_mechanisms.py` and `EpisodicMemory` from `memory_systems.py`.
    2.  Modify the `__init__` method signature of `DU_Core_V1` to accept these two new components: `__init__(self, ..., plasticity_rule: PlasticityRuleInterface, memory_system: EpisodicMemory)`.
    3.  Store these injected components as private attributes: `self._plasticity_rule = plasticity_rule` and `self._memory_system = memory_system`.
    4.  Add a new method `apply_learning(self, **kwargs)` which will simply call `self._plasticity_rule.apply(**kwargs)`. This delegates the learning logic to the attached rule.

#### **3. Task 3.3b: Implementing the `Continual_Trainer` with GEM**

* **Action:** You will create a new file, `gif_framework/training/trainer.py`, and implement the `Continual_Trainer` class.
* **Justification:** This class will encapsulate the entire complex logic of an online training step, including the GEM algorithm. This keeps our main experiment scripts (which we will build later) clean and high-level.
* **Implementation Steps:**

    * **`__init__(self, gif_model: GIF, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module)`:**
        * The trainer is initialized with the main `GIF` model instance (which contains the DU Core and its memory), a PyTorch optimizer (e.g., `torch.optim.Adam`), and a loss function (e.g., `torch.nn.CrossEntropyLoss`).

    * **`train_step(self, sample_data: Any, target: torch.Tensor, task_id: str) -> float`:**
        * This method performs a single, complete, GEM-constrained training step.
        * **A. Forward Pass:** `output = self.gif_model.process_single_input(sample_data)`.
        * **B. Store Experience:** Create an `ExperienceTuple` with the relevant data from the forward pass and store it in the DU Core's memory system via `self.gif_model._du_core._memory_system.add(...)`.
        * **C. Calculate Current Gradient:**
            1.  Calculate the loss for the current task: `loss_current = self.criterion(output, target)`.
            2.  Call `loss_current.backward()` to compute the gradients.
            3.  **Important:** You must then manually extract these gradients for every parameter in the model and store them in a single, flattened tensor, let's call it `g_current`. You can do this by iterating through `self.gif_model.parameters()`.
        * **D. Sample from Memory:** Get a batch of past experiences: `memories = self.gif_model._du_core._memory_system.sample(...)`.
        * **E. Calculate Past Gradients & Check for Interference:**
            1.  Loop through each `memory` in the retrieved batch.
            2.  For each memory, perform a forward pass and calculate the loss.
            3.  Compute the gradients for this past experience, creating a flattened tensor `g_past`.
            4.  Calculate the dot product: `dot_product = torch.dot(g_current, g_past)`.
            5.  **If `dot_product < 0`**, it signifies interference. You must then perform the projection.
        * **F. Project Gradient (The Core GEM Logic):**
            1.  If interference was detected, adjust the current gradient using the GEM projection formula: `g_current = g_current - (dot_product / torch.dot(g_past, g_past)) * g_past`.
        * **G. Apply Final Update:**
            1.  Manually copy the final, potentially projected `g_current` back into the `.grad` attribute of each model parameter.
            2.  Call `self.optimizer.step()` to apply the update.
            3.  Call `self.optimizer.zero_grad()` to clear the gradients for the next step.
            4.  Return the `loss_current.item()`.

---

**Summary of your task:**

1.  Modify `gif_framework/core/du_core.py` to allow a `PlasticityRuleInterface` and an `EpisodicMemory` to be injected into `DU_Core_V1`.
2.  Create the new file `gif_framework/training/trainer.py`.
3.  Implement the `Continual_Trainer` class.
4.  The core of your work will be implementing the `train_step` method, which must correctly execute the full Gradient Episodic Memory algorithm as detailed in steps A through G above.
5.  Ensure all new and modified code is professionally documented.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**