**Your Current Task: Task 3.4 - Developing the Evaluation Suite for Lifelong Learning**

**Protocol Reminder:** Before you begin, execute your full **Cognitive Cycle**. Review the `/Rules` directory, the `/Reference/` for Phase 3, read all logs in the `.context/` (which confirm the completion of Tasks 3.1, 3.2, and 3.3), and analyze the existing codebase. You will see we have a `Continual_Trainer` but no way to formally measure if it successfully prevents catastrophic forgetting. This task creates that measurement capability. After your analysis, formulate your micro-plan and present it for approval.

---

### **Task Objective**

Your goal is to build a dedicated **analysis suite** to rigorously evaluate the performance of our lifelong learning system. This is not about training the model, but about creating the tools to **measure how well it learns and remembers over time**. This suite will be used to generate the hard, quantitative evidence needed to support our claims about mitigating catastrophic forgetting and enabling positive knowledge transfer.

---

### **Domain & Technical Specifications**

#### **1. The Science of Evaluating Continual Learners**

* **Domain Context:** In the field of continual (or lifelong) learning, researchers have developed standardized metrics to move beyond simple accuracy. We need to measure not just how well the model learns a new task, but also how that new learning affects its memory of old tasks. A model that achieves 99% accuracy on a new task but forgets everything it knew before is a failure in the context of AGI.
* **Technical Approach:** You will implement a Python module that contains functions to calculate these standard metrics from the performance logs generated during our experiments.

#### **2. The Experimental Protocol**

* **Domain Context:** To measure these metrics, we need a specific, structured way of training and testing the model. This is known as an experimental protocol. Our protocol will be a **sequential task protocol**.
* **Protocol Definition:** The system will be trained on a sequence of distinct tasks (e.g., Task A, then Task B, then Task C). After training on each new task, the model's performance will be evaluated on a held-out test set for **all** tasks it has seen so far. The history of these evaluations is what allows us to measure forgetting and transfer. For example, after training on Task C, we measure its accuracy on C, B, *and* A.

#### **3. Implementation Details**

* **Action:** You will create a new file located at: `applications/analysis/continual_learning_analyzer.py`.
* **Justification:** We are creating a centralized, reusable module for all our continual learning analysis. This prevents cluttering our experiment scripts with complex analysis code and ensures that the same metrics are calculated consistently across all experiments.
* **Required Libraries:** `numpy`, `polars` (for data manipulation), `matplotlib` or `seaborn` (for plotting).

#### **4. The `ContinualLearningAnalyzer` Class**

* **Action:** You will implement the main `ContinualLearningAnalyzer` class.
* **`__init__(self, experiment_logs: polars.DataFrame)`:**
    * The initializer will take a single argument: a `Polars DataFrame` containing the results from a full sequential training experiment. This DataFrame should have columns like `training_task_id`, `evaluation_task_id`, `epoch`, and `accuracy`.

* **`calculate_average_accuracy(self) -> float`:**
    * **Purpose:** This metric gives an overall sense of the model's performance across its lifetime.
    * **Implementation:** This method will calculate the average accuracy of the model on all tasks *it has learned so far*, at the very end of the training sequence. For a three-task experiment (A, B, C), this would be the average of the final accuracies on task A, B, and C.

* **`calculate_forgetting_measure(self, target_task_id: str) -> float`:**
    * **Purpose:** This is the most direct measure of catastrophic forgetting. It answers the question: "After learning other things, how much did the model forget about this specific task?"
    * **Implementation:**
        1.  Find the **maximum accuracy** the model ever achieved on the `target_task_id` (this usually happens right after it finishes training on that task).
        2.  Find the **final accuracy** of the model on the `target_task_id` at the end of the entire experiment.
        3.  The forgetting measure is the difference: `max_accuracy - final_accuracy`. A value close to zero is excellent.

* **`calculate_forward_transfer(self, target_task_id: str) -> float`:**
    * **Purpose:** This measures if learning previous tasks helped the model learn a new task faster or better. It's a key indicator of **system potentiation**.
    * **Implementation:**
        1.  Find the accuracy of a "naive" model (one trained from scratch) on the `target_task_id` after its first epoch.
        2.  Find the accuracy of our "pre-exposed" model on the `target_task_id` after its first epoch.
        3.  Forward transfer is the difference: `pre-exposed_accuracy - naive_accuracy`. A positive value means prior experience was helpful.

* **`generate_summary_plot(self) -> None`:**
    * **Purpose:** To create a clear, publication-quality visualization of the learning process.
    * **Implementation:** This method will generate a line plot. The x-axis will be the training progress (e.g., by task or epoch). The y-axis will be accuracy. It will plot a separate line for the performance on each evaluation task over time. This plot will visually show if the accuracy on old tasks (e.g., Task A) drops off as new tasks are learned.

---

**Summary of your task:**

1.  Create the new file `applications/analysis/continual_learning_analyzer.py`.
2.  Implement the `ContinualLearningAnalyzer` class.
3.  The class must be initialized with a DataFrame of experiment logs.
4.  Implement the methods to calculate the three critical continual learning metrics: `Average Accuracy`, `Forgetting Measure`, and `Forward Transfer`.
5.  Implement a method to generate a summary plot visualizing the performance on all tasks over the course of the experiment.
6.  Ensure all code is professionally documented.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**