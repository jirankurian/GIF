**Your Current Task: Task 3.2 - Engineering the Episodic Memory System**

**Protocol Reminder:** Before you begin, execute your full **Cognitive Cycle**. Review the `/Rules` directory, the `/docs` directory, the `/Reference/` for Phase 3, read all logs in `.context/` (including the outcome previous tasks), and analyze the existing codebase. You will note that we have learning mechanisms but no system for storing the experiences they need to learn from. This task fills that critical gap. After your analysis, formulate your micro-plan and present it for approval.

---

### **Task Objective**

Your goal is to build a robust and computationally efficient **Episodic Memory system**. This is not just a simple data log; it is a core cognitive component of the DU module. Its purpose is to store structured "memories" of the framework's past experiences, which will be used by advanced Real-Time Learning mechanisms (like Gradient Episodic Memory) to learn continuously without catastrophically forgetting what it has learned before.

---

### **Domain & Technical Specifications**

#### **1. The Concept of Episodic Memory in AGI**

* **Domain Context:** In cognitive science, episodic memory is the memory of everyday events (the "what, when, where"). For our AGI, it serves a similar purpose: it's the agent's personal history. When the DU Core processes an input and produces an output, that entire interaction is an "experience." By storing these experiences, the agent can later "reflect" on them to refine its knowledge. This is the foundation for moving from a purely reactive system to one that learns from its own unique history.
* **Technical Approach:** We will implement this as a dedicated Python class that manages a memory buffer of past experiences. The system must be efficient, as it will be accessed during the real-time learning loop.

#### **2. Implementation Details**

* **Action:** You will create a new file located at: `gif_framework/core/memory_systems.py`.
* **Justification:** We are creating a new, dedicated module for memory, separating the logic of "remembering" from the logic of "learning" (`rtl_mechanisms.py`) and "processing" (`du_core.py`). This adheres to the Single Responsibility Principle and our modular design philosophy.
* **Required Libraries:** You will need `collections` (for `deque`), `typing` (for `NamedTuple`, `List`), and `torch`.

#### **3. The `ExperienceTuple` Data Structure**

* **Action:** First, you will define a `NamedTuple` called `ExperienceTuple`.
* **Justification:** Using a `NamedTuple` is a lightweight and professional way to create an immutable data structure for our memories. It makes the code more readable and less error-prone than using a generic tuple, as we can access fields by name (e.g., `experience.task_id`).
* **Required Fields:**
    * `input_spikes`: `torch.Tensor` - The spike train that was the input for this experience.
    * `internal_state`: `Any` - A placeholder for the DU Core's internal state during the experience (we can flesh this out later). For now, it can be `None`.
    * `output_spikes`: `torch.Tensor` - The resulting output spike train.
    * `task_id`: `str` - A string identifier for the task being performed (e.g., "exoplanet_classification" or "ecg_arrhythmia_detection"). This is critical for organizing memories and sampling them for specific tasks later.

#### **4. The `EpisodicMemory` Class**

* **Action:** You will implement the main `EpisodicMemory` class.
* **`__init__(self, capacity: int)`:**
    * The initializer will take a single integer, `capacity`, which defines the maximum number of experiences the memory can hold.
    * Inside the initializer, you will create a **`collections.deque`** with its `maxlen` attribute set to this `capacity`.
    * **Justification:** A `deque` with a `maxlen` is the perfect data structure for our memory buffer. It is highly optimized in C for fast appends. When the deque is full and a new item is added, it automatically and efficiently discards the oldest item from the other end, creating a perfect "sliding window" or FIFO (First-In-First-Out) memory buffer.

* **`add(self, experience: ExperienceTuple) -> None`:**
    * This method will take a single `ExperienceTuple` object as input.
    * It will simply append this experience to the internal `deque`.

* **`sample(self, batch_size: int) -> List[ExperienceTuple]`:**
    * **This is the most critical method for our future learning algorithms.**
    * It takes an integer `batch_size`.
    * It must first check if the requested `batch_size` is larger than the number of memories currently stored. If so, it should just return all the memories it has.
    * It will then use Python's `random.sample()` function to randomly select a `batch_size` number of unique experiences from the internal `deque`.
    * **Return Value:** It must return a list of `ExperienceTuple` objects.

---

**Summary of your task:**

1.  Create the new file `gif_framework/core/memory_systems.py`.
2.  Import the necessary libraries.
3.  Define the `ExperienceTuple` `NamedTuple` with its specified fields.
4.  Implement the `EpisodicMemory` class.
5.  The class must use a `collections.deque` with a `maxlen` for its internal storage.
6.  Implement the `add` method to store new experiences.
7.  Implement the `sample` method to retrieve a random batch of past experiences.
8.  Ensure all code is professionally documented with clear docstrings explaining the purpose of the episodic memory system and its methods.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**