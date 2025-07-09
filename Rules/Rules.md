### **The Strategic Prompting & Development Protocol for GIF-DU**

This protocol is designed to ensure maximum context retention and precision throughout the development of the General Intelligence Framework. It combines a static knowledge base, a dynamic session log, and a rigorous cognitive cycle for the AI Agent.

---

#### **1. Guiding Principles**

* **Clarity over Speed:** Every step will be deliberate and explicit. The AI Agent will prioritize clear, well-defined tasks over rapid, potentially-confused generation.
* **User as the Final Authority:** The AI agent will always present its micro-plan for a given task and wait for the users explicit approval before generating any code or modifying the project structure. The user is the project lead.
* **Single Task per Prompt:** Each prompt the AI Agent provide will correspond to a single, well-defined task from the phase breakdowns. This ensures a tight, focused context for every interaction.
* **Traceability:** Every action taken by the AI agent will be logged, creating a fully traceable history of the project's development.

---

#### **2. The Project Knowledge Base**

To maintain perfect context, Establish a two-part knowledge base within the project directory.

* **A. The `/Reference/` Directory (The "Bible")**
    * **Purpose:** This folder will contain the **static, canonical "source of truth"** for the project. It is AI Agent's constitution. It will not be modified during development unless the user explicitly agree to amend the plan.
    * **Contents:** A series of Markdown (`.md`) files, one for each phase of our development plan (e.g., `Phase1.md`, `Phase2.md`, etc.). Each file will contain the detailed task-by-task breakdown for that phase, including objectives, key activities, justifications, and required libraries.
    * **Core Research:** A series of docx and pdf files containing the core research and theoretical underpinnings of the project.
    * **Usage:** At the beginning of every task, The AI Agent's first action will be to read the relevant reference document to re-ground itself in the high-level goals of the current phase.

* **B. The `.context/` Directory (The "Session Log")**
    * **Purpose:** This folder will serve as the **dynamic, short-term memory or session log** of the AI agent. It is a chronological record of actions taken and their outcomes.
    * **Contents:** A series of timestamped log files (e.g., `20250709_1230_Task_2.1_Complete.log`). Each file will be a concise summary written by the AI Agent upon the completion of a task, detailing what was done, what files were created or modified, and the outcome (e.g., "Task 2.1 complete. Created `base_interfaces.py` with `EncoderInterface` and `DecoderInterface`. All tests passed.").
    * **Usage:** After reviewing the "Bible," The AI Agent's next step will always be to sequentially read all logs in this directory to reconstruct the full history of the current development session.

---

#### **3. The AI Agent's Cognitive Cycle**

For every new task prompt the user provide, The AI Agent will execute the following rigorous cognitive cycle before generating any output. The AI Agent will explicitly state which step it is on in its thought process.

1.  **Step 1: Re-establish Foundational Knowledge (`Reading the Bible`)**
    * Action: The AI Agent will first access and review the relevant Markdown file in the `/Reference/` directory that corresponds to the current development phase.

2.  **Step 2: Reconstruct Session History (`Reading the Log`)**
    * Action: The AI Agent will then access and sequentially review all log files in the `.context/` directory to understand the precise state of our progress so far.

3.  **Step 3: Analyze Current Codebase (`Scanning the Code`)**
    * Action: The AI Agent will perform a full analysis of the current project directory, paying close attention to the files relevant to the upcoming task.

4.  **Step 4: Formulate a Micro-Plan (`Building the Plan`)**
    * Action: Based on the synthesis of the above three steps, The AI Agent will create a detailed, step-by-step micro-plan for executing the *current task only*. This plan will be presented to the user for approval. It will detail what files The AI Agent intend to create or modify and the logic The Agent will implement.

5.  **Step 5: Await Approval (`Awaiting Your Command`)**
    * Action: The AI Agent will present the micro-plan to the user and stop. The AI Agent will not proceed until The AI Agent receive user's explicit confirmation (e.g., "Approved," "Proceed," "Looks good").

6.  **Step 6: Execute the Task (`Generating the Solution`)**
    * Action: Upon users approval, The AI Agent will execute the micro-plan and generate the required code, documentation, or other artifacts.

7.  **Step 7: Log the Outcome (`Updating the Log`)**
    * Action: Immediately after generating the solution, The AI Agent will create a new, timestamped log file in the `.context/` directory, summarizing what was just accomplished.

---

#### **4. Our Interaction Model**

Our workflow will be a structured, sequential execution of the phase breakdowns.

1.  The AI Agent will initiate each task by providing a prompt, e.g., **"Let's start Task 1.1."**
2.  The AI Agent will execute its full **Cognitive Cycle** (Steps 1-5), culminating in presenting the user with a micro-plan for Task 1.1.
3.  The user will review the plan and give their approval.
4.  The AI Agent will execute Step 6 (generating the code) and Step 7 (logging the outcome).
5.  This process will repeat for Task 1.2, and so on.
