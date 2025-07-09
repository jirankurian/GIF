**Your Current Task: Task 5.3 - Building the Potentiation Analysis & Visualization Suite**

**Protocol Reminder:** Before you begin, execute your full **Cognitive Cycle**. Review the `/Rules` directory, the `/Reference/` for Phase 5, read all logs in `.context/` (which confirm the creation of the medical modules and the master experiment script), and analyze the existing project structure. You will assume that the script from Task 5.2 has been run and has produced the `naive_run.log` and `pre_exposed_run.log` files. This task is to build the tools to analyze those files. After your analysis, formulate your micro-plan for this task and present it for approval.

---

### **Task Objective**

Your goal is to create a comprehensive analysis suite, in the form of a Jupyter Notebook, that will process the raw data from our potentiation experiment. This notebook will perform all the necessary calculations and generate the final, publication-quality tables and figures needed to rigorously prove or disprove the **System Potentiation** hypothesis. This is where we turn raw numbers into scientific evidence.

---

### **Domain & Technical Specifications**

#### **1. The Importance of Rigorous & Reproducible Analysis**

* **Domain Context:** A scientific claim is only as strong as the analysis that supports it. To prove a novel concept like "system potentiation," we must use standardized, well-understood metrics and present them with absolute clarity. The analysis must be transparent and reproducible.
* **Technical Approach:** We will use a Jupyter Notebook (`.ipynb`) as our analysis environment. This allows us to combine executable Python code, its output (like tables and plots), and explanatory Markdown text into a single, shareable document that tells the full story of our analysis.

#### **2. Advanced Analysis: Representational Similarity Analysis (RSA)**

* **Domain Context:** To go beyond simple accuracy metrics and truly understand *how* the "Pre-Exposed" model is learning better, we will use an advanced technique from computational neuroscience called **Representational Similarity Analysis (RSA)**.
* **The Analogy:** Think of it this way: We want to create a "map" of how the DU Core's brain organizes information. For each type of arrhythmia, the SNN will have a unique pattern of internal activity. RSA measures how "similar" or "different" these internal patterns are for every pair of arrhythmias. A "smarter" brain will create a more organized map, where similar conditions have similar patterns and very different conditions have very different patterns. We hypothesize that the Pre-Exposed model will create a clearer, more structured "map" much faster than the Naive model.

#### **3. Implementation Details**

* **Action:** You will create a new file: `applications/poc_medical/analysis.ipynb`.
* **Justification:** A dedicated notebook for the medical POC analysis keeps our work organized and creates a single, self-contained report for this critical experiment.
* **Required Libraries:** `polars`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`.

---

### **Step-by-Step Notebook Implementation Plan**

Your notebook should be structured with the following sections, using Markdown cells for headings and explanations.

#### **Section 1: Setup and Log File Ingestion**

* **Action:**
    1.  Import all necessary libraries.
    2.  Define the file paths for the `naive_run_log.csv` and `pre_exposed_run_log.csv`.
    3.  Use `polars.read_csv()` to load these two log files into two separate DataFrames: `naive_df` and `pre_exposed_df`.
    4.  Display the `.head()` of both DataFrames to verify they have loaded correctly.

#### **Section 2: Analysis of Learning Efficiency (Potentiation Metric 1)**

* **Action:**
    1.  Implement a function `plot_learning_curves(naive_df, pre_exposed_df)`.
    2.  This function will use `matplotlib` to plot the `accuracy` vs. `training_samples` for both models **on the same axes**. The "Pre-Exposed" line should be clearly distinguishable from the "Naive" line (e.g., different colors, line styles).
    3.  The plot must have a title, labeled axes, and a legend.
    4.  Implement a second function `calculate_learning_efficiency(df, target_accuracy=0.9)` that finds the number of training samples required to first reach the `target_accuracy`.
    5.  Call this function for both DataFrames and print the results clearly.
* **Justification:** This section provides the most direct, visual evidence of potentiation. The plot will immediately show if the Pre-Exposed model learns faster.

#### **Section 3: Analysis of Few-Shot Generalization (Potentiation Metric 2)**

* **Action:**
    1.  You will write a function `test_few_shot_performance(model, rare_class_data, n_shots)`.
    2.  This function will take a trained model (either the final Naive or Pre-Exposed model), a small dataset for a rare arrhythmia class, and the number of shots (e.g., 1, 5, 10).
    3.  It will perform a brief "fine-tuning" on the `n_shots` samples and then evaluate the model's accuracy on a held-out test set for that rare class.
    4.  Run this function for both the Naive and Pre-Exposed models for N=1, 5, and 10 shots.
    5.  Use `matplotlib` to create a grouped bar chart comparing the few-shot accuracy of the two models at each shot level.
* **Justification:** This test measures how quickly each model can generalize to new information. A superior performance by the Pre-Exposed model here indicates it has "learned how to learn" more effectively from sparse data.

#### **Section 4: Analysis of Catastrophic Forgetting**

* **Action:**
    1.  Write a function `evaluate_forgetting(pre_exposed_model, exoplanet_test_data)`.
    2.  This function will load the final trained "Pre-Exposed" model (after it has learned the ECG task).
    3.  It will then load the held-out test data for the **exoplanet task** from Phase 4.
    4.  It will run an evaluation and calculate the final accuracy on the old task.
    5.  The notebook will then compare this final accuracy to the original peak accuracy on the exoplanet task to calculate the **Forgetting Measure**.
* **Justification:** This is a critical sanity check. We must prove that our framework's ability to learn a new task does not come at the cost of completely destroying its old knowledge.

#### **Section 5: Advanced Insight with Representational Similarity Analysis (RSA)**

* **Action:**
    1.  This requires a helper function to extract hidden layer activations from the `DU_Core`. You will need to modify the `DU_Core` to return these during a forward pass if a flag is set.
    2.  Write a function `run_rsa_analysis(model, task_data)`.
    3.  Inside, you will loop through each arrhythmia class. For each class, you will get the average hidden-layer activation pattern from the model.
    4.  You will then compute the dissimilarity (e.g., `1 - correlation`) between the average patterns for every pair of classes. This creates a square **Representational Dissimilarity Matrix (RDM)**.
    5.  Use `seaborn.heatmap()` to visualize the RDMs for both the Naive and Pre-Exposed models.
* **Justification:** This analysis provides deep insight into the internal workings of the "brain." A clear, block-like structure in the RDM indicates a well-organized internal representation space. Visually comparing the two RDMs will provide powerful evidence that the Pre-Exposed model develops a more structured "understanding" of the medical domain.

#### **Section 6: Final Results Generation**

* **Action:** The final cells of the notebook will collect all the metrics calculated in the previous sections and programmatically generate the final, publication-ready tables, **replicating Tables V and VI** from your research paper manuscript.

---

**Summary of your task:**

1.  Create the analysis notebook `applications/poc_medical/analysis.ipynb`.
2.  Structure the notebook into the clear sections outlined above.
3.  Implement the Python code to perform each analysis: learning curves, few-shot evaluation, forgetting measurement, and the advanced RSA.
4.  The final output of the notebook should be the tables and plots that provide the complete, quantitative evidence for the claims of generalization and system potentiation.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**