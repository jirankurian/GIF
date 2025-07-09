**Your Current Task: Task 5.1 - Implementing the Medical Domain Application Modules**

**Protocol Reminder:** Before you begin, you must execute your full **Cognitive Cycle**. Review the `/Rules` directory, the `/Reference/` for Phase 5, read all logs in the `.context/` to understand the completion of the exoplanet POC, and analyze the existing project codebase. You will see that we have proven the framework on one domain; this task is the first step in proving its generality. After your analysis, formulate your micro-plan for this task and present it for approval.

---

### **Task Objective**

Your goal is to build the domain-specific modules required for our second proof-of-concept: **ECG arrhythmia detection**. You will implement a new `ECG_Encoder` and a new `Arrhythmia_Decoder`. This is the practical test of the GIF's "plug-and-play" architecture, where we will "unplug" the astronomy modules and "plug in" these new medical diagnostic modules without changing the core framework.

---

### **Domain & Technical Specifications**

#### **1. The Challenge: Adapting to a New Domain**

* **Domain Context:** An electrocardiogram (ECG) is a time-series signal representing the heart's electrical activity. While it is a time-series like an exoplanet light curve, its features are entirely different. Instead of looking for dips in brightness, we are looking for subtle changes in the shape, timing, and rhythm of specific waveforms (the P-wave, QRS complex, and T-wave).
* **Technical Approach:** You will create a new set of classes that implement our established `EncoderInterface` and `DecoderInterface`. This demonstrates that as long as a module adheres to the contract, the GIF can work with it, regardless of the domain.

#### **2. Implementation Details for the `ECG_Encoder`**

* **Action:** You will create a new application package at `applications/poc_medical/` and within it, the file `applications/poc_medical/encoders/ecg_encoder.py`.
* **Class Definition:** Implement a class `ECG_Encoder` that inherits directly from `EncoderInterface`.
* **`encode(self, raw_data: polars.DataFrame) -> torch.Tensor`:**
    * **Purpose:** To convert a continuous ECG voltage signal into a meaningful spike train.
    * **Input:** A `polars.DataFrame` from our `RealisticECGGenerator` containing "time" and "voltage" columns.
    * **Encoding Scheme: Threshold-Based Feature Encoding.** This is a more sophisticated method than we used for the exoplanet data and is better suited for ECGs. The logic is as follows:
        1.  First, you must process the raw ECG signal to detect the locations of the key peaks: the R-peaks (the most prominent spike) and potentially the P and T waves. You can use a standard peak detection algorithm from a library like `scipy.signal.find_peaks`.
        2.  You will generate a **multi-channel spike train**. Each channel will represent a specific feature. For example:
            * **Channel 0 (R-peaks):** Fire a spike in this channel at the time-step corresponding to each detected R-peak.
            * **Channel 1 (Heart Rate):** Calculate the instantaneous heart rate (the R-R interval) and encode this value using a rate-based code (i.e., the number of spikes in a small time window is proportional to the heart rate).
            * **Channel 2 (QRS Duration):** Measure the width of the QRS complex and encode this duration using a latency code (i.e., the timing of a spike relative to the R-peak encodes the duration).
    * **Return:** A `torch.Tensor` of shape `[num_time_steps, num_channels]` representing the feature-rich spike train.
* **Justification:** This feature-based encoding is more powerful than simple delta modulation because it provides the SNN with information that is already known to be clinically relevant (heart rate, peak locations, etc.). This allows the DU Core to focus on learning the complex *patterns* between these features, which is a more efficient use of its resources.

#### **3. Implementation Details for the `Arrhythmia_Decoder`**

* **Action:** Create the file `applications/poc_medical/decoders/arrhythmia_decoder.py`.
* **Class Definition:** Implement a class `Arrhythmia_Decoder` that inherits from `DecoderInterface` and `torch.nn.Module`.
* **`decode(self, spike_train: torch.Tensor) -> str`:**
    * **Purpose:** To interpret the output spike pattern from the DU Core and classify it as a specific cardiac arrhythmia.
    * **Input:** The output `SpikeTrain` from the DU Core.
    * **Readout Logic:**
        1.  **Spike Integration:** First, sum the spikes for each output neuron over the entire simulation window to get a vector of total spike counts. This vector represents the DU Core's final "vote."
        2.  **Linear Readout Layer:** In the decoder's `__init__`, you will define a `torch.nn.Linear` layer. This layer will take the vector of spike counts as input and map it to a set of output logits, one for each possible arrhythmia class (e.g., Normal Sinus Rhythm, Atrial Fibrillation, etc., according to the AAMI standard).
        3.  **Classification:** Apply a `softmax` function to the logits to get the final class probabilities and return the name of the class with the highest probability.
* **Justification:** The decoder acts as the final, trainable "decision-making" layer. While the SNN learns the complex temporal patterns, this simple linear layer learns to map those patterns to the specific class labels we care about. This separation of concerns is an efficient and standard practice in SNN design.

---

**Summary of your task:**

1.  Create the new application package `applications/poc_medical/`.
2.  In this package, create `encoders/ecg_encoder.py` and implement the `ECG_Encoder` class using a feature-based encoding scheme.
3.  Create `decoders/arrhythmia_decoder.py` and implement the `Arrhythmia_Decoder` class with a trainable linear readout layer for classification.
4.  Ensure both new classes correctly inherit from their respective interfaces and are professionally documented.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**