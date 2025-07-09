**Your Current Task: Task 2.4 - Establishing a Rigorous Testing Suite**

**Protocol Reminder:** Before you begin, you must execute your full **Cognitive Cycle**. Review the `/Rules` directory, the `/Reference/` for Phase 2, read all logs in the `.context/` to confirm the completion of Tasks 2.1, 2.2, and 2.3, and analyze the existing project structure, noting the `interfaces`, `orchestrator`, and `core` modules. After your analysis, formulate your micro-plan for this task and present it for approval.

-----

### **Task Objective**

Your goal is to build a comprehensive suite of automated tests for the GIF framework. This suite will serve two critical purposes:

1.  **Verification:** To prove that the components we have built so far (`GIF`, `DU_Core_V1`, and the interfaces) function exactly as designed.
2.  **Regression Prevention:** To create a "safety net" that automatically checks our entire framework every time we make changes, ensuring that future development doesn't accidentally break existing functionality.

We will use the **`pytest`** framework, which is the industry standard for testing in Python due to its power, simplicity, and rich ecosystem.

-----

### **Domain & Technical Specifications**

#### **1. The Principle of Mocking for Unit Tests**

  * **Domain Context:** When we test a single component (a "unit"), we want to test it in isolation. For example, when testing the `GIF` orchestrator, we don't care about the complex internal logic of a real `Encoder` or `DU_Core`. We only care if the `GIF` class calls their methods correctly.
  * **Technical Approach:** To achieve this isolation, we use **Mock Objects**. A mock is a "fake" object that we create for the purpose of a test. It mimics the behavior of a real object but has a very simple, predictable implementation. You will create mock versions of our encoders and decoders.

#### **2. Test-Driven Development (TDD) Structure**

  * **Action:** You will create a root-level `/tests/` directory. All test files must be named `test_*.py` or `*_test.py` for `pytest` to discover them automatically.

-----

### **Step-by-Step Implementation Plan**

#### **Step 1: Project and Tooling Setup**

  * **Action:**
    1.  Create the `/tests/` directory in the project root.
    2.  Inside `/tests/`, create `__init__.py` to make it a package. Also create subdirectories `tests/core/` and `tests/interfaces/` with their own `__init__.py` files to mirror our framework's structure.
    3.  Modify the `pyproject.toml` file. Add `pytest` to the project's development dependencies. It should look something like this:
        ```toml
        [project.optional-dependencies]
        dev = ["pytest", "ruff"]
        ```
  * **Justification:** This sets up the professional structure required for an organized and discoverable test suite. Adding `pytest` as a development dependency ensures that anyone who works on the project can easily install the necessary testing tools with `uv pip install -e .[dev]`.

#### **Step 2: Create Mock Components**

  * **Action:**
    1.  Create a new file: `tests/mocks.py`.
    2.  In this file, implement a `MockEncoder` class that inherits from `EncoderInterface`. Its `encode` method should not do any real work; it should simply return a pre-defined `torch.Tensor` of a specific shape. Its `get_config` and `calibrate` methods can return empty dictionaries or `None`.
    3.  Implement a `MockDecoder` class that inherits from `DecoderInterface`. Its `decode` method should simply return a fixed, predictable value, for example, the string `"mock_action_success"`.
  * **Justification:** These mock objects are essential for writing fast, reliable unit tests. They allow us to test the `GIF` orchestrator's logic without needing to run a complex SNN simulation or a real data encoding process, isolating the component we are actually testing.

#### **Step 3: Write Unit Tests for the `DU_Core_V1`**

  * **Action:**
    1.  Create the file `tests/core/test_du_core.py`.
    2.  Write a test function `test_du_core_initialization()` that creates instances of `DU_Core_V1` with different valid configurations (e.g., varying numbers of hidden layers) and asserts that they are created without errors.
    3.  Write a test function `test_du_core_forward_pass_shape()` that:
          * Initializes a `DU_Core_V1` with a known configuration (e.g., `input_size=10, hidden_sizes=[20], output_size=5`).
          * Creates a dummy input spike tensor of the correct shape (e.g., `shape=[50, 1, 10]` for `num_steps=50, batch_size=1`).
          * Calls the `forward()` method.
          * Asserts that the returned output tensor has the expected shape (`[50, 1, 5]`).
  * **Justification:** These tests verify that our "brain" module is structurally sound and that its input/output dimensions are handled correctly, which is a common source of bugs in neural network code.

#### **Step 4: Write Unit Tests for the `GIF` Orchestrator**

  * **Action:**
    1.  Create the file `tests/test_orchestrator.py`.
    2.  Write `test_attach_valid_modules()`: Instantiates the `GIF` and `MockEncoder`/`MockDecoder` and asserts that the `attach_` methods run without error.
    3.  Write `test_attach_invalid_modules()`: Creates a simple dummy class that does *not* inherit from our interfaces. Use `pytest.raises(TypeError)` to assert that passing this invalid object to an `attach_` method correctly raises a `TypeError`. This proves our interface contracts are being enforced.
    4.  Write `test_run_cycle_without_modules_fails()`: Instantiates a `GIF` and calls `process_single_input()` *before* attaching modules. Use `pytest.raises(RuntimeError)` to assert that the method fails with the expected error. This tests our error-handling logic.
  * **Justification:** These tests validate the "contractual" and "coordination" logic of the `GIF` class. They prove that it correctly manages its modular components and handles invalid configurations gracefully.

#### **Step 5: Write the Core Integration Test**

  * **Action:**
    1.  In the same `tests/test_orchestrator.py` file, write a final, critical test: `test_full_integration_cycle()`.
    2.  This test will bring everything together:
        a.  Instantiate a real `DU_Core_V1`.
        b.  Instantiate the `GIF` and inject the `DU_Core_V1`.
        c.  Instantiate the `MockEncoder` and `MockDecoder` from `tests/mocks.py`.
        d.  Attach the mock encoder and decoder to the `GIF` instance.
        e.  Call `gif.process_single_input()` with some dummy raw data.
        f.  Assert that the final returned value is exactly `"mock_action_success"`.
  * **Justification:** This is the most important test in this phase. It doesn't just test one component; it tests the **entire architectural pipeline**. A passing integration test proves that the data flows correctly from the outside world, through the encoder's contract, through the real DU Core, through the decoder's contract, and back out as a final action. It confirms our core architecture is sound.

-----

**Summary of your task:**

1.  Set up the `/tests/` directory and `pytest` configuration.
2.  Create the `tests/mocks.py` file with mock components.
3.  Create `tests/core/test_du_core.py` and write unit tests for the SNN.
4.  Create `tests/test_orchestrator.py` and write unit tests for the GIF's logic.
5.  In the same file, write the final integration test to validate the entire pipeline.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**