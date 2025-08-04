### **Prompt 6.4: Implementing Web-Augmented Knowledge & Final Feature Validation**

**Protocol Reminder:** Before you begin, you must execute your full **Cognitive Cycle**. Review the `/Rules` directory, the `/Reference/` for Phase 6, read all logs in the `.context/`, and analyze the entire project codebase. This is a critical two-part task: first, you will add the final knowledge-gathering feature, and second, you will validate all advanced features to ensure the framework is 100% correct.

**Objective:**

1.  **Enhance the Framework:** Upgrade the `KnowledgeAugmenter` to perform live **Retrieval-Augmented Generation (RAG) from the web**, prioritizing it over the existing database methods.
2.  **Validate All Advanced Features:** Write and execute a final, exhaustive suite of automated tests that rigorously verify the correctness of every feature developed in Phase 6: the `DU_Core_V2`, the database RAG/CAG, the new Web RAG, Meta-Cognitive Routing, and Interface Self-Generation.

**Context & Rationale:**
For the GIF/DU to be a truly general intelligence, it must have access to the most current information possible. Grounding its knowledge in live web data is the ultimate test of this capability. After implementing this final feature, we will conduct a thorough testing and validation pass. This is the final quality gate at the feature level. A passing test suite from this prompt will certify that the framework is robust, correct, and ready for the final system-wide integration tests and scientific result generation phases.

**Architectural Adherence:**
The new Web RAG logic will be encapsulated within the existing `KnowledgeAugmenter` module. The new tests will be added to the appropriate files within the `/tests/` directory, adhering to our established `pytest` structure.

-----

### **Part 1: Development - Implementing Web-Augmented RAG**

#### **Step 1.1: Update Dependencies**

  * **File to Modify:** `pyproject.toml`.
  * **Action:** Add the necessary libraries for web searching and parsing to the `[project.optional-dependencies]` section under the `knowledge` key.
  * **Libraries to Add:** `"duckduckgo-search"`, `"requests"`, `"beautifulsoup4"`.

#### **Step 1.2: Upgrade the `KnowledgeAugmenter`**

  * **File to Modify:** `gif_framework/core/knowledge_augmenter.py`.
  * **Action:** Add a new method for web retrieval and update the main entry point to prioritize it.
  * **Code to Implement:**
    ```python
    # Add new imports at the top
    import requests
    from bs4 import BeautifulSoup
    from duckduckgo_search import DDGS

    # Inside the KnowledgeAugmenter class:
    def retrieve_web_context(self, query: str) -> str:
        """Performs a web search and scrapes the top result for context."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=1))
                if not results:
                    return "No web results found."
                
                top_result_url = results[0]['href']
                
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(top_result_url, headers=headers, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract text from paragraph tags, a simple but effective method
                paragraphs = [p.get_text() for p in soup.find_all('p')]
                return " ".join(paragraphs)
        except Exception as e:
            # Handle potential errors gracefully
            return f"Failed to retrieve web context: {e}"

    # You will also create a new primary retrieval method
    def retrieve_context(self, query: str) -> str:
        """Primary method to get context, prioritizing the web."""
        web_context = self.retrieve_web_context(query)
        if "Failed to retrieve" not in web_context and "No web results" not in web_context:
            return web_context
        else:
            # Fallback to existing database RAG
            return " ".join(self.retrieve_unstructured_context(query))
    ```

-----

### **Part 2: Validation of All Advanced Features**

#### **Step 2.1: Test the New Web RAG Capability**

  * **File to Modify:** `tests/core/test_knowledge_augmenter.py`.
  * **Action:** Add a new test for the web retrieval function, mocking the external web calls.
  * **Code to Implement:**
    ```python
    from unittest.mock import patch, MagicMock

    def test_web_rag_retrieval():
        """Tests that the web RAG method correctly searches, fetches, and parses."""
        augmenter = KnowledgeAugmenter(...)

        mock_search_results = [{'href': 'http://mock-url.com'}]
        mock_html_content = "<html><body><p>This is the first paragraph.</p><p>This is the second.</p></body></html>"
        expected_text = "This is the first paragraph. This is the second."

        # Use patch to mock the external libraries
        with patch('gif_framework.core.knowledge_augmenter.DDGS') as mock_ddgs, \
             patch('gif_framework.core.knowledge_augmenter.requests.get') as mock_get:
            
            # Configure the mocks
            mock_ddgs.return_value.__enter__.return_value.text.return_value = mock_search_results
            mock_get.return_value.content = mock_html_content
            mock_get.return_value.raise_for_status = MagicMock()

            # Run the method
            retrieved_context = augmenter.retrieve_web_context("test query")
            
            # Assertions
            mock_ddgs.return_value.__enter__.return_value.text.assert_called_with("test query", max_results=1)
            mock_get.assert_called_with('http://mock-url.com', headers=ANY, timeout=10)
            assert retrieved_context == expected_text
    ```

#### **Step 2.2: Implement the Full Advanced Feature Test Suite**

  * **Action:** You are to implement the **complete test suite for all of Phase 6**, as defined in the prompt `Prompt6.4.md` that I provided previously. I am including the summary of required tests here for your convenience. **If any of these tests fail, you must analyze the traceback, identify the bug in the corresponding module, and implement the necessary fix.**
  * **Summary of Required Tests (from previous prompt):**
    1.  **`DU_Core_V2` Validation (`tests/core/test_du_core_v2.py`):**
          * Implement `test_du_core_v2_has_heterogeneous_architecture()` to verify the mix of SNN/SSM and Attention layers.
          * Implement `test_du_core_v2_forward_pass_runs_without_error()` to ensure the complex data flow executes.
    2.  **Database RAG/CAG Validation (`tests/core/test_knowledge_augmenter.py`):**
          * Implement `test_rag_retrieval_constructs_correct_query()` to validate Milvus query construction using a mock client.
          * Implement `test_cag_update_constructs_correct_cypher_query()` to validate Neo4j query construction using a mock driver.
    3.  **Meta-Cognition Validation (`tests/applications/test_meta_cognition.py`):**
          * Implement `test_meta_controller_selects_correct_encoder()` to prove the `GIF` can dynamically choose the right tool from the `ModuleLibrary`.
    4.  **Self-Generation Validation (`tests/applications/test_self_generation.py`):**
          * Implement `test_generates_and_imports_valid_decoder()` to validate the full NL -\> Code -\> Import -\> Validation pipeline using `tempfile` and `importlib`.

**Final Deliverable:**
A feature-complete framework that now includes live Web RAG capabilities. More importantly, a complete and **100% passing `pytest` suite** that validates every advanced feature from Phase 6. This provides the final layer of assurance that the framework is not only architecturally sound but that its most sophisticated, AGI-related capabilities are implemented correctly and are ready for the final system-wide validation.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**