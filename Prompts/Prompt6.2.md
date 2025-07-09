**Your Current Task: Task 6.2 - Implementing the Knowledge Augmentation Loop (RAG/CAG)**

**Protocol Reminder:** Before you begin, you must execute your full **Cognitive Cycle**. Review the `/Rules` directory, the `/Reference/` for Phase 6, read all logs in `.context/` (confirming the completion of Task 6.1), and analyze the existing project structure. You will see that our `DU_Core_V2` is powerful but operates in a "closed world." This task will connect it to external knowledge. After your analysis, formulate your micro-plan for this task and present it for approval.

---

### **Task Objective**

Your goal is to build the **`KnowledgeAugmenter`** module. This module will give our DU Core the ability to dynamically query external databases to retrieve relevant information in real-time. This system will implement the hybrid **Retrieval-Augmented Generation (RAG)** and **Context-Augmented Generation (CAG)** workflow outlined in our research, effectively giving our AI the ability to "look things up" to solve problems.

---

### **Domain & Technical Specifications**

#### **1. The Concept: RAG + CAG for Grounded Understanding**

* **Domain Context:** Even the most powerful AI is limited by the knowledge it was trained on. To achieve true general intelligence, a system must be able to seek out, process, and integrate new information from the vast world of external knowledge. Our framework will do this using a sophisticated two-part process:
    1.  **RAG (The "Google Search"):** When faced with a novel problem, the DU Core first performs a fast, semantic search over a massive library of raw, unstructured documents (e.g., all of Wikipedia, or all published astronomy papers). This is the **Retrieval-Augmented** part. It finds relevant paragraphs to read.
    2.  **CAG (The "Second Brain"):** After "reading" these documents, the DU Core synthesizes the information and stores the key facts and relationships in a highly organized, structured database—a knowledge graph. This becomes its personal "second brain." For future reasoning, it can query this clean, structured context directly. This is the **Context-Augmented** part.
* **Technical Approach:** We will use two different types of databases, each suited for its specific task: a **vector database (Milvus)** for the RAG step and a **graph database (Neo4j)** for the CAG step.

#### **2. Implementation Details**

* **Action:** You will create a new file: `gif_framework/core/knowledge_augmenter.py`.
* **Justification:** We are encapsulating all database connection and querying logic into a single, dedicated module. This separates the concerns of "knowledge access" from the core cognitive processing of the DU, keeping the architecture clean and modular.

#### **Step 1: Setup New Dependencies**

* **Action:** First, you must add the necessary database clients and embedding libraries to our `pyproject.toml` configuration file.
* **Required Libraries:**
    * `pymilvus`: The official Python client for the Milvus vector database.
    * `neo4j`: The official Python driver for the Neo4j graph database.
    * `sentence-transformers`: A high-performance library for creating the vector embeddings needed for semantic search.

#### **Step 2: The `KnowledgeAugmenter` Class Implementation**

* **Action:** You will implement the main `KnowledgeAugmenter` class.
* **`__init__(self, milvus_config: dict, neo4j_config: dict, embedding_model_name: str)`:**
    * The initializer will take configuration dictionaries for connecting to the Milvus and Neo4j databases (e.g., host, port, credentials).
    * It will also take the name of a pre-trained `sentence-transformers` model (e.g., `'all-MiniLM-L6-v2'`).
    * Inside, it will initialize the database clients and load the embedding model.

* **The RAG Method: `retrieve_unstructured_context(self, query_text: str, top_k: int = 5) -> List[str]`:**
    * **Purpose:** To perform the fast semantic search over raw documents.
    * **Implementation:**
        1.  Take a string `query_text` as input.
        2.  Use the loaded `sentence-transformers` model to convert the `query_text` into a vector embedding.
        3.  Use the `milvus` client's `search()` method to find the `top_k` most similar document vectors in your pre-populated Milvus collection.
        4.  Return a list containing the raw text of the `top_k` retrieved document chunks.

* **The CAG Methods:**

    * **`update_knowledge_graph(self, structured_knowledge: Dict)`:**
        * **Purpose:** To take newly synthesized information and store it in the structured Neo4j knowledge graph.
        * **Input:** A dictionary representing a single fact, e.g., `{'subject': 'Kepler-90', 'relation': 'HAS_PLANET', 'object': 'Kepler-90i'}`.
        * **Implementation:** This method will use the `neo4j` driver to execute a **Cypher query**. The query will use the `MERGE` command to create the subject and object nodes if they don't exist, and then create the relationship between them. This prevents duplicate entries and efficiently builds the graph.

    * **`retrieve_structured_context(self, entity_name: str) -> Dict`:**
        * **Purpose:** To query the organized "second brain" for high-quality context.
        * **Input:** The name of an entity (e.g., "Kepler-90i").
        * **Implementation:** This method will execute a Cypher query to find the specified node in the graph and return all of its properties and direct relationships.
        * **Return:** A dictionary representing the local subgraph around the entity.

#### **Step 3: Integration with `DU_Core_V2`**

* **Action:** You will modify the `DU_Core_V2` class in `gif_framework/core/du_core_v2.py`.
* **Implementation:**
    1.  The `__init__` method will be updated to accept an optional `KnowledgeAugmenter` instance via Dependency Injection.
    2.  The main `forward` processing loop will be modified. You will add a condition where, if the network's internal state reflects low confidence or high surprise (e.g., high prediction error), it will trigger the `KnowledgeAugmenter`.
    3.  It will use its current internal state to formulate a `query_text`, call `retrieve_unstructured_context`, process the results, and then feed this new contextual information back into its own processing loop as an additional input, allowing it to refine its "understanding" in real-time.

---

**Summary of your task:**

1.  Update `pyproject.toml` with the new database and embedding libraries.
2.  Create the new file `gif_framework/core/knowledge_augmenter.py`.
3.  Implement the `KnowledgeAugmenter` class with methods for connecting to databases, performing semantic search in Milvus (RAG), and updating/querying the Neo4j knowledge graph (CAG).
4.  Modify the `DU_Core_V2` to accept and use an instance of the `KnowledgeAugmenter`, allowing it to ground its reasoning in external knowledge.
5.  Ensure all new code is professionally documented.

Now, following your protocol, please formulate your micro-plan for this task.

**Awaiting approval to proceed.**