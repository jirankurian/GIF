"""
Knowledge Augmentation Module - RAG/CAG Implementation
=====================================================

This module implements the Knowledge Augmentation Loop that connects the DU Core
to external knowledge sources through a hybrid Retrieval-Augmented Generation (RAG)
and Context-Augmented Generation (CAG) workflow.

The system provides two complementary knowledge access patterns:
1. **RAG (Retrieval-Augmented Generation)**: Fast semantic search over vast
   collections of unstructured documents using vector embeddings and Milvus
2. **CAG (Context-Augmented Generation)**: Structured knowledge retrieval from
   an organized knowledge graph stored in Neo4j

Key Features:
- **Semantic Search**: Vector-based similarity search using sentence transformers
- **Knowledge Graphs**: Structured entity-relationship storage and querying
- **Hybrid Workflow**: Seamless integration of unstructured and structured knowledge
- **Real-time Integration**: Dynamic knowledge retrieval during DU Core processing
- **Scalable Architecture**: Handle massive knowledge bases efficiently

Technical Architecture:
The KnowledgeAugmenter acts as a bridge between the DU Core's internal processing
and external knowledge sources. It monitors the DU Core's internal state for
uncertainty signals and dynamically retrieves relevant context to enhance
reasoning capabilities.

RAG Pipeline:
1. Query formulation from DU Core internal state
2. Embedding generation using sentence transformers
3. Vector similarity search in Milvus database
4. Context integration back into DU Core processing

CAG Pipeline:
1. Knowledge extraction from retrieved documents
2. Entity and relationship identification
3. Structured storage in Neo4j knowledge graph
4. Efficient querying for organized context

Author: GIF Development Team
Phase: 6.2 - Advanced Features Implementation
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import json
import hashlib

# Core dependencies with graceful fallback
try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is required for the Knowledge Augmenter. Please install it with: "
        "pip install torch>=2.0.0"
    )

# Knowledge augmentation dependencies with informative error messages
try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
    from pymilvus.exceptions import MilvusException
except ImportError:
    raise ImportError(
        "Milvus is required for vector storage. Please install it with: "
        "pip install pymilvus>=2.3.0"
    )

try:
    from neo4j import GraphDatabase, Driver
    from neo4j.exceptions import Neo4jError
except ImportError:
    raise ImportError(
        "Neo4j is required for graph storage. Please install it with: "
        "pip install neo4j>=5.0.0"
    )

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "Sentence Transformers is required for embeddings. Please install it with: "
        "pip install sentence-transformers>=2.2.0"
    )

# Web scraping dependencies with graceful fallback
try:
    import requests
    from bs4 import BeautifulSoup
    from duckduckgo_search import DDGS
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    requests = None
    BeautifulSoup = None
    DDGS = None


class KnowledgeAugmenter:
    """
    Knowledge Augmentation system implementing hybrid RAG/CAG workflow.
    
    This class provides the DU Core with the ability to access and integrate
    external knowledge from both unstructured documents (via Milvus vector search)
    and structured knowledge graphs (via Neo4j). The system enables real-time
    knowledge retrieval and integration during cognitive processing.
    
    The KnowledgeAugmenter implements two complementary knowledge access patterns:
    
    1. **RAG (Retrieval-Augmented Generation)**:
       - Semantic search over large document collections
       - Vector embeddings for similarity matching
       - Fast retrieval of relevant text chunks
       
    2. **CAG (Context-Augmented Generation)**:
       - Structured knowledge graph storage and querying
       - Entity-relationship modeling
       - Organized context retrieval
    
    Args:
        milvus_config (dict): Configuration for Milvus vector database connection
        neo4j_config (dict): Configuration for Neo4j graph database connection
        embedding_model_name (str): Name of sentence transformer model for embeddings
        logger (Optional[logging.Logger]): Custom logger instance
        
    Example:
        # Configure database connections
        milvus_config = {
            "host": "localhost",
            "port": 19530,
            "collection_name": "knowledge_base"
        }
        
        neo4j_config = {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password"
        }
        
        # Create knowledge augmenter
        augmenter = KnowledgeAugmenter(
            milvus_config=milvus_config,
            neo4j_config=neo4j_config,
            embedding_model_name="all-MiniLM-L6-v2"
        )
        
        # RAG: Retrieve unstructured context
        context = augmenter.retrieve_unstructured_context(
            query_text="exoplanet detection methods",
            top_k=5
        )
        
        # CAG: Store structured knowledge
        augmenter.update_knowledge_graph({
            "subject": "Kepler-90",
            "relation": "HAS_PLANET", 
            "object": "Kepler-90i"
        })
        
        # CAG: Retrieve structured context
        entity_context = augmenter.retrieve_structured_context("Kepler-90")
    """
    
    def __init__(
        self,
        milvus_config: Dict[str, Any],
        neo4j_config: Dict[str, Any],
        embedding_model_name: str = "all-MiniLM-L6-v2",
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Initialize the Knowledge Augmenter with database connections and embedding model."""
        
        # Setup logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Store configurations
        self.milvus_config = milvus_config.copy()
        self.neo4j_config = neo4j_config.copy()
        self.embedding_model_name = embedding_model_name
        
        # Initialize components
        self._milvus_collection = None
        self._neo4j_driver = None
        self._embedding_model = None
        
        # Connection status tracking
        self._milvus_connected = False
        self._neo4j_connected = False
        self._embedding_loaded = False
        self._web_scraping_available = WEB_SCRAPING_AVAILABLE
        
        # Initialize all components
        self._initialize_embedding_model()
        self._initialize_milvus_connection()
        self._initialize_neo4j_connection()
        
        self.logger.info("KnowledgeAugmenter initialized successfully")
        
    def _initialize_embedding_model(self) -> None:
        """Initialize the sentence transformer model for embeddings."""
        try:
            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            self._embedding_loaded = True
            self.logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Could not initialize embedding model: {e}")
            
    def _initialize_milvus_connection(self) -> None:
        """Initialize connection to Milvus vector database."""
        try:
            # Extract connection parameters
            host = self.milvus_config.get("host", "localhost")
            port = self.milvus_config.get("port", 19530)
            collection_name = self.milvus_config.get("collection_name", "knowledge_base")
            
            self.logger.info(f"Connecting to Milvus at {host}:{port}")
            
            # Establish connection
            connections.connect("default", host=host, port=port)
            
            # Initialize or connect to collection
            self._setup_milvus_collection(collection_name)
            
            self._milvus_connected = True
            self.logger.info("Milvus connection established successfully")
            
        except MilvusException as e:
            self.logger.warning(f"Milvus connection failed: {e}")
            self.logger.warning("RAG functionality will be disabled")
            self._milvus_connected = False
            
        except Exception as e:
            self.logger.warning(f"Unexpected error connecting to Milvus: {e}")
            self.logger.warning("RAG functionality will be disabled")
            self._milvus_connected = False

    def _setup_milvus_collection(self, collection_name: str) -> None:
        """Setup or connect to Milvus collection for vector storage."""
        try:
            # Check if collection exists
            from pymilvus import utility

            if utility.has_collection(collection_name):
                # Connect to existing collection
                self._milvus_collection = Collection(collection_name)
                self.logger.info(f"Connected to existing Milvus collection: {collection_name}")
            else:
                # Create new collection
                self._create_milvus_collection(collection_name)
                self.logger.info(f"Created new Milvus collection: {collection_name}")

        except Exception as e:
            self.logger.error(f"Failed to setup Milvus collection: {e}")
            raise

    def _create_milvus_collection(self, collection_name: str) -> None:
        """Create a new Milvus collection with appropriate schema."""
        # Define collection schema for document storage
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # MiniLM dimension
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000)
        ]

        schema = CollectionSchema(fields, "Knowledge base for RAG retrieval")
        self._milvus_collection = Collection(collection_name, schema)

        # Create index for vector search
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self._milvus_collection.create_index("embedding", index_params)

    def _initialize_neo4j_connection(self) -> None:
        """Initialize connection to Neo4j graph database."""
        try:
            # Extract connection parameters
            uri = self.neo4j_config.get("uri", "bolt://localhost:7687")
            user = self.neo4j_config.get("user", "neo4j")
            password = self.neo4j_config.get("password", "password")

            self.logger.info(f"Connecting to Neo4j at {uri}")

            # Establish connection
            self._neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))

            # Test connection
            with self._neo4j_driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    self._neo4j_connected = True
                    self.logger.info("Neo4j connection established successfully")

        except Neo4jError as e:
            self.logger.warning(f"Neo4j connection failed: {e}")
            self.logger.warning("CAG functionality will be disabled")
            self._neo4j_connected = False

        except Exception as e:
            self.logger.warning(f"Unexpected error connecting to Neo4j: {e}")
            self.logger.warning("CAG functionality will be disabled")
            self._neo4j_connected = False

    def retrieve_unstructured_context(
        self,
        query_text: str,
        top_k: int = 5
    ) -> List[str]:
        """
        Retrieve relevant unstructured context using RAG (Retrieval-Augmented Generation).

        This method implements the RAG pipeline by converting the query text into
        vector embeddings and performing semantic similarity search against the
        Milvus vector database to find the most relevant document chunks. If no
        local results are found or if the database is unavailable, it automatically
        falls back to web-based retrieval for real-time information.

        The RAG process follows these steps:
        1. Convert query text to vector embedding using sentence transformer
        2. Perform vector similarity search in Milvus collection
        3. Retrieve top-k most similar document chunks
        4. If no results found, fallback to web search and content extraction
        5. Return raw text content for integration into DU Core processing

        Args:
            query_text (str): The query text to search for relevant context
            top_k (int): Number of top similar documents to retrieve (default: 5)

        Returns:
            List[str]: List of relevant document text chunks

        Raises:
            RuntimeError: If Milvus connection is not available
            ValueError: If query_text is empty or top_k is invalid

        Example:
            # Retrieve context for exoplanet research
            context = augmenter.retrieve_unstructured_context(
                query_text="transit photometry exoplanet detection",
                top_k=3
            )

            # Context will contain relevant document chunks like:
            # ["Transit photometry is a method of detecting exoplanets...",
            #  "The Kepler mission used transit photometry to discover...",
            #  "Light curve analysis reveals planetary characteristics..."]
        """
        # Validate inputs
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # Check if RAG is available
        if not self._milvus_connected:
            self.logger.warning("Milvus not connected, returning empty context")
            return []

        if not self._embedding_loaded:
            self.logger.warning("Embedding model not loaded, returning empty context")
            return []

        try:
            # Step 1: Generate query embedding
            self.logger.debug(f"Generating embedding for query: {query_text[:100]}...")
            query_embedding = self._embedding_model.encode([query_text])[0].tolist()

            # Step 2: Perform vector similarity search
            self.logger.debug(f"Searching for top-{top_k} similar documents")
            self._milvus_collection.load()  # Ensure collection is loaded

            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = self._milvus_collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "source"]
            )

            # Step 3: Extract text content from results
            retrieved_texts = []
            for hit in results[0]:
                text_content = hit.entity.get("text", "")
                if text_content:
                    retrieved_texts.append(text_content)

            self.logger.info(f"Retrieved {len(retrieved_texts)} relevant documents from vector database")

            # If we have good results from vector database, return them
            if retrieved_texts:
                return retrieved_texts

            # Fallback to web search if no local results found
            self.logger.info("No local results found, falling back to web search")
            return self.retrieve_web_context(query_text, top_k)

        except Exception as e:
            self.logger.error(f"Error during RAG retrieval: {e}")

            # Fallback to web search on error
            if self._web_scraping_available:
                self.logger.info("Falling back to web search due to database error")
                return self.retrieve_web_context(query_text, top_k)

            return []

    def update_knowledge_graph(self, structured_knowledge: Dict[str, Any]) -> None:
        """
        Update the knowledge graph with structured information using CAG.

        This method implements the CAG (Context-Augmented Generation) storage pipeline
        by taking structured knowledge in the form of entity-relationship triples
        and storing them in the Neo4j graph database using efficient MERGE operations.

        The CAG storage process:
        1. Validate the structured knowledge format
        2. Extract subject, relation, and object components
        3. Use Cypher MERGE operations to create/update nodes and relationships
        4. Handle duplicate prevention and graph consistency

        Args:
            structured_knowledge (Dict[str, Any]): Knowledge triple with keys:
                - "subject": The subject entity (string)
                - "relation": The relationship type (string)
                - "object": The object entity (string)
                - "properties": Optional additional properties (dict)

        Raises:
            RuntimeError: If Neo4j connection is not available
            ValueError: If structured_knowledge format is invalid

        Example:
            # Store astronomical knowledge
            augmenter.update_knowledge_graph({
                "subject": "Kepler-90",
                "relation": "HAS_PLANET",
                "object": "Kepler-90i",
                "properties": {
                    "discovery_year": 2017,
                    "orbital_period": 14.4,
                    "planet_type": "super-Earth"
                }
            })

            # Store medical knowledge
            augmenter.update_knowledge_graph({
                "subject": "Atrial Fibrillation",
                "relation": "CAUSES",
                "object": "Irregular Heart Rhythm",
                "properties": {
                    "severity": "high",
                    "treatment": "anticoagulation"
                }
            })
        """
        # Validate inputs
        if not isinstance(structured_knowledge, dict):
            raise ValueError("structured_knowledge must be a dictionary")

        required_keys = ["subject", "relation", "object"]
        for key in required_keys:
            if key not in structured_knowledge:
                raise ValueError(f"structured_knowledge must contain '{key}' key")
            if not structured_knowledge[key]:
                raise ValueError(f"'{key}' cannot be empty")

        # Check if CAG is available
        if not self._neo4j_connected:
            self.logger.warning("Neo4j not connected, skipping knowledge graph update")
            return

        try:
            subject = structured_knowledge["subject"]
            relation = structured_knowledge["relation"]
            obj = structured_knowledge["object"]
            properties = structured_knowledge.get("properties", {})

            # Build Cypher query with MERGE operations for consistency
            cypher_query = """
            MERGE (s:Entity {name: $subject})
            MERGE (o:Entity {name: $object})
            MERGE (s)-[r:%s]->(o)
            SET r += $properties
            RETURN s, r, o
            """ % relation

            # Execute the query
            with self._neo4j_driver.session() as session:
                result = session.run(
                    cypher_query,
                    subject=subject,
                    object=obj,
                    properties=properties
                )

                # Verify the operation
                record = result.single()
                if record:
                    self.logger.debug(f"Updated knowledge graph: {subject} -{relation}-> {obj}")
                else:
                    self.logger.warning("Knowledge graph update returned no results")

        except Neo4jError as e:
            self.logger.error(f"Neo4j error during knowledge graph update: {e}")

        except Exception as e:
            self.logger.error(f"Error updating knowledge graph: {e}")

    def retrieve_structured_context(self, entity_name: str) -> Dict[str, Any]:
        """
        Retrieve structured context from the knowledge graph using CAG.

        This method implements the CAG (Context-Augmented Generation) retrieval pipeline
        by querying the Neo4j graph database for all relationships and properties
        associated with a specific entity, returning organized contextual information.

        The CAG retrieval process:
        1. Query Neo4j for the specified entity node
        2. Retrieve all incoming and outgoing relationships
        3. Collect entity properties and relationship details
        4. Return structured context dictionary

        Args:
            entity_name (str): Name of the entity to retrieve context for

        Returns:
            Dict[str, Any]: Structured context containing:
                - "entity": The entity name
                - "properties": Entity properties (if any)
                - "outgoing_relations": List of outgoing relationships
                - "incoming_relations": List of incoming relationships
                - "related_entities": Set of connected entity names

        Raises:
            RuntimeError: If Neo4j connection is not available
            ValueError: If entity_name is empty

        Example:
            # Retrieve context for astronomical entity
            context = augmenter.retrieve_structured_context("Kepler-90")

            # Returns structured information like:
            # {
            #     "entity": "Kepler-90",
            #     "properties": {"star_type": "G-type", "distance": "2545 ly"},
            #     "outgoing_relations": [
            #         {"relation": "HAS_PLANET", "target": "Kepler-90i", "properties": {...}},
            #         {"relation": "HAS_PLANET", "target": "Kepler-90b", "properties": {...}}
            #     ],
            #     "incoming_relations": [],
            #     "related_entities": ["Kepler-90i", "Kepler-90b"]
            # }
        """
        # Validate inputs
        if not entity_name or not entity_name.strip():
            raise ValueError("entity_name cannot be empty")

        # Check if CAG is available
        if not self._neo4j_connected:
            self.logger.warning("Neo4j not connected, returning empty context")
            return {
                "entity": entity_name,
                "properties": {},
                "outgoing_relations": [],
                "incoming_relations": [],
                "related_entities": set()
            }

        try:
            # Query for entity and all its relationships
            cypher_query = """
            MATCH (e:Entity {name: $entity_name})
            OPTIONAL MATCH (e)-[r_out]->(target)
            OPTIONAL MATCH (source)-[r_in]->(e)
            RETURN e,
                   collect(DISTINCT {relation: type(r_out), target: target.name, properties: properties(r_out)}) as outgoing,
                   collect(DISTINCT {relation: type(r_in), source: source.name, properties: properties(r_in)}) as incoming
            """

            with self._neo4j_driver.session() as session:
                result = session.run(cypher_query, entity_name=entity_name)
                record = result.single()

                if not record or not record["e"]:
                    self.logger.debug(f"Entity '{entity_name}' not found in knowledge graph")
                    return {
                        "entity": entity_name,
                        "properties": {},
                        "outgoing_relations": [],
                        "incoming_relations": [],
                        "related_entities": set()
                    }

                # Extract entity properties
                entity_node = record["e"]
                entity_properties = dict(entity_node)
                entity_properties.pop("name", None)  # Remove the name property

                # Process outgoing relationships
                outgoing_relations = []
                related_entities = set()

                for rel in record["outgoing"]:
                    if rel["relation"] and rel["target"]:  # Filter out null relationships
                        outgoing_relations.append({
                            "relation": rel["relation"],
                            "target": rel["target"],
                            "properties": rel["properties"] or {}
                        })
                        related_entities.add(rel["target"])

                # Process incoming relationships
                incoming_relations = []

                for rel in record["incoming"]:
                    if rel["relation"] and rel["source"]:  # Filter out null relationships
                        incoming_relations.append({
                            "relation": rel["relation"],
                            "source": rel["source"],
                            "properties": rel["properties"] or {}
                        })
                        related_entities.add(rel["source"])

                context = {
                    "entity": entity_name,
                    "properties": entity_properties,
                    "outgoing_relations": outgoing_relations,
                    "incoming_relations": incoming_relations,
                    "related_entities": related_entities
                }

                self.logger.debug(f"Retrieved context for '{entity_name}': "
                                f"{len(outgoing_relations)} outgoing, "
                                f"{len(incoming_relations)} incoming relations")

                return context

        except Neo4jError as e:
            self.logger.error(f"Neo4j error during context retrieval: {e}")
            return {
                "entity": entity_name,
                "properties": {},
                "outgoing_relations": [],
                "incoming_relations": [],
                "related_entities": set()
            }

        except Exception as e:
            self.logger.error(f"Error retrieving structured context: {e}")
            return {
                "entity": entity_name,
                "properties": {},
                "outgoing_relations": [],
                "incoming_relations": [],
                "related_entities": set()
            }

    def add_documents_to_vector_store(
        self,
        documents: List[Dict[str, Any]]
    ) -> bool:
        """
        Add documents to the Milvus vector store for RAG retrieval.

        This utility method allows populating the vector database with new documents
        for semantic search. Each document is embedded using the sentence transformer
        and stored with metadata for future retrieval.

        Args:
            documents (List[Dict[str, Any]]): List of documents, each containing:
                - "text": The document text content (required)
                - "source": Source identifier (optional)
                - "metadata": Additional metadata (optional)

        Returns:
            bool: True if documents were successfully added, False otherwise

        Example:
            documents = [
                {
                    "text": "Exoplanets are planets outside our solar system...",
                    "source": "astronomy_textbook_ch3",
                    "metadata": {"topic": "exoplanets", "difficulty": "beginner"}
                },
                {
                    "text": "Transit photometry measures the dimming of starlight...",
                    "source": "kepler_mission_paper",
                    "metadata": {"topic": "detection_methods", "year": 2010}
                }
            ]

            success = augmenter.add_documents_to_vector_store(documents)
        """
        if not self._milvus_connected or not self._embedding_loaded:
            self.logger.warning("Cannot add documents: Milvus or embedding model not available")
            return False

        try:
            # Prepare data for insertion
            texts = []
            embeddings = []
            sources = []
            metadata_list = []

            for doc in documents:
                if "text" not in doc or not doc["text"]:
                    self.logger.warning("Skipping document without text content")
                    continue

                text = doc["text"]
                source = doc.get("source", "unknown")
                metadata = json.dumps(doc.get("metadata", {}))

                # Generate embedding
                embedding = self._embedding_model.encode([text])[0].tolist()

                texts.append(text)
                embeddings.append(embedding)
                sources.append(source)
                metadata_list.append(metadata)

            if not texts:
                self.logger.warning("No valid documents to insert")
                return False

            # Insert into Milvus
            entities = [texts, embeddings, sources, metadata_list]
            self._milvus_collection.insert(entities)
            self._milvus_collection.flush()

            self.logger.info(f"Successfully added {len(texts)} documents to vector store")
            return True

        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {e}")
            return False

    def retrieve_context(self, query: str) -> str:
        """
        Primary method to get context, prioritizing the web.

        This method implements the enhanced RAG pipeline that prioritizes live web data
        over local database content. It first attempts to retrieve current information
        from the web, and only falls back to the local vector database if web retrieval
        fails or returns insufficient results.

        This prioritization ensures that the DU Core has access to the most current
        information available, which is crucial for tasks requiring up-to-date knowledge.

        Args:
            query (str): Search query for context retrieval

        Returns:
            str: Retrieved context text, prioritizing web sources

        Example:
            # Get current context with web priority
            context = augmenter.retrieve_context("latest exoplanet discoveries")
            # Will first search web, then fallback to local database if needed
        """
        try:
            # Step 1: Try web retrieval first (prioritized)
            web_context = self.retrieve_web_context(query)

            # Check if web retrieval was successful
            if web_context and len(web_context) > 0:
                # Filter out failed retrievals
                valid_contexts = [ctx for ctx in web_context
                                if ctx and "Failed to retrieve" not in ctx and "No web results" not in ctx]

                if valid_contexts:
                    self.logger.info(f"Retrieved {len(valid_contexts)} web contexts for query: {query[:50]}...")
                    return " ".join(valid_contexts)

            # Step 2: Fallback to existing database RAG
            self.logger.info("Web retrieval unsuccessful, falling back to database RAG")
            database_contexts = self.retrieve_unstructured_context(query)

            if database_contexts:
                self.logger.info(f"Retrieved {len(database_contexts)} database contexts for query: {query[:50]}...")
                return " ".join(database_contexts)

            # Step 3: No context available
            self.logger.warning(f"No context available for query: {query[:50]}...")
            return "No relevant context found."

        except Exception as e:
            self.logger.error(f"Error during context retrieval: {e}")
            return "Context retrieval failed."

    def retrieve_web_context(
        self,
        query: str,
        max_results: int = 3,
        timeout: int = 10
    ) -> List[str]:
        """
        Retrieve relevant context from the web using real-time web scraping.

        This method implements web-based RAG by searching the internet for relevant
        information and extracting text content from web pages. It serves as a
        fallback or supplement to the local vector database when external knowledge
        is needed.

        The web RAG process follows these steps:
        1. Perform web search using search engines or APIs
        2. Extract URLs from search results
        3. Scrape and extract text content from relevant web pages
        4. Return cleaned text content for integration into DU Core processing

        Args:
            query (str): Search query for web retrieval
            max_results (int): Maximum number of web pages to retrieve
            timeout (int): Timeout in seconds for web requests

        Returns:
            List[str]: List of extracted text content from web pages

        Example:
            # Retrieve real-time context from the web
            context = augmenter.retrieve_web_context(
                query="latest exoplanet discoveries 2024",
                max_results=3
            )

            # Context will contain current web content like:
            # ["Recent discoveries by JWST have revealed...",
            #  "The TESS mission continues to find new exoplanets...",
            #  "Breakthrough in atmospheric analysis techniques..."]
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        if max_results <= 0:
            raise ValueError("max_results must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        # Check if web scraping is available
        if not self._web_scraping_available:
            self.logger.warning("Web scraping dependencies not available, returning empty context")
            return []

        try:
            self.logger.info(f"Retrieving web context for query: {query[:100]}...")

            # Step 1: Search the web for relevant URLs
            search_results = self._search_web(query, max_results)

            if not search_results:
                self.logger.warning("No web search results found")
                return []

            # Step 2: Extract content from web pages
            web_contexts = []
            for result in search_results[:max_results]:
                url = result.get('url', '')
                if url:
                    content = self._extract_text_from_url(url, timeout)
                    if content:
                        web_contexts.append(content)

            self.logger.info(f"Retrieved {len(web_contexts)} web contexts")
            return web_contexts

        except Exception as e:
            self.logger.error(f"Error during web context retrieval: {e}")
            return []

    def _search_web(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Search the web for relevant URLs using DuckDuckGo search.

        This method performs a web search to find relevant URLs for the given query.
        It uses the duckduckgo-search library as a privacy-friendly search engine
        that doesn't require API keys or have rate limiting issues.

        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return

        Returns:
            List[Dict[str, str]]: List of search results with 'title' and 'url' keys
        """
        try:
            # Use DuckDuckGo search library (no API key required)
            if DDGS is None:
                self.logger.warning("DuckDuckGo search not available, falling back to empty results")
                return []

            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=max_results))

                results = []
                for result in search_results:
                    if 'href' in result and 'title' in result:
                        results.append({
                            'title': result['title'],
                            'url': result['href']
                        })

            self.logger.debug(f"Found {len(results)} search results for query: {query}")
            return results

        except Exception as e:
            self.logger.error(f"Error during web search: {e}")
            return []

    def _extract_text_from_url(self, url: str, timeout: int = 10) -> str:
        """
        Extract clean text content from a web page URL.

        This method fetches the content of a web page and extracts the main
        text content while filtering out navigation, ads, and other non-content
        elements.

        Args:
            url (str): URL to extract content from
            timeout (int): Request timeout in seconds

        Returns:
            str: Cleaned text content from the web page
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()

            # Extract text from main content areas
            content_selectors = [
                'main', 'article', '.content', '.main-content',
                '.post-content', '.entry-content', '#content'
            ]

            text_content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    text_content = content_elem.get_text(separator=' ', strip=True)
                    break

            # Fallback to body if no main content found
            if not text_content:
                body = soup.find('body')
                if body:
                    text_content = body.get_text(separator=' ', strip=True)

            # Clean up the text
            lines = text_content.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and len(line) > 20:  # Filter out very short lines
                    cleaned_lines.append(line)

            # Limit content length to avoid overwhelming the system
            cleaned_text = ' '.join(cleaned_lines)
            if len(cleaned_text) > 2000:
                cleaned_text = cleaned_text[:2000] + "..."

            self.logger.debug(f"Extracted {len(cleaned_text)} characters from {url}")
            return cleaned_text

        except Exception as e:
            self.logger.error(f"Error extracting text from {url}: {e}")
            return ""

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Knowledge Augmenter.

        Returns:
            Dict[str, Any]: Status information including connection states and capabilities
        """
        return {
            "milvus_connected": self._milvus_connected,
            "neo4j_connected": self._neo4j_connected,
            "embedding_loaded": self._embedding_loaded,
            "web_scraping_available": self._web_scraping_available,
            "rag_available": self._milvus_connected and self._embedding_loaded,
            "cag_available": self._neo4j_connected,
            "web_rag_available": self._web_scraping_available,
            "embedding_model": self.embedding_model_name,
            "knowledge_augmentation_ready": (
                self._milvus_connected and
                self._neo4j_connected and
                self._embedding_loaded
            ),
            "full_capabilities": (
                self._milvus_connected and
                self._neo4j_connected and
                self._embedding_loaded and
                self._web_scraping_available
            ),
            "milvus_config": {k: v for k, v in self.milvus_config.items() if k != "password"},
            "neo4j_config": {k: v for k, v in self.neo4j_config.items() if k != "password"}
        }

    def close(self) -> None:
        """Close database connections and clean up resources."""
        try:
            if self._neo4j_driver:
                self._neo4j_driver.close()
                self.logger.info("Neo4j connection closed")

            if self._milvus_connected:
                connections.disconnect("default")
                self.logger.info("Milvus connection closed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()

    def __repr__(self) -> str:
        """String representation of the Knowledge Augmenter."""
        status = self.get_status()
        return (
            f"KnowledgeAugmenter(\n"
            f"  embedding_model: {self.embedding_model_name}\n"
            f"  rag_available: {status['rag_available']}\n"
            f"  cag_available: {status['cag_available']}\n"
            f"  web_rag_available: {status['web_rag_available']}\n"
            f"  milvus_connected: {status['milvus_connected']}\n"
            f"  neo4j_connected: {status['neo4j_connected']}\n"
            f"  full_capabilities: {status['full_capabilities']}\n"
            f")"
        )
