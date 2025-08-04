"""
Unit Tests for KnowledgeAugmenter - RAG/CAG Implementation
=========================================================

This module contains comprehensive unit tests for the KnowledgeAugmenter class,
which implements the hybrid Retrieval-Augmented Generation (RAG) and 
Context-Augmented Generation (CAG) workflow for external knowledge integration.

Test Categories:
- KnowledgeAugmenter initialization and configuration
- RAG functionality (vector database retrieval)
- CAG functionality (knowledge graph operations)
- Web scraping capabilities
- Integration with DU_Core_V2
- Error handling and edge cases
- Mock database operations

The tests use mocked database connections to avoid requiring actual Milvus/Neo4j
instances during testing, ensuring reliable and fast test execution.
"""

import pytest
import torch
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import the KnowledgeAugmenter class to test
try:
    from gif_framework.core.knowledge_augmenter import KnowledgeAugmenter
    KNOWLEDGE_AUGMENTER_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AUGMENTER_AVAILABLE = False

# Import DU_Core_V2 for integration testing
try:
    from gif_framework.core.du_core_v2 import DU_Core_V2
    DU_CORE_V2_AVAILABLE = True
except ImportError:
    DU_CORE_V2_AVAILABLE = False


class TestKnowledgeAugmenterInitialization:
    """Test KnowledgeAugmenter initialization and configuration."""
    
    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    @patch('gif_framework.core.knowledge_augmenter.SentenceTransformer')
    @patch('gif_framework.core.knowledge_augmenter.connections')
    @patch('gif_framework.core.knowledge_augmenter.GraphDatabase')
    def test_valid_initialization(self, mock_neo4j, mock_milvus, mock_transformer):
        """Test basic valid initialization of KnowledgeAugmenter."""
        # Mock successful initialization
        mock_transformer.return_value = Mock()
        mock_milvus.connect.return_value = None
        mock_neo4j.driver.return_value = Mock()
        
        milvus_config = {
            "host": "localhost",
            "port": 19530,
            "collection_name": "test_collection"
        }
        
        neo4j_config = {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password"
        }
        
        augmenter = KnowledgeAugmenter(
            milvus_config=milvus_config,
            neo4j_config=neo4j_config,
            embedding_model_name="all-MiniLM-L6-v2"
        )
        
        assert augmenter.embedding_model_name == "all-MiniLM-L6-v2"
        assert augmenter.milvus_config["host"] == "localhost"
        assert augmenter.neo4j_config["uri"] == "bolt://localhost:7687"
    
    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_invalid_milvus_config(self):
        """Test initialization with invalid Milvus configuration."""
        with pytest.raises((ValueError, TypeError)):
            KnowledgeAugmenter(
                milvus_config=None,  # Invalid config
                neo4j_config={"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"},
                embedding_model_name="all-MiniLM-L6-v2"
            )
    
    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_invalid_neo4j_config(self):
        """Test initialization with invalid Neo4j configuration."""
        with pytest.raises((ValueError, TypeError)):
            KnowledgeAugmenter(
                milvus_config={"host": "localhost", "port": 19530, "collection_name": "test"},
                neo4j_config=None,  # Invalid config
                embedding_model_name="all-MiniLM-L6-v2"
            )


class TestRAGFunctionality:
    """Test RAG (Retrieval-Augmented Generation) functionality."""
    
    @pytest.fixture
    def mock_augmenter(self):
        """Create a mocked KnowledgeAugmenter for testing."""
        with patch('gif_framework.core.knowledge_augmenter.SentenceTransformer'), \
             patch('gif_framework.core.knowledge_augmenter.connections'), \
             patch('gif_framework.core.knowledge_augmenter.GraphDatabase'):
            
            milvus_config = {"host": "localhost", "port": 19530, "collection_name": "test"}
            neo4j_config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"}
            
            augmenter = KnowledgeAugmenter(
                milvus_config=milvus_config,
                neo4j_config=neo4j_config,
                embedding_model_name="all-MiniLM-L6-v2"
            )
            
            # Mock successful connections
            augmenter._milvus_connected = True
            augmenter._neo4j_connected = True
            augmenter._embedding_loaded = True
            augmenter._web_scraping_available = True
            
            return augmenter
    
    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_retrieve_unstructured_context_success(self, mock_augmenter):
        """Test successful retrieval of unstructured context."""
        # Mock Milvus search results
        mock_hit = Mock()
        mock_hit.entity.get.return_value = "Sample document text about exoplanets"
        
        mock_results = [[mock_hit]]
        
        with patch.object(mock_augmenter, '_milvus_collection') as mock_collection:
            mock_collection.search.return_value = mock_results
            mock_collection.load.return_value = None
            
            with patch.object(mock_augmenter, '_embedding_model') as mock_model:
                mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
                
                result = mock_augmenter.retrieve_unstructured_context(
                    query_text="exoplanet detection methods",
                    top_k=3
                )
                
                assert len(result) == 1
                assert "exoplanets" in result[0]
    
    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_retrieve_unstructured_context_empty_query(self, mock_augmenter):
        """Test retrieval with empty query text."""
        with pytest.raises(ValueError, match="query_text cannot be empty"):
            mock_augmenter.retrieve_unstructured_context(
                query_text="",
                top_k=3
            )
    
    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_retrieve_unstructured_context_invalid_top_k(self, mock_augmenter):
        """Test retrieval with invalid top_k parameter."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            mock_augmenter.retrieve_unstructured_context(
                query_text="test query",
                top_k=0
            )


class TestCAGFunctionality:
    """Test CAG (Context-Augmented Generation) functionality."""
    
    @pytest.fixture
    def mock_augmenter(self):
        """Create a mocked KnowledgeAugmenter for testing."""
        with patch('gif_framework.core.knowledge_augmenter.SentenceTransformer'), \
             patch('gif_framework.core.knowledge_augmenter.connections'), \
             patch('gif_framework.core.knowledge_augmenter.GraphDatabase'):
            
            milvus_config = {"host": "localhost", "port": 19530, "collection_name": "test"}
            neo4j_config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"}
            
            augmenter = KnowledgeAugmenter(
                milvus_config=milvus_config,
                neo4j_config=neo4j_config,
                embedding_model_name="all-MiniLM-L6-v2"
            )
            
            # Mock successful connections
            augmenter._milvus_connected = True
            augmenter._neo4j_connected = True
            augmenter._embedding_loaded = True
            augmenter._web_scraping_available = True
            
            return augmenter
    
    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_update_knowledge_graph_success(self, mock_augmenter):
        """Test successful knowledge graph update."""
        structured_knowledge = {
            "subject": "Kepler-90",
            "relation": "HAS_PLANET",
            "object": "Kepler-90i",
            "properties": {"discovery_year": 2017}
        }
        
        mock_session = Mock()
        mock_result = Mock()
        mock_result.single.return_value = {"created": True}
        mock_session.run.return_value = mock_result
        
        with patch.object(mock_augmenter, '_neo4j_driver') as mock_driver:
            mock_driver.session.return_value.__enter__.return_value = mock_session
            
            # Should not raise any exception
            mock_augmenter.update_knowledge_graph(structured_knowledge)
            
            # Verify session.run was called
            mock_session.run.assert_called_once()
    
    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_update_knowledge_graph_invalid_structure(self, mock_augmenter):
        """Test knowledge graph update with invalid structure."""
        invalid_knowledge = {
            "subject": "Kepler-90",
            # Missing required "relation" and "object" keys
        }
        
        with pytest.raises(ValueError, match="must contain"):
            mock_augmenter.update_knowledge_graph(invalid_knowledge)
    
    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_retrieve_structured_context_success(self, mock_augmenter):
        """Test successful structured context retrieval."""
        mock_record = {
            "e": {"name": "Kepler-90", "star_type": "G-type"},
            "outgoing": [
                {"relation": "HAS_PLANET", "target": "Kepler-90i", "properties": {"orbital_period": 14.4}}
            ],
            "incoming": []
        }
        
        mock_session = Mock()
        mock_result = Mock()
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        
        with patch.object(mock_augmenter, '_neo4j_driver') as mock_driver:
            mock_driver.session.return_value.__enter__.return_value = mock_session
            
            result = mock_augmenter.retrieve_structured_context("Kepler-90")
            
            assert result["entity"] == "Kepler-90"
            assert "star_type" in result["properties"]
            assert len(result["outgoing_relations"]) == 1
            assert result["outgoing_relations"][0]["relation"] == "HAS_PLANET"
    
    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_retrieve_structured_context_empty_entity(self, mock_augmenter):
        """Test structured context retrieval with empty entity name."""
        with pytest.raises(ValueError, match="entity_name cannot be empty"):
            mock_augmenter.retrieve_structured_context("")


class TestWebScrapingFunctionality:
    """Test web scraping capabilities."""
    
    @pytest.fixture
    def mock_augmenter(self):
        """Create a mocked KnowledgeAugmenter for testing."""
        with patch('gif_framework.core.knowledge_augmenter.SentenceTransformer'), \
             patch('gif_framework.core.knowledge_augmenter.connections'), \
             patch('gif_framework.core.knowledge_augmenter.GraphDatabase'):
            
            milvus_config = {"host": "localhost", "port": 19530, "collection_name": "test"}
            neo4j_config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"}
            
            augmenter = KnowledgeAugmenter(
                milvus_config=milvus_config,
                neo4j_config=neo4j_config,
                embedding_model_name="all-MiniLM-L6-v2"
            )
            
            # Mock successful connections
            augmenter._milvus_connected = True
            augmenter._neo4j_connected = True
            augmenter._embedding_loaded = True
            augmenter._web_scraping_available = True
            
            return augmenter

    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    @patch('gif_framework.core.knowledge_augmenter.requests')
    @patch('gif_framework.core.knowledge_augmenter.BeautifulSoup')
    def test_retrieve_web_context_success(self, mock_bs4, mock_requests, mock_augmenter):
        """Test successful web context retrieval."""
        # Mock search results
        mock_search_response = Mock()
        mock_search_response.content = b'<div class="result"><a class="result__a" href="https://example.com">Test Result</a></div>'
        mock_search_response.raise_for_status.return_value = None

        # Mock content extraction
        mock_content_response = Mock()
        mock_content_response.content = b'<main>This is test content about exoplanets and their detection methods.</main>'
        mock_content_response.raise_for_status.return_value = None

        mock_requests.get.side_effect = [mock_search_response, mock_content_response]

        # Mock BeautifulSoup parsing
        mock_search_soup = Mock()
        mock_result_div = Mock()
        mock_link = Mock()
        mock_link.get_text.return_value = "Test Result"
        mock_link.get.return_value = "https://example.com"
        mock_result_div.find.return_value = mock_link
        mock_search_soup.find_all.return_value = [mock_result_div]

        mock_content_soup = Mock()
        mock_main = Mock()
        mock_main.get_text.return_value = "This is test content about exoplanets and their detection methods."
        mock_content_soup.select_one.return_value = mock_main
        mock_content_soup.return_value = [Mock(), Mock()]  # For script removal

        mock_bs4.side_effect = [mock_search_soup, mock_content_soup]

        result = mock_augmenter.retrieve_web_context(
            query="exoplanet detection methods",
            max_results=1
        )

        assert len(result) == 1
        assert "exoplanets" in result[0]

    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_retrieve_web_context_empty_query(self, mock_augmenter):
        """Test web context retrieval with empty query."""
        with pytest.raises(ValueError, match="query cannot be empty"):
            mock_augmenter.retrieve_web_context(
                query="",
                max_results=3
            )

    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_retrieve_web_context_invalid_max_results(self, mock_augmenter):
        """Test web context retrieval with invalid max_results."""
        with pytest.raises(ValueError, match="max_results must be positive"):
            mock_augmenter.retrieve_web_context(
                query="test query",
                max_results=0
            )

    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_retrieve_web_context_unavailable(self, mock_augmenter):
        """Test web context retrieval when web scraping is unavailable."""
        mock_augmenter._web_scraping_available = False

        result = mock_augmenter.retrieve_web_context(
            query="test query",
            max_results=3
        )

        assert result == []


class TestIntegrationWithDUCoreV2:
    """Test integration between KnowledgeAugmenter and DU_Core_V2."""

    @pytest.mark.skipif(not (KNOWLEDGE_AUGMENTER_AVAILABLE and DU_CORE_V2_AVAILABLE),
                       reason="KnowledgeAugmenter or DU_Core_V2 not available")
    @patch('gif_framework.core.knowledge_augmenter.SentenceTransformer')
    @patch('gif_framework.core.knowledge_augmenter.connections')
    @patch('gif_framework.core.knowledge_augmenter.GraphDatabase')
    def test_du_core_v2_with_knowledge_augmenter(self, mock_neo4j, mock_milvus, mock_transformer):
        """Test DU_Core_V2 initialization with KnowledgeAugmenter."""
        # Mock successful initialization
        mock_transformer.return_value = Mock()
        mock_milvus.connect.return_value = None
        mock_neo4j.driver.return_value = Mock()

        milvus_config = {"host": "localhost", "port": 19530, "collection_name": "test"}
        neo4j_config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"}

        augmenter = KnowledgeAugmenter(
            milvus_config=milvus_config,
            neo4j_config=neo4j_config,
            embedding_model_name="all-MiniLM-L6-v2"
        )

        du_core_v2 = DU_Core_V2(
            input_size=100,
            hidden_sizes=[64, 32],
            output_size=10,
            knowledge_augmenter=augmenter
        )

        assert du_core_v2._knowledge_augmenter is augmenter

    @pytest.mark.skipif(not (KNOWLEDGE_AUGMENTER_AVAILABLE and DU_CORE_V2_AVAILABLE),
                       reason="KnowledgeAugmenter or DU_Core_V2 not available")
    def test_uncertainty_detection_triggers_knowledge_retrieval(self):
        """Test that uncertainty detection triggers knowledge retrieval."""
        # This would require more complex mocking of the forward pass
        # For now, we'll test the uncertainty detection method directly

        with patch('gif_framework.core.knowledge_augmenter.SentenceTransformer'), \
             patch('gif_framework.core.knowledge_augmenter.connections'), \
             patch('gif_framework.core.knowledge_augmenter.GraphDatabase'):

            milvus_config = {"host": "localhost", "port": 19530, "collection_name": "test"}
            neo4j_config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"}

            augmenter = KnowledgeAugmenter(
                milvus_config=milvus_config,
                neo4j_config=neo4j_config,
                embedding_model_name="all-MiniLM-L6-v2"
            )

            du_core_v2 = DU_Core_V2(
                input_size=10,
                hidden_sizes=[8],
                output_size=5,
                knowledge_augmenter=augmenter
            )

            # Test uncertainty detection with mock hidden states
            hidden_states = [torch.rand(2, 8)]  # batch_size=2, hidden_size=8

            should_augment, uncertainty_score = du_core_v2._detect_uncertainty(hidden_states)

            assert isinstance(should_augment, bool)
            assert isinstance(uncertainty_score, float)
            assert 0.0 <= uncertainty_score <= 1.0


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    @pytest.fixture
    def mock_augmenter(self):
        """Create a mocked KnowledgeAugmenter for testing."""
        with patch('gif_framework.core.knowledge_augmenter.SentenceTransformer'), \
             patch('gif_framework.core.knowledge_augmenter.connections'), \
             patch('gif_framework.core.knowledge_augmenter.GraphDatabase'):

            milvus_config = {"host": "localhost", "port": 19530, "collection_name": "test"}
            neo4j_config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"}

            augmenter = KnowledgeAugmenter(
                milvus_config=milvus_config,
                neo4j_config=neo4j_config,
                embedding_model_name="all-MiniLM-L6-v2"
            )

            return augmenter

    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_rag_with_disconnected_milvus(self, mock_augmenter):
        """Test RAG functionality when Milvus is disconnected."""
        mock_augmenter._milvus_connected = False
        mock_augmenter._web_scraping_available = False

        result = mock_augmenter.retrieve_unstructured_context(
            query_text="test query",
            top_k=3
        )

        assert result == []

    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_cag_with_disconnected_neo4j(self, mock_augmenter):
        """Test CAG functionality when Neo4j is disconnected."""
        mock_augmenter._neo4j_connected = False

        # Update should not raise exception
        mock_augmenter.update_knowledge_graph({
            "subject": "test",
            "relation": "TEST_RELATION",
            "object": "test_object"
        })

        # Retrieve should return empty structure
        result = mock_augmenter.retrieve_structured_context("test_entity")

        assert result["entity"] == "test_entity"
        assert result["properties"] == {}
        assert result["outgoing_relations"] == []
        assert result["incoming_relations"] == []

    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_get_status_method(self, mock_augmenter):
        """Test the get_status method returns correct information."""
        # Set known states
        mock_augmenter._milvus_connected = True
        mock_augmenter._neo4j_connected = True
        mock_augmenter._embedding_loaded = True
        mock_augmenter._web_scraping_available = True

        status = mock_augmenter.get_status()

        assert status["milvus_connected"] is True
        assert status["neo4j_connected"] is True
        assert status["embedding_loaded"] is True
        assert status["web_scraping_available"] is True
        assert status["rag_available"] is True
        assert status["cag_available"] is True
        assert status["web_rag_available"] is True
        assert status["knowledge_augmentation_ready"] is True
        assert status["full_capabilities"] is True

    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_repr_method(self, mock_augmenter):
        """Test the string representation method."""
        mock_augmenter._milvus_connected = True
        mock_augmenter._neo4j_connected = True
        mock_augmenter._embedding_loaded = True
        mock_augmenter._web_scraping_available = True

        repr_str = repr(mock_augmenter)

        assert "KnowledgeAugmenter" in repr_str
        assert "all-MiniLM-L6-v2" in repr_str
        assert "rag_available: True" in repr_str
        assert "cag_available: True" in repr_str
        assert "web_rag_available: True" in repr_str
        assert "full_capabilities: True" in repr_str


class TestDocumentProcessing:
    """Test document processing and vector store operations."""

    @pytest.fixture
    def mock_augmenter(self):
        """Create a mocked KnowledgeAugmenter for testing."""
        with patch('gif_framework.core.knowledge_augmenter.SentenceTransformer'), \
             patch('gif_framework.core.knowledge_augmenter.connections'), \
             patch('gif_framework.core.knowledge_augmenter.GraphDatabase'):

            milvus_config = {"host": "localhost", "port": 19530, "collection_name": "test"}
            neo4j_config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"}

            augmenter = KnowledgeAugmenter(
                milvus_config=milvus_config,
                neo4j_config=neo4j_config,
                embedding_model_name="all-MiniLM-L6-v2"
            )

            # Mock successful connections
            augmenter._milvus_connected = True
            augmenter._neo4j_connected = True
            augmenter._embedding_loaded = True
            augmenter._web_scraping_available = True

            return augmenter

    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_add_documents_to_vector_store_success(self, mock_augmenter):
        """Test successful addition of documents to vector store."""
        documents = [
            {
                "text": "Exoplanets are planets outside our solar system.",
                "source": "astronomy_textbook",
                "metadata": {"topic": "exoplanets", "difficulty": "beginner"}
            },
            {
                "text": "Transit photometry measures the dimming of starlight.",
                "source": "kepler_mission_paper",
                "metadata": {"topic": "detection_methods", "year": 2010}
            }
        ]

        with patch.object(mock_augmenter, '_milvus_collection') as mock_collection:
            mock_collection.insert.return_value = Mock()

            with patch.object(mock_augmenter, '_embedding_model') as mock_model:
                mock_model.encode.return_value = [[0.1, 0.2, 0.3]]

                result = mock_augmenter.add_documents_to_vector_store(documents)

                assert result is True
                mock_collection.insert.assert_called_once()

    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_add_documents_empty_list(self, mock_augmenter):
        """Test adding empty document list."""
        result = mock_augmenter.add_documents_to_vector_store([])

        assert result is False

    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_add_documents_invalid_format(self, mock_augmenter):
        """Test adding documents with invalid format."""
        invalid_documents = [
            {"source": "test"},  # Missing "text" field
            {"text": ""},        # Empty text
        ]

        with patch.object(mock_augmenter, '_milvus_collection') as mock_collection:
            with patch.object(mock_augmenter, '_embedding_model') as mock_model:
                result = mock_augmenter.add_documents_to_vector_store(invalid_documents)

                assert result is False
                mock_collection.insert.assert_not_called()


class TestWebRAGCapabilities:
    """Test Web RAG functionality with DuckDuckGo search."""

    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    @patch('gif_framework.core.knowledge_augmenter.SentenceTransformer')
    @patch('gif_framework.core.knowledge_augmenter.connections')
    @patch('gif_framework.core.knowledge_augmenter.GraphDatabase')
    def test_web_rag_retrieval(self, mock_neo4j, mock_milvus, mock_transformer):
        """Tests that the web RAG method correctly searches, fetches, and parses."""
        # Create augmenter with mocked dependencies
        milvus_config = {"host": "localhost", "port": 19530, "collection_name": "test"}
        neo4j_config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "test"}

        augmenter = KnowledgeAugmenter(
            milvus_config=milvus_config,
            neo4j_config=neo4j_config,
            embedding_model_name="test-model"
        )

        mock_search_results = [{'url': 'http://mock-url.com', 'title': 'Test Result'}]
        mock_html_content = "<html><body><p>This is the first paragraph.</p><p>This is the second.</p></body></html>"
        expected_text = "This is the first paragraph. This is the second."

        # Mock the web retrieval methods directly to avoid import issues
        with patch.object(augmenter, '_search_web') as mock_search, \
             patch.object(augmenter, '_extract_text_from_url') as mock_extract:

            # Set web scraping as available
            augmenter._web_scraping_available = True

            # Configure the mocks
            mock_search.return_value = mock_search_results
            mock_extract.return_value = expected_text

            # Run the method
            retrieved_context = augmenter.retrieve_web_context("test query")

            # Assertions
            mock_search.assert_called_with("test query", 3)
            mock_extract.assert_called_with('http://mock-url.com', 10)
            assert len(retrieved_context) > 0
            assert expected_text in retrieved_context[0]

    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    @patch('gif_framework.core.knowledge_augmenter.SentenceTransformer')
    @patch('gif_framework.core.knowledge_augmenter.connections')
    @patch('gif_framework.core.knowledge_augmenter.GraphDatabase')
    def test_prioritized_context_retrieval(self, mock_neo4j, mock_milvus, mock_transformer):
        """Test that retrieve_context prioritizes web over database."""
        # Create augmenter with mocked dependencies
        milvus_config = {"host": "localhost", "port": 19530, "collection_name": "test"}
        neo4j_config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "test"}

        augmenter = KnowledgeAugmenter(
            milvus_config=milvus_config,
            neo4j_config=neo4j_config,
            embedding_model_name="test-model"
        )

        # Mock successful web retrieval
        with patch.object(augmenter, 'retrieve_web_context') as mock_web, \
             patch.object(augmenter, 'retrieve_unstructured_context') as mock_db:

            mock_web.return_value = ["Web context from live search"]
            mock_db.return_value = ["Database context from vector store"]

            # Test prioritization
            result = augmenter.retrieve_context("test query")

            # Should call web first and return web result
            mock_web.assert_called_once_with("test query")
            mock_db.assert_not_called()  # Should not fallback to database
            assert "Web context from live search" in result

    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    @patch('gif_framework.core.knowledge_augmenter.SentenceTransformer')
    @patch('gif_framework.core.knowledge_augmenter.connections')
    @patch('gif_framework.core.knowledge_augmenter.GraphDatabase')
    def test_fallback_to_database_rag(self, mock_neo4j, mock_milvus, mock_transformer):
        """Test fallback to database when web retrieval fails."""
        # Create augmenter with mocked dependencies
        milvus_config = {"host": "localhost", "port": 19530, "collection_name": "test"}
        neo4j_config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "test"}

        augmenter = KnowledgeAugmenter(
            milvus_config=milvus_config,
            neo4j_config=neo4j_config,
            embedding_model_name="test-model"
        )

        # Mock failed web retrieval and successful database retrieval
        with patch.object(augmenter, 'retrieve_web_context') as mock_web, \
             patch.object(augmenter, 'retrieve_unstructured_context') as mock_db:

            mock_web.return_value = []  # Empty web results
            mock_db.return_value = ["Database context from vector store"]

            # Test fallback
            result = augmenter.retrieve_context("test query")

            # Should call web first, then fallback to database
            mock_web.assert_called_once_with("test query")
            mock_db.assert_called_once_with("test query")
            assert "Database context from vector store" in result


class TestDatabaseRAGCAGValidation:
    """Test database RAG/CAG query construction validation."""

    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_rag_retrieval_constructs_correct_query(self):
        """Test that RAG retrieval constructs correct Milvus query."""
        # Create a minimal augmenter instance
        milvus_config = {"host": "localhost", "port": 19530, "collection_name": "test"}
        neo4j_config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "test"}

        with patch('gif_framework.core.knowledge_augmenter.connections'), \
             patch('gif_framework.core.knowledge_augmenter.GraphDatabase'), \
             patch('gif_framework.core.knowledge_augmenter.SentenceTransformer'):
            augmenter = KnowledgeAugmenter(milvus_config, neo4j_config)

        # Mock the internal method directly to avoid connection issues
        with patch.object(augmenter, '_embedding_model') as mock_embedding_model, \
             patch.object(augmenter, '_milvus_collection') as mock_collection:

            # Set up mocks
            mock_embedding = Mock()
            mock_embedding.tolist.return_value = [0.1] * 384
            mock_embedding_model.encode.return_value = [mock_embedding]

            mock_hit = Mock()
            mock_hit.entity.get = lambda x: "Test document text" if x == "text" else "test_source"
            mock_collection.search.return_value = [[mock_hit]]
            mock_collection.load = Mock()

            # Set connection state
            augmenter._milvus_connected = True

            # Test RAG retrieval
            result = augmenter.retrieve_unstructured_context("test query", top_k=5)

            # Verify embedding generation (called with list)
            mock_embedding_model.encode.assert_called_once_with(["test query"])

            # Verify Milvus search was called with correct parameters
            mock_collection.search.assert_called_once()
            search_args = mock_collection.search.call_args

            # Check search parameters - handle both positional and keyword args
            if len(search_args[0]) > 0:
                # Called with positional args
                assert search_args[0][0] == [[0.1] * 384]  # data parameter (list of embeddings)
                assert search_args[1]['anns_field'] == 'embedding'
                assert search_args[1]['param'] == {'metric_type': 'COSINE', 'params': {'nprobe': 10}}
                assert search_args[1]['limit'] == 5
                assert search_args[1]['output_fields'] == ['text', 'source']
            else:
                # Called with keyword args only
                assert search_args[1]['data'] == [[0.1] * 384]
                assert search_args[1]['anns_field'] == 'embedding'
                assert search_args[1]['param'] == {'metric_type': 'COSINE', 'params': {'nprobe': 10}}
                assert search_args[1]['limit'] == 5
                assert search_args[1]['output_fields'] == ['text', 'source']

    @pytest.mark.skipif(not KNOWLEDGE_AUGMENTER_AVAILABLE, reason="KnowledgeAugmenter dependencies not available")
    def test_cag_update_constructs_correct_cypher_query(self):
        """Test that CAG update constructs correct Neo4j Cypher query."""
        # Create a minimal augmenter instance
        milvus_config = {"host": "localhost", "port": 19530, "collection_name": "test"}
        neo4j_config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "test"}

        with patch('gif_framework.core.knowledge_augmenter.connections'), \
             patch('gif_framework.core.knowledge_augmenter.GraphDatabase'), \
             patch('gif_framework.core.knowledge_augmenter.SentenceTransformer'):
            augmenter = KnowledgeAugmenter(milvus_config, neo4j_config)

        # Mock the Neo4j driver and session directly
        with patch.object(augmenter, '_neo4j_driver') as mock_driver:
            mock_session = Mock()
            mock_context_manager = Mock()
            mock_context_manager.__enter__ = Mock(return_value=mock_session)
            mock_context_manager.__exit__ = Mock(return_value=None)
            mock_driver.session.return_value = mock_context_manager

            # Set connection state
            augmenter._neo4j_connected = True

            # Test knowledge graph update
            structured_knowledge = {
                "subject": "Kepler-90",
                "relation": "HAS_PLANET",
                "object": "Kepler-90i",
                "properties": {"discovery_year": 2017}
            }

            augmenter.update_knowledge_graph(structured_knowledge)

            # Verify session was used
            mock_driver.session.assert_called_once()

            # Verify Cypher query was executed
            mock_session.run.assert_called_once()
            cypher_call = mock_session.run.call_args

            # Check that the Cypher query contains expected elements
            cypher_query = cypher_call[0][0]
            assert "MERGE" in cypher_query
            assert "Kepler-90" in cypher_call[1]['subject']
            assert "Kepler-90i" in cypher_call[1]['object']
            assert "HAS_PLANET" in cypher_query
            assert cypher_call[1]['properties'] == {"discovery_year": 2017}


if __name__ == "__main__":
    pytest.main([__file__])
