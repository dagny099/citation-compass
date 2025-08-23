"""
Pytest configuration and shared fixtures for Academic Citation Platform.

Provides comprehensive test fixtures for all components including:
- Mock data for papers, authors, venues
- Database connections and test data
- API client mocks
- ML model fixtures
"""

import pytest
import os
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, List, Any, Generator
import numpy as np

# Import all models for fixture creation
from src.models.paper import Paper, PaperCreate
from src.models.author import Author
from src.models.venue import Venue
from src.models.ml import PaperEmbedding, CitationPrediction, ModelType, PredictionConfidence
from src.models.network import NetworkNode, NetworkEdge, NodeType, EdgeType
from src.models.api import APIResponse, APIStatus
from src.data.api_config import SemanticScholarConfig, Neo4jConfig, MLConfig
from src.data.unified_api_client import UnifiedSemanticScholarClient
from src.data.unified_database import UnifiedDatabaseManager


# ====================================================================
# BASIC DATA FIXTURES
# ====================================================================

@pytest.fixture
def sample_author_data() -> Dict[str, Any]:
    """Sample author data for testing."""
    return {
        "author_id": "author123",
        "name": "Dr. Jane Smith",
        "paper_count": 45,
        "citation_count": 1250,
        "h_index": 18,
        "url": "https://example.com/author/jane-smith",
        "affiliations": ["MIT", "Stanford University"]
    }


@pytest.fixture
def sample_paper_data() -> Dict[str, Any]:
    """Sample paper data for testing."""
    return {
        "paper_id": "paper123",
        "title": "Advanced Machine Learning Techniques for Citation Prediction",
        "abstract": "This paper presents novel approaches to predicting academic citations using graph neural networks.",
        "year": 2023,
        "authors": ["Dr. Jane Smith", "Prof. John Doe"],
        "venues": ["International Conference on Machine Learning"],
        "citation_count": 42,
        "reference_count": 15,
        "fields": ["Computer Science", "Machine Learning"],
        "doi": "10.1000/example123"
    }


@pytest.fixture
def sample_venue_data() -> Dict[str, Any]:
    """Sample venue data for testing."""
    return {
        "name": "International Conference on Machine Learning",
        "venue_type": "Conference",
        "url": "https://icml.cc",
        "issn": "1234-5678"
    }


@pytest.fixture
def sample_embedding_data() -> List[float]:
    """Sample embedding vector for testing."""
    return [0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, 0.8]


# ====================================================================
# MODEL FIXTURES
# ====================================================================

@pytest.fixture
def sample_paper(sample_paper_data: Dict[str, Any]) -> Paper:
    """Create a sample Paper model instance."""
    return Paper(**sample_paper_data)


@pytest.fixture
def sample_author(sample_author_data: Dict[str, Any]) -> Author:
    """Create a sample Author model instance."""
    return Author(**sample_author_data)


@pytest.fixture
def sample_venue(sample_venue_data: Dict[str, Any]) -> Venue:
    """Create a sample Venue model instance."""
    return Venue(**sample_venue_data)


@pytest.fixture
def sample_paper_embedding(sample_embedding_data: List[float]) -> PaperEmbedding:
    """Create a sample PaperEmbedding instance."""
    return PaperEmbedding(
        paper_id="paper123",
        embedding=sample_embedding_data,
        model_name="TransE",
        embedding_dim=len(sample_embedding_data),
        created_at=datetime.now()
    )


@pytest.fixture
def sample_citation_prediction() -> CitationPrediction:
    """Create a sample CitationPrediction instance."""
    return CitationPrediction(
        source_paper_id="paper123",
        target_paper_id="paper456",
        prediction_score=0.85,
        model_name="TransE",
        created_at=datetime.now()
    )


@pytest.fixture
def sample_network_node() -> NetworkNode:
    """Create a sample NetworkNode instance."""
    return NetworkNode(
        id="paper123",
        label="ML Paper",
        node_type=NodeType.PAPER,
        citation_count=42,
        properties={"year": 2023, "field": "ML"}
    )


@pytest.fixture
def sample_network_edge() -> NetworkEdge:
    """Create a sample NetworkEdge instance."""
    return NetworkEdge(
        source="paper123",
        target="paper456",
        edge_type=EdgeType.CITES,
        weight=1.0,
        properties={"citation_context": "methodology"}
    )


# ====================================================================
# MULTIPLE ITEMS FIXTURES
# ====================================================================

@pytest.fixture
def sample_papers_list(sample_paper_data: Dict[str, Any]) -> List[Paper]:
    """Create a list of sample papers with different IDs."""
    papers = []
    for i in range(5):
        data = sample_paper_data.copy()
        data["paper_id"] = f"paper{i+1}"
        data["title"] = f"Test Paper {i+1}"
        data["year"] = 2020 + i
        data["citation_count"] = 10 * (i + 1)
        papers.append(Paper(**data))
    return papers


@pytest.fixture
def sample_authors_list(sample_author_data: Dict[str, Any]) -> List[Author]:
    """Create a list of sample authors with different IDs."""
    authors = []
    for i in range(3):
        data = sample_author_data.copy()
        data["author_id"] = f"author{i+1}"
        data["name"] = f"Author {i+1}"
        data["paper_count"] = 20 + i * 10
        data["citation_count"] = 500 + i * 300
        authors.append(Author(**data))
    return authors


# ====================================================================
# CONFIG FIXTURES
# ====================================================================

@pytest.fixture
def semantic_scholar_config() -> SemanticScholarConfig:
    """Create test configuration for Semantic Scholar API."""
    return SemanticScholarConfig(
        base_url="https://api.semanticscholar.org/graph/v1",
        requests_per_minute=60,
        requests_per_second=1,
        timeout=30,
        max_retries=3,
        retry_delay=1.0,
        user_agent="test-agent",
        cache_ttl=300
    )


@pytest.fixture
def neo4j_config() -> Neo4jConfig:
    """Create test configuration for Neo4j."""
    return Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="test_password",
        database="test_db",
        connection_timeout=10,
        max_connection_pool_size=50,
        encrypted=False
    )


@pytest.fixture
def ml_config() -> MLConfig:
    """Create test configuration for ML components."""
    return MLConfig(
        model_path="/tmp/test_model.pkl",
        embedding_dim=128,
        batch_size=32,
        prediction_threshold=0.5,
        cache_predictions=True,
        max_cache_size=1000
    )


# ====================================================================
# MOCK FIXTURES
# ====================================================================

@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    mock_driver = Mock()
    mock_session = Mock()
    mock_driver.session.return_value = mock_session
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock(return_value=None)
    return mock_driver


@pytest.fixture
def mock_api_response() -> Mock:
    """Mock HTTP response for API testing."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"paperId": "test123", "title": "Test Paper"}],
        "total": 1,
        "offset": 0,
        "next": None
    }
    mock_response.headers = {"x-ratelimit-remaining": "100"}
    return mock_response


@pytest.fixture
def mock_requests_session():
    """Mock requests session for HTTP testing."""
    session = Mock()
    session.get = Mock()
    session.post = Mock()
    return session


# ====================================================================
# FILE AND DIRECTORY FIXTURES
# ====================================================================

@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def temp_model_file(temp_dir: str) -> str:
    """Create temporary model file for testing."""
    model_path = os.path.join(temp_dir, "test_model.pkl")
    # Create a dummy model file
    with open(model_path, 'wb') as f:
        f.write(b'dummy model data')
    return model_path


@pytest.fixture
def temp_config_file(temp_dir: str) -> str:
    """Create temporary config file for testing."""
    config_path = os.path.join(temp_dir, "test_config.json")
    config_data = {
        "api": {"base_url": "https://test.api.com"},
        "database": {"uri": "bolt://localhost:7687"},
        "ml": {"model_path": "/tmp/model.pkl"}
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    return config_path


# ====================================================================
# DATABASE FIXTURES
# ====================================================================

@pytest.fixture
def mock_database_manager(mock_neo4j_driver) -> UnifiedDatabaseManager:
    """Mock database manager for testing."""
    with pytest.MonkeyPatch().context() as m:
        # Mock the driver creation
        m.setattr("neo4j.GraphDatabase.driver", lambda *args, **kwargs: mock_neo4j_driver)
        
        config = Neo4jConfig(
            uri="bolt://test:7687",
            username="test",
            password="test",
            database="test"
        )
        return UnifiedDatabaseManager(config)


@pytest.fixture
def sample_neo4j_records() -> List[Dict[str, Any]]:
    """Sample Neo4j query results."""
    return [
        {
            "paper": {
                "paperId": "paper1",
                "title": "Test Paper 1",
                "year": 2023,
                "citationCount": 10
            },
            "author": {
                "authorId": "author1",
                "name": "Test Author",
                "paperCount": 20
            }
        },
        {
            "paper": {
                "paperId": "paper2", 
                "title": "Test Paper 2",
                "year": 2022,
                "citationCount": 5
            },
            "author": {
                "authorId": "author2",
                "name": "Another Author",
                "paperCount": 15
            }
        }
    ]


# ====================================================================
# API CLIENT FIXTURES
# ====================================================================

@pytest.fixture
def mock_api_client(semantic_scholar_config: SemanticScholarConfig, mock_requests_session) -> UnifiedSemanticScholarClient:
    """Mock API client for testing."""
    client = UnifiedSemanticScholarClient(semantic_scholar_config)
    client.session = mock_requests_session
    return client


# ====================================================================
# ERROR FIXTURES
# ====================================================================

@pytest.fixture
def api_error_response() -> Mock:
    """Mock API error response."""
    mock_response = Mock()
    mock_response.status_code = 429
    mock_response.json.return_value = {"error": "Rate limit exceeded"}
    mock_response.raise_for_status.side_effect = Exception("HTTP 429")
    return mock_response


@pytest.fixture
def network_error():
    """Mock network connection error."""
    return ConnectionError("Unable to connect to API")


# ====================================================================
# ML FIXTURES
# ====================================================================

@pytest.fixture
def sample_embeddings_matrix() -> np.ndarray:
    """Sample embeddings matrix for testing."""
    return np.random.rand(10, 128)  # 10 papers, 128 dimensions


@pytest.fixture
def sample_predictions_data() -> List[Dict[str, Any]]:
    """Sample prediction results."""
    return [
        {
            "source_paper_id": "paper1",
            "target_paper_id": "paper2", 
            "score": 0.85,
            "confidence": "high"
        },
        {
            "source_paper_id": "paper1",
            "target_paper_id": "paper3",
            "score": 0.23,
            "confidence": "low"
        }
    ]


# ====================================================================
# PERFORMANCE FIXTURES
# ====================================================================

@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = datetime.now()
        
        def stop(self):
            self.end_time = datetime.now()
        
        @property
        def elapsed(self) -> timedelta:
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return timedelta(0)
    
    return Timer()


# ====================================================================
# INTEGRATION FIXTURES
# ====================================================================

@pytest.fixture
def integration_test_data() -> Dict[str, Any]:
    """Complete test data for integration testing."""
    return {
        "search_query": "machine learning citation prediction",
        "expected_results": 10,
        "test_paper_ids": ["paper1", "paper2", "paper3"],
        "test_author_ids": ["author1", "author2"],
        "expected_embeddings_dim": 128,
        "prediction_threshold": 0.5
    }


# ====================================================================
# CLEANUP FIXTURES
# ====================================================================

@pytest.fixture(autouse=True)
def cleanup_test_environment():
    """Automatic cleanup after each test."""
    yield
    # Cleanup operations can be added here
    # For now, just ensure no hanging connections
    pass


# ====================================================================
# PARAMETRIZED FIXTURES
# ====================================================================

@pytest.fixture(params=[
    {"model_name": "TransE", "dim": 128},
    {"model_name": "ComplEx", "dim": 256},
    {"model_name": "RotatE", "dim": 512}
])
def ml_model_params(request):
    """Parametrized ML model configurations."""
    return request.param


@pytest.fixture(params=[NodeType.PAPER, NodeType.AUTHOR, NodeType.VENUE])
def node_types(request):
    """Parametrized node types for network testing."""
    return request.param


@pytest.fixture(params=[EdgeType.CITES, EdgeType.AUTHORED, EdgeType.PUBLISHED_IN])
def edge_types(request):
    """Parametrized edge types for network testing.""" 
    return request.param