"""
Simplified pytest configuration and shared fixtures for Academic Citation Platform.

Provides essential test fixtures organized by component:
- Basic test data and models
- Mock services and connections  
- Integration test setup
"""

import pytest
import os
import tempfile
from datetime import datetime
from unittest.mock import Mock
from typing import Dict, List, Any, Generator
import numpy as np

# Core model imports
from src.models.paper import Paper
from src.models.author import Author
from src.models.ml import PaperEmbedding, CitationPrediction
from src.data.api_config import Neo4jConfig, MLConfig


# ====================================================================
# CORE TEST DATA
# ====================================================================

@pytest.fixture
def sample_paper_data() -> Dict[str, Any]:
    """Sample paper data for testing."""
    return {
        "paper_id": "paper123",
        "title": "Advanced ML Techniques for Citation Prediction",
        "abstract": "Novel approaches to predicting citations using GNNs.",
        "year": 2023,
        "authors": ["Dr. Jane Smith", "Prof. John Doe"],
        "citation_count": 42,
        "fields": ["Computer Science", "Machine Learning"]
    }

@pytest.fixture
def sample_author_data() -> Dict[str, Any]:
    """Sample author data for testing."""
    return {
        "author_id": "author123",
        "name": "Dr. Jane Smith",
        "paper_count": 45,
        "citation_count": 1250,
        "h_index": 18
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
def sample_papers_list(sample_paper_data: Dict[str, Any]) -> List[Paper]:
    """Create a list of sample papers for batch testing."""
    papers = []
    for i in range(3):  # Reduced from 5 to 3
        data = sample_paper_data.copy()
        data["paper_id"] = f"paper{i+1}"
        data["title"] = f"Test Paper {i+1}"
        data["year"] = 2020 + i
        papers.append(Paper(**data))
    return papers


# ====================================================================
# CONFIGURATION FIXTURES
# ====================================================================

@pytest.fixture
def neo4j_config() -> Neo4jConfig:
    """Create test Neo4j configuration."""
    return Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="test_password",
        database="test_db"
    )

@pytest.fixture
def ml_config() -> MLConfig:
    """Create test ML configuration."""
    return MLConfig(
        model_path="/tmp/test_model.pkl",
        embedding_dim=128,
        batch_size=32
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
        "total": 1
    }
    return mock_response

@pytest.fixture
def api_error_response() -> Mock:
    """Mock API error response."""
    mock_response = Mock()
    mock_response.status_code = 429
    mock_response.json.return_value = {"error": "Rate limit exceeded"}
    return mock_response


# ====================================================================
# UTILITY FIXTURES
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
    with open(model_path, 'wb') as f:
        f.write(b'dummy model data')
    return model_path

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
            }
        },
        {
            "paper": {
                "paperId": "paper2", 
                "title": "Test Paper 2",
                "year": 2022,
                "citationCount": 5
            }
        }
    ]

@pytest.fixture
def sample_embeddings_matrix() -> np.ndarray:
    """Sample embeddings matrix for testing."""
    return np.random.rand(5, 8)  # 5 papers, 8 dimensions

@pytest.fixture
def sample_predictions_data() -> List[Dict[str, Any]]:
    """Sample prediction results."""
    return [
        {"source_paper_id": "paper1", "target_paper_id": "paper2", "score": 0.85},
        {"source_paper_id": "paper1", "target_paper_id": "paper3", "score": 0.23}
    ]