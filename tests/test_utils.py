"""
Test utilities and helper functions for Academic Citation Platform tests.

Provides common testing utilities including:
- Data validation helpers
- Mock response generators  
- Test assertion helpers
- Performance measurement tools
"""

import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable
from unittest.mock import Mock, MagicMock
from contextlib import contextmanager
import numpy as np

from src.models.paper import Paper
from src.models.author import Author
from src.models.venue import Venue
from src.models.ml import PaperEmbedding, CitationPrediction
from src.models.api import APIResponse, APIStatus


# ====================================================================
# DATA VALIDATION HELPERS
# ====================================================================

def validate_paper_model(paper: Paper) -> bool:
    """Validate Paper model has required fields."""
    required_fields = ['paper_id', 'title']
    return all(hasattr(paper, field) and getattr(paper, field) for field in required_fields)


def validate_author_model(author: Author) -> bool:
    """Validate Author model has required fields."""
    required_fields = ['name']
    return all(hasattr(author, field) and getattr(author, field) for field in required_fields)


def validate_embedding_model(embedding: PaperEmbedding) -> bool:
    """Validate PaperEmbedding model."""
    return (
        hasattr(embedding, 'paper_id') and 
        hasattr(embedding, 'embedding') and
        hasattr(embedding, 'embedding_dim') and
        len(embedding.embedding) == embedding.embedding_dim
    )


def validate_prediction_model(prediction: CitationPrediction) -> bool:
    """Validate CitationPrediction model."""
    return (
        hasattr(prediction, 'source_paper_id') and
        hasattr(prediction, 'target_paper_id') and
        hasattr(prediction, 'prediction_score') and
        0.0 <= prediction.prediction_score <= 1.0
    )


# ====================================================================
# MOCK RESPONSE GENERATORS
# ====================================================================

def generate_mock_semantic_scholar_response(
    papers: List[Dict[str, Any]], 
    total: Optional[int] = None,
    offset: int = 0
) -> Dict[str, Any]:
    """Generate mock Semantic Scholar API response."""
    return {
        "data": papers,
        "total": total or len(papers),
        "offset": offset,
        "next": f"/papers?offset={offset + len(papers)}" if total and offset + len(papers) < total else None
    }


def generate_mock_paper_data(
    paper_id: str = "test_paper",
    title: str = "Test Paper",
    year: int = 2023,
    citation_count: int = 10
) -> Dict[str, Any]:
    """Generate mock paper data."""
    return {
        "paperId": paper_id,
        "title": title,
        "abstract": f"Abstract for {title}",
        "year": year,
        "authors": [
            {"authorId": "author1", "name": "Test Author"}
        ],
        "venue": {"name": "Test Venue", "type": "Conference"},
        "citationCount": citation_count,
        "references": [],
        "citations": [],
        "fieldsOfStudy": ["Computer Science"],
        "url": f"https://example.com/{paper_id}",
        "doi": f"10.1000/{paper_id}"
    }


def generate_mock_author_data(
    author_id: str = "test_author",
    name: str = "Test Author",
    paper_count: int = 20,
    citation_count: int = 500
) -> Dict[str, Any]:
    """Generate mock author data."""
    return {
        "authorId": author_id,
        "name": name,
        "paperCount": paper_count,
        "citationCount": citation_count,
        "hIndex": min(paper_count, int(citation_count ** 0.5)),
        "url": f"https://example.com/author/{author_id}",
        "affiliations": ["Test University"]
    }


def generate_mock_neo4j_record(data: Dict[str, Any]) -> Mock:
    """Generate mock Neo4j record."""
    mock_record = Mock()
    for key, value in data.items():
        setattr(mock_record, key, value)
    mock_record.data.return_value = data
    return mock_record


# ====================================================================
# HTTP MOCK HELPERS
# ====================================================================

class MockHTTPResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self, json_data: Dict[str, Any], status_code: int = 200, headers: Dict[str, str] = None):
        self.json_data = json_data
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(json_data)
    
    def json(self):
        return self.json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def create_rate_limit_response(remaining: int = 0) -> MockHTTPResponse:
    """Create mock rate limit response."""
    return MockHTTPResponse(
        {"error": "Rate limit exceeded"},
        status_code=429,
        headers={"x-ratelimit-remaining": str(remaining)}
    )


def create_success_response(data: Any) -> MockHTTPResponse:
    """Create mock success response."""
    return MockHTTPResponse(data, status_code=200)


# ====================================================================
# ASSERTION HELPERS
# ====================================================================

def assert_paper_fields(paper: Paper, expected: Dict[str, Any]):
    """Assert paper fields match expected values."""
    for field, expected_value in expected.items():
        actual_value = getattr(paper, field, None)
        assert actual_value == expected_value, f"Paper.{field}: expected {expected_value}, got {actual_value}"


def assert_author_fields(author: Author, expected: Dict[str, Any]):
    """Assert author fields match expected values."""
    for field, expected_value in expected.items():
        actual_value = getattr(author, field, None)
        assert actual_value == expected_value, f"Author.{field}: expected {expected_value}, got {actual_value}"


def assert_api_response_structure(response: APIResponse):
    """Assert API response has correct structure."""
    assert hasattr(response, 'status')
    assert hasattr(response, 'data')
    assert hasattr(response, 'message')
    assert isinstance(response.status, APIStatus)


def assert_embedding_valid(embedding: PaperEmbedding):
    """Assert embedding is valid."""
    assert embedding.paper_id is not None
    assert embedding.embedding is not None
    assert len(embedding.embedding) == embedding.embedding_dim
    assert all(isinstance(x, (int, float)) for x in embedding.embedding)


def assert_prediction_valid(prediction: CitationPrediction):
    """Assert prediction is valid."""
    assert prediction.source_paper_id is not None
    assert prediction.target_paper_id is not None
    assert 0.0 <= prediction.prediction_score <= 1.0
    assert prediction.model_name is not None


# ====================================================================
# PERFORMANCE HELPERS
# ====================================================================

@contextmanager
def time_limit(seconds: float):
    """Context manager to enforce time limits on tests."""
    start_time = time.time()
    yield
    elapsed = time.time() - start_time
    assert elapsed <= seconds, f"Test took {elapsed:.2f}s, limit was {seconds}s"


def measure_execution_time(func: Callable, *args, **kwargs) -> tuple:
    """Measure function execution time."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


async def measure_async_execution_time(coro) -> tuple:
    """Measure async function execution time."""
    start_time = time.time()
    result = await coro
    end_time = time.time()
    return result, end_time - start_time


# ====================================================================
# DATA GENERATION HELPERS
# ====================================================================

def generate_test_papers(count: int = 10) -> List[Dict[str, Any]]:
    """Generate list of test paper data."""
    papers = []
    for i in range(count):
        papers.append(generate_mock_paper_data(
            paper_id=f"paper_{i+1}",
            title=f"Test Paper {i+1}",
            year=2020 + (i % 4),
            citation_count=i * 5
        ))
    return papers


def generate_test_authors(count: int = 5) -> List[Dict[str, Any]]:
    """Generate list of test author data."""
    authors = []
    for i in range(count):
        authors.append(generate_mock_author_data(
            author_id=f"author_{i+1}",
            name=f"Test Author {i+1}",
            paper_count=10 + i * 5,
            citation_count=100 + i * 50
        ))
    return authors


def generate_test_embeddings(paper_ids: List[str], dim: int = 128) -> List[PaperEmbedding]:
    """Generate test embeddings for papers."""
    embeddings = []
    for paper_id in paper_ids:
        embedding_vector = np.random.rand(dim).tolist()
        embeddings.append(PaperEmbedding(
            paper_id=paper_id,
            embedding=embedding_vector,
            model_name="TestModel",
            embedding_dim=dim
        ))
    return embeddings


def generate_test_predictions(
    source_papers: List[str], 
    target_papers: List[str]
) -> List[CitationPrediction]:
    """Generate test citation predictions."""
    predictions = []
    for source in source_papers:
        for target in target_papers:
            if source != target:  # Don't predict self-citations
                score = np.random.rand()
                predictions.append(CitationPrediction(
                    source_paper_id=source,
                    target_paper_id=target,
                    prediction_score=score,
                    model_name="TestModel"
                ))
    return predictions


# ====================================================================
# DATABASE HELPERS
# ====================================================================

def create_mock_neo4j_session():
    """Create mock Neo4j session."""
    session = Mock()
    session.run = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=None)
    return session


def create_mock_query_result(records: List[Dict[str, Any]]):
    """Create mock Neo4j query result."""
    mock_result = Mock()
    mock_result.records = [generate_mock_neo4j_record(record) for record in records]
    mock_result.single.return_value = mock_result.records[0] if mock_result.records else None
    mock_result.data.return_value = records
    return mock_result


# ====================================================================
# FILE HELPERS
# ====================================================================

def create_temp_model_file(model_data: bytes, file_path: str):
    """Create temporary model file for testing."""
    with open(file_path, 'wb') as f:
        f.write(model_data)


def create_temp_config_file(config_data: Dict[str, Any], file_path: str):
    """Create temporary config file for testing."""
    with open(file_path, 'w') as f:
        json.dump(config_data, f, indent=2)


# ====================================================================
# COMPARISON HELPERS
# ====================================================================

def compare_papers(paper1: Paper, paper2: Paper, ignore_fields: List[str] = None) -> bool:
    """Compare two Paper instances."""
    ignore_fields = ignore_fields or []
    
    for field in paper1.__fields__:
        if field in ignore_fields:
            continue
        
        val1 = getattr(paper1, field, None)
        val2 = getattr(paper2, field, None)
        
        if val1 != val2:
            return False
    
    return True


def compare_authors(author1: Author, author2: Author, ignore_fields: List[str] = None) -> bool:
    """Compare two Author instances."""
    ignore_fields = ignore_fields or []
    
    for field in author1.__fields__:
        if field in ignore_fields:
            continue
        
        val1 = getattr(author1, field, None)
        val2 = getattr(author2, field, None)
        
        if val1 != val2:
            return False
    
    return True


# ====================================================================
# ERROR SIMULATION
# ====================================================================

class SimulatedNetworkError(Exception):
    """Simulated network error for testing."""
    pass


class SimulatedDatabaseError(Exception):
    """Simulated database error for testing."""
    pass


class SimulatedAPIError(Exception):
    """Simulated API error for testing."""
    pass


def simulate_intermittent_failure(failure_rate: float = 0.3):
    """Decorator to simulate intermittent failures."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if np.random.rand() < failure_rate:
                raise SimulatedNetworkError("Simulated network failure")
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ====================================================================
# ASYNC HELPERS
# ====================================================================

def run_async_test(coro):
    """Helper to run async test functions."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def async_mock_response(data: Any, delay: float = 0.1):
    """Create async mock response with delay."""
    await asyncio.sleep(delay)
    return data