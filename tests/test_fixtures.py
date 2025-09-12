"""
Test the test fixtures and infrastructure to ensure they work correctly.

This test file validates that all fixtures in conftest.py work as expected
and provide valid test data for the other test suites.
"""

import pytest
import os
import tempfile
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

from src.models.paper import Paper
from src.models.author import Author
from src.models.venue import Venue
from src.models.ml import PaperEmbedding, CitationPrediction, ModelType, PredictionConfidence
from src.models.network import NetworkNode, NetworkEdge, NodeType, EdgeType
from src.models.api import APIResponse, APIStatus
from src.data.api_config import SemanticScholarConfig, Neo4jConfig, MLConfig

from tests.test_utils import (
    validate_paper_model,
    validate_author_model, 
    validate_embedding_model,
    validate_prediction_model,
    assert_paper_fields,
    assert_author_fields
)


@pytest.mark.unit
class TestDataFixtures:
    """Test basic data fixtures."""

    def test_sample_author_data_fixture(self, sample_author_data):
        """Test sample author data fixture."""
        assert isinstance(sample_author_data, dict)
        assert "author_id" in sample_author_data
        assert "name" in sample_author_data
        assert sample_author_data["name"] == "Dr. Jane Smith"
        assert sample_author_data["paper_count"] == 45

    def test_sample_paper_data_fixture(self, sample_paper_data):
        """Test sample paper data fixture.""" 
        assert isinstance(sample_paper_data, dict)
        assert "paper_id" in sample_paper_data
        assert "title" in sample_paper_data
        assert sample_paper_data["paper_id"] == "paper123"
        assert sample_paper_data["year"] == 2023

    def test_sample_venue_data_fixture(self, sample_venue_data):
        """Test sample venue data fixture."""
        assert isinstance(sample_venue_data, dict)
        assert "name" in sample_venue_data
        assert "venue_type" in sample_venue_data
        assert sample_venue_data["name"] == "International Conference on Machine Learning"

    def test_sample_embedding_data_fixture(self, sample_embedding_data):
        """Test sample embedding data fixture."""
        assert isinstance(sample_embedding_data, list)
        assert len(sample_embedding_data) == 8
        assert all(isinstance(x, float) for x in sample_embedding_data)


@pytest.mark.unit
class TestModelFixtures:
    """Test model instance fixtures."""

    def test_sample_paper_fixture(self, sample_paper):
        """Test sample paper model fixture."""
        assert isinstance(sample_paper, Paper)
        assert validate_paper_model(sample_paper)
        assert sample_paper.paper_id == "paper123"
        assert sample_paper.title == "Advanced Machine Learning Techniques for Citation Prediction"

    def test_sample_author_fixture(self, sample_author):
        """Test sample author model fixture.""" 
        assert isinstance(sample_author, Author)
        assert validate_author_model(sample_author)
        assert sample_author.author_id == "author123"
        assert sample_author.name == "Dr. Jane Smith"

    def test_sample_venue_fixture(self, sample_venue):
        """Test sample venue model fixture."""
        assert isinstance(sample_venue, Venue)
        assert sample_venue.name == "International Conference on Machine Learning"
        assert sample_venue.venue_type == "Conference"

    def test_sample_paper_embedding_fixture(self, sample_paper_embedding):
        """Test sample paper embedding fixture."""
        assert isinstance(sample_paper_embedding, PaperEmbedding)
        assert validate_embedding_model(sample_paper_embedding)
        assert sample_paper_embedding.paper_id == "paper123"
        assert sample_paper_embedding.model_name == "TransE"
        assert len(sample_paper_embedding.embedding) == sample_paper_embedding.embedding_dim

    def test_sample_citation_prediction_fixture(self, sample_citation_prediction):
        """Test sample citation prediction fixture."""
        assert isinstance(sample_citation_prediction, CitationPrediction)
        assert validate_prediction_model(sample_citation_prediction)
        assert sample_citation_prediction.source_paper_id == "paper123"
        assert sample_citation_prediction.target_paper_id == "paper456"
        assert 0.0 <= sample_citation_prediction.prediction_score <= 1.0

    def test_sample_network_node_fixture(self, sample_network_node):
        """Test sample network node fixture."""
        assert isinstance(sample_network_node, NetworkNode)
        assert sample_network_node.id == "paper123"
        assert sample_network_node.node_type == NodeType.PAPER
        assert sample_network_node.citation_count == 42

    def test_sample_network_edge_fixture(self, sample_network_edge):
        """Test sample network edge fixture."""
        assert isinstance(sample_network_edge, NetworkEdge)
        assert sample_network_edge.source == "paper123"
        assert sample_network_edge.target == "paper456"
        assert sample_network_edge.edge_type == EdgeType.CITES


@pytest.mark.unit
class TestListFixtures:
    """Test fixtures that provide lists of items."""

    def test_sample_papers_list_fixture(self, sample_papers_list):
        """Test sample papers list fixture."""
        assert isinstance(sample_papers_list, list)
        assert len(sample_papers_list) == 5
        assert all(isinstance(paper, Paper) for paper in sample_papers_list)
        assert all(validate_paper_model(paper) for paper in sample_papers_list)
        
        # Check that papers have different IDs
        paper_ids = [paper.paper_id for paper in sample_papers_list]
        assert len(set(paper_ids)) == 5  # All unique

    def test_sample_authors_list_fixture(self, sample_authors_list):
        """Test sample authors list fixture."""
        assert isinstance(sample_authors_list, list)
        assert len(sample_authors_list) == 3
        assert all(isinstance(author, Author) for author in sample_authors_list)
        assert all(validate_author_model(author) for author in sample_authors_list)
        
        # Check that authors have different IDs
        author_ids = [author.author_id for author in sample_authors_list]
        assert len(set(author_ids)) == 3  # All unique


@pytest.mark.unit
class TestConfigFixtures:
    """Test configuration fixtures."""

    def test_semantic_scholar_config_fixture(self, semantic_scholar_config):
        """Test Semantic Scholar config fixture."""
        assert isinstance(semantic_scholar_config, SemanticScholarConfig)
        assert semantic_scholar_config.base_url == "https://api.semanticscholar.org/graph/v1"
        assert semantic_scholar_config.requests_per_minute == 60
        assert semantic_scholar_config.timeout == 30

    def test_neo4j_config_fixture(self, neo4j_config):
        """Test Neo4j config fixture."""
        assert isinstance(neo4j_config, Neo4jConfig)
        assert neo4j_config.uri == "bolt://localhost:7687"
        assert neo4j_config.username == "neo4j"
        assert neo4j_config.database == "test_db"

    def test_ml_config_fixture(self, ml_config):
        """Test ML config fixture."""
        assert isinstance(ml_config, MLConfig)
        assert ml_config.model_path == "/tmp/test_model.pkl"
        assert ml_config.embedding_dim == 128
        assert ml_config.batch_size == 32


@pytest.mark.unit
class TestMockFixtures:
    """Test mock object fixtures."""

    def test_mock_neo4j_driver_fixture(self, mock_neo4j_driver):
        """Test mock Neo4j driver fixture."""
        assert hasattr(mock_neo4j_driver, 'session')
        session = mock_neo4j_driver.session()
        assert session is not None

    def test_mock_api_response_fixture(self, mock_api_response):
        """Test mock API response fixture."""
        assert hasattr(mock_api_response, 'status_code')
        assert hasattr(mock_api_response, 'json')
        assert mock_api_response.status_code == 200
        
        json_data = mock_api_response.json()
        assert "data" in json_data
        assert "total" in json_data

    def test_mock_requests_session_fixture(self, mock_requests_session):
        """Test mock requests session fixture."""
        assert hasattr(mock_requests_session, 'get')
        assert hasattr(mock_requests_session, 'post')


@pytest.mark.unit
class TestFileFixtures:
    """Test file and directory fixtures."""

    def test_temp_dir_fixture(self, temp_dir):
        """Test temporary directory fixture."""
        assert isinstance(temp_dir, str)
        assert os.path.exists(temp_dir)
        assert os.path.isdir(temp_dir)

    def test_temp_model_file_fixture(self, temp_model_file):
        """Test temporary model file fixture."""
        assert isinstance(temp_model_file, str)
        assert os.path.exists(temp_model_file)
        assert os.path.isfile(temp_model_file)
        
        # Check file has some content
        assert os.path.getsize(temp_model_file) > 0

    def test_temp_config_file_fixture(self, temp_config_file):
        """Test temporary config file fixture."""
        assert isinstance(temp_config_file, str)
        assert os.path.exists(temp_config_file)
        assert os.path.isfile(temp_config_file)
        
        # Check it's valid JSON
        import json
        with open(temp_config_file, 'r') as f:
            config_data = json.load(f)
        assert isinstance(config_data, dict)
        assert "api" in config_data


@pytest.mark.unit
class TestDatabaseFixtures:
    """Test database-related fixtures."""

    def test_sample_neo4j_records_fixture(self, sample_neo4j_records):
        """Test sample Neo4j records fixture."""
        assert isinstance(sample_neo4j_records, list)
        assert len(sample_neo4j_records) == 2
        
        for record in sample_neo4j_records:
            assert isinstance(record, dict)
            assert "paper" in record
            assert "author" in record
            assert "paperId" in record["paper"]
            assert "authorId" in record["author"]

    def test_mock_database_manager_fixture(self, mock_database_manager):
        """Test mock database manager fixture."""
        # Just verify it was created without error
        assert mock_database_manager is not None
        assert hasattr(mock_database_manager, 'config')


@pytest.mark.unit
class TestAPIFixtures:
    """Test API-related fixtures."""

    def test_mock_api_client_fixture(self, mock_api_client):
        """Test mock API client fixture."""
        assert mock_api_client is not None
        assert hasattr(mock_api_client, 'config')
        assert hasattr(mock_api_client, 'session')

    def test_api_error_response_fixture(self, api_error_response):
        """Test API error response fixture."""
        assert hasattr(api_error_response, 'status_code')
        assert api_error_response.status_code == 429
        
        json_data = api_error_response.json()
        assert "error" in json_data

    def test_network_error_fixture(self, network_error):
        """Test network error fixture."""
        assert isinstance(network_error, ConnectionError)


@pytest.mark.unit
class TestMLFixtures:
    """Test ML-related fixtures."""

    def test_sample_embeddings_matrix_fixture(self, sample_embeddings_matrix):
        """Test sample embeddings matrix fixture."""
        assert isinstance(sample_embeddings_matrix, np.ndarray)
        assert sample_embeddings_matrix.shape == (10, 128)
        assert sample_embeddings_matrix.dtype == np.float64

    def test_sample_predictions_data_fixture(self, sample_predictions_data):
        """Test sample predictions data fixture."""
        assert isinstance(sample_predictions_data, list)
        assert len(sample_predictions_data) == 2
        
        for prediction in sample_predictions_data:
            assert isinstance(prediction, dict)
            assert "source_paper_id" in prediction
            assert "target_paper_id" in prediction
            assert "score" in prediction
            assert 0.0 <= prediction["score"] <= 1.0


@pytest.mark.unit
class TestPerformanceFixtures:
    """Test performance-related fixtures."""

    def test_performance_timer_fixture(self, performance_timer):
        """Test performance timer fixture."""
        assert hasattr(performance_timer, 'start')
        assert hasattr(performance_timer, 'stop')
        assert hasattr(performance_timer, 'elapsed')
        
        # Test basic functionality
        performance_timer.start()
        import time
        time.sleep(0.01)  # Sleep for 10ms
        performance_timer.stop()
        
        assert performance_timer.elapsed.total_seconds() > 0


@pytest.mark.unit
class TestIntegrationFixtures:
    """Test integration testing fixtures."""

    def test_integration_test_data_fixture(self, integration_test_data):
        """Test integration test data fixture."""
        assert isinstance(integration_test_data, dict)
        assert "search_query" in integration_test_data
        assert "expected_results" in integration_test_data
        assert "test_paper_ids" in integration_test_data
        assert isinstance(integration_test_data["test_paper_ids"], list)


@pytest.mark.unit 
class TestParametrizedFixtures:
    """Test parametrized fixtures."""

    def test_ml_model_params_fixture(self, ml_model_params):
        """Test ML model params fixture."""
        assert isinstance(ml_model_params, dict)
        assert "model_name" in ml_model_params
        assert "dim" in ml_model_params
        assert ml_model_params["model_name"] in ["TransE", "ComplEx", "RotatE"]

    def test_node_types_fixture(self, node_types):
        """Test node types fixture."""
        assert node_types in [NodeType.PAPER, NodeType.AUTHOR, NodeType.VENUE]

    def test_edge_types_fixture(self, edge_types):
        """Test edge types fixture."""
        assert edge_types in [EdgeType.CITES, EdgeType.AUTHORED, EdgeType.PUBLISHED_IN]


@pytest.mark.integration
class TestFixtureIntegration:
    """Test that fixtures work together correctly."""

    def test_paper_and_embedding_integration(self, sample_paper, sample_paper_embedding):
        """Test that paper and embedding fixtures work together."""
        # Both should reference the same paper ID
        assert sample_paper.paper_id == sample_paper_embedding.paper_id

    def test_prediction_integration(self, sample_papers_list, sample_citation_prediction):
        """Test prediction works with paper list."""
        paper_ids = [paper.paper_id for paper in sample_papers_list]
        
        # Prediction should reference valid papers (at least source should exist)
        # Note: target might not be in our sample list, which is fine
        # We're just checking the integration works without errors
        assert sample_citation_prediction.source_paper_id is not None
        assert sample_citation_prediction.target_paper_id is not None

    def test_config_and_client_integration(self, semantic_scholar_config, mock_api_client):
        """Test config and client fixtures work together."""
        assert mock_api_client.config.base_url == semantic_scholar_config.base_url
        assert mock_api_client.config.timeout == semantic_scholar_config.timeout

    def test_database_fixtures_integration(self, mock_database_manager, sample_neo4j_records):
        """Test database fixtures work together."""
        # Verify manager can be used with sample records
        assert mock_database_manager is not None
        assert len(sample_neo4j_records) > 0
        
        # This is a basic integration check - more detailed tests would be in 
        # actual database test files


if __name__ == "__main__":
    pytest.main([__file__, "-v"])