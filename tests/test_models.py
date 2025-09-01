"""
Test suite for unified data models.

Tests all the models including ML, network, and API models
to ensure proper validation and functionality.
"""

import pytest
import numpy as np
from datetime import datetime
from typing import List
from pydantic import ValidationError

from src.models import (
    # Core models
    Paper, PaperCreate, PaperUpdate,
    
    # ML models
    PaperEmbedding, CitationPrediction, ModelMetadata, TrainingConfig, EvaluationMetrics,
    BatchPredictionRequest, BatchPredictionResponse, ModelType, PredictionConfidence,
    
    # Network models
    NetworkNode, NetworkEdge, NetworkGraph, VisualizationConfig, NetworkAnalysis,
    NodeType, EdgeType, VisualizationBackend, LayoutAlgorithm, NodeSize, EdgeWidth,
    
    # API models
    APIResponse, PaginatedResponse, APIError, SearchRequest, BatchRequest,
    CitationSearchRequest, PredictionRequest, NetworkAnalysisRequest,
    PaginationParams, PaginationMeta, APIStatus, ErrorType
)


class TestCoreModels:
    """Test core entity models."""
    
    def test_paper_model_basic(self):
        """Test basic Paper model functionality."""
        paper = Paper(
            paper_id="12345",
            title="Test Paper",
            abstract="This is a test abstract",
            year=2023,
            citation_count=42
        )
        
        assert paper.paper_id == "12345"
        assert paper.title == "Test Paper"
        assert paper.has_abstract is True
        assert paper.is_highly_cited(threshold=40) is True
        assert paper.is_highly_cited(threshold=50) is False
    
    def test_paper_from_semantic_scholar(self):
        """Test Paper creation from Semantic Scholar API response."""
        api_data = {
            "paperId": "67890",
            "title": "Another Test Paper",
            "abstract": "Test abstract",
            "year": 2022,
            "citationCount": 15,
            "authors": [{"name": "John Doe"}, {"name": "Jane Smith"}],
            "venue": "Test Conference",
            "fieldsOfStudy": ["Computer Science", "Machine Learning"]
        }
        
        paper = Paper.from_semantic_scholar_response(api_data)
        
        assert paper.paper_id == "67890"
        assert paper.title == "Another Test Paper"
        assert len(paper.authors) == 2
        assert "John Doe" in paper.authors
        assert len(paper.venues) == 1
        assert "Test Conference" in paper.venues
        assert len(paper.fields) == 2
    
    def test_paper_validation(self):
        """Test Paper model validation.""" 
        # Test invalid year
        with pytest.raises((ValueError, ValidationError)):
            Paper(paper_id="123", title="Test", year=1800)
        
        # Test empty title
        with pytest.raises((ValueError, ValidationError)):
            Paper(paper_id="123", title="")


class TestMLModels:
    """Test machine learning related models."""
    
    def test_paper_embedding_basic(self):
        """Test PaperEmbedding model functionality."""
        embedding = PaperEmbedding(
            paper_id="test_paper",
            embedding=[0.1, 0.2, 0.3, 0.4],
            model_name="TransE",
            embedding_dim=4
        )
        
        assert embedding.paper_id == "test_paper"
        assert len(embedding.embedding) == 4
        assert embedding.embedding_dim == 4
        
        # Test numpy conversion
        np_array = embedding.to_numpy()
        assert isinstance(np_array, np.ndarray)
        assert len(np_array) == 4
    
    def test_paper_embedding_from_numpy(self):
        """Test PaperEmbedding creation from numpy array."""
        np_embedding = np.array([0.5, 0.6, 0.7])
        
        embedding = PaperEmbedding.from_numpy(
            paper_id="numpy_test",
            embedding=np_embedding,
            model_name="TestModel"
        )
        
        assert embedding.paper_id == "numpy_test"
        assert embedding.embedding_dim == 3
        assert embedding.embedding == [0.5, 0.6, 0.7]
    
    def test_paper_embedding_cosine_similarity(self):
        """Test cosine similarity calculation."""
        embedding1 = PaperEmbedding(
            paper_id="p1",
            embedding=[1.0, 0.0, 0.0],
            model_name="test",
            embedding_dim=3
        )
        
        embedding2 = PaperEmbedding(
            paper_id="p2", 
            embedding=[0.0, 1.0, 0.0],
            model_name="test",
            embedding_dim=3
        )
        
        # Orthogonal vectors should have 0 similarity
        similarity = embedding1.cosine_similarity(embedding2)
        assert abs(similarity) < 1e-10
        
        # Identical vectors should have similarity 1
        similarity_self = embedding1.cosine_similarity(embedding1)
        assert abs(similarity_self - 1.0) < 1e-10
    
    def test_citation_prediction_model(self):
        """Test CitationPrediction model functionality."""
        prediction = CitationPrediction(
            source_paper_id="source123",
            target_paper_id="target456", 
            prediction_score=0.85,
            model_name="TransE",
            source_title="Source Paper",
            target_title="Target Paper"
        )
        
        assert prediction.source_paper_id == "source123"
        assert prediction.target_paper_id == "target456"
        assert prediction.prediction_score == 0.85
        assert prediction.confidence_level == PredictionConfidence.HIGH
        assert prediction.is_positive_prediction() is True
    
    def test_training_config_validation(self):
        """Test TrainingConfig validation."""
        # Valid config
        config = TrainingConfig(
            embedding_dim=128,
            epochs=100,
            batch_size=512,
            learning_rate=0.01
        )
        
        assert config.embedding_dim == 128
        assert config.epochs == 100
        
        # Test invalid learning rate
        with pytest.raises(ValueError, match="Learning rate must be between"):
            TrainingConfig(learning_rate=1.5)
        
        # Test invalid split ratio
        with pytest.raises(ValueError, match="Train and validation splits"):
            TrainingConfig(train_test_split=0.9, validation_split=0.2)
    
    def test_evaluation_metrics_comparison(self):
        """Test EvaluationMetrics comparison functionality."""
        metrics1 = EvaluationMetrics(
            model_name="Model1",
            model_version="1.0",
            evaluation_set="test",
            num_test_edges=1000,
            num_entities=500,
            mean_reciprocal_rank=0.3
        )
        
        metrics2 = EvaluationMetrics(
            model_name="Model2", 
            model_version="1.0",
            evaluation_set="test",
            num_test_edges=1000,
            num_entities=500,
            mean_reciprocal_rank=0.25
        )
        
        assert metrics1.is_better_than(metrics2) is True
        assert metrics2.is_better_than(metrics1) is False


class TestNetworkModels:
    """Test network visualization models."""
    
    def test_network_node_basic(self):
        """Test NetworkNode model functionality."""
        node = NetworkNode(
            id="paper123",
            label="Test Paper Node",
            node_type=NodeType.PAPER,
            title="Full Paper Title",
            citation_count=25,
            year=2023,
            authors=["Author 1", "Author 2"]
        )
        
        assert node.id == "paper123"
        assert node.node_type == NodeType.PAPER
        assert node.citation_count == 25
        
        # Test display label truncation
        long_label = "A" * 100
        node.label = long_label
        display_label = node.get_display_label(max_length=20)
        assert len(display_label) <= 20
        assert display_label.endswith("...")
    
    def test_network_edge_basic(self):
        """Test NetworkEdge model functionality."""
        edge = NetworkEdge(
            source="paper1",
            target="paper2",
            edge_type=EdgeType.CITES,
            weight=1.5,
            confidence=0.8,
            source_label="Source Paper",
            target_label="Target Paper"
        )
        
        assert edge.source == "paper1"
        assert edge.target == "paper2"
        assert edge.edge_type == EdgeType.CITES
        assert edge.weight == 1.5
        
        # Test edge ID generation
        edge_id = edge.get_edge_id()
        assert "paper1" in edge_id
        assert "CITES" in edge_id
        assert "paper2" in edge_id
    
    def test_network_graph_basic(self):
        """Test NetworkGraph model functionality."""
        nodes = [
            NetworkNode(id="n1", label="Node 1", node_type=NodeType.PAPER),
            NetworkNode(id="n2", label="Node 2", node_type=NodeType.PAPER),
            NetworkNode(id="n3", label="Node 3", node_type=NodeType.AUTHOR)
        ]
        
        edges = [
            NetworkEdge(source="n1", target="n2", edge_type=EdgeType.CITES),
            NetworkEdge(source="n2", target="n1", edge_type=EdgeType.CITES)
        ]
        
        graph = NetworkGraph(
            nodes=nodes,
            edges=edges,
            name="Test Graph"
        )
        
        assert graph.num_nodes == 3
        assert graph.num_edges == 2
        assert graph.density is not None
        
        # Test node retrieval
        node = graph.get_node_by_id("n1")
        assert node is not None
        assert node.label == "Node 1"
        
        # Test neighbor finding
        neighbors = graph.get_neighbors("n1")
        assert "n2" in neighbors
    
    def test_network_graph_filtering(self):
        """Test NetworkGraph filtering functionality."""
        nodes = [
            NetworkNode(id="p1", label="Paper 1", node_type=NodeType.PAPER),
            NetworkNode(id="p2", label="Paper 2", node_type=NodeType.PAPER),
            NetworkNode(id="a1", label="Author 1", node_type=NodeType.AUTHOR)
        ]
        
        edges = [
            NetworkEdge(source="p1", target="p2", edge_type=EdgeType.CITES),
            NetworkEdge(source="a1", target="p1", edge_type=EdgeType.AUTHORED)
        ]
        
        graph = NetworkGraph(nodes=nodes, edges=edges)
        
        # Filter to only papers
        paper_graph = graph.filter_by_node_type([NodeType.PAPER])
        
        assert paper_graph.num_nodes == 2
        assert paper_graph.num_edges == 1  # Only paper-to-paper edges remain
    
    def test_visualization_config_validation(self):
        """Test VisualizationConfig validation."""
        config = VisualizationConfig(
            backend=VisualizationBackend.PYVIS,
            width=1000,
            height=800,
            min_node_size=5.0,
            max_node_size=50.0
        )
        
        assert config.backend == VisualizationBackend.PYVIS
        assert config.width == 1000
        
        # Test invalid size configuration
        with pytest.raises(ValueError, match="Maximum node size must be"):
            VisualizationConfig(min_node_size=60.0, max_node_size=50.0)


class TestAPIModels:
    """Test API request and response models."""
    
    def test_api_response_success(self):
        """Test successful API response creation."""
        data = {"test": "data"}
        response = APIResponse.success(data, message="Success")
        
        assert response.status == APIStatus.SUCCESS
        assert response.data == data
        assert response.message == "Success"
        assert response.errors is None
    
    def test_api_response_error(self):
        """Test error API response creation."""
        error = APIError.validation_error("Invalid input", field="title")
        response = APIResponse.error(error, message="Validation failed")
        
        assert response.status == APIStatus.ERROR
        assert response.data is None
        assert len(response.errors) == 1
        assert response.errors[0].error_type == ErrorType.VALIDATION_ERROR
    
    def test_paginated_response(self):
        """Test paginated response functionality."""
        data = [{"id": 1}, {"id": 2}, {"id": 3}]
        pagination = PaginationMeta(
            total_count=100,
            page_count=10,
            current_page=1,
            per_page=10,
            has_next=True,
            has_previous=False
        )
        
        response = PaginatedResponse.success(data, pagination)
        
        assert response.status == APIStatus.SUCCESS
        assert len(response.data) == 3
        assert response.pagination.total_count == 100
    
    def test_search_request_validation(self):
        """Test SearchRequest validation."""
        # Valid request
        request = SearchRequest(
            query="machine learning",
            min_year=2020,
            max_year=2023
        )
        
        assert request.query == "machine learning"
        assert request.min_year == 2020
        
        # Test invalid year range
        with pytest.raises(ValueError, match="max_year must be"):
            SearchRequest(query="test", min_year=2025, max_year=2020)
        
        # Test empty query
        with pytest.raises(ValueError, match="Query cannot be empty"):
            SearchRequest(query="   ")
    
    def test_citation_search_request(self):
        """Test CitationSearchRequest functionality."""
        request = CitationSearchRequest(
            query="neural networks",
            min_citations=10,
            max_citations=1000,
            citation_depth=2,
            year_range=(2020, 2023),
            authors=["John Doe", "Jane Smith"]
        )
        
        assert request.query == "neural networks"
        assert request.citation_depth == 2
        assert len(request.authors) == 2
        
        # Test validation
        with pytest.raises(ValueError, match="max_citations must be"):
            CitationSearchRequest(min_citations=100, max_citations=50)
    
    def test_prediction_request(self):
        """Test PredictionRequest functionality."""
        request = PredictionRequest(
            source_paper_ids=["paper1", "paper2"],
            target_paper_ids=["paper3", "paper4"],
            model_name="TransE",
            top_k=5,
            confidence_threshold=0.7
        )
        
        assert len(request.source_paper_ids) == 2
        assert request.top_k == 5
        assert request.confidence_threshold == 0.7
        
        # Test validation
        with pytest.raises(ValueError, match="Must provide at least one"):
            PredictionRequest(source_paper_ids=[])
    
    def test_pagination_params(self):
        """Test PaginationParams functionality."""
        params = PaginationParams(
            offset=20,
            limit=50,
            sort_by="citation_count"
        )
        
        assert params.offset == 20
        assert params.limit == 50
        
        # Test limit validation
        with pytest.raises(ValueError, match="Limit cannot exceed"):
            PaginationParams(limit=2000)
    
    def test_pagination_meta_from_params(self):
        """Test PaginationMeta creation from params."""
        params = PaginationParams(offset=50, limit=25)
        meta = PaginationMeta.from_params(params, total_count=200)
        
        assert meta.total_count == 200
        assert meta.current_page == 3  # (50 / 25) + 1
        assert meta.per_page == 25
        assert meta.has_next is True
        assert meta.has_previous is True


class TestModelIntegration:
    """Test integration between different model types."""
    
    def test_paper_to_network_node_conversion(self):
        """Test converting Paper to NetworkNode."""
        paper = Paper(
            paper_id="integration_test",
            title="Integration Test Paper",
            citation_count=30,
            year=2023,
            authors=["Test Author"]
        )
        
        # Convert to network node
        node = NetworkNode(
            id=paper.paper_id,
            label=paper.title,
            node_type=NodeType.PAPER,
            title=paper.title,
            citation_count=paper.citation_count,
            year=paper.year,
            authors=paper.authors
        )
        
        assert node.id == paper.paper_id
        assert node.label == paper.title
        assert node.citation_count == paper.citation_count
    
    def test_embedding_to_prediction_workflow(self):
        """Test workflow from embedding to prediction."""
        # Create embeddings
        embedding1 = PaperEmbedding(
            paper_id="paper1",
            embedding=[0.1, 0.2, 0.3],
            model_name="TransE",
            embedding_dim=3
        )
        
        embedding2 = PaperEmbedding(
            paper_id="paper2",
            embedding=[0.4, 0.5, 0.6],
            model_name="TransE", 
            embedding_dim=3
        )
        
        # Calculate similarity (mock prediction score)
        similarity = embedding1.cosine_similarity(embedding2)
        
        # Create prediction
        prediction = CitationPrediction(
            source_paper_id=embedding1.paper_id,
            target_paper_id=embedding2.paper_id,
            prediction_score=abs(similarity),  # Use absolute value
            model_name="TransE"
        )
        
        assert prediction.source_paper_id == "paper1"
        assert prediction.target_paper_id == "paper2"
        assert 0.0 <= prediction.prediction_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])