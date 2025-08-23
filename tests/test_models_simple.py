"""
Simple test for the new ML and Network models.
Tests the models we added without touching the existing legacy models.
"""

import pytest
import numpy as np

from src.models.ml import PaperEmbedding, CitationPrediction, ModelType, PredictionConfidence
from src.models.network import NetworkNode, NetworkEdge, NodeType, EdgeType
from src.models.api import APIResponse, APIStatus


def test_paper_embedding_basic():
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


def test_citation_prediction():
    """Test CitationPrediction model."""
    prediction = CitationPrediction(
        source_paper_id="source123",
        target_paper_id="target456", 
        prediction_score=0.85,
        model_name="TransE"
    )
    
    assert prediction.source_paper_id == "source123"
    assert prediction.prediction_score == 0.85
    assert prediction.confidence_level == PredictionConfidence.HIGH
    assert prediction.is_positive_prediction is True


def test_network_node():
    """Test NetworkNode model."""
    node = NetworkNode(
        id="paper123",
        label="Test Paper Node",
        node_type=NodeType.PAPER,
        citation_count=25
    )
    
    assert node.id == "paper123"
    assert node.node_type == NodeType.PAPER
    assert node.citation_count == 25


def test_network_edge():
    """Test NetworkEdge model."""
    edge = NetworkEdge(
        source="paper1",
        target="paper2",
        edge_type=EdgeType.CITES,
        weight=1.5
    )
    
    assert edge.source == "paper1"
    assert edge.target == "paper2"
    assert edge.edge_type == EdgeType.CITES


def test_api_response():
    """Test APIResponse model."""
    data = {"test": "data"}
    response = APIResponse.success(data, message="Success")
    
    assert response.status == APIStatus.SUCCESS
    assert response.data == data
    assert response.message == "Success"


if __name__ == "__main__":
    pytest.main([__file__])