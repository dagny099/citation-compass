"""
Validation tests for the Academic Citation Platform.

These tests validate:
- Data integrity and consistency
- Model correctness and behavior
- API response validation
- Configuration validation
- Security and input sanitization
"""

import pytest
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Optional
import json
import re
from pathlib import Path

from src.models.paper import Paper, PaperCreate
from src.models.author import Author, AuthorCreate
from src.models.venue import Venue, VenueCreate
from src.models.ml import (
    PaperEmbedding, 
    CitationPrediction, 
    ModelMetadata,
    PredictionConfidence,
    ModelType
)
from src.models.network import NetworkNode, NetworkEdge, NodeType, EdgeType
from src.models.api import APIResponse, APIStatus, SearchRequest
from src.data.api_config import (
    SemanticScholarConfig, 
    Neo4jConfig, 
    MLConfig, 
    PlatformConfig
)


@pytest.mark.validation
class TestDataModelValidation:
    """Validate data model integrity and constraints."""
    
    def test_paper_model_validation(self):
        """Test Paper model validation rules."""
        # Valid paper
        paper = Paper(
            paper_id="valid_id_123",
            title="A Valid Research Paper Title",
            abstract="This is a valid abstract with sufficient content for testing.",
            year=2023,
            authors=["Author One", "Author Two"],
            citation_count=42,
            reference_count=25
        )
        
        assert paper.paper_id == "valid_id_123"
        assert paper.year == 2023
        assert len(paper.authors) == 2
        
        # Test validation constraints
        with pytest.raises(ValueError):
            Paper(
                paper_id="",  # Empty ID should fail
                title="Title",
                year=2023
            )
        
        with pytest.raises(ValueError):
            Paper(
                paper_id="valid_id",
                title="",  # Empty title should fail
                year=2023
            )
        
        with pytest.raises(ValueError):
            Paper(
                paper_id="valid_id",
                title="Title",
                year=1800  # Year too early should fail
            )
        
        with pytest.raises(ValueError):
            Paper(
                paper_id="valid_id",
                title="Title", 
                year=2030,
                citation_count=-1  # Negative citations should fail
            )
    
    def test_author_model_validation(self):
        """Test Author model validation rules."""
        # Valid author
        author = Author(
            name="Dr. Jane Smith",
            author_id="author123",
            paper_count=50,
            citation_count=1500,
            h_index=25
        )
        
        assert author.display_name == "Dr. Jane Smith"
        assert author.is_prolific(threshold=25) is True
        assert author.is_highly_cited(threshold=1000) is True
        
        # Test validation constraints
        with pytest.raises(ValueError):
            Author(name="")  # Empty name should fail
        
        with pytest.raises(ValueError):
            Author(
                name="Valid Name",
                paper_count=-1  # Negative count should fail
            )
        
        with pytest.raises(ValueError):
            Author(
                name="Valid Name",
                h_index=-5  # Negative h-index should fail
            )
    
    def test_venue_model_validation(self):
        """Test Venue model validation rules."""
        # Valid venue
        venue = Venue(
            name="International Conference on Machine Learning",
            venue_type="Conference",
            paper_count=1000,
            total_citations=25000
        )
        
        assert venue.name == "International Conference on Machine Learning"
        assert venue.venue_type == "Conference"
        
        # Test validation constraints
        with pytest.raises(ValueError):
            Venue(name="")  # Empty name should fail
        
        with pytest.raises(ValueError):
            Venue(
                name="Valid Venue",
                paper_count=-1  # Negative count should fail
            )
    
    def test_ml_model_validation(self):
        """Test ML model validation rules."""
        # Valid paper embedding
        embedding = PaperEmbedding(
            paper_id="paper123",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            model_name="TransE",
            embedding_dim=5
        )
        
        assert len(embedding.embedding) == embedding.embedding_dim
        assert embedding.model_name == "TransE"
        
        # Valid citation prediction
        prediction = CitationPrediction(
            source_paper_id="paper1",
            target_paper_id="paper2",
            prediction_score=0.85,
            model_name="TransE"
        )
        
        assert prediction.confidence_level == PredictionConfidence.HIGH
        assert prediction.is_positive_prediction is True
        
        # Test validation constraints
        with pytest.raises(ValueError):
            PaperEmbedding(
                paper_id="",  # Empty paper_id should fail
                embedding=[0.1, 0.2],
                model_name="TransE",
                embedding_dim=2
            )
        
        with pytest.raises(ValueError):
            CitationPrediction(
                source_paper_id="paper1",
                target_paper_id="paper2", 
                prediction_score=1.5,  # Score > 1.0 should fail
                model_name="TransE"
            )
        
        with pytest.raises(ValueError):
            CitationPrediction(
                source_paper_id="paper1",
                target_paper_id="paper2",
                prediction_score=-0.1,  # Negative score should fail
                model_name="TransE"
            )
    
    def test_network_model_validation(self):
        """Test Network model validation rules."""
        # Valid network node
        node = NetworkNode(
            id="node123",
            label="Test Node",
            node_type=NodeType.PAPER,
            citation_count=10
        )
        
        assert node.id == "node123"
        assert node.node_type == NodeType.PAPER
        
        # Valid network edge
        edge = NetworkEdge(
            source="node1",
            target="node2",
            edge_type=EdgeType.CITES,
            weight=1.0
        )
        
        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.edge_type == EdgeType.CITES
        
        # Test validation constraints
        with pytest.raises(ValueError):
            NetworkNode(
                id="",  # Empty ID should fail
                label="Label",
                node_type=NodeType.PAPER
            )
        
        with pytest.raises(ValueError):
            NetworkEdge(
                source="node1",
                target="node1",  # Self-loop might be restricted
                edge_type=EdgeType.CITES,
                weight=-1.0  # Negative weight should fail
            )


@pytest.mark.validation
class TestConfigurationValidation:
    """Validate configuration integrity and security."""
    
    def test_semantic_scholar_config_validation(self):
        """Test Semantic Scholar configuration validation."""
        # Valid configuration
        config = SemanticScholarConfig(
            base_url="https://api.semanticscholar.org/graph/v1",
            requests_per_minute=60,
            timeout=30
        )
        
        assert config.base_url.startswith("https://")
        assert config.requests_per_minute > 0
        assert config.timeout > 0
        
        # Test invalid configurations
        with pytest.raises(ValueError):
            SemanticScholarConfig(
                base_url="http://unsecure.api.com",  # Should prefer HTTPS
                requests_per_minute=0,  # Should be positive
                timeout=-1  # Should be positive
            )
    
    def test_neo4j_config_validation(self):
        """Test Neo4j configuration validation."""
        # Valid configuration
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="secure_password",
            database="test_db"
        )
        
        assert config.validate() is True
        
        # Missing parameters
        incomplete_config = Neo4jConfig(
            uri="bolt://localhost:7687"
            # Missing username and password
        )
        
        assert incomplete_config.validate() is False
        missing = incomplete_config.get_missing_params()
        assert len(missing) > 0
    
    def test_ml_config_validation(self):
        """Test ML configuration validation."""
        config = MLConfig(
            embedding_dim=128,
            batch_size=32,
            prediction_threshold=0.5
        )
        
        assert config.embedding_dim > 0
        assert config.batch_size > 0
        assert 0.0 <= config.prediction_threshold <= 1.0
        
        # Test invalid configurations
        with pytest.raises(ValueError):
            MLConfig(
                embedding_dim=0,  # Should be positive
                batch_size=-1,  # Should be positive  
                prediction_threshold=1.5  # Should be <= 1.0
            )
    
    def test_platform_config_integration(self):
        """Test platform configuration integration."""
        # Test with overrides
        overrides = {
            "semantic_scholar": {"requests_per_minute": 100},
            "ml": {"embedding_dim": 256}
        }
        
        config = PlatformConfig(overrides)
        
        assert config.semantic_scholar.requests_per_minute == 100
        assert config.ml.embedding_dim == 256
        
        # Test configuration consistency
        api_config = config.get_api_config()
        ml_config = config.get_ml_config()
        
        assert api_config.requests_per_minute == 100
        assert ml_config.embedding_dim == 256


@pytest.mark.validation 
class TestDataIntegrityValidation:
    """Validate data integrity and consistency."""
    
    def test_embedding_consistency(self):
        """Test embedding vector consistency."""
        # Create embedding
        embedding_vector = np.random.rand(128).tolist()
        
        embedding = PaperEmbedding(
            paper_id="test_paper",
            embedding=embedding_vector,
            model_name="TransE",
            embedding_dim=128
        )
        
        # Test consistency
        assert len(embedding.embedding) == embedding.embedding_dim
        assert all(isinstance(x, (int, float)) for x in embedding.embedding)
        
        # Test numpy conversion
        np_array = embedding.to_numpy()
        assert isinstance(np_array, np.ndarray)
        assert np_array.shape == (128,)
        assert np.allclose(np_array, embedding_vector)
    
    def test_prediction_consistency(self):
        """Test prediction consistency and constraints."""
        predictions = [
            CitationPrediction(
                source_paper_id="paper1",
                target_paper_id=f"target_{i}",
                prediction_score=0.9 - i * 0.1,
                model_name="TransE"
            )
            for i in range(5)
        ]
        
        # Test score ordering
        scores = [p.prediction_score for p in predictions]
        assert scores == sorted(scores, reverse=True)
        
        # Test score bounds
        for pred in predictions:
            assert 0.0 <= pred.prediction_score <= 1.0
            
        # Test confidence levels
        high_conf_count = sum(1 for p in predictions 
                             if p.confidence_level == PredictionConfidence.HIGH)
        assert high_conf_count > 0  # At least one should be high confidence
    
    def test_citation_network_consistency(self):
        """Test citation network data consistency."""
        # Create test network
        papers = ["paper_1", "paper_2", "paper_3"]
        
        # Create citation edges (should be acyclic in ideal case)
        edges = [
            ("paper_1", "paper_2"),
            ("paper_1", "paper_3"),
            ("paper_2", "paper_3")
        ]
        
        # Test network properties
        sources = set(edge[0] for edge in edges)
        targets = set(edge[1] for edge in edges)
        
        # All papers should be covered
        all_papers_in_network = sources.union(targets)
        assert len(all_papers_in_network) == len(papers)
        
        # Test no self-citations
        for source, target in edges:
            assert source != target
    
    def test_temporal_consistency(self):
        """Test temporal data consistency."""
        # Create papers with different years
        papers = [
            Paper(
                paper_id=f"paper_{i}",
                title=f"Paper {i}",
                year=2020 + i
            )
            for i in range(5)
        ]
        
        # Test year progression
        years = [p.year for p in papers]
        assert years == sorted(years)
        
        # Test realistic year bounds
        for paper in papers:
            assert 1990 <= paper.year <= 2025


@pytest.mark.validation
class TestSecurityValidation:
    """Validate security aspects and input sanitization."""
    
    def test_input_sanitization(self):
        """Test input sanitization for common attacks."""
        # Test SQL injection patterns
        malicious_inputs = [
            "'; DROP TABLE papers; --",
            "' OR 1=1 --",
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "../../../etc/passwd",
            "{{7*7}}"  # Template injection
        ]
        
        for malicious_input in malicious_inputs:
            # Paper title should sanitize input
            with pytest.raises((ValueError, TypeError)):
                Paper(
                    paper_id=malicious_input,
                    title="Valid Title",
                    year=2023
                )
    
    def test_api_key_handling(self):
        """Test secure API key handling."""
        config = SemanticScholarConfig(api_key="secret_key_12345")
        
        # API key should not appear in string representation
        config_str = str(config)
        assert "secret_key_12345" not in config_str
        
        # API key should be redacted in JSON serialization
        config_dict = config.__dict__.copy()
        if 'api_key' in config_dict and config_dict['api_key']:
            # In production, this should be redacted
            pass  # Would test redaction logic
    
    def test_path_traversal_protection(self):
        """Test protection against path traversal attacks."""
        from src.services.ml_service import TransEModelService
        
        # Test various path traversal attempts
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM"
        ]
        
        for malicious_path in malicious_paths:
            with pytest.raises((ValueError, FileNotFoundError, PermissionError)):
                service = TransEModelService(
                    model_path=Path(malicious_path),
                    entity_mapping_path=Path("test.pkl"),
                    metadata_path=Path("test.pkl")
                )
                service.ensure_loaded()  # This should fail safely
    
    def test_resource_limits(self):
        """Test resource usage limits."""
        # Test embedding vector size limits
        oversized_embedding = [0.1] * 10000  # Very large embedding
        
        with pytest.warns(UserWarning):  # Should warn about large embedding
            PaperEmbedding(
                paper_id="test",
                embedding=oversized_embedding,
                model_name="Test",
                embedding_dim=10000
            )


@pytest.mark.validation
class TestBusinessLogicValidation:
    """Validate business logic and domain rules."""
    
    def test_citation_prediction_logic(self):
        """Test citation prediction business logic."""
        # High confidence predictions should have high scores
        high_conf_pred = CitationPrediction(
            source_paper_id="paper1",
            target_paper_id="paper2",
            prediction_score=0.95,
            model_name="TransE"
        )
        
        assert high_conf_pred.confidence_level == PredictionConfidence.HIGH
        assert high_conf_pred.is_positive_prediction is True
        
        # Low confidence predictions
        low_conf_pred = CitationPrediction(
            source_paper_id="paper1", 
            target_paper_id="paper3",
            prediction_score=0.15,
            model_name="TransE"
        )
        
        assert low_conf_pred.confidence_level == PredictionConfidence.LOW
        assert low_conf_pred.is_positive_prediction is False
    
    def test_author_metrics_logic(self):
        """Test author metrics business logic."""
        # Prolific author
        prolific_author = Author(
            name="Prolific Researcher",
            paper_count=100,
            citation_count=5000,
            h_index=50
        )
        
        assert prolific_author.is_prolific(threshold=50) is True
        assert prolific_author.is_highly_cited(threshold=1000) is True
        
        # Early career researcher
        early_career = Author(
            name="Early Career Researcher",
            paper_count=5,
            citation_count=50,
            h_index=3
        )
        
        assert early_career.is_prolific(threshold=50) is False
        assert early_career.is_highly_cited(threshold=1000) is False
    
    def test_venue_quality_metrics(self):
        """Test venue quality assessment logic."""
        # High-quality venue
        high_quality_venue = Venue(
            name="Nature",
            venue_type="Journal",
            paper_count=1000,
            total_citations=100000,
            avg_citations_per_paper=100.0
        )
        
        assert high_quality_venue.avg_citations_per_paper >= 50.0
        
        # Calculate impact metrics
        if (high_quality_venue.paper_count and 
            high_quality_venue.total_citations):
            calculated_avg = (high_quality_venue.total_citations / 
                            high_quality_venue.paper_count)
            assert abs(calculated_avg - 100.0) < 0.01
    
    def test_network_analysis_logic(self):
        """Test network analysis business logic."""
        # Create test network nodes
        paper_nodes = [
            NetworkNode(
                id=f"paper_{i}",
                label=f"Paper {i}",
                node_type=NodeType.PAPER,
                citation_count=10 * i
            )
            for i in range(1, 6)
        ]
        
        # Test centrality-based importance
        citation_counts = [node.citation_count for node in paper_nodes]
        assert citation_counts == sorted(citation_counts)
        
        # Most cited paper should be most important
        most_cited = max(paper_nodes, key=lambda x: x.citation_count)
        assert most_cited.id == "paper_5"
        assert most_cited.citation_count == 50


@pytest.mark.validation
class TestPerformanceValidation:
    """Validate performance characteristics."""
    
    def test_prediction_performance_bounds(self):
        """Test prediction performance is within acceptable bounds."""
        import time
        
        # Mock a prediction operation
        start_time = time.time()
        
        # Simulate prediction computation
        prediction = CitationPrediction(
            source_paper_id="paper1",
            target_paper_id="paper2", 
            prediction_score=0.85,
            model_name="TransE"
        )
        
        end_time = time.time()
        
        # Object creation should be very fast
        creation_time = end_time - start_time
        assert creation_time < 0.001  # Less than 1ms
        
        # Test confidence calculation performance
        start_time = time.time()
        confidence = prediction.confidence_level
        end_time = time.time()
        
        calc_time = end_time - start_time
        assert calc_time < 0.001  # Less than 1ms
    
    def test_embedding_performance_bounds(self):
        """Test embedding operations performance."""
        import time
        
        # Test large embedding creation
        large_embedding = np.random.rand(512).tolist()
        
        start_time = time.time()
        embedding = PaperEmbedding(
            paper_id="test_paper",
            embedding=large_embedding,
            model_name="TransE",
            embedding_dim=512
        )
        end_time = time.time()
        
        creation_time = end_time - start_time
        assert creation_time < 0.01  # Less than 10ms
        
        # Test numpy conversion performance
        start_time = time.time()
        np_array = embedding.to_numpy()
        end_time = time.time()
        
        conversion_time = end_time - start_time
        assert conversion_time < 0.001  # Less than 1ms
    
    def test_memory_usage_validation(self):
        """Test memory usage is within acceptable bounds."""
        import sys
        
        # Test model size estimation
        paper = Paper(
            paper_id="test_paper",
            title="A" * 1000,  # Long title
            abstract="B" * 5000,  # Long abstract
            year=2023
        )
        
        paper_size = sys.getsizeof(paper)
        
        # Should be reasonable size (less than 10KB for single paper)
        assert paper_size < 10000
        
        # Test embedding memory usage
        embedding = PaperEmbedding(
            paper_id="test",
            embedding=[0.1] * 512,  # 512-dim embedding
            model_name="TransE",
            embedding_dim=512
        )
        
        embedding_size = sys.getsizeof(embedding)
        
        # Should be reasonable (less than 5KB)
        assert embedding_size < 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])