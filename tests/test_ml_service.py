"""
Tests for the ML service layer.

These tests validate the TransE model service functionality including
model loading, predictions, caching, and error handling.
"""

import pytest
import tempfile
import pickle
import torch
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.services.ml_service import (
    TransEModel,
    PredictionCache, 
    TransEModelService,
    get_ml_service,
    reset_ml_service
)
from src.models.ml import PaperEmbedding, CitationPrediction, ModelMetadata


@pytest.mark.ml
class TestTransEModel:
    """Test the TransE model implementation."""
    
    def test_model_initialization(self):
        """Test model can be initialized with correct parameters."""
        model = TransEModel(num_entities=100, embedding_dim=64)
        
        assert model.num_entities == 100
        assert model.embedding_dim == 64
        assert model.entity_embeddings.num_embeddings == 100
        assert model.entity_embeddings.embedding_dim == 64
        assert model.relation_embedding.num_embeddings == 1
        assert model.relation_embedding.embedding_dim == 64
    
    def test_forward_pass(self):
        """Test model forward pass produces correct output shapes."""
        model = TransEModel(num_entities=10, embedding_dim=8)
        
        sources = torch.tensor([0, 1, 2])
        targets = torch.tensor([3, 4, 5])
        
        scores = model.forward(sources, targets)
        
        assert scores.shape == (3,)
        assert torch.all(scores >= 0)  # Norms should be non-negative
    
    def test_predict_batch(self):
        """Test batch prediction functionality."""
        model = TransEModel(num_entities=10, embedding_dim=8)
        
        sources = torch.tensor([0, 1])
        candidates = torch.tensor([2, 3, 4])
        
        scores = model.predict_batch(sources, candidates)
        
        assert scores.shape == (2, 3)  # 2 sources x 3 candidates
        assert torch.all(scores >= 0)


@pytest.mark.ml
class TestPredictionCache:
    """Test the prediction caching functionality."""
    
    def setup_method(self):
        """Set up test cache."""
        self.cache = PredictionCache(max_size=5, ttl_seconds=60)
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        result = self.cache.get("paper123", top_k=10)
        assert result is None
    
    def test_cache_set_and_hit(self):
        """Test setting and retrieving cached predictions."""
        predictions = [
            CitationPrediction(
                source_paper_id="paper1",
                target_paper_id="paper2", 
                prediction_score=0.85,
                model_name="TransE"
            )
        ]
        
        # Set cache
        self.cache.set("paper123", 10, predictions)
        
        # Get from cache
        cached = self.cache.get("paper123", 10)
        assert cached is not None
        assert len(cached) == 1
        assert cached[0].source_paper_id == "paper1"
    
    def test_cache_expiry(self):
        """Test cache entries expire correctly."""
        cache = PredictionCache(max_size=5, ttl_seconds=0)  # Immediate expiry
        
        predictions = [
            CitationPrediction(
                source_paper_id="paper1",
                target_paper_id="paper2",
                prediction_score=0.85,
                model_name="TransE"
            )
        ]
        
        cache.set("paper123", 10, predictions)
        
        # Should be expired immediately
        result = cache.get("paper123", 10)
        assert result is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        predictions = [
            CitationPrediction(
                source_paper_id="paper1",
                target_paper_id="paper2",
                prediction_score=0.85, 
                model_name="TransE"
            )
        ]
        
        # Fill cache to capacity
        for i in range(5):
            self.cache.set(f"paper{i}", 10, predictions)
        
        # Add one more - should evict oldest
        self.cache.set("paper_new", 10, predictions)
        
        # First entry should be evicted
        assert self.cache.get("paper0", 10) is None
        # New entry should be present
        assert self.cache.get("paper_new", 10) is not None


@pytest.mark.ml
class TestTransEModelService:
    """Test the TransE model service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global service
        reset_ml_service()
        
        # Create temporary files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "test_model.pt"
        self.entity_mapping_path = Path(self.temp_dir) / "entity_mapping.pkl" 
        self.metadata_path = Path(self.temp_dir) / "metadata.json"
        
        # Create test entity mapping
        self.test_entity_mapping = {
            "paper1": 0,
            "paper2": 1, 
            "paper3": 2,
            "paper4": 3,
            "paper5": 4
        }
        
        with open(self.entity_mapping_path, 'wb') as f:
            pickle.dump(self.test_entity_mapping, f)
        
        # Create test metadata
        test_metadata = {
            "training_date": "2024-01-01",
            "num_epochs": 100,
            "loss": 0.15,
            "dataset": {
                "num_papers": 5,
                "num_citations": 10,
                "total_training_samples": 100
            },
            "model_config": {
                "embedding_dim": 8,
                "margin": 1.0,
                "learning_rate": 0.01,
                "norm_p": 1
            },
            "training_config": {
                "batch_size": 1024,
                "epochs": 100
            },
            "training_results": {
                "epochs_completed": 100,
                "final_loss": 0.15
            }
        }
        
        import json
        with open(self.metadata_path, 'w') as f:
            json.dump(test_metadata, f)
        
        # Create test model checkpoint
        model = TransEModel(num_entities=5, embedding_dim=8)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "num_entities": 5,
                "embedding_dim": 8,
                "margin": 1.0,
                "p_norm": 1
            }
        }
        
        torch.save(checkpoint, self.model_path)
    
    def test_service_initialization(self):
        """Test service initializes correctly."""
        service = TransEModelService(
            model_path=self.model_path,
            entity_mapping_path=self.entity_mapping_path,
            metadata_path=self.metadata_path,
            device="cpu"
        )
        
        assert service.model_path == self.model_path
        assert service.device == "cpu"
        assert service.cache is not None
    
    def test_lazy_loading(self):
        """Test components are loaded lazily."""
        service = TransEModelService(
            model_path=self.model_path,
            entity_mapping_path=self.entity_mapping_path,
            metadata_path=self.metadata_path
        )
        
        # Initially nothing loaded
        assert service.model is None
        assert service.entity_mapping is None
        
        # Loading triggered by ensure_loaded
        service.ensure_loaded()
        
        assert service.model is not None
        assert service.entity_mapping is not None
        assert len(service.entity_mapping) == 5
    
    def test_get_paper_embedding(self):
        """Test retrieving paper embeddings."""
        service = TransEModelService(
            model_path=self.model_path,
            entity_mapping_path=self.entity_mapping_path,
            metadata_path=self.metadata_path
        )
        
        # Test valid paper
        embedding = service.get_paper_embedding("paper1")
        assert embedding is not None
        assert isinstance(embedding, PaperEmbedding)
        assert embedding.paper_id == "paper1"
        assert embedding.model_name == "TransE"
        assert len(embedding.embedding) == 8  # embedding_dim
        
        # Test invalid paper
        embedding = service.get_paper_embedding("nonexistent")
        assert embedding is None
    
    def test_predict_citations(self):
        """Test citation prediction functionality."""
        service = TransEModelService(
            model_path=self.model_path,
            entity_mapping_path=self.entity_mapping_path,
            metadata_path=self.metadata_path
        )
        
        # Test prediction with valid source
        predictions = service.predict_citations("paper1", top_k=3)
        
        assert isinstance(predictions, list)
        assert len(predictions) <= 3
        
        for pred in predictions:
            assert isinstance(pred, CitationPrediction)
            assert pred.source_paper_id == "paper1"
            assert 0.0 <= pred.prediction_score <= 1.0
            assert pred.model_name == "TransE"
        
        # Should be sorted by score (highest first)
        if len(predictions) > 1:
            for i in range(1, len(predictions)):
                assert predictions[i-1].prediction_score >= predictions[i].prediction_score
    
    def test_predict_citations_with_candidates(self):
        """Test prediction with specific candidates."""
        service = TransEModelService(
            model_path=self.model_path,
            entity_mapping_path=self.entity_mapping_path,
            metadata_path=self.metadata_path
        )
        
        candidates = ["paper2", "paper3"]
        predictions = service.predict_citations("paper1", candidate_paper_ids=candidates)
        
        assert len(predictions) <= len(candidates)
        target_ids = [pred.target_paper_id for pred in predictions]
        
        for target_id in target_ids:
            assert target_id in candidates
    
    def test_predict_citations_invalid_source(self):
        """Test prediction with invalid source paper."""
        service = TransEModelService(
            model_path=self.model_path,
            entity_mapping_path=self.entity_mapping_path,
            metadata_path=self.metadata_path
        )
        
        predictions = service.predict_citations("nonexistent")
        assert predictions == []
    
    def test_batch_predictions(self):
        """Test batch prediction functionality."""
        service = TransEModelService(
            model_path=self.model_path,
            entity_mapping_path=self.entity_mapping_path,
            metadata_path=self.metadata_path
        )
        
        source_papers = ["paper1", "paper2"]
        results = service.batch_predict_citations(source_papers, top_k=2)
        
        assert isinstance(results, dict)
        assert len(results) == 2
        assert "paper1" in results
        assert "paper2" in results
        
        for paper_id, predictions in results.items():
            assert isinstance(predictions, list)
            assert len(predictions) <= 2
    
    def test_caching(self):
        """Test prediction caching works correctly."""
        service = TransEModelService(
            model_path=self.model_path,
            entity_mapping_path=self.entity_mapping_path,
            metadata_path=self.metadata_path,
            cache_predictions=True
        )
        
        # First call - should cache
        predictions1 = service.predict_citations("paper1", top_k=2)
        
        # Second call - should use cache
        predictions2 = service.predict_citations("paper1", top_k=2)
        
        # Should be identical (from cache)
        assert len(predictions1) == len(predictions2)
        for p1, p2 in zip(predictions1, predictions2):
            assert p1.target_paper_id == p2.target_paper_id
            assert p1.prediction_score == p2.prediction_score
    
    def test_get_model_info(self):
        """Test getting model metadata."""
        service = TransEModelService(
            model_path=self.model_path,
            entity_mapping_path=self.entity_mapping_path,
            metadata_path=self.metadata_path
        )
        
        info = service.get_model_info()
        assert isinstance(info, ModelMetadata)
        assert info.model_name == "TransE Citation Predictor"
        assert info.embedding_dim == 8
        assert info.num_entities == 5
    
    def test_health_check(self):
        """Test health check functionality."""
        service = TransEModelService(
            model_path=self.model_path,
            entity_mapping_path=self.entity_mapping_path,
            metadata_path=self.metadata_path
        )
        
        health = service.health_check()
        
        assert isinstance(health, dict)
        assert health["status"] == "healthy"
        assert health["model_loaded"] is True
        assert health["entity_mapping_loaded"] is True
        assert health["num_entities"] == 5
        assert health["prediction_test"] is True
    
    def test_health_check_missing_files(self):
        """Test health check with missing files."""
        service = TransEModelService(
            model_path=Path("nonexistent.pt"),
            entity_mapping_path=Path("nonexistent.pkl"),
            metadata_path=Path("nonexistent.pkl")
        )
        
        health = service.health_check()
        
        assert health["status"] == "unhealthy"
        assert "error" in health


@pytest.mark.ml 
class TestGlobalMLService:
    """Test global ML service functions."""
    
    def setup_method(self):
        """Reset global state."""
        reset_ml_service()
    
    def test_get_ml_service_singleton(self):
        """Test global service instance is singleton."""
        service1 = get_ml_service()
        service2 = get_ml_service()
        
        assert service1 is service2
    
    def test_force_reload(self):
        """Test force reload creates new instance."""
        service1 = get_ml_service()
        service2 = get_ml_service(force_reload=True)
        
        assert service1 is not service2
    
    def test_reset_service(self):
        """Test reset clears global instance."""
        service1 = get_ml_service()
        reset_ml_service()
        service2 = get_ml_service()
        
        assert service1 is not service2


@pytest.mark.performance
class TestMLServicePerformance:
    """Performance tests for ML service."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        reset_ml_service()
        
        # Create larger test setup
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "test_model.pt"
        self.entity_mapping_path = Path(self.temp_dir) / "entity_mapping.pkl"
        
        # Create larger entity mapping
        self.test_entity_mapping = {f"paper{i}": i for i in range(1000)}
        
        with open(self.entity_mapping_path, 'wb') as f:
            pickle.dump(self.test_entity_mapping, f)
        
        # Create larger model
        model = TransEModel(num_entities=1000, embedding_dim=128)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "num_entities": 1000,
                "embedding_dim": 128,
                "margin": 1.0,
                "p_norm": 1
            }
        }
        
        torch.save(checkpoint, self.model_path)
    
    def test_prediction_performance(self):
        """Test prediction performance with timing."""
        service = TransEModelService(
            model_path=self.model_path,
            entity_mapping_path=self.entity_mapping_path,
            cache_predictions=False  # Test raw performance
        )
        
        import time
        start_time = time.time()
        
        # Make predictions
        predictions = service.predict_citations("paper1", top_k=10)
        
        end_time = time.time()
        prediction_time = end_time - start_time
        
        # Should complete reasonably quickly
        assert prediction_time < 5.0  # 5 seconds max
        assert len(predictions) == 10
    
    def test_cache_performance(self):
        """Test caching improves performance.""" 
        service = TransEModelService(
            model_path=self.model_path,
            entity_mapping_path=self.entity_mapping_path,
            cache_predictions=True
        )
        
        import time
        
        # First prediction (cold)
        start1 = time.time()
        predictions1 = service.predict_citations("paper1", top_k=10)
        time1 = time.time() - start1
        
        # Second prediction (cached)
        start2 = time.time()
        predictions2 = service.predict_citations("paper1", top_k=10)
        time2 = time.time() - start2
        
        # Cached should be faster
        assert time2 < time1
        assert len(predictions1) == len(predictions2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])