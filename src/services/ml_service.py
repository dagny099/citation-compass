"""
Machine Learning service layer for TransE citation prediction model.

This service provides a high-level interface for loading and serving locally
trained TransE models, with caching optimizations for web application performance.
"""

import os
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import json

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Configure PyTorch to allow loading older model checkpoints
# This is needed for PyTorch 2.6+ compatibility with models saved in older versions
try:
    import collections
    import torch._utils
    
    # Add safe globals for loading older model checkpoints
    torch.serialization.add_safe_globals([
        collections.OrderedDict,
        torch._utils._rebuild_tensor_v2,
        torch._utils._rebuild_parameter,
        torch.Size
    ])
    
    # Add numpy globals if available
    try:
        import numpy
        torch.serialization.add_safe_globals([
            numpy.ndarray,
            numpy.dtype,
        ])
        # Try to add multiarray functions with fallback for newer numpy versions
        try:
            torch.serialization.add_safe_globals([
                numpy._core.multiarray._reconstruct,
                numpy._core.multiarray.scalar,
            ])
        except AttributeError:
            # Fallback for older numpy versions
            try:
                torch.serialization.add_safe_globals([
                    numpy.core.multiarray._reconstruct,
                    numpy.core.multiarray.scalar,
                ])
            except AttributeError:
                pass
    except (ImportError, AttributeError):
        pass
        
except (AttributeError, ImportError):
    # add_safe_globals might not be available in all PyTorch versions
    pass

from ..models.ml import (
    PaperEmbedding, 
    CitationPrediction, 
    ModelMetadata, 
    PredictionConfidence,
    ModelType
)
from ..data.api_config import get_config


class TransEModel(nn.Module):
    """
    TransE model implementation for citation prediction.
    
    This is a clean implementation of the TransE model that can be trained
    locally and used for citation prediction tasks.
    """
    
    def __init__(self, 
                 num_entities: int,
                 embedding_dim: int = 128,
                 margin: float = 1.0,
                 p_norm: int = 1):
        super().__init__()
        
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.p_norm = p_norm
        
        # Entity embeddings (papers)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        
        # Relation embedding (we only have "CITES" relation)
        self.relation_embedding = nn.Embedding(1, embedding_dim)
    
    def forward(self, sources: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute scores for (source, target) pairs.
        
        Args:
            sources: Tensor of source entity IDs [batch_size]
            targets: Tensor of target entity IDs [batch_size]
        
        Returns:
            Scores (lower = more likely citation) [batch_size]
        """
        # Get embeddings
        source_emb = self.entity_embeddings(sources)  # [batch_size, embedding_dim]
        target_emb = self.entity_embeddings(targets)  # [batch_size, embedding_dim] 
        relation_emb = self.relation_embedding(torch.zeros_like(sources))  # [batch_size, embedding_dim]
        
        # TransE score: ||source + relation - target||
        score = source_emb + relation_emb - target_emb  # [batch_size, embedding_dim]
        score = torch.norm(score, p=self.p_norm, dim=1)  # [batch_size]
        
        return score
    
    def predict_batch(self, sources: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """
        Predict citation probabilities for source papers to candidate targets.
        
        Args:
            sources: Source paper IDs [num_sources]
            candidates: Candidate target paper IDs [num_candidates]
        
        Returns:
            Scores matrix [num_sources, num_candidates] (lower = more likely)
        """
        self.eval()
        with torch.no_grad():
            sources = sources.unsqueeze(1)  # [num_sources, 1]
            candidates = candidates.unsqueeze(0)  # [1, num_candidates]
            
            # Broadcast to [num_sources, num_candidates]
            sources_expanded = sources.expand(-1, candidates.size(1))
            candidates_expanded = candidates.expand(sources.size(0), -1)
            
            # Flatten for forward pass
            sources_flat = sources_expanded.flatten()
            candidates_flat = candidates_expanded.flatten()
            
            scores = self.forward(sources_flat, candidates_flat)
            return scores.view(sources.size(0), candidates.size(1))


class PredictionCache:
    """
    In-memory cache for ML predictions with TTL and LRU eviction.
    
    Optimized for web application performance where users might request
    the same predictions multiple times.
    """
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        """
        Initialize prediction cache.
        
        Args:
            max_size: Maximum number of cached prediction sets
            ttl_seconds: Time-to-live for cached predictions in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[List[CitationPrediction], datetime]] = {}
        self.access_order: List[str] = []
        
        self.logger = logging.getLogger(__name__)
    
    def _generate_cache_key(self, paper_id: str, top_k: int, filters: Dict = None) -> str:
        """Generate cache key from prediction parameters."""
        key_data = {
            "paper_id": paper_id,
            "top_k": top_k,
            "filters": filters or {}
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, paper_id: str, top_k: int, filters: Dict = None) -> Optional[List[CitationPrediction]]:
        """
        Retrieve cached predictions if available and not expired.
        
        Args:
            paper_id: Source paper ID
            top_k: Number of top predictions requested
            filters: Additional filters applied
            
        Returns:
            Cached predictions or None if not found/expired
        """
        cache_key = self._generate_cache_key(paper_id, top_k, filters)
        
        if cache_key in self.cache:
            predictions, timestamp = self.cache[cache_key]
            
            # Check if expired
            if datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds):
                del self.cache[cache_key]
                if cache_key in self.access_order:
                    self.access_order.remove(cache_key)
                return None
            
            # Update access order for LRU
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            
            self.logger.debug(f"Cache hit for paper {paper_id} (top_k={top_k})")
            return predictions
        
        return None
    
    def set(self, paper_id: str, top_k: int, predictions: List[CitationPrediction], 
            filters: Dict = None) -> None:
        """
        Store predictions in cache.
        
        Args:
            paper_id: Source paper ID
            top_k: Number of top predictions
            predictions: List of predictions to cache
            filters: Additional filters applied
        """
        cache_key = self._generate_cache_key(paper_id, top_k, filters)
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and cache_key not in self.cache:
            if self.access_order:
                oldest_key = self.access_order.pop(0)
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
        
        self.cache[cache_key] = (predictions, datetime.now())
        
        # Update access order
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)
        
        self.logger.debug(f"Cached predictions for paper {paper_id} (top_k={top_k})")


class TransEModelService:
    """
    High-level service for TransE model predictions with caching and optimization.
    
    This service loads locally trained models and provides a clean interface for
    citation predictions with performance optimizations suitable for web applications.
    """
    
    def __init__(self, 
                 model_path: Optional[Path] = None,
                 entity_mapping_path: Optional[Path] = None,
                 metadata_path: Optional[Path] = None,
                 cache_predictions: bool = True,
                 device: Optional[str] = None):
        """
        Initialize TransE model service.
        
        Args:
            model_path: Path to trained TransE model (.pt file)
            entity_mapping_path: Path to entity mapping (.pkl file)
            metadata_path: Path to training metadata (.pkl file)
            cache_predictions: Whether to cache predictions
            device: Computing device ('cpu', 'cuda', 'mps')
        """
        self.config = get_config()
        self.ml_config = self.config.get_ml_config()
        self.logger = logging.getLogger(__name__)
        
        # Set up paths
        self.model_path = model_path or self._get_reference_model_path()
        self.entity_mapping_path = entity_mapping_path or self._get_reference_entity_mapping_path()
        self.metadata_path = metadata_path or self._get_reference_metadata_path()
        
        # Set up device
        self.device = device or self.ml_config.device
        
        # Initialize cache
        self.cache = PredictionCache() if cache_predictions else None
        
        # Model components (loaded lazily)
        self.model: Optional[TransEModel] = None
        self.entity_mapping: Optional[Dict[str, int]] = None
        self.reverse_mapping: Optional[Dict[int, str]] = None
        self.metadata: Optional[Dict] = None
        
        self.logger.info(f"TransE service initialized (device: {self.device})")
    
    def _get_reference_model_path(self) -> Path:
        """Get path to the trained model in local models directory."""
        return Path("models/transe_citation_model.pt")
    
    def _get_reference_entity_mapping_path(self) -> Path:
        """Get path to entity mapping in local models directory."""
        return Path("models/entity_mapping.pkl")
    
    def _get_reference_metadata_path(self) -> Path:
        """Get path to training metadata in local models directory."""
        return Path("models/training_metadata.json")
    
    def _load_entity_mapping(self) -> None:
        """Load entity mapping from pickle file."""
        if not self.entity_mapping_path.exists():
            raise FileNotFoundError(f"Entity mapping not found: {self.entity_mapping_path}")
        
        self.logger.info(f"Loading entity mapping from {self.entity_mapping_path}")
        try:
            with open(self.entity_mapping_path, 'rb') as f:
                self.entity_mapping = pickle.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load entity mapping: {e}")
            raise RuntimeError(f"Cannot load entity mapping from {self.entity_mapping_path}: {e}")
        
        # Create reverse mapping
        self.reverse_mapping = {v: k for k, v in self.entity_mapping.items()}
        self.logger.info(f"Loaded {len(self.entity_mapping)} entity mappings")
    
    def _load_metadata(self) -> None:
        """Load training metadata from JSON file."""
        if self.metadata_path.exists():
            self.logger.info(f"Loading metadata from {self.metadata_path}")
            import json
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.logger.warning(f"Metadata file not found: {self.metadata_path}")
            self.metadata = {}
    
    def _load_model(self) -> None:
        """Load the trained TransE model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.logger.info(f"Loading TransE model from {self.model_path}")
        
        # Load checkpoint with explicit weights_only=False for compatibility with older models
        try:
            # For PyTorch 2.8+ we need to be very explicit about unsafe loading
            checkpoint = torch.load(
                self.model_path, 
                map_location=self.device, 
                weights_only=False,
                pickle_module=pickle  # Explicitly use standard pickle module
            )
        except Exception as e:
            self.logger.error(f"Failed to load model checkpoint with standard method: {e}")
            # Try alternative loading approach for compatibility
            try:
                import pickle as pkl
                with open(self.model_path, 'rb') as f:
                    # Load using standard pickle first to get the data
                    checkpoint = pkl.load(f)
                    # Then move tensors to appropriate device if needed
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        for key, value in checkpoint['model_state_dict'].items():
                            if hasattr(value, 'to'):
                                checkpoint['model_state_dict'][key] = value.to(self.device)
                    self.logger.info("Successfully loaded checkpoint using alternative method")
            except Exception as e2:
                raise RuntimeError(f"Cannot load model from {self.model_path}. Primary error: {e}. Alternative method error: {e2}")
        
        # Extract model configuration
        if "model_config" in checkpoint:
            saved_config = checkpoint["model_config"]
            # Filter config to only include parameters that TransEModel expects
            config = {
                "num_entities": len(self.entity_mapping),
                "embedding_dim": saved_config.get("embedding_dim", 128),
                "margin": saved_config.get("margin", 1.0),
                "p_norm": saved_config.get("norm_p", 1)  # Note: norm_p in config becomes p_norm in model
            }
        else:
            # Fallback configuration if not saved in checkpoint
            config = {
                "num_entities": len(self.entity_mapping),
                "embedding_dim": 128,
                "margin": 1.0,
                "p_norm": 1
            }
            self.logger.warning("Using default model configuration (not found in checkpoint)")
        
        # Create and load model
        self.model = TransEModel(**config)
        
        # Fix state dict key mismatch (saved model has "relation_embeddings" but class expects "relation_embedding")
        state_dict = checkpoint["model_state_dict"]
        if "relation_embeddings.weight" in state_dict and "relation_embedding.weight" not in state_dict:
            state_dict["relation_embedding.weight"] = state_dict.pop("relation_embeddings.weight")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"Model loaded successfully (entities: {config['num_entities']}, "
                        f"embedding_dim: {config['embedding_dim']})")
    
    def ensure_loaded(self) -> None:
        """Ensure all model components are loaded."""
        if self.entity_mapping is None:
            self._load_entity_mapping()
        
        if self.metadata is None:
            self._load_metadata()
        
        if self.model is None:
            self._load_model()
    
    def get_paper_embedding(self, paper_id: str) -> Optional[PaperEmbedding]:
        """
        Get embedding vector for a specific paper.
        
        Args:
            paper_id: Paper ID (from Semantic Scholar or internal)
            
        Returns:
            PaperEmbedding instance or None if paper not found
        """
        self.ensure_loaded()
        
        if paper_id not in self.entity_mapping:
            return None
        
        entity_idx = self.entity_mapping[paper_id]
        
        with torch.no_grad():
            embedding_tensor = self.model.entity_embeddings.weight[entity_idx]
            embedding_vector = embedding_tensor.cpu().numpy().tolist()
        
        return PaperEmbedding(
            paper_id=paper_id,
            embedding=embedding_vector,
            model_name="TransE",
            embedding_dim=len(embedding_vector),
            created_at=datetime.now()
        )
    
    def predict_citations(self, 
                         source_paper_id: str, 
                         candidate_paper_ids: Optional[List[str]] = None,
                         top_k: int = 10,
                         score_threshold: Optional[float] = None) -> List[CitationPrediction]:
        """
        Predict citation likelihood from a source paper to candidate papers.
        
        Args:
            source_paper_id: ID of the source paper
            candidate_paper_ids: Optional list of candidate papers to rank.
                                If None, ranks against all papers in the model.
            top_k: Number of top predictions to return
            score_threshold: Optional minimum score threshold
            
        Returns:
            List of CitationPrediction objects sorted by likelihood
        """
        self.ensure_loaded()
        
        # Check cache first
        if self.cache:
            cached_predictions = self.cache.get(source_paper_id, top_k)
            if cached_predictions:
                return cached_predictions
        
        # Validate source paper
        if source_paper_id not in self.entity_mapping:
            self.logger.warning(f"Source paper not found in model: {source_paper_id}")
            return []
        
        source_idx = self.entity_mapping[source_paper_id]
        
        # Determine candidates
        if candidate_paper_ids is None:
            # Use all papers in the model
            candidate_indices = list(range(len(self.entity_mapping)))
            candidate_paper_ids = [self.reverse_mapping[i] for i in candidate_indices]
        else:
            # Filter to valid candidates
            valid_candidates = []
            candidate_indices = []
            for paper_id in candidate_paper_ids:
                if paper_id in self.entity_mapping:
                    valid_candidates.append(paper_id)
                    candidate_indices.append(self.entity_mapping[paper_id])
            candidate_paper_ids = valid_candidates
        
        if not candidate_paper_ids:
            return []
        
        # Make predictions
        source_tensor = torch.tensor([source_idx], device=self.device)
        candidate_tensor = torch.tensor(candidate_indices, device=self.device)
        
        with torch.no_grad():
            # Get prediction scores (lower = more likely)
            scores = self.model.predict_batch(source_tensor, candidate_tensor)
            scores = scores.squeeze(0).cpu().numpy()
        
        # Create predictions
        predictions = []
        for i, (paper_id, score) in enumerate(zip(candidate_paper_ids, scores)):
            # Convert TransE distance to probability-like score (higher = more likely)
            probability_score = 1.0 / (1.0 + float(score))
            
            if score_threshold is None or probability_score >= score_threshold:
                prediction = CitationPrediction(
                    source_paper_id=source_paper_id,
                    target_paper_id=paper_id,
                    prediction_score=probability_score,
                    model_name="TransE",
                    raw_score=float(score),
                    predicted_at=datetime.now()
                )
                predictions.append(prediction)
        
        # Sort by likelihood (highest probability first)
        predictions.sort(key=lambda x: x.prediction_score, reverse=True)
        
        # Return top_k predictions
        top_predictions = predictions[:top_k]
        
        # Cache the results
        if self.cache:
            self.cache.set(source_paper_id, top_k, top_predictions)
        
        self.logger.info(f"Generated {len(top_predictions)} predictions for {source_paper_id}")
        return top_predictions
    
    def batch_predict_citations(self, 
                               source_paper_ids: List[str],
                               top_k: int = 10) -> Dict[str, List[CitationPrediction]]:
        """
        Generate predictions for multiple source papers efficiently.
        
        Args:
            source_paper_ids: List of source paper IDs
            top_k: Number of top predictions per source paper
            
        Returns:
            Dictionary mapping source paper IDs to their predictions
        """
        results = {}
        
        for source_id in source_paper_ids:
            try:
                predictions = self.predict_citations(source_id, top_k=top_k)
                results[source_id] = predictions
            except Exception as e:
                self.logger.error(f"Error predicting for {source_id}: {e}")
                results[source_id] = []
        
        return results
    
    def get_model_info(self) -> ModelMetadata:
        """
        Get information about the loaded model.
        
        Returns:
            ModelMetadata with model information
        """
        self.ensure_loaded()
        
        # Get training metadata from the JSON file if available
        metadata = self.metadata if self.metadata else {}
        training_config = metadata.get("training_config", {})
        data_config = metadata.get("data_config", {})
        model_config = metadata.get("model_config", {})
        training_results = metadata.get("training_results", {})
        
        return ModelMetadata(
            model_name="TransE Citation Predictor",
            model_type=ModelType.TRANSE,
            model_version="1.0",
            embedding_dim=self.model.embedding_dim,
            num_entities=len(self.entity_mapping),
            num_relations=1,  # TransE typically uses one relation type for citations
            training_dataset_size=metadata.get("dataset", {}).get("total_training_samples", 0),
            training_epochs=training_results.get("epochs_completed", training_config.get("epochs", 0)),
            learning_rate=model_config.get("learning_rate", 0.01),
            batch_size=training_config.get("batch_size", 1024),
            created_at=datetime.now()
        )
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the model service.
        
        Returns:
            Dictionary with health status information
        """
        try:
            self.ensure_loaded()
            
            # Test a simple prediction if we have entities
            if len(self.entity_mapping) > 0:
                sample_paper = next(iter(self.entity_mapping.keys()))
                test_predictions = self.predict_citations(sample_paper, top_k=1)
                prediction_test = len(test_predictions) > 0
            else:
                prediction_test = False
            
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "entity_mapping_loaded": self.entity_mapping is not None,
                "num_entities": len(self.entity_mapping) if self.entity_mapping else 0,
                "device": str(self.device),
                "cache_enabled": self.cache is not None,
                "prediction_test": prediction_test,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Global service instance
_ml_service: Optional[TransEModelService] = None


def get_ml_service(force_reload: bool = False) -> TransEModelService:
    """
    Get the global ML service instance.
    
    Args:
        force_reload: Force reloading of the service
        
    Returns:
        TransEModelService instance
    """
    global _ml_service
    
    if _ml_service is None or force_reload:
        _ml_service = TransEModelService()
    
    return _ml_service


def reset_ml_service() -> None:
    """Reset the global ML service instance. Useful for testing."""
    global _ml_service
    _ml_service = None