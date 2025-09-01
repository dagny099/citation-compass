"""
Demo Service Integration for Academic Citation Platform.

This service integrates demo datasets with existing ML models and analytics,
providing realistic demonstrations of platform capabilities without requiring
users to import real data or train models.

Features:
- Synthetic embeddings that work with existing ML service
- Demo predictions that showcase citation prediction capabilities
- Integration with analytics service for network analysis
- Realistic but fast-to-generate sample data
"""

import logging
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

from .ml_service import get_ml_service, TransEModelService
from .analytics_service import get_analytics_service
from ..data.demo_loader import get_demo_loader, DemoDataLoader
from ..data.fixtures import get_fixture_manager, FixtureManager
from ..models.ml import CitationPrediction, PaperEmbedding
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DemoEmbeddingGenerator:
    """
    Generates realistic embeddings for demo papers that work with existing ML models.
    
    Creates embeddings that:
    - Have realistic similarity patterns based on paper content
    - Work with existing TransE model architecture
    - Provide meaningful clustering for visualization
    - Enable realistic citation predictions
    """
    
    def __init__(self, embedding_dim: int = 128):
        """
        Initialize demo embedding generator.
        
        Args:
            embedding_dim: Dimension of generated embeddings
        """
        self.embedding_dim = embedding_dim
        self.logger = get_logger(__name__)
        
        # Field-based base embeddings for realistic clustering
        self.field_embeddings = {
            "Machine Learning": np.random.normal(0, 0.1, embedding_dim),
            "Computer Vision": np.random.normal(0.2, 0.1, embedding_dim),
            "Natural Language Processing": np.random.normal(-0.2, 0.1, embedding_dim),
            "Robotics": np.random.normal(0.1, 0.1, embedding_dim),
            "Medical Informatics": np.random.normal(-0.1, 0.1, embedding_dim),
            "AI Ethics": np.random.normal(0, 0.15, embedding_dim),
            "Neuroscience": np.random.normal(0.15, 0.1, embedding_dim),
            "Psychology": np.random.normal(-0.15, 0.1, embedding_dim),
        }
        
        # Normalize base embeddings
        for field in self.field_embeddings:
            self.field_embeddings[field] = self.field_embeddings[field] / np.linalg.norm(self.field_embeddings[field])
    
    def generate_paper_embedding(self, 
                                paper_data: Dict[str, Any],
                                add_noise: bool = True) -> np.ndarray:
        """
        Generate embedding for a paper based on its metadata.
        
        Args:
            paper_data: Paper metadata dictionary
            add_noise: Whether to add random noise for realism
            
        Returns:
            Embedding vector as numpy array
        """
        field = paper_data.get("field", "Machine Learning")
        year = paper_data.get("year", 2020)
        citation_count = paper_data.get("citation_count", 0)
        
        # Start with field-based embedding
        base_embedding = self.field_embeddings.get(field, self.field_embeddings["Machine Learning"]).copy()
        
        # Add temporal component (recent papers slightly different)
        temporal_factor = (year - 2015) / 10.0  # 0.0 to 0.9 for 2015-2024
        temporal_component = np.random.normal(0, 0.02, self.embedding_dim)
        base_embedding += temporal_factor * temporal_component
        
        # Add citation influence (highly cited papers have slightly different embeddings)
        citation_factor = min(np.log(citation_count + 1) / 10.0, 0.3)  # Cap at 0.3
        citation_component = np.random.normal(0, 0.03, self.embedding_dim)
        base_embedding += citation_factor * citation_component
        
        # Add noise for uniqueness
        if add_noise:
            noise = np.random.normal(0, 0.05, self.embedding_dim)
            base_embedding += noise
        
        # Normalize to unit vector
        embedding = base_embedding / np.linalg.norm(base_embedding)
        
        return embedding
    
    def generate_batch_embeddings(self, papers: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for a batch of papers.
        
        Args:
            papers: List of paper data dictionaries
            
        Returns:
            Dictionary mapping paper_id to embedding array
        """
        embeddings = {}
        
        for paper in papers:
            paper_id = paper["paper_id"]
            embedding = self.generate_paper_embedding(paper)
            embeddings[paper_id] = embedding
        
        self.logger.info(f"Generated embeddings for {len(embeddings)} papers")
        return embeddings
    
    def create_similarity_based_embeddings(self, 
                                         papers: List[Dict[str, Any]],
                                         citations: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings that reflect citation relationships.
        
        Papers that cite each other will have more similar embeddings.
        
        Args:
            papers: List of paper data
            citations: List of citation relationships
            
        Returns:
            Dictionary mapping paper_id to embedding array
        """
        # Generate initial embeddings
        embeddings = self.generate_batch_embeddings(papers)
        
        # Create citation graph for similarity adjustment
        citation_graph = {}
        for citation in citations:
            source_id = citation["source_id"]
            target_id = citation["target_id"]
            
            if source_id not in citation_graph:
                citation_graph[source_id] = []
            citation_graph[source_id].append(target_id)
        
        # Adjust embeddings based on citations (iterative process)
        for iteration in range(3):  # 3 iterations for convergence
            for source_id, targets in citation_graph.items():
                if source_id in embeddings:
                    source_embedding = embeddings[source_id]
                    
                    # Average with cited papers' embeddings
                    adjustments = []
                    for target_id in targets:
                        if target_id in embeddings:
                            target_embedding = embeddings[target_id]
                            # Move source slightly towards target
                            adjustment = 0.1 * (target_embedding - source_embedding)
                            adjustments.append(adjustment)
                    
                    if adjustments:
                        mean_adjustment = np.mean(adjustments, axis=0)
                        embeddings[source_id] = source_embedding + mean_adjustment
                        # Re-normalize
                        embeddings[source_id] = embeddings[source_id] / np.linalg.norm(embeddings[source_id])
        
        self.logger.info(f"Adjusted embeddings based on {len(citations)} citations")
        return embeddings


class DemoPredictionGenerator:
    """
    Generates realistic citation predictions for demo datasets.
    
    Creates predictions that:
    - Follow realistic patterns (newer papers cite older ones)
    - Respect field boundaries (ML papers more likely to cite ML papers)
    - Have realistic confidence scores
    - Provide meaningful results for demonstration
    """
    
    def __init__(self, embeddings: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize prediction generator.
        
        Args:
            embeddings: Optional pre-computed embeddings for papers
        """
        self.embeddings = embeddings or {}
        self.logger = get_logger(__name__)
    
    def generate_predictions(self,
                           source_paper_id: str,
                           papers: List[Dict[str, Any]], 
                           top_k: int = 10,
                           exclude_existing_citations: bool = True) -> List[CitationPrediction]:
        """
        Generate citation predictions for a source paper.
        
        Args:
            source_paper_id: ID of source paper
            papers: All available papers
            top_k: Number of predictions to return
            exclude_existing_citations: Whether to exclude existing citations
            
        Returns:
            List of citation predictions
        """
        # Find source paper
        source_paper = None
        for paper in papers:
            if paper["paper_id"] == source_paper_id:
                source_paper = paper
                break
        
        if source_paper is None:
            self.logger.warning(f"Source paper {source_paper_id} not found")
            return []
        
        # Generate candidate predictions
        candidates = []
        source_field = source_paper.get("field", "Unknown")
        source_year = source_paper.get("year", 2020)
        
        for paper in papers:
            if paper["paper_id"] == source_paper_id:
                continue  # Skip self
            
            # Calculate base similarity score
            score = self._calculate_similarity_score(source_paper, paper)
            
            # Add some randomness for realism
            score += np.random.normal(0, 0.1)
            
            # Clamp score between 0 and 1
            score = max(0.0, min(1.0, score))
            
            prediction = CitationPrediction(
                source_paper_id=source_paper_id,
                target_paper_id=paper["paper_id"],
                prediction_score=score,
                model_name="Demo_TransE",
                raw_score=1.0 - score,  # TransE uses distance (lower = better)
                predicted_at=datetime.now()
            )
            
            candidates.append(prediction)
        
        # Sort by score and return top k
        candidates.sort(key=lambda x: x.prediction_score, reverse=True)
        
        self.logger.info(f"Generated {len(candidates[:top_k])} predictions for {source_paper_id}")
        return candidates[:top_k]
    
    def _calculate_similarity_score(self, source_paper: Dict, target_paper: Dict) -> float:
        """Calculate similarity score between two papers."""
        score = 0.5  # Base score
        
        # Field similarity (most important factor)
        if source_paper.get("field") == target_paper.get("field"):
            score += 0.3
        elif self._fields_related(source_paper.get("field"), target_paper.get("field")):
            score += 0.15
        
        # Temporal factor (newer papers cite older papers)
        source_year = source_paper.get("year", 2020)
        target_year = target_paper.get("year", 2020)
        
        if target_year < source_year:  # Target is older
            year_diff = source_year - target_year
            if year_diff <= 5:  # Recent papers
                score += 0.2
            elif year_diff <= 10:  # Somewhat recent
                score += 0.1
        elif target_year > source_year:  # Target is newer (less likely)
            score -= 0.1
        
        # Citation count influence (highly cited papers more likely to be cited)
        target_citations = target_paper.get("citation_count", 0)
        if target_citations > 1000:
            score += 0.15
        elif target_citations > 100:
            score += 0.1
        elif target_citations > 10:
            score += 0.05
        
        # Embedding similarity if available
        source_id = source_paper["paper_id"]
        target_id = target_paper["paper_id"]
        
        if source_id in self.embeddings and target_id in self.embeddings:
            source_emb = self.embeddings[source_id]
            target_emb = self.embeddings[target_id]
            
            # Cosine similarity
            similarity = np.dot(source_emb, target_emb) / (
                np.linalg.norm(source_emb) * np.linalg.norm(target_emb)
            )
            
            # Convert to 0-1 range and add to score
            score += (similarity + 1) / 4.0  # Maps [-1,1] to [0,0.5]
        
        return score
    
    def _fields_related(self, field1: Optional[str], field2: Optional[str]) -> bool:
        """Check if two fields are related."""
        if not field1 or not field2:
            return False
        
        related_fields = {
            "Machine Learning": ["Computer Vision", "Natural Language Processing", "AI Ethics"],
            "Computer Vision": ["Machine Learning", "Robotics"],
            "Natural Language Processing": ["Machine Learning", "AI Ethics"],
            "Robotics": ["Computer Vision", "Machine Learning"],
            "Medical Informatics": ["Machine Learning", "Computer Vision"],
            "Neuroscience": ["Psychology", "Machine Learning"],
            "Psychology": ["Neuroscience", "AI Ethics"]
        }
        
        return field2 in related_fields.get(field1, [])


class DemoMLService:
    """
    Demo ML service that provides realistic predictions using demo data.
    
    This service mimics the real ML service but works with demo datasets,
    providing immediate functionality for users who want to explore the
    platform without setting up real models.
    """
    
    def __init__(self):
        """Initialize demo ML service."""
        self.logger = get_logger(__name__)
        self.embeddings: Dict[str, np.ndarray] = {}
        self.papers: List[Dict[str, Any]] = []
        self.citations: List[Dict[str, Any]] = []
        
        self.embedding_generator = DemoEmbeddingGenerator()
        self.prediction_generator: Optional[DemoPredictionGenerator] = None
        
        self.loaded_dataset: Optional[str] = None
        self.is_loaded = False
    
    def load_demo_dataset(self, dataset_name: Optional[str] = None) -> bool:
        """
        Load demo dataset for ML predictions.
        
        Args:
            dataset_name: Name of dataset to load
            
        Returns:
            True if loaded successfully
        """
        try:
            # Load demo data
            demo_loader = get_demo_loader()
            success = demo_loader.load_demo_dataset(dataset_name, force_offline=True)
            
            if not success:
                self.logger.error("Failed to load demo dataset")
                return False
            
            # Get data interface
            data_interface = demo_loader.get_data_interface()
            
            # Load papers and citations from offline store
            if hasattr(data_interface, 'store'):
                store = data_interface.store
                self.papers = list(store.papers.values())
                self.citations = store.citations
                
                # Generate embeddings
                self.embeddings = self.embedding_generator.create_similarity_based_embeddings(
                    self.papers, self.citations
                )
                
                # Initialize prediction generator
                self.prediction_generator = DemoPredictionGenerator(self.embeddings)
                
                self.loaded_dataset = dataset_name or store.loaded_dataset
                self.is_loaded = True
                
                self.logger.info(f"Demo ML service loaded with {len(self.papers)} papers, "
                               f"{len(self.embeddings)} embeddings")
                
                return True
            else:
                self.logger.error("Could not access offline data store")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load demo dataset: {e}")
            return False
    
    def predict_citations(self, 
                         source_paper_id: str,
                         top_k: int = 10) -> List[CitationPrediction]:
        """
        Generate citation predictions for a paper.
        
        Args:
            source_paper_id: ID of source paper
            top_k: Number of predictions to return
            
        Returns:
            List of citation predictions
        """
        if not self.is_loaded:
            self.logger.warning("Demo ML service not loaded")
            return []
        
        if self.prediction_generator is None:
            self.logger.error("Prediction generator not initialized")
            return []
        
        return self.prediction_generator.generate_predictions(
            source_paper_id, self.papers, top_k
        )
    
    def get_paper_embedding(self, paper_id: str) -> Optional[PaperEmbedding]:
        """
        Get embedding for a paper.
        
        Args:
            paper_id: Paper ID
            
        Returns:
            PaperEmbedding or None if not found
        """
        if not self.is_loaded or paper_id not in self.embeddings:
            return None
        
        embedding_vector = self.embeddings[paper_id].tolist()
        
        return PaperEmbedding(
            paper_id=paper_id,
            embedding=embedding_vector,
            model_name="Demo_TransE",
            embedding_dim=len(embedding_vector),
            created_at=datetime.now()
        )
    
    def health_check(self) -> Dict[str, Any]:
        """Get health status of demo ML service."""
        return {
            "status": "healthy" if self.is_loaded else "not_loaded",
            "model_loaded": self.is_loaded,
            "entity_mapping_loaded": self.is_loaded,
            "num_entities": len(self.embeddings),
            "device": "cpu",
            "cache_enabled": False,
            "prediction_test": self.is_loaded,
            "timestamp": datetime.now().isoformat(),
            "mode": "demo",
            "loaded_dataset": self.loaded_dataset
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get demo model information."""
        return {
            "model_name": "Demo TransE Citation Predictor",
            "model_type": "demo",
            "model_version": "1.0.0",
            "embedding_dim": self.embedding_generator.embedding_dim,
            "num_entities": len(self.embeddings),
            "training_dataset_size": len(self.papers),
            "mode": "demo",
            "loaded_dataset": self.loaded_dataset,
            "created_at": datetime.now().isoformat()
        }


class DemoServiceManager:
    """
    Manager for coordinating demo services across the platform.
    
    Provides a unified interface for:
    - Loading demo datasets
    - Coordinating between demo ML and analytics services  
    - Managing demo mode state
    - Switching between demo and production modes
    """
    
    def __init__(self):
        """Initialize demo service manager."""
        self.logger = get_logger(__name__)
        
        # Demo services
        self.demo_ml_service: Optional[DemoMLService] = None
        self.demo_mode_active = False
        self.loaded_dataset: Optional[str] = None
        
        # Original services (for fallback)
        self._original_ml_service = None
    
    def enable_demo_mode(self, dataset_name: Optional[str] = None) -> bool:
        """
        Enable demo mode with specified dataset.
        
        Args:
            dataset_name: Dataset to load for demo mode
            
        Returns:
            True if demo mode enabled successfully
        """
        try:
            self.logger.info("Enabling demo mode...")
            
            # Initialize demo ML service
            self.demo_ml_service = DemoMLService()
            success = self.demo_ml_service.load_demo_dataset(dataset_name)
            
            if success:
                self.demo_mode_active = True
                self.loaded_dataset = dataset_name
                self.logger.info(f"Demo mode enabled with dataset: {dataset_name}")
                return True
            else:
                self.logger.error("Failed to enable demo mode")
                return False
                
        except Exception as e:
            self.logger.error(f"Error enabling demo mode: {e}")
            return False
    
    def disable_demo_mode(self) -> None:
        """Disable demo mode and return to production services."""
        self.demo_mode_active = False
        self.demo_ml_service = None
        self.loaded_dataset = None
        self.logger.info("Demo mode disabled")
    
    def is_demo_mode_active(self) -> bool:
        """Check if demo mode is currently active."""
        return self.demo_mode_active
    
    def get_ml_service(self):
        """Get appropriate ML service (demo or production)."""
        if self.demo_mode_active and self.demo_ml_service:
            return self.demo_ml_service
        else:
            return get_ml_service()  # Fall back to production service
    
    def get_demo_status(self) -> Dict[str, Any]:
        """Get current demo mode status."""
        return {
            "demo_mode_active": self.demo_mode_active,
            "loaded_dataset": self.loaded_dataset,
            "demo_ml_service_loaded": self.demo_ml_service is not None and self.demo_ml_service.is_loaded,
            "available_datasets": [info.name for info in get_fixture_manager().list_available_fixtures()]
        }


# Global demo service manager
_demo_manager: Optional[DemoServiceManager] = None


def get_demo_manager() -> DemoServiceManager:
    """Get global demo service manager."""
    global _demo_manager
    
    if _demo_manager is None:
        _demo_manager = DemoServiceManager()
    
    return _demo_manager


def enable_demo_mode(dataset_name: Optional[str] = None) -> bool:
    """
    Quick function to enable demo mode.
    
    Args:
        dataset_name: Dataset to load
        
    Returns:
        True if enabled successfully
    """
    return get_demo_manager().enable_demo_mode(dataset_name)


def disable_demo_mode() -> None:
    """Quick function to disable demo mode."""
    get_demo_manager().disable_demo_mode()


def is_demo_mode_active() -> bool:
    """Quick function to check if demo mode is active."""
    return get_demo_manager().is_demo_mode_active()