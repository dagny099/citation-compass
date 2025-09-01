"""
Machine Learning models for the Academic Citation Platform.

This module provides data models specifically for ML workflows, including:
- Embedding storage and retrieval (for TransE models)
- Prediction results and confidence scores
- Model metadata and training configurations
- Evaluation metrics and performance tracking

These models support the integration of citation-map-dashboard ML capabilities
with the interactive platform.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field as PydanticField, field_validator, ConfigDict
from pathlib import Path


class ModelType(str, Enum):
    """Supported ML model types."""
    TRANSE = "TransE"
    COMPLEX = "ComplEx"
    ROTAT = "RotatE"
    DISTMULT = "DistMult"


class PredictionConfidence(str, Enum):
    """Prediction confidence levels."""
    HIGH = "high"       # > 0.8
    MEDIUM = "medium"   # 0.5 - 0.8
    LOW = "low"         # < 0.5


class PaperEmbedding(BaseModel):
    """
    Model for storing and managing paper embeddings from ML models.
    
    This supports the TransE model integration from citation-map-dashboard
    and provides a unified interface for embedding storage.
    """
    
    paper_id: str = PydanticField(..., description="Unique paper identifier")
    embedding: List[float] = PydanticField(..., description="Dense embedding vector")
    model_name: str = PydanticField(..., description="Name of the model that generated this embedding")
    model_version: Optional[str] = PydanticField(None, description="Version of the model")
    embedding_dim: int = PydanticField(..., description="Dimensionality of the embedding")
    created_at: datetime = PydanticField(default_factory=datetime.now, description="When embedding was created")
    updated_at: datetime = PydanticField(default_factory=datetime.now, description="When embedding was last updated")
    
    @field_validator('embedding')
    @classmethod
    def validate_embedding_dimension(cls, v, info):
        """Ensure embedding dimension matches declared dimension."""
        if hasattr(info, 'data') and 'embedding_dim' in info.data and len(v) != info.data['embedding_dim']:
            raise ValueError(f"Embedding length {len(v)} doesn't match declared dimension {info.data['embedding_dim']}")
        return v
    
    @field_validator('embedding_dim')
    @classmethod
    def validate_embedding_dim_positive(cls, v):
        """Ensure embedding dimension is positive."""
        if v <= 0:
            raise ValueError("Embedding dimension must be positive")
        return v
    
    def to_numpy(self) -> np.ndarray:
        """Convert embedding to numpy array."""
        return np.array(self.embedding, dtype=np.float32)
    
    @classmethod
    def from_numpy(cls, paper_id: str, embedding: np.ndarray, 
                   model_name: str, model_version: Optional[str] = None) -> PaperEmbedding:
        """Create PaperEmbedding from numpy array."""
        return cls(
            paper_id=paper_id,
            embedding=embedding.tolist(),
            model_name=model_name,
            model_version=model_version,
            embedding_dim=len(embedding)
        )
    
    def cosine_similarity(self, other: PaperEmbedding) -> float:
        """Calculate cosine similarity with another embedding."""
        if self.embedding_dim != other.embedding_dim:
            raise ValueError("Cannot compare embeddings of different dimensions")
        
        a = self.to_numpy()
        b = other.to_numpy()
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))


class CitationPrediction(BaseModel):
    """
    Model for citation prediction results.
    
    Represents the output of ML models predicting whether one paper
    should cite another, with confidence scores and metadata.
    """
    
    source_paper_id: str = PydanticField(..., description="Paper that would do the citing")
    target_paper_id: str = PydanticField(..., description="Paper that would be cited")
    prediction_score: float = PydanticField(..., ge=0.0, le=1.0, description="Prediction confidence score (0-1)")
    model_name: str = PydanticField(..., description="Name of the model making the prediction")
    model_version: Optional[str] = PydanticField(None, description="Version of the model")
    predicted_at: datetime = PydanticField(default_factory=datetime.now, description="When prediction was made")
    
    # Additional context
    source_title: Optional[str] = PydanticField(None, description="Title of source paper")
    target_title: Optional[str] = PydanticField(None, description="Title of target paper")
    explanation: Optional[str] = PydanticField(None, description="Human-readable explanation of prediction")
    
    @property
    def confidence_level(self) -> PredictionConfidence:
        """Get categorical confidence level."""
        if self.prediction_score >= 0.8:
            return PredictionConfidence.HIGH
        elif self.prediction_score >= 0.5:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW
    
    def is_positive_prediction(self, threshold: float = 0.5) -> bool:
        """Check if this is a positive prediction above threshold."""
        return self.prediction_score >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage or API responses."""
        return {
            "source_paper_id": self.source_paper_id,
            "target_paper_id": self.target_paper_id,
            "prediction_score": self.prediction_score,
            "confidence_level": self.confidence_level.value,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "predicted_at": self.predicted_at.isoformat(),
            "source_title": self.source_title,
            "target_title": self.target_title,
            "explanation": self.explanation
        }


class ModelMetadata(BaseModel):
    """
    Metadata for trained ML models.
    
    Stores information about model architecture, training, and performance
    to support model versioning and reproducibility.
    """
    
    model_name: str = PydanticField(..., description="Unique model name")
    model_type: ModelType = PydanticField(..., description="Type of ML model")
    model_version: str = PydanticField(..., description="Model version string")
    
    # Model architecture
    embedding_dim: int = PydanticField(..., description="Embedding dimension")
    num_entities: int = PydanticField(..., description="Number of entities in training data")
    num_relations: int = PydanticField(..., description="Number of relation types")
    
    # Training information
    training_dataset_size: int = PydanticField(..., description="Number of training examples")
    training_epochs: int = PydanticField(..., description="Number of training epochs")
    learning_rate: float = PydanticField(..., description="Learning rate used")
    batch_size: int = PydanticField(..., description="Training batch size")
    margin: Optional[float] = PydanticField(None, description="Margin for ranking loss (TransE)")
    
    # Performance metrics
    mrr: Optional[float] = PydanticField(None, ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    hits_at_1: Optional[float] = PydanticField(None, ge=0.0, le=1.0, description="Hits@1 score")
    hits_at_3: Optional[float] = PydanticField(None, ge=0.0, le=1.0, description="Hits@3 score")
    hits_at_10: Optional[float] = PydanticField(None, ge=0.0, le=1.0, description="Hits@10 score")
    auc: Optional[float] = PydanticField(None, ge=0.0, le=1.0, description="Area Under Curve")
    
    # File paths
    model_path: Optional[str] = PydanticField(None, description="Path to saved model file")
    entity_mapping_path: Optional[str] = PydanticField(None, description="Path to entity mapping file")
    
    # Timestamps
    created_at: datetime = PydanticField(default_factory=datetime.now, description="Model creation time")
    trained_at: Optional[datetime] = PydanticField(None, description="Model training completion time")
    
    @field_validator('embedding_dim', 'num_entities', 'num_relations', 'training_dataset_size', 'training_epochs', 'batch_size')
    @classmethod
    def validate_positive_integers(cls, v):
        """Ensure key metrics are positive."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v
    
    @field_validator('learning_rate')
    @classmethod
    def validate_learning_rate(cls, v):
        """Ensure learning rate is reasonable."""
        if v <= 0 or v > 1:
            raise ValueError("Learning rate must be between 0 and 1")
        return v
    
    def performance_summary(self) -> Dict[str, float]:
        """Get summary of performance metrics."""
        metrics = {}
        for metric_name in ['mrr', 'hits_at_1', 'hits_at_3', 'hits_at_10', 'auc']:
            value = getattr(self, metric_name)
            if value is not None:
                metrics[metric_name] = value
        return metrics
    
    def is_better_than(self, other: ModelMetadata, primary_metric: str = 'mrr') -> bool:
        """Compare performance with another model."""
        self_metric = getattr(self, primary_metric, None)
        other_metric = getattr(other, primary_metric, None)
        
        if self_metric is None or other_metric is None:
            return False
        
        return self_metric > other_metric


class TrainingConfig(BaseModel):
    """
    Configuration for model training.
    
    Encapsulates all parameters needed to train a citation prediction model,
    supporting reproducible training workflows.
    """
    
    # Model configuration
    model_type: ModelType = PydanticField(default=ModelType.TRANSE, description="Type of model to train")
    embedding_dim: int = PydanticField(default=128, ge=1, description="Embedding dimension")
    margin: float = PydanticField(default=1.0, ge=0.0, description="Margin for ranking loss")
    p_norm: int = PydanticField(default=1, ge=1, le=2, description="Norm to use (1 or 2)")
    
    # Training parameters
    epochs: int = PydanticField(default=100, ge=1, description="Number of training epochs")
    batch_size: int = PydanticField(default=1024, ge=1, description="Training batch size")
    learning_rate: float = PydanticField(default=0.01, gt=0.0, le=1.0, description="Learning rate")
    negative_sampling_ratio: int = PydanticField(default=1, ge=1, description="Negative samples per positive sample")
    
    # Data configuration
    train_test_split: float = PydanticField(default=0.8, gt=0.0, lt=1.0, description="Training data fraction")
    validation_split: float = PydanticField(default=0.1, ge=0.0, lt=1.0, description="Validation data fraction")
    
    # Device and performance
    device: str = PydanticField(default="cpu", description="Training device (cpu, cuda, mps)")
    num_workers: int = PydanticField(default=0, ge=0, description="Number of data loading workers")
    
    # Regularization
    weight_decay: float = PydanticField(default=0.0, ge=0.0, description="Weight decay for regularization")
    dropout: float = PydanticField(default=0.0, ge=0.0, le=1.0, description="Dropout rate")
    
    # Early stopping
    early_stopping_patience: int = PydanticField(default=10, ge=1, description="Epochs to wait before early stopping")
    early_stopping_metric: str = PydanticField(default="mrr", description="Metric to monitor for early stopping")
    
    # Output configuration
    save_model: bool = PydanticField(default=True, description="Whether to save trained model")
    model_save_path: Optional[str] = PydanticField(None, description="Path to save model")
    log_interval: int = PydanticField(default=10, ge=1, description="Epochs between progress logs")
    
    @field_validator('train_test_split', 'validation_split')
    @classmethod
    def validate_splits(cls, v, info):
        """Ensure data splits are valid."""
        if 'train_test_split' in values:
            total = values['train_test_split'] + v
            if total >= 1.0:
                raise ValueError("Train and validation splits must sum to less than 1.0")
        return v
    
    def get_test_split(self) -> float:
        """Calculate test split fraction."""
        return 1.0 - self.train_test_split - self.validation_split


class EvaluationMetrics(BaseModel):
    """
    Comprehensive evaluation metrics for citation prediction models.
    
    Provides a standardized way to store and compare model performance
    across different evaluation runs and model types.
    """
    
    # Basic identification
    model_name: str = PydanticField(..., description="Name of evaluated model")
    model_version: str = PydanticField(..., description="Version of evaluated model")
    evaluation_set: str = PydanticField(..., description="Dataset used for evaluation (test/validation)")
    evaluated_at: datetime = PydanticField(default_factory=datetime.now, description="Evaluation timestamp")
    
    # Dataset statistics
    num_test_edges: int = PydanticField(..., ge=0, description="Number of test edges")
    num_entities: int = PydanticField(..., ge=0, description="Number of unique entities")
    
    # Ranking metrics
    mean_rank: Optional[float] = PydanticField(None, ge=1.0, description="Mean rank of correct answers")
    mean_reciprocal_rank: Optional[float] = PydanticField(None, ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    hits_at_1: Optional[float] = PydanticField(None, ge=0.0, le=1.0, description="Hits@1 accuracy")
    hits_at_3: Optional[float] = PydanticField(None, ge=0.0, le=1.0, description="Hits@3 accuracy")
    hits_at_5: Optional[float] = PydanticField(None, ge=0.0, le=1.0, description="Hits@5 accuracy")
    hits_at_10: Optional[float] = PydanticField(None, ge=0.0, le=1.0, description="Hits@10 accuracy")
    
    # Classification metrics (for threshold-based evaluation)
    precision: Optional[float] = PydanticField(None, ge=0.0, le=1.0, description="Precision score")
    recall: Optional[float] = PydanticField(None, ge=0.0, le=1.0, description="Recall score")
    f1_score: Optional[float] = PydanticField(None, ge=0.0, le=1.0, description="F1 score")
    auc_roc: Optional[float] = PydanticField(None, ge=0.0, le=1.0, description="Area Under ROC Curve")
    auc_pr: Optional[float] = PydanticField(None, ge=0.0, le=1.0, description="Area Under Precision-Recall Curve")
    
    # Additional context
    evaluation_duration_seconds: Optional[float] = PydanticField(None, ge=0.0, description="Time taken for evaluation")
    threshold_used: Optional[float] = PydanticField(None, ge=0.0, le=1.0, description="Threshold for binary classification")
    notes: Optional[str] = PydanticField(None, description="Additional notes about evaluation")
    
    def ranking_metrics_summary(self) -> Dict[str, float]:
        """Get summary of ranking metrics."""
        metrics = {}
        for metric in ['mean_rank', 'mean_reciprocal_rank', 'hits_at_1', 'hits_at_3', 'hits_at_5', 'hits_at_10']:
            value = getattr(self, metric)
            if value is not None:
                metrics[metric] = value
        return metrics
    
    def classification_metrics_summary(self) -> Dict[str, float]:
        """Get summary of classification metrics."""
        metrics = {}
        for metric in ['precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr']:
            value = getattr(self, metric)
            if value is not None:
                metrics[metric] = value
        return metrics
    
    def is_better_than(self, other: EvaluationMetrics, 
                      primary_metric: str = 'mean_reciprocal_rank') -> bool:
        """Compare performance with another evaluation."""
        self_metric = getattr(self, primary_metric, None)
        other_metric = getattr(other, primary_metric, None)
        
        if self_metric is None or other_metric is None:
            return False
        
        # For mean_rank, lower is better
        if primary_metric == 'mean_rank':
            return self_metric < other_metric
        else:
            # For most other metrics, higher is better
            return self_metric > other_metric
    
    def get_performance_grade(self, metric: str = 'mean_reciprocal_rank') -> str:
        """Get letter grade for performance."""
        value = getattr(self, metric, None)
        if value is None:
            return "N/A"
        
        if metric == 'mean_rank':
            # For mean rank, lower is better
            if value <= 5:
                return "A"
            elif value <= 10:
                return "B"
            elif value <= 20:
                return "C"
            else:
                return "D"
        else:
            # For other metrics, higher is better
            if value >= 0.8:
                return "A"
            elif value >= 0.6:
                return "B"
            elif value >= 0.4:
                return "C"
            else:
                return "D"


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch prediction operations.
    
    Supports efficient batch processing of citation predictions
    for large-scale analysis and evaluation.
    """
    
    source_paper_ids: List[str] = PydanticField(..., min_items=1, description="Papers that would do the citing")
    target_paper_ids: Optional[List[str]] = PydanticField(None, description="Specific target papers (if None, use all)")
    model_name: str = PydanticField(..., description="Name of model to use for predictions")
    
    # Filtering options
    top_k: int = PydanticField(default=10, ge=1, le=1000, description="Return top K predictions per source")
    threshold: float = PydanticField(default=0.5, ge=0.0, le=1.0, description="Minimum prediction score")
    exclude_existing_citations: bool = PydanticField(default=True, description="Filter out existing citations")
    
    # Processing options
    batch_size: int = PydanticField(default=1000, ge=1, description="Batch size for processing")
    include_explanations: bool = PydanticField(default=False, description="Include prediction explanations")
    
    @field_validator('source_paper_ids')
    @classmethod
    def validate_source_papers(cls, v):
        """Ensure source papers list is not empty."""
        if not v:
            raise ValueError("Must provide at least one source paper")
        return v


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch prediction operations.
    
    Contains prediction results along with metadata about the operation.
    """
    
    predictions: List[CitationPrediction] = PydanticField(..., description="Prediction results")
    request_id: Optional[str] = PydanticField(None, description="Unique request identifier")
    
    # Processing metadata
    total_predictions: int = PydanticField(..., description="Total number of predictions made")
    processing_time_seconds: float = PydanticField(..., description="Time taken to process request")
    model_used: str = PydanticField(..., description="Model name used for predictions")
    
    # Statistics
    high_confidence_count: int = PydanticField(default=0, description="Number of high confidence predictions")
    medium_confidence_count: int = PydanticField(default=0, description="Number of medium confidence predictions")  
    low_confidence_count: int = PydanticField(default=0, description="Number of low confidence predictions")
    
    @field_validator('predictions')
    @classmethod
    def compute_confidence_counts(cls, v, info):
        """Compute confidence level counts from predictions."""
        if 'high_confidence_count' not in values:
            high_count = sum(1 for p in v if p.confidence_level == PredictionConfidence.HIGH)
            medium_count = sum(1 for p in v if p.confidence_level == PredictionConfidence.MEDIUM)
            low_count = sum(1 for p in v if p.confidence_level == PredictionConfidence.LOW)
            
            values['high_confidence_count'] = high_count
            values['medium_confidence_count'] = medium_count
            values['low_confidence_count'] = low_count
        
        return v
    
    def get_predictions_by_confidence(self, confidence: PredictionConfidence) -> List[CitationPrediction]:
        """Filter predictions by confidence level."""
        return [p for p in self.predictions if p.confidence_level == confidence]
    
    def get_top_predictions(self, k: int = 10) -> List[CitationPrediction]:
        """Get top K predictions by score."""
        return sorted(self.predictions, key=lambda p: p.prediction_score, reverse=True)[:k]