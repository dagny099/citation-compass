"""
Comprehensive data models for Academic Citation Platform.

This module provides unified data models supporting:
- Interactive exploration (from knowledge-cartography)
- ML prediction workflows (from citation-map-dashboard)  
- Data ingestion and ETL (from academic-citation-prediction)
- Network visualization and analysis
- API requests and responses

All models use Pydantic for validation and serialization.
"""

# Core entity models
from .paper import Paper, PaperCreate, PaperUpdate, PaperDataClass
from .author import Author, AuthorCreate, AuthorUpdate  
from .venue import Venue, VenueCreate, VenueUpdate
from .field import Field, FieldCreate, FieldUpdate
from .citation import Citation, CitationCreate

# ML and prediction models
from .ml import (
    PaperEmbedding,
    CitationPrediction,
    ModelMetadata,
    TrainingConfig,
    EvaluationMetrics,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelType,
    PredictionConfidence
)

# Network visualization models
from .network import (
    NetworkNode,
    NetworkEdge,
    NetworkGraph,
    VisualizationConfig,
    NetworkAnalysis,
    NodeType,
    EdgeType,
    VisualizationBackend,
    LayoutAlgorithm,
    NodeSize,
    EdgeWidth
)

# API models
from .api import (
    APIResponse,
    PaginatedResponse,
    APIError,
    SearchRequest,
    BatchRequest,
    BatchResponse,
    CitationSearchRequest,
    PredictionRequest,
    NetworkAnalysisRequest,
    HealthCheckResponse,
    PaginationParams,
    PaginationMeta,
    APIStatus,
    ErrorType
)

__all__ = [
    # Core entity models
    "Paper", "PaperCreate", "PaperUpdate", "PaperDataClass",
    "Author", "AuthorCreate", "AuthorUpdate", 
    "Venue", "VenueCreate", "VenueUpdate",
    "Field", "FieldCreate", "FieldUpdate",
    "Citation", "CitationCreate",
    
    # ML models
    "PaperEmbedding",
    "CitationPrediction", 
    "ModelMetadata",
    "TrainingConfig",
    "EvaluationMetrics",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "ModelType",
    "PredictionConfidence",
    
    # Network models
    "NetworkNode",
    "NetworkEdge", 
    "NetworkGraph",
    "VisualizationConfig",
    "NetworkAnalysis",
    "NodeType",
    "EdgeType",
    "VisualizationBackend", 
    "LayoutAlgorithm",
    "NodeSize",
    "EdgeWidth",
    
    # API models
    "APIResponse",
    "PaginatedResponse",
    "APIError",
    "SearchRequest",
    "BatchRequest",
    "BatchResponse", 
    "CitationSearchRequest",
    "PredictionRequest",
    "NetworkAnalysisRequest",
    "HealthCheckResponse",
    "PaginationParams",
    "PaginationMeta",
    "APIStatus",
    "ErrorType",
]