"""
Configuration management for API clients and external services.

This module centralizes all configuration for the Academic Citation Platform,
including API endpoints, rate limits, field specifications, and connection parameters.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
from pathlib import Path


@dataclass
class SemanticScholarConfig:
    """
    Configuration for Semantic Scholar API integration.
    
    Consolidates configuration from all three reference codebases:
    - knowledge-cartography: Basic API usage
    - citation-map-dashboard: Graph extraction needs  
    - academic-citation-prediction: Advanced pagination and rate limiting
    """
    
    # Base configuration
    base_url: str = "https://api.semanticscholar.org/graph/v1"
    api_key: Optional[str] = None
    rate_limit_pause: float = 1.0  # Seconds between requests
    request_timeout: int = 30      # Request timeout in seconds
    
    # Pagination settings
    max_batch_size: int = 500      # Maximum items per batch request
    max_pagination_limit: int = 10000  # Maximum total items to retrieve
    default_page_size: int = 100   # Default items per page
    
    # Field specifications for different use cases
    paper_fields: str = "paperId,title,abstract,citationCount,publicationDate,year,authors,venues,fieldsOfStudy,references,citations"
    author_fields: str = "authorId,name,paperCount,citationCount,hIndex"
    citation_fields: str = "paperId,title,citationCount"
    
    # Endpoints
    endpoints: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize endpoints after dataclass creation."""
        if self.endpoints is None:
            self.endpoints = {
                # Paper endpoints
                'paper_search': '/paper/search',
                'paper_search_bulk': '/paper/search/bulk',
                'paper_batch': '/paper/batch',
                'paper_details': '/paper/{paper_id}',
                'paper_citations': '/paper/{paper_id}/citations',
                'paper_references': '/paper/{paper_id}/references',
                
                # Author endpoints  
                'author_search': '/author/search',
                'author_search_bulk': '/author/search/bulk',
                'author_batch': '/author/batch',
                'author_details': '/author/{author_id}',
                'author_papers': '/author/{author_id}/papers',
            }
        
        # Load API key from environment if not provided
        if self.api_key is None:
            self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")


@dataclass  
class Neo4jConfig:
    """
    Configuration for Neo4j database connections.
    
    Supports both local and cloud (Aura) deployments used across reference codebases.
    """
    
    uri: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: str = "neo4j"  # Default database name
    
    # Connection settings
    connection_timeout: int = 30
    max_retry_attempts: int = 3
    
    def __post_init__(self):
        """Load credentials from environment variables."""
        if self.uri is None:
            self.uri = os.getenv("NEO4J_URI")
        if self.username is None:
            self.username = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
        if self.password is None:
            self.password = os.getenv("NEO4J_PWD") or os.getenv("NEO4J_PASSWORD")
    
    def validate(self) -> bool:
        """Validate that all required connection parameters are present."""
        return all([self.uri, self.username, self.password])
    
    def get_missing_params(self) -> list[str]:
        """Return list of missing required parameters."""
        missing = []
        if not self.uri:
            missing.append("NEO4J_URI")
        if not self.username:
            missing.append("NEO4J_USER/NEO4J_USERNAME")
        if not self.password:
            missing.append("NEO4J_PWD/NEO4J_PASSWORD")
        return missing


@dataclass
class CacheConfig:
    """Configuration for caching layer to improve performance."""
    
    # Cache settings for different data types
    api_response_ttl: int = 300      # 5 minutes for API responses
    db_query_ttl: int = 600          # 10 minutes for database queries
    prediction_ttl: int = 3600       # 1 hour for ML predictions
    
    # Cache size limits
    max_cache_size: int = 1000       # Maximum cached items
    
    # Cache backends
    use_redis: bool = False          # Use Redis if available, else in-memory
    redis_url: Optional[str] = None
    
    def __post_init__(self):
        """Load Redis configuration from environment."""
        if self.redis_url is None:
            self.redis_url = os.getenv("REDIS_URL")
            if self.redis_url:
                self.use_redis = True


@dataclass
class MLConfig:
    """Configuration for machine learning model integration."""
    
    # Model paths (will be populated from reference codebases)
    model_dir: Path = Path("models")
    transE_model_path: Optional[Path] = None
    entity_mapping_path: Optional[Path] = None
    training_metadata_path: Optional[Path] = None
    
    # Model parameters
    embedding_dim: int = 128
    prediction_batch_size: int = 1000
    top_k_predictions: int = 10
    
    # Device configuration
    device: str = "cpu"  # Will auto-detect GPU if available
    
    def __post_init__(self):
        """Initialize model paths and device detection."""
        # Set up model paths relative to project root
        if self.transE_model_path is None:
            self.transE_model_path = self.model_dir / "transe_citation_model.pt"
        if self.entity_mapping_path is None:
            self.entity_mapping_path = self.model_dir / "entity_mapping.pkl"
        if self.training_metadata_path is None:
            self.training_metadata_path = self.model_dir / "training_metadata.pkl"
        
        # Auto-detect device (will check for CUDA/MPS in future)
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
        except ImportError:
            pass  # PyTorch not available, stick with CPU


class PlatformConfig:
    """
    Master configuration class that aggregates all component configurations.
    
    This provides a single entry point for all configuration needs across
    the integrated platform, making it easy to manage settings and ensure
    consistency across all modules.
    """
    
    def __init__(self, config_overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize platform configuration with optional overrides.
        
        Args:
            config_overrides: Dictionary of configuration overrides for testing
                             or environment-specific settings
        """
        # Initialize component configurations
        self.semantic_scholar = SemanticScholarConfig()
        self.neo4j = Neo4jConfig()
        self.cache = CacheConfig()
        self.ml = MLConfig()
        
        # Apply any overrides
        if config_overrides:
            self._apply_overrides(config_overrides)
        
        # Validate configuration
        self._validate_config()
    
    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply configuration overrides for specific components."""
        for component, settings in overrides.items():
            if hasattr(self, component):
                component_config = getattr(self, component)
                for key, value in settings.items():
                    if hasattr(component_config, key):
                        setattr(component_config, key, value)
    
    def _validate_config(self) -> None:
        """Validate all configuration components and log warnings for issues."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Validate Neo4j configuration
        if not self.neo4j.validate():
            missing = self.neo4j.get_missing_params()
            logger.warning(f"Neo4j configuration incomplete. Missing: {missing}")
        
        # Validate model paths exist
        if not self.ml.transE_model_path.exists():
            logger.warning(f"TransE model not found at {self.ml.transE_model_path}")
        
        # Log configuration summary
        logger.info(f"Platform configuration loaded:")
        logger.info(f"  - Neo4j: {self.neo4j.uri}")
        logger.info(f"  - ML Device: {self.ml.device}")
        logger.info(f"  - Cache: {'Redis' if self.cache.use_redis else 'Memory'}")
    
    def get_api_config(self) -> SemanticScholarConfig:
        """Get API configuration for external services."""
        return self.semantic_scholar
    
    def get_db_config(self) -> Neo4jConfig:
        """Get database configuration."""
        return self.neo4j
    
    def get_cache_config(self) -> CacheConfig:
        """Get caching configuration."""
        return self.cache
    
    def get_ml_config(self) -> MLConfig:
        """Get machine learning configuration."""
        return self.ml


# Global configuration instance
_config_instance: Optional[PlatformConfig] = None

def get_config(config_overrides: Optional[Dict[str, Any]] = None) -> PlatformConfig:
    """
    Get the global platform configuration instance.
    
    This implements a singleton pattern to ensure consistent configuration
    across the entire platform while allowing for testing overrides.
    
    Args:
        config_overrides: Optional overrides for testing or environment-specific settings
        
    Returns:
        PlatformConfig instance
    """
    global _config_instance
    
    if _config_instance is None or config_overrides is not None:
        _config_instance = PlatformConfig(config_overrides)
    
    return _config_instance


def reset_config() -> None:
    """Reset global configuration instance. Useful for testing."""
    global _config_instance
    _config_instance = None