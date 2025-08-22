"""
Centralized configuration management for Academic Citation Platform.

This module provides unified configuration management combining patterns from:
- academic-citation-predictions: Environment-based API configuration
- citation-map-dashboard: dotenv integration and ML parameters  
- knowledge-cartography: Production logging and caching settings

Configuration sections:
- Database: Neo4j connection parameters
- API: Semantic Scholar API settings
- ML: Machine learning model parameters
- Logging: Application logging configuration
- Cache: Data caching settings
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ====================================================================
# CONFIGURATION DATACLASSES
# ====================================================================

@dataclass
class DatabaseConfig:
    """Neo4j database configuration."""
    uri: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    database: str = "neo4j"
    max_connection_lifetime: int = 300
    max_connection_pool_size: int = 100
    connection_timeout: int = 30
    trust: str = "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"
    
    def __post_init__(self):
        """Load configuration from environment variables."""
        # Support multiple naming conventions
        self.uri = self.uri or os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL")
        self.user = self.user or os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
        self.password = self.password or os.getenv("NEO4J_PWD") or os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", self.database)
        
    def validate(self) -> bool:
        """Validate required configuration parameters."""
        return all([self.uri, self.user, self.password])
        
    def get_connection_params(self) -> Dict[str, Any]:
        """Get Neo4j driver connection parameters."""
        return {
            "uri": self.uri,
            "auth": (self.user, self.password),
            "max_connection_lifetime": self.max_connection_lifetime,
            "max_connection_pool_size": self.max_connection_pool_size,
            "connection_timeout": self.connection_timeout,
            "trust": self.trust
        }


@dataclass  
class APIConfig:
    """Semantic Scholar API configuration."""
    base_url: str = "https://api.semanticscholar.org/graph/v1"
    api_key: Optional[str] = None
    rate_limit_requests: int = 100
    rate_limit_period: int = 300  # 5 minutes
    timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 0.3
    batch_size: int = 500
    max_papers_per_request: int = 1000
    
    def __post_init__(self):
        """Load API configuration from environment."""
        self.api_key = self.api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")
        
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {
            "User-Agent": "Academic-Citation-Platform/1.0",
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers


@dataclass
class MLConfig:
    """Machine learning model configuration."""
    # TransE Model Parameters
    embedding_dim: int = 128
    margin: float = 1.0
    learning_rate: float = 0.001
    batch_size: int = 1024
    num_epochs: int = 100
    
    # Training Parameters  
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    negative_sampling_ratio: int = 1
    
    # Model Persistence
    models_dir: str = "models"
    checkpoint_frequency: int = 10
    early_stopping_patience: int = 10
    
    # Evaluation Parameters
    top_k_values: list = field(default_factory=lambda: [1, 3, 5, 10, 20])
    prediction_batch_size: int = 10000
    
    def __post_init__(self):
        """Validate ML configuration."""
        if abs(self.train_split + self.validation_split + self.test_split - 1.0) > 0.001:
            raise ValueError("Train/validation/test splits must sum to 1.0")
        
        # Create models directory if it doesn't exist
        Path(self.models_dir).mkdir(exist_ok=True)


@dataclass
class LoggingConfig:
    """Application logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_file: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    def __post_init__(self):
        """Load logging configuration from environment."""
        self.level = os.getenv("LOG_LEVEL", self.level).upper()
        self.log_file = os.getenv("LOG_FILE", self.log_file)
        
    def setup_logging(self):
        """Configure application logging."""
        logging.basicConfig(
            level=getattr(logging, self.level),
            format=self.format,
            datefmt=self.date_format
        )
        
        # Add file handler if specified
        if self.log_file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            file_handler.setFormatter(
                logging.Formatter(self.format, self.date_format)
            )
            logging.getLogger().addHandler(file_handler)


@dataclass
class CacheConfig:
    """Data caching configuration."""
    enabled: bool = True
    default_ttl: int = 300  # 5 minutes
    max_size: int = 1000
    
    # Specific TTL settings for different data types
    database_query_ttl: int = 300  # 5 minutes
    api_response_ttl: int = 3600   # 1 hour  
    model_prediction_ttl: int = 1800  # 30 minutes
    network_visualization_ttl: int = 600  # 10 minutes
    
    def __post_init__(self):
        """Load cache configuration from environment."""
        self.enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.default_ttl = int(os.getenv("CACHE_DEFAULT_TTL", self.default_ttl))


@dataclass
class PathsConfig:
    """File paths and directory configuration."""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    outputs_dir: Path = field(default_factory=lambda: Path("outputs"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    temp_dir: Path = field(default_factory=lambda: Path("temp"))
    
    def __post_init__(self):
        """Ensure all directories exist."""
        for dir_path in [self.data_dir, self.models_dir, self.outputs_dir, 
                        self.logs_dir, self.temp_dir]:
            dir_path.mkdir(exist_ok=True)
            
    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert relative path to absolute path from project root."""
        return self.project_root / relative_path


# ====================================================================
# MAIN CONFIGURATION CLASS
# ====================================================================

class Settings:
    """
    Main configuration class aggregating all configuration sections.
    
    This class provides a single point of access for all application
    configuration, combining database, API, ML, logging, and other settings.
    """
    
    def __init__(self):
        """Initialize all configuration sections."""
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.ml = MLConfig()
        self.logging = LoggingConfig()
        self.cache = CacheConfig()
        self.paths = PathsConfig()
        
        # Set up logging immediately
        self.logging.setup_logging()
        
    def validate(self) -> Dict[str, bool]:
        """
        Validate all configuration sections.
        
        Returns:
            Dictionary with validation results for each section
        """
        return {
            "database": self.database.validate(),
            "api": True,  # API key is optional
            "ml": True,   # ML config is always valid after __post_init__
            "logging": True,
            "cache": True,
            "paths": True
        }
        
    def get_validation_errors(self) -> list:
        """
        Get list of configuration validation errors.
        
        Returns:
            List of error messages for invalid configurations
        """
        errors = []
        validation = self.validate()
        
        if not validation["database"]:
            errors.append("Database configuration invalid: missing required Neo4j credentials")
            
        return errors
        
    def is_valid(self) -> bool:
        """Check if all configuration is valid."""
        return len(self.get_validation_errors()) == 0
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary (excluding sensitive information).
        
        Returns:
            Dictionary representation of non-sensitive configuration
        """
        return {
            "database": {
                "uri": self.database.uri,
                "user": self.database.user,
                "database": self.database.database,
                "has_password": bool(self.database.password)
            },
            "api": {
                "base_url": self.api.base_url,
                "has_api_key": bool(self.api.api_key),
                "rate_limit_requests": self.api.rate_limit_requests,
                "timeout": self.api.timeout
            },
            "ml": {
                "embedding_dim": self.ml.embedding_dim,
                "batch_size": self.ml.batch_size,
                "num_epochs": self.ml.num_epochs,
                "models_dir": self.ml.models_dir
            },
            "logging": {
                "level": self.logging.level,
                "log_file": self.logging.log_file
            },
            "cache": {
                "enabled": self.cache.enabled,
                "default_ttl": self.cache.default_ttl
            }
        }


# ====================================================================
# GLOBAL SETTINGS INSTANCE
# ====================================================================

# Create global settings instance
settings = Settings()

# Validate configuration on import
if not settings.is_valid():
    errors = settings.get_validation_errors()
    logging.warning(f"Configuration validation errors: {errors}")


# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def get_env_info() -> Dict[str, str]:
    """
    Get information about the current environment.
    
    Returns:
        Dictionary with environment information
    """
    return {
        "python_version": os.sys.version,
        "cwd": os.getcwd(),
        "user": os.getenv("USER", "unknown"),
        "home": os.getenv("HOME", "unknown"),
        "path": os.getenv("PATH", "")[:100] + "..." if len(os.getenv("PATH", "")) > 100 else os.getenv("PATH", "")
    }


def create_sample_env_file(file_path: str = ".env.example") -> None:
    """
    Create a sample .env file with all required environment variables.
    
    Args:
        file_path: Path where to create the sample file
    """
    sample_content = """# Academic Citation Platform Environment Configuration

# Neo4j Database Configuration
NEO4J_URI=neo4j+s://your-database-url
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

# Semantic Scholar API Configuration (optional)
SEMANTIC_SCHOLAR_API_KEY=your-api-key

# Logging Configuration (optional)
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Cache Configuration (optional)  
CACHE_ENABLED=true
CACHE_DEFAULT_TTL=300

# Development/Production Mode
ENVIRONMENT=development
DEBUG=false
"""
    
    with open(file_path, 'w') as f:
        f.write(sample_content)
        
    logging.info(f"Created sample environment file: {file_path}")


if __name__ == "__main__":
    # Print configuration summary when run directly
    print("Academic Citation Platform Configuration")
    print("=" * 50)
    print(f"Configuration valid: {settings.is_valid()}")
    
    if not settings.is_valid():
        print("Validation errors:")
        for error in settings.get_validation_errors():
            print(f"  - {error}")
    
    print("\nConfiguration Summary:")
    import json
    print(json.dumps(settings.to_dict(), indent=2))