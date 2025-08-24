# Services API Reference

Comprehensive API documentation for the Academic Citation Platform service layer.

## Analytics Service

The core analytics service orchestrates network analysis, community detection, and temporal analysis operations.

::: src.services.analytics_service.AnalyticsService
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members:
        - __init__
        - get_network_overview
        - analyze_network
        - detect_communities
        - compute_centrality_metrics
        - analyze_temporal_trends
        - generate_report
        - run_comprehensive_analysis

---

### Analytics Service Factory

::: src.services.analytics_service.get_analytics_service
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

---

## ML Service

Machine learning service for citation prediction, model training, and embedding management.

::: src.services.ml_service.MLService
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members:
        - __init__
        - predict_citations
        - predict_batch_citations
        - compute_embeddings
        - find_similar_papers
        - train_model
        - evaluate_model
        - load_model
        - save_model

---

### TransE Model Service

::: src.services.ml_service.TransEModelService
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members:
        - __init__
        - train
        - predict
        - evaluate
        - save_model
        - load_model

---

### ML Service Factory

::: src.services.ml_service.get_ml_service
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

---

## Service Configuration

### Analytics Configuration

```python
@dataclass
class AnalyticsConfig:
    max_workers: int = 4
    cache_enabled: bool = True
    cache_ttl: int = 3600
    request_timeout: int = 300
    
    # Network analysis settings
    community_detection_method: str = 'louvain'
    centrality_measures: List[str] = field(default_factory=lambda: ['pagerank', 'betweenness'])
    
    # Temporal analysis settings
    time_window_days: int = 365
    trend_detection_threshold: float = 0.05
```

### ML Configuration

```python
@dataclass
class MLConfig:
    model_path: str = "models/transe_citation_model.pt"
    embedding_dim: int = 128
    batch_size: int = 1024
    device: str = "auto"  # 'auto', 'cpu', 'cuda', 'mps'
    
    # Training settings
    learning_rate: float = 0.01
    epochs: int = 100
    margin: float = 1.0
    negative_sampling_ratio: int = 1
    
    # Prediction settings
    prediction_threshold: float = 0.5
    max_predictions: int = 1000
    confidence_interval: float = 0.95
```

## Service Error Handling

### Common Exceptions

```python
class ServiceError(Exception):
    """Base exception for service layer errors."""
    pass

class ModelNotFoundError(ServiceError):
    """Raised when ML model is not available."""
    pass

class DataValidationError(ServiceError):
    """Raised when input data fails validation."""
    pass

class NetworkAnalysisError(ServiceError):
    """Raised when network analysis operations fail."""
    pass

class PredictionError(ServiceError):
    """Raised when ML predictions fail."""
    pass
```

### Error Recovery

Services implement graceful degradation:

```python
async def predict_citations_with_fallback(paper_id: str) -> List[Prediction]:
    try:
        # Try ML-based prediction
        return await ml_service.predict_citations(paper_id)
    except ModelNotFoundError:
        # Fallback to similarity-based prediction
        logger.warning("ML model unavailable, using similarity fallback")
        return await similarity_service.predict_citations(paper_id)
    except Exception as e:
        # Final fallback to cached results
        logger.error(f"All prediction methods failed: {e}")
        return await cache_service.get_cached_predictions(paper_id)
```

## Performance Monitoring

### Service Metrics

```python
class ServiceMetrics:
    def __init__(self):
        self.request_count = Counter()
        self.request_duration = Histogram()
        self.error_count = Counter()
    
    def record_request(self, service: str, operation: str, duration: float):
        self.request_count.inc({'service': service, 'operation': operation})
        self.request_duration.observe(duration, {'service': service})
    
    def record_error(self, service: str, error_type: str):
        self.error_count.inc({'service': service, 'error_type': error_type})
```

### Health Checks

```python
async def check_service_health() -> Dict[str, bool]:
    """Check health of all services."""
    health_status = {}
    
    # Check analytics service
    try:
        analytics = get_analytics_service()
        await analytics.get_network_overview()
        health_status['analytics'] = True
    except Exception:
        health_status['analytics'] = False
    
    # Check ML service
    try:
        ml = get_ml_service()
        await ml.load_model()
        health_status['ml'] = True
    except Exception:
        health_status['ml'] = False
    
    return health_status
```

## Usage Examples

### Basic Analytics Workflow

```python
from src.services.analytics_service import get_analytics_service

# Initialize service
analytics = get_analytics_service()

# Get network overview
overview = await analytics.get_network_overview()
print(f"Network has {overview.num_papers} papers and {overview.num_citations} citations")

# Detect communities
communities = await analytics.detect_communities(method='louvain')
print(f"Found {len(communities.communities)} research communities")

# Analyze temporal trends
trends = await analytics.analyze_temporal_trends(
    time_window=TimeWindow(start_year=2020, end_year=2023)
)
print(f"Citation growth rate: {trends.growth_rate:.2%}")
```

### Basic ML Workflow

```python
from src.services.ml_service import get_ml_service

# Initialize service  
ml = get_ml_service()

# Generate predictions for a paper
predictions = await ml.predict_citations(
    paper_id="abc123",
    top_k=10
)

for pred in predictions:
    print(f"Paper: {pred.target_title}")
    print(f"Confidence: {pred.confidence:.3f}")
    print(f"Score: {pred.score:.3f}")
    print("---")

# Find similar papers
similar = await ml.find_similar_papers(
    paper_id="abc123", 
    threshold=0.8
)
print(f"Found {len(similar)} similar papers")
```

### Combined Analysis

```python
async def comprehensive_paper_analysis(paper_id: str):
    """Combine analytics and ML for comprehensive analysis."""
    
    # Get services
    analytics = get_analytics_service()
    ml = get_ml_service()
    
    # Run analyses in parallel
    network_task = analytics.analyze_paper_network(paper_id)
    prediction_task = ml.predict_citations(paper_id)
    similarity_task = ml.find_similar_papers(paper_id)
    
    # Wait for completion
    network_analysis, predictions, similar_papers = await asyncio.gather(
        network_task, prediction_task, similarity_task
    )
    
    # Generate comprehensive report
    return {
        'paper_id': paper_id,
        'network_position': network_analysis,
        'predicted_citations': predictions,
        'similar_papers': similar_papers,
        'analysis_timestamp': datetime.now()
    }
```

## Service Integration Patterns

### Dependency Injection

```python
class AnalyticsService:
    def __init__(self, 
                 database: UnifiedDatabase,
                 ml_service: MLService,
                 cache: CacheService):
        self.database = database
        self.ml_service = ml_service  
        self.cache = cache
```

### Factory Pattern

```python
def create_analytics_service(config: AnalyticsConfig) -> AnalyticsService:
    """Factory function for analytics service with full dependency injection."""
    
    database = UnifiedDatabase(config.database_config)
    ml_service = MLService(config.ml_config)
    cache = CacheService(config.cache_config)
    
    return AnalyticsService(
        database=database,
        ml_service=ml_service,
        cache=cache
    )
```

### Service Registry

```python
class ServiceRegistry:
    def __init__(self):
        self._services = {}
    
    def register(self, name: str, service: Any):
        self._services[name] = service
    
    def get(self, name: str) -> Any:
        if name not in self._services:
            raise ServiceNotFoundError(f"Service '{name}' not registered")
        return self._services[name]

# Global registry
registry = ServiceRegistry()
registry.register('analytics', get_analytics_service())
registry.register('ml', get_ml_service())
```