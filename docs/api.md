# API Reference

## Overview

Citation Compass provides several APIs for interacting with citation data, machine learning models, and analytics services. This document covers the main API components and their usage.

## Core Services API

### ML Service API

#### `get_ml_service() -> TransEModelService`
Get the global ML service instance.

```python
from src.services.ml_service import get_ml_service

ml_service = get_ml_service()
```

#### `TransEModelService.predict_citations()`
Generate citation predictions for a paper.

```python
def predict_citations(
    source_paper_id: str, 
    candidate_paper_ids: Optional[List[str]] = None,
    top_k: int = 10,
    score_threshold: Optional[float] = None
) -> List[CitationPrediction]
```

**Parameters:**
- `source_paper_id`: Semantic Scholar paper ID
- `candidate_paper_ids`: Optional list of candidate papers (uses all if None)
- `top_k`: Number of top predictions to return
- `score_threshold`: Minimum prediction score threshold

**Returns:** List of `CitationPrediction` objects with scores and metadata

**Example:**
```python
predictions = ml_service.predict_citations(
    "649def34f8be52c8b66281af98ae884c09aef38f9", 
    top_k=5
)
for pred in predictions:
    print(f"Paper: {pred.target_paper_id}, Score: {pred.prediction_score:.3f}")
```

#### `TransEModelService.get_paper_embedding()`
Get embedding vector for a specific paper.

```python
def get_paper_embedding(paper_id: str) -> Optional[PaperEmbedding]
```

**Example:**
```python
embedding = ml_service.get_paper_embedding("649def34f8be52c8b66281af98ae884c09aef38f9")
if embedding:
    print(f"Embedding dimension: {embedding.embedding_dim}")
```

#### `TransEModelService.health_check()`
Perform health check on the model service.

```python
health = ml_service.health_check()
print(f"Status: {health['status']}")
print(f"Entities: {health['num_entities']}")
```

### Database API

#### `Neo4jConnection`
Main database connection class.

```python
from src.database.connection import Neo4jConnection

# Create connection
db = Neo4jConnection()

# Query database
papers = db.query("MATCH (p:Paper) RETURN p.title, p.citationCount LIMIT 10")

# Execute command
stats = db.execute("CREATE (p:Paper {title: $title})", {"title": "New Paper"})
```

#### Common Database Queries

**Get paper details:**
```python
from src.database.connection import get_paper_details

paper_data = get_paper_details(db, "649def34f8be52c8b66281af98ae884c09aef38f9")
```

**Search papers by keyword:**
```python
from src.database.connection import find_papers_by_keyword

results = find_papers_by_keyword(db, "machine learning")
```

**Get citation network:**
```python
from src.database.connection import get_citation_network

network = get_citation_network(db, "649def34f8be52c8b66281af98ae884c09aef38f9", depth=2)
```

### API Client

#### `UnifiedSemanticScholarClient`
Unified API client for Semantic Scholar integration.

```python
from src.data.unified_api_client import UnifiedSemanticScholarClient

client = UnifiedSemanticScholarClient()
```

#### Search Papers
```python
# Search by query
results = client.search_papers("attention mechanism", limit=10)

# Get paper details
paper = client.get_paper_details("649def34f8be52c8b66281af98ae884c09aef38f9")

# Get citations
citations = client.get_paper_citations("649def34f8be52c8b66281af98ae884c09aef38f9")

# Batch paper details
papers = client.batch_paper_details(["paper_id_1", "paper_id_2"])
```

#### Network Expansion
```python
# Expand citation network from seed papers
expanded_papers, stats = client.expand_citation_network(
    seed_papers=[paper1, paper2],
    max_depth=2,
    citation_threshold=10
)
```

## Data Models

### CitationPrediction
```python
@dataclass
class CitationPrediction:
    source_paper_id: str
    target_paper_id: str
    prediction_score: float
    model_name: str
    raw_score: Optional[float] = None
    confidence_level: PredictionConfidence = PredictionConfidence.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
```

### PaperEmbedding
```python
@dataclass
class PaperEmbedding:
    paper_id: str
    embedding: List[float]
    model_name: str
    embedding_dim: int
    created_at: datetime
```

### Paper
```python
class Paper(BaseModel):
    paperId: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    year: Optional[int] = None
    citationCount: Optional[int] = None
    authors: List[Author] = []
    venues: List[str] = []
    fields: List[str] = []
```

## Analytics API

### Network Analysis

#### `NetworkAnalyzer`
```python
from src.analytics.network_analysis import NetworkAnalyzer

analyzer = NetworkAnalyzer()

# Load citation network
graph = analyzer.load_citation_network(paper_ids=["id1", "id2"])

# Calculate centrality measures
centrality = analyzer.calculate_centrality_measures(graph)

# Detect communities
communities = analyzer.detect_communities(graph)

# Generate network statistics
stats = analyzer.generate_network_statistics(graph)
```

### Performance Metrics

#### `ModelEvaluator`
```python
from src.analytics.performance_metrics import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate model performance
metrics = evaluator.evaluate_model(predictions, ground_truth)

# Calculate ranking metrics
ranking_metrics = evaluator.calculate_ranking_metrics(predictions, ground_truth)
```

### Export Engine

#### `ExportEngine`
```python
from src.analytics.export_engine import ExportEngine

exporter = ExportEngine()

# Export to different formats
exporter.export_to_csv(data, "results.csv")
exporter.export_to_json(data, "results.json")
exporter.export_to_latex(data, "table.tex")
```

## Configuration API

### Settings
```python
from src.config.settings import get_config

config = get_config()

# Access different config sections
db_config = config.get_database_config()
ml_config = config.get_ml_config()
api_config = config.get_api_config()
cache_config = config.get_cache_config()
```

### Environment Variables

Required environment variables:

```bash
# Database
NEO4J_URI=neo4j+s://your-database-url
NEO4J_USER=neo4j  
NEO4J_PASSWORD=your-password

# API (optional)
SEMANTIC_SCHOLAR_API_KEY=your-api-key

# Logging (optional)
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

## Error Handling

### Custom Exceptions

#### `Neo4jError`
Database-related errors.

```python
from src.database.connection import Neo4jError

try:
    result = db.query("INVALID QUERY")
except Neo4jError as e:
    print(f"Database error: {e}")
```

#### `APIError` 
API client errors (implicit in requests exceptions).

```python
import requests

try:
    paper = client.get_paper_details("invalid_id")
except requests.exceptions.RequestException as e:
    print(f"API error: {e}")
```

## Rate Limiting

The API client includes automatic rate limiting:

```python
# Rate limiting is handled automatically
client = UnifiedSemanticScholarClient()

# This will automatically pace requests
for paper_id in large_paper_list:
    paper = client.get_paper_details(paper_id)
```

## Caching

### API Response Caching
```python
# Caching is automatic for API responses
client = UnifiedSemanticScholarClient()

# First call hits API
paper1 = client.get_paper_details("paper_id")  # API call

# Second call uses cache (within TTL)
paper2 = client.get_paper_details("paper_id")  # Cached
```

### ML Service Caching
```python
# Predictions are cached automatically
ml_service = get_ml_service()

# First call computes predictions
preds1 = ml_service.predict_citations("paper_id", top_k=10)  # Computed

# Second call uses cache
preds2 = ml_service.predict_citations("paper_id", top_k=10)  # Cached
```

## Performance Monitoring

### API Metrics
```python
client = UnifiedSemanticScholarClient()

# Get usage metrics
metrics = client.get_metrics()
print(f"Success rate: {metrics.success_rate:.1f}%")
print(f"Cache hit rate: {metrics.cache_hit_rate:.1f}%")
print(f"Average response time: {metrics.average_response_time:.2f}s")
```

### Health Checks
```python
# ML Service health
ml_health = ml_service.health_check()

# Database health  
db_health = db.test_connection()
```

## Usage Examples

### Complete Workflow Example
```python
from src.services.ml_service import get_ml_service
from src.data.unified_api_client import UnifiedSemanticScholarClient
from src.database.connection import Neo4jConnection

# Initialize services
ml_service = get_ml_service()
api_client = UnifiedSemanticScholarClient()
db = Neo4jConnection()

# Search for papers
search_results = api_client.search_papers("machine learning", limit=5)
paper_ids = [p['paperId'] for p in search_results.get('data', [])]

# Generate predictions for first paper
if paper_ids:
    predictions = ml_service.predict_citations(paper_ids[0], top_k=10)
    
    # Get details for predicted papers
    predicted_ids = [pred.target_paper_id for pred in predictions[:3]]
    paper_details = api_client.batch_paper_details(predicted_ids)
    
    # Store results in database (if needed)
    # ... custom storage logic
    
    print(f"Generated {len(predictions)} predictions")
    for pred in predictions[:3]:
        print(f"  {pred.target_paper_id}: {pred.prediction_score:.3f}")
```

---

*This API reference is updated as the system evolves. Last updated: August 2025*
