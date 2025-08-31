# Data Models API Reference

Comprehensive documentation for data models, schemas, and data structures used throughout the Academic Citation Platform.

## Core Data Models

### Paper Model

The central data structure representing academic papers.

::: models.paper.Paper
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### Citation Model

Represents citation relationships between papers.

::: models.citation.Citation
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### Author Model

Represents academic authors and their affiliations.

::: models.author.Author
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### Venue Model

Represents publication venues (journals, conferences).

::: models.venue.Venue
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### Field Model

Represents academic fields of study.

::: models.field.Field
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

---

## Network Models

### Network Graph

::: models.network.NetworkGraph
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### Network Analysis

::: models.network.NetworkAnalysis
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

---

## Machine Learning Models

### Citation Prediction

::: models.ml.CitationPrediction
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### Training Configuration

::: models.ml.TrainingConfig
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### Evaluation Metrics

::: models.ml.EvaluationMetrics
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### Paper Embedding

::: models.ml.PaperEmbedding
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

---

## API Models

### API Response Models

::: models.api.APIResponse
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### API Error

::: models.api.APIError
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### Pagination

::: models.api.PaginatedResponse
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

---

## Model Schemas and Validation

### Pydantic Base Models

All models inherit from enhanced Pydantic base classes:

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import re

class CitationPlatformBaseModel(BaseModel):
    """Base model with common configuration."""
    
    class Config:
        # Enable ORM mode for database integration
        orm_mode = True
        
        # Allow population by field name or alias
        allow_population_by_field_name = True
        
        # Validate assignment to ensure data integrity
        validate_assignment = True
        
        # Use enum values for serialization
        use_enum_values = True
        
        # Generate JSON schema
        schema_extra = {
            "example": {}
        }
```

### Field Validation Examples

```python
class Paper(CitationPlatformBaseModel):
    paper_id: str = Field(..., regex=r'^[a-zA-Z0-9\-_]+$', description="Unique paper identifier")
    title: str = Field(..., min_length=1, max_length=500, description="Paper title")
    abstract: Optional[str] = Field(None, max_length=5000, description="Paper abstract")
    year: Optional[int] = Field(None, ge=1900, le=2030, description="Publication year")
    citation_count: int = Field(0, ge=0, description="Number of citations")
    
    @validator('title')
    def validate_title(cls, v):
        """Ensure title is properly formatted."""
        if not v.strip():
            raise ValueError('Title cannot be empty or whitespace only')
        return v.strip()
    
    @validator('abstract')
    def validate_abstract(cls, v):
        """Clean and validate abstract."""
        if v:
            # Remove excessive whitespace
            v = re.sub(r'\s+', ' ', v.strip())
            if len(v) < 10:
                raise ValueError('Abstract too short (minimum 10 characters)')
        return v
```

### Custom Field Types

```python
from pydantic import BaseModel, Field
from typing import NewType, List
from decimal import Decimal

# Custom types for semantic clarity
PaperID = NewType('PaperID', str)
AuthorID = NewType('AuthorID', str)
VenueID = NewType('VenueID', str)
ConfidenceScore = NewType('ConfidenceScore', float)

class Prediction(BaseModel):
    source_paper: PaperID = Field(..., description="Source paper ID")
    target_paper: PaperID = Field(..., description="Target paper ID")
    confidence: ConfidenceScore = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    score: float = Field(..., description="Raw model score")
    model_version: str = Field(..., description="Model version used")
    
    @validator('confidence')
    def round_confidence(cls, v):
        """Round confidence to 3 decimal places."""
        return round(v, 3)
```

## Model Relationships

### Entity Relationships

```python
from typing import List, Optional, ForwardRef
from pydantic import BaseModel

# Forward references for circular dependencies
AuthorRef = ForwardRef('Author')
VenueRef = ForwardRef('Venue')
FieldRef = ForwardRef('Field')

class Paper(BaseModel):
    paper_id: str
    title: str
    
    # Relationships
    authors: List[AuthorRef] = []
    venue: Optional[VenueRef] = None
    fields: List[FieldRef] = []
    citations: List['Citation'] = []
    references: List['Citation'] = []
    
class Author(BaseModel):
    author_id: str
    name: str
    
    # Relationships  
    papers: List['Paper'] = []
    affiliations: List[str] = []

# Update forward references
Paper.model_rebuild()
Author.model_rebuild()
```

### Database Integration

```python
from src.database.connection import Neo4jConnection
from typing import Optional

class PaperRepository:
    def __init__(self, connection: Neo4jConnection):
        self.conn = connection
    
    async def get_by_id(self, paper_id: str) -> Optional[Paper]:
        """Retrieve paper by ID with full relationships."""
        
        query = """
        MATCH (p:Paper {paper_id: $paper_id})
        OPTIONAL MATCH (p)<-[:AUTHORED]-(a:Author)
        OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
        OPTIONAL MATCH (p)-[:BELONGS_TO]->(f:Field)
        OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
        OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
        
        RETURN p,
               collect(DISTINCT a) as authors,
               v as venue,
               collect(DISTINCT f) as fields,
               collect(DISTINCT cited) as citations,
               collect(DISTINCT citing) as cited_by
        """
        
        result = await self.conn.run(query, paper_id=paper_id)
        record = await result.single()
        
        if not record:
            return None
            
        return Paper(
            paper_id=record['p']['paper_id'],
            title=record['p']['title'],
            abstract=record['p'].get('abstract'),
            year=record['p'].get('year'),
            authors=[Author(**author) for author in record['authors']],
            venue=Venue(**record['venue']) if record['venue'] else None,
            fields=[Field(**field) for field in record['fields']],
            citation_count=len(record['citations'])
        )
```

## Serialization and Export

### JSON Serialization

```python
import json
from typing import Any, Dict
from datetime import datetime

class CitationPlatformJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for platform models."""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'dict'):
            # Pydantic models
            return obj.dict()
        elif hasattr(obj, '__dict__'):
            # Other objects with __dict__
            return obj.__dict__
        else:
            return super().default(obj)

# Usage
paper = Paper(paper_id="123", title="Example Paper")
json_string = json.dumps(paper, cls=CitationPlatformJSONEncoder, indent=2)
```

### Export Formats

```python
from typing import Union, List
import pandas as pd

class ModelExporter:
    @staticmethod
    def to_dataframe(models: List[BaseModel]) -> pd.DataFrame:
        """Convert list of models to pandas DataFrame."""
        data = [model.dict() for model in models]
        return pd.DataFrame(data)
    
    @staticmethod
    def to_csv(models: List[BaseModel], file_path: str):
        """Export models to CSV file."""
        df = ModelExporter.to_dataframe(models)
        df.to_csv(file_path, index=False)
    
    @staticmethod
    def to_latex(models: List[BaseModel], caption: str = "") -> str:
        """Convert models to LaTeX table."""
        df = ModelExporter.to_dataframe(models)
        return df.to_latex(index=False, caption=caption)

# Usage
papers = [Paper(paper_id=f"{i}", title=f"Paper {i}") for i in range(10)]
ModelExporter.to_csv(papers, "papers.csv")
latex_table = ModelExporter.to_latex(papers, "Sample Papers")
```

## Model Validation and Testing

### Unit Tests for Models

```python
import pytest
from pydantic import ValidationError

class TestPaperModel:
    def test_valid_paper_creation(self):
        paper = Paper(
            paper_id="valid123",
            title="A Valid Paper Title",
            abstract="This is a valid abstract with sufficient length.",
            year=2023,
            citation_count=5
        )
        assert paper.paper_id == "valid123"
        assert paper.citation_count == 5
    
    def test_invalid_paper_id(self):
        with pytest.raises(ValidationError) as exc_info:
            Paper(
                paper_id="invalid@id!",  # Contains invalid characters
                title="Valid Title"
            )
        assert "paper_id" in str(exc_info.value)
    
    def test_title_validation(self):
        with pytest.raises(ValidationError):
            Paper(paper_id="123", title="   ")  # Whitespace only
        
        with pytest.raises(ValidationError):
            Paper(paper_id="123", title="")  # Empty string
```

### Model Factory for Testing

```python
import factory
from factory import fuzzy
from datetime import datetime

class PaperFactory(factory.Factory):
    class Meta:
        model = Paper
    
    paper_id = factory.Sequence(lambda n: f"paper_{n}")
    title = factory.Faker('sentence', nb_words=6)
    abstract = factory.Faker('paragraph')
    year = fuzzy.FuzzyInteger(2000, 2023)
    citation_count = fuzzy.FuzzyInteger(0, 100)

class AuthorFactory(factory.Factory):
    class Meta:
        model = Author
    
    author_id = factory.Sequence(lambda n: f"author_{n}")
    name = factory.Faker('name')
    
class CitationFactory(factory.Factory):
    class Meta:
        model = Citation
    
    source_paper = factory.SubFactory(PaperFactory)
    target_paper = factory.SubFactory(PaperFactory)
    citation_context = factory.Faker('sentence')

# Usage in tests
def test_paper_with_citations():
    paper = PaperFactory()
    citations = CitationFactory.create_batch(5, source_paper=paper)
    
    assert len(citations) == 5
    assert all(c.source_paper.paper_id == paper.paper_id for c in citations)
```

## Performance Considerations

### Model Optimization

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class OptimizedPaper(BaseModel):
    """Memory-optimized paper model for large-scale processing."""
    
    # Use slots to reduce memory overhead
    __slots__ = ('paper_id', 'title', 'year', 'citation_count')
    
    paper_id: str
    title: str
    year: Optional[int] = None
    citation_count: int = 0
    
    class Config:
        # Disable arbitrary types for performance
        arbitrary_types_allowed = False
        
        # Validate only on assignment, not creation
        validate_assignment = True
        validate_all = False
        
        # Skip validation for known safe data
        allow_reuse = True

# Batch processing optimization
def process_papers_batch(papers: List[dict]) -> List[OptimizedPaper]:
    """Efficiently process large batches of paper data."""
    return [OptimizedPaper.parse_obj(paper) for paper in papers]
```

### Lazy Loading

```python
from typing import Optional, Callable

class LazyPaper(BaseModel):
    """Paper model with lazy loading of expensive relationships."""
    
    paper_id: str
    title: str
    
    # Lazy-loaded fields
    _authors: Optional[List[Author]] = None
    _citations: Optional[List[Citation]] = None
    _loader: Optional[Callable] = None
    
    def __init__(self, **data):
        self._loader = data.pop('loader', None)
        super().__init__(**data)
    
    @property
    def authors(self) -> List[Author]:
        if self._authors is None and self._loader:
            self._authors = self._loader.load_authors(self.paper_id)
        return self._authors or []
    
    @property
    def citations(self) -> List[Citation]:
        if self._citations is None and self._loader:
            self._citations = self._loader.load_citations(self.paper_id)
        return self._citations or []
```

This comprehensive model system provides type safety, validation, and performance optimization for the entire Academic Citation Platform.