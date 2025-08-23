# Academic Citation Platform Architecture

## Overview

The Academic Citation Platform is a comprehensive system for academic citation network analysis, prediction, and exploration. It integrates the best features from three reference codebases to provide a unified solution for citation analysis, machine learning predictions, and interactive visualization.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface Layer                       │
├─────────────────────────────────────────────────────────────┤
│  Streamlit Application (app.py)                             │
│  ├── Home Page                                              │
│  ├── ML Predictions                                         │
│  ├── Embedding Explorer                                     │
│  ├── Enhanced Visualizations                                │
│  ├── Results Interpretation                                 │
│  └── Analysis Pipeline                                      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                            │
├─────────────────────────────────────────────────────────────┤
│  ML Service (TransE Model)    │  Analytics Service          │
│  ├── Model Loading            │  ├── Network Analysis       │
│  ├── Predictions              │  ├── Performance Metrics    │
│  ├── Embeddings               │  ├── Temporal Analysis      │
│  └── Caching                  │  └── Export Engine         │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                              │
├─────────────────────────────────────────────────────────────┤
│  Unified API Client           │  Database Connection        │
│  ├── Semantic Scholar API     │  ├── Neo4j Driver          │
│  ├── Rate Limiting            │  ├── Query Execution       │
│  ├── Caching                  │  ├── Schema Management     │
│  └── Batch Processing         │  └── Connection Pooling    │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Storage Layer                            │
├─────────────────────────────────────────────────────────────┤
│  Neo4j Graph Database         │  ML Model Files            │
│  ├── Paper Nodes              │  ├── TransE Model (.pt)    │
│  ├── Author Nodes             │  ├── Entity Mapping (.pkl) │
│  ├── Citation Edges           │  └── Metadata (.pkl)       │
│  └── Venue/Field/Year Nodes   │                            │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Configuration Management (`src/config/`)
- **settings.py**: Environment-based configuration with validation
- **database.py**: Neo4j queries and database configuration

### 2. Data Models (`src/models/`)
- **paper.py**: Paper entity models with Pydantic validation
- **author.py**: Author and collaboration models  
- **citation.py**: Citation relationship models
- **ml.py**: Machine learning model interfaces
- **network.py**: Network analysis data structures

### 3. Database Layer (`src/database/`)
- **connection.py**: Enhanced Neo4j connection with error handling and caching
- **schema.py**: Unified database schema definitions
- **migrations/**: Schema migration scripts (placeholder)

### 4. Data Access (`src/data/`)
- **unified_api_client.py**: Comprehensive Semantic Scholar API client
- **api_config.py**: API configuration and rate limiting settings
- **unified_database.py**: Database abstraction layer

### 5. Service Layer (`src/services/`)
- **ml_service.py**: TransE model service with caching and optimization
- **analytics_service.py**: Network analysis and metrics computation

### 6. Analytics Engine (`src/analytics/`)
- **network_analysis.py**: Graph analysis algorithms
- **contextual_explanations.py**: Academic result interpretation  
- **performance_metrics.py**: Model evaluation metrics
- **temporal_analysis.py**: Time-series citation analysis
- **export_engine.py**: Multi-format result export

### 7. Web Interface (`src/streamlit_app/`)
- **ML_Predictions.py**: Interactive citation prediction interface
- **Embedding_Explorer.py**: Paper embedding visualization
- **Enhanced_Visualizations.py**: Advanced network visualizations
- **Results_Interpretation.py**: Contextual result interpretation
- **Notebook_Pipeline.py**: Interactive analysis workflows

## Data Flow

### 1. Data Ingestion
```
External APIs → Unified API Client → Data Models → Neo4j Database
```

### 2. ML Pipeline
```
Neo4j Data → ML Service → TransE Model → Predictions → Cache
```

### 3. Analysis Pipeline
```
Database Query → Analytics Service → Metrics Computation → Export Engine
```

### 4. User Interface
```
User Input → Streamlit Pages → Services → Database/ML → Results Display
```

## Key Design Patterns

### 1. Dependency Injection
- Services are injected into components for testability
- Configuration is environment-driven with validation

### 2. Caching Strategy
- **API Client**: Response caching with TTL and LRU eviction
- **ML Service**: Prediction caching for performance
- **Database**: Query result caching

### 3. Error Handling
- Comprehensive exception handling at all layers
- Graceful degradation for missing dependencies
- Detailed logging for debugging

### 4. Type Safety
- Pydantic models for data validation
- Type hints throughout codebase
- Runtime validation for critical paths

## Scalability Considerations

### 1. Database
- Connection pooling for concurrent access
- Indexed queries for performance
- Batch operations for bulk updates

### 2. API Client
- Rate limiting to respect API constraints
- Pagination for large datasets
- Retry logic for transient failures

### 3. ML Service
- Model caching to avoid repeated loading
- Batch prediction capabilities
- Device-aware computation (CPU/GPU)

### 4. Web Interface
- Streamlit caching for expensive operations
- Progressive loading for large datasets
- Session state management

## Security Considerations

### 1. Credentials Management
- Environment variable configuration
- No hardcoded credentials
- Support for multiple naming conventions

### 2. Input Validation
- Pydantic model validation
- SQL injection prevention through parameterized queries
- API input sanitization

### 3. Error Information
- Sanitized error messages for users
- Detailed logging for developers
- No sensitive data in logs

## Deployment Architecture

### Development Environment
```
Local Machine
├── Python Virtual Environment
├── Local Neo4j Instance (or Neo4j AuraDB)
├── Streamlit Development Server
└── ML Model Files (local)
```

### Production Environment
```
Cloud Infrastructure
├── Container Orchestration (Docker/Kubernetes)
├── Managed Neo4j Database (AuraDB)
├── Application Server (Streamlit/FastAPI)
├── Model Storage (S3/GCS)
└── Monitoring & Logging
```

## Testing Strategy

### 1. Unit Tests
- Individual component testing
- Mock dependencies for isolation
- Comprehensive model validation

### 2. Integration Tests
- End-to-end workflow testing
- Database integration validation
- API client testing with live services

### 3. Performance Tests
- Load testing for database queries
- ML service benchmarking
- API rate limit validation

## Future Enhancements

### 1. Microservices Architecture
- Split monolithic services into focused microservices
- API gateway for service orchestration
- Independent scaling and deployment

### 2. Real-time Updates
- Streaming data ingestion
- Live model retraining
- WebSocket-based UI updates

### 3. Advanced Analytics
- Graph neural networks for citation prediction
- Multi-modal analysis (text + citations)
- Collaborative filtering approaches

---

*This architecture document is maintained as the system evolves. Last updated: August 2025*