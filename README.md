# Academic Citation Platform

A comprehensive platform for academic citation network analysis, prediction, and exploration. This project integrates the best features from three reference codebases to provide a unified solution for citation analysis, machine learning predictions, and interactive visualization.

## 🚀 Project Overview

The Academic Citation Platform combines:
- **Data Pipeline**: Robust API integration with Semantic Scholar
- **Graph Database**: Neo4j-based storage and analytics
- **Machine Learning**: TransE-based citation prediction models
- **Visualization**: Multi-backend network visualization
- **Web Interface**: Interactive Streamlit application

## 🏗️ Architecture

```
src/
├── config/                    # Centralized configuration management
│   ├── settings.py           # Environment-based settings
│   └── database.py           # Neo4j queries and DB config
├── database/
│   ├── connection.py         # Robust Neo4j connection layer
│   ├── schema.py             # Unified database schema
│   └── migrations/           # Schema migration scripts
├── models/
│   ├── paper.py              # Paper entity models
│   ├── author.py             # Author and collaboration models
│   ├── venue.py              # Publication venue models
│   ├── field.py              # Research field models
│   └── citation.py           # Citation relationship models
└── utils/
    ├── logging.py            # Centralized logging
    └── validation.py         # Data validation utilities
```

## 🔧 Phase 1: Foundation Setup ✅ Complete

### ✅ Completed Components

1. **Directory Structure**: Clean, modular organization
2. **Database Layer**: Enhanced Neo4j connection with error handling and caching
3. **Unified Schema**: Combined schema from all reference codebases
4. **Configuration Management**: Environment-based settings with validation
5. **Data Models**: Pydantic models for all entities with validation
6. **Utilities**: Logging, validation, and common functions

### 🎯 Key Features

- **Robust Database Connection**: Enhanced error handling, connection pooling, and automatic retry
- **Unified Schema**: Supports all node types and relationships from reference codebases
- **Type Safety**: Pydantic models with validation for all data entities
- **Configuration Management**: Centralized settings with environment variable support
- **Logging**: Structured logging with file rotation and multiple levels

## 🚀 Phase 2: Data Pipeline & ML Integration ✅ Complete

### ✅ Completed Components

1. **Unified API Client**: Semantic Scholar integration with rate limiting and caching
2. **TransE Model Service**: Production-ready ML service with intelligent caching
3. **Streamlit Web Interface**: 4-page interactive application
4. **Comprehensive Testing**: 100+ tests covering all modules
5. **Integration Layer**: Seamless service coordination and error handling

### 🎯 Key Features

- **ML-Powered Predictions**: TransE model for citation prediction with confidence scoring
- **Interactive Web Interface**: Streamlit application with ML predictions and embedding explorer
- **Production Optimization**: Caching, error handling, and performance monitoring
- **Comprehensive Testing**: Unit, integration, and validation testing across all components
- **Service Architecture**: Modular design with clear separation of concerns

## 📋 Current Phase

### Phase 3: Advanced Analytics & Production Features ✅ Complete

**✅ Completed Components:**
1. **Advanced Analytics Engine**: Complete implementation in `src/analytics/`
   - Network analysis with centrality measures and community detection
   - Temporal analysis for citation trends and growth patterns  
   - Performance metrics and benchmarking framework
   - Multi-format export engine (HTML, JSON, CSV, LaTeX)

2. **Interactive Notebook Pipeline**: Full-featured Streamlit interface
   - Interactive notebook execution with 3 pre-built analysis notebooks
   - Configurable parameters and real-time execution
   - Advanced visualizations and export capabilities
   - Analytics service integration with caching and error handling

3. **Production Features**: Enterprise-ready capabilities
   - Comprehensive test coverage (100+ tests across all modules)
   - Performance optimization with intelligent caching
   - Error handling and graceful degradation
   - Health monitoring and system diagnostics

### Phase 4: Contextual Documentation & User Guidance (Planned)

**🎯 Vision:** "From Data to Understanding" - Transform raw analytics into actionable research insights

**Focus Areas:**
- **Contextual Result Interpretation**: Real-time explanations for every metric with academic benchmarking
- **Interactive Exploration**: Drill-down capabilities and comparative analysis against published studies
- **Research Use Case Library**: Domain-specific interpretation guides with practical examples
- **Actionable Insights Generation**: AI-powered recommendations and next-step suggestions
- **Academic Integration**: Export templates for papers, presentations, and grant proposals

**Key Deliverables:**
- Metric explanation system with traffic light performance indicators
- Comparative benchmarking against 10+ academic domains
- LaTeX/PowerPoint export templates ready for academic use
- Research assistant features for hypothesis generation

See `PHASE_4_PLAN.md` for detailed implementation roadmap.

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- Neo4j Database (local or cloud)
- Git

### Environment Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Configure your environment variables:
```env
# Neo4j Database Configuration
NEO4J_URI=neo4j+s://your-database-url
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# API Configuration (optional)
SEMANTIC_SCHOLAR_API_KEY=your-api-key

# Logging Configuration (optional)
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd academic-citation-platform

# Install core dependencies
pip install -e .

# Install optional dependencies for different phases
pip install -e ".[ml,viz,web]"    # All components
pip install -e ".[ml]"            # ML components only
pip install -e ".[web]"           # Web interface only

# For development
pip install -e ".[dev,all]"

# Test the configuration
python -c "from src.data.api_config import get_config; print('✅ Configuration loaded successfully')"

# Test ML service
python -c "from src.services.ml_service import get_ml_service; print('✅ ML service initialized')"

# Launch Streamlit application
streamlit run app.py
```

## 📊 Database Schema

### Node Types
- **Paper**: Academic papers with metadata (title, abstract, citations, etc.)
- **Author**: Researchers with profiles and metrics
- **PubVenue**: Publication venues (journals, conferences)
- **PubYear**: Publication years for temporal analysis
- **Field**: Research fields and categories
- **Institution**: Academic institutions and affiliations

### Relationship Types
- **CITES**: Paper citation relationships
- **AUTHORED**: Author-paper relationships
- **PUBLISHED_IN**: Paper-venue relationships
- **PUB_YEAR**: Paper-year relationships
- **IS_ABOUT**: Paper-field relationships
- **AFFILIATED_WITH**: Author-institution relationships

## 🧪 Testing

```bash
# Run complete test suite (100+ tests)
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_models_simple.py -v          # Data model tests
python -m pytest tests/test_ml_service.py -v             # ML service tests
python -m pytest tests/test_unified_api_client.py -v     # API client tests
python -m pytest tests/test_integration.py -v            # Integration tests

# Run with coverage analysis
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Performance and validation testing
python -m pytest tests/test_validation.py -v             # Security & validation
python -m pytest tests/test_fixtures.py -v               # Fixture integrity

# Manual Streamlit testing
streamlit run app.py
# Then test: ML Predictions, Embedding Explorer, Enhanced Visualizations, Notebook Pipeline

# Health checks
python -c "from src.services.ml_service import get_ml_service; print(get_ml_service().health_check())"
```

### Test Coverage
- **100+ comprehensive tests** across all modules
- **Unit tests**: Data models, utilities, configuration
- **Integration tests**: Service coordination, end-to-end workflows
- **Validation tests**: Security, performance, data integrity
- **ML tests**: Model functionality, predictions, caching
- **API tests**: Rate limiting, error handling, response validation

## 📖 Documentation

- **Architecture**: See `/docs/architecture.md` (coming soon)
- **API Reference**: See `/docs/api.md` (coming soon)
- **Data Models**: See model docstrings in `/src/models/`
- **Configuration**: See `src/config/settings.py`

## 🤝 Contributing

This project integrates features from three reference codebases:
1. **academic-citation-predictions**: API integration and data pipeline
2. **citation-map-dashboard**: ML models and prediction algorithms  
3. **knowledge-cartography**: Web interface and visualization

## 📝 License

[License details to be added]

## 🙏 Acknowledgments

This project builds upon and integrates work from three excellent academic citation analysis codebases. Special thanks to the original authors for their contributions to the field.

---

**Status**: Phase 3 Complete ✅ | Phase 4 Planning 📋
**Current Focus**: Contextual Documentation & User Guidance
**Next**: Enhanced Result Interpretation & Academic Integration  
**Last Updated**: August 2025

## 🚀 Quick Start Commands

```bash
# Complete setup and testing
pip install -e ".[all]" && python -m pytest tests/ -v && streamlit run app.py

# ML predictions demo
python -c "
from src.services.ml_service import get_ml_service
ml = get_ml_service()
predictions = ml.predict_citations('649def34f8be52c8b66281af98ae884c09aef38f9', top_k=5)
print(f'Generated {len(predictions)} predictions')
"
```