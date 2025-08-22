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

## 🔧 Phase 1: Foundation Setup (Current)

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

## 📋 Next Phases

### Phase 2: Data Pipeline (Planned)
- API client for Semantic Scholar integration
- Graph builder for Neo4j import
- ETL pipeline orchestration

### Phase 3: Machine Learning (Planned)  
- TransE model implementation for citation prediction
- Training pipeline and evaluation metrics
- Model persistence and versioning

### Phase 4: Visualization & Web Interface (Planned)
- Multi-backend network visualization
- Interactive Streamlit application
- Real-time analytics dashboard

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

# Install dependencies (coming in Phase 2)
pip install -r requirements.txt

# Test the configuration
python -m src.config.settings
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
# Run unit tests (basic tests included)
pytest src/tests/

# Test database connection
python -c "from src.database.connection import create_connection; create_connection()"

# Validate configuration
python -m src.config.settings
```

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

**Status**: Phase 1 Foundation Setup Complete ✅
**Next**: Phase 2 Data Pipeline Implementation
**Last Updated**: December 2024