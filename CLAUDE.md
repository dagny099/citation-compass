# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_models.py -v
python -m pytest tests/test_ml_service.py -v
python -m pytest tests/test_unified_database.py -v

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=html

# Run tests by markers
python -m pytest -m "not slow" -v  # Skip slow tests
python -m pytest -m integration -v  # Integration tests only
```

### Code Quality
```bash
# Format code with black
black src/ tests/ --line-length 100

# Sort imports
isort src/ tests/ --profile black

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Development Server & Documentation
```bash
# Start Streamlit app
streamlit run app.py

# Start documentation server
mkdocs serve

# Build documentation
mkdocs build
```

### Database Operations
```bash
# Setup database schema and sample data
python setup_database.py

# CLI tool for database operations
acp health              # Health check
acp db setup           # Setup schema
acp db stats           # Database statistics
acp ml info            # ML model information
acp config show        # Show configuration
```

### Model Training Pipeline
```bash
# Run Jupyter notebooks for model training
jupyter notebook notebooks/01_comprehensive_exploration.ipynb
jupyter notebook notebooks/02_model_training_pipeline.ipynb
jupyter notebook notebooks/03_prediction_evaluation.ipynb
jupyter notebook notebooks/04_narrative_presentation.ipynb
```

## Architecture Overview

This is an academic citation analysis platform with ML prediction capabilities, structured as a modular Python application with multiple interfaces (CLI, Streamlit web app, Jupyter notebooks).

### Core Architecture Layers

**Services Layer (`src/services/`)**
- `ml_service.py`: TransE model serving, citation prediction, embeddings
- `analytics_service.py`: Network analysis, community detection, centrality measures
- Business logic coordination and caching

**Data Layer (`src/database/`)**
- Neo4j graph database integration
- Citation networks stored as graph relationships
- Papers, Authors, Venues, Fields as nodes

**Models (`src/models/`)**
- Pydantic data models for all entities (Paper, Author, Citation, etc.)
- ML models (TransE embeddings, prediction models)
- Network visualization models
- API request/response models

**Data Ingestion (`src/data/`)**
- Semantic Scholar API integration with rate limiting
- Batch processing with streaming performance optimizations
- ETL pipelines for academic paper data

### Key Components

**CLI Tool (`src/cli/main.py`)**
- Health checks, database setup, model operations
- Accessible via `acp` command after installation

**Streamlit Web App (`app.py` + `src/streamlit_app/`)**
- Interactive visualization and analysis interface
- Demo mode with curated datasets
- File upload for paper collections
- Real-time ML predictions and network analysis

**Training Pipeline (Notebooks)**
- 4-notebook pipeline for model training and evaluation
- Comprehensive exploration → Training → Evaluation → Presentation

### Environment Setup Requirements

**Required Environment Variables** (create `.env` file):
```bash
NEO4J_URI=neo4j+s://your-database-url
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

**Installation Options:**
- `pip install -e ".[all]"` - Full installation
- `pip install -e ".[ml]"` - ML features only
- `pip install -e ".[web]"` - Web interface only

### Database Schema

- **Paper** nodes with citation relationships (`CITES`)
- **Author** nodes linked to papers (`AUTHORED`)
- **Venue** and **Field** nodes for metadata
- Constraints on unique identifiers, indices on common query fields

### ML Model Architecture

- **TransE Model**: Paper embeddings for citation prediction
- Models saved in `models/` directory after training
- Prediction confidence scoring and batch processing
- Demo mode includes synthetic embeddings for testing

### Testing Structure

- Unit tests for models, services, database operations
- Integration tests for full workflows
- Performance tests for large datasets
- Fixtures for consistent test data
- Markers: `slow`, `integration`, `performance`

### Key Files to Understand First

1. `src/models/__init__.py` - Complete data model overview
2. `src/services/` - Business logic and ML serving
3. `app.py` - Main application entry point
4. `setup_database.py` - Database initialization
5. `src/cli/main.py` - Command-line interface

The platform integrates three merged codebases (knowledge cartography, citation mapping, prediction system) into a unified academic research tool.