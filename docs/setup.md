# Setup Guide

## Prerequisites

- Python 3.8+ (recommended: Python 3.10+)
- Neo4j Database (local installation or Neo4j AuraDB cloud instance)
- Git
- 4GB+ RAM (for ML model operations)

## Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd academic-citation-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[all]"
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your database credentials
# NEO4J_URI=neo4j+s://your-database-url
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=your-password
```

### 3. Setup Database

```bash
# Run database setup script
python setup_database.py
```

### 4. Verify Models

```bash
# Verify ML model files are accessible
python verify_models.py
```

### 5. Launch Application

```bash
# Start Streamlit application
streamlit run app.py
```

## Detailed Setup

### Environment Configuration

#### Required Variables
```bash
# Neo4j Database (required)
NEO4J_URI=neo4j+s://your-database-url  # or neo4j://localhost:7687 for local
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j  # optional, defaults to 'neo4j'
```

#### Optional Variables
```bash
# Semantic Scholar API (optional, improves rate limits)
SEMANTIC_SCHOLAR_API_KEY=your-api-key

# Logging configuration
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
LOG_FILE=logs/app.log

# Cache settings
CACHE_ENABLED=true
CACHE_DEFAULT_TTL=300  # seconds

# Development mode
ENVIRONMENT=development  # or production
DEBUG=false
```

### Database Setup Options

#### Option 1: Neo4j AuraDB (Recommended for beginners)

1. Go to [Neo4j AuraDB](https://neo4j.com/cloud/aura/)
2. Create a free account and database instance
3. Download connection credentials
4. Use the provided URI, username, and password in `.env`

#### Option 2: Local Neo4j Installation

```bash
# Download and install Neo4j Desktop
# Or use Docker:
docker run \
    --name neo4j \
    -p7474:7474 -p7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    --env NEO4J_AUTH=neo4j/your-password \
    neo4j:latest
```

Set in `.env`:
```bash
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

### Installation Options

#### Core Installation (minimal)
```bash
pip install -e .
```

#### ML Components Only
```bash
pip install -e ".[ml]"
```

#### Web Interface Only  
```bash
pip install -e ".[web]"
```

#### Development Environment
```bash
pip install -e ".[dev,all]"
```

#### All Components
```bash
pip install -e ".[all]"
```

### Model Files Setup

The ML models are included in the `models/` directory. Verify they're accessible:

```bash
python -c "
from src.services.ml_service import get_ml_service
ml = get_ml_service()
health = ml.health_check()
print('Model Status:', health['status'])
print('Entities:', health.get('num_entities', 0))
"
```

If models are missing or corrupted, check the `models/` directory contains:
- `transe_citation_model.pt` (~19MB)
- `entity_mapping.pkl` (~577KB) 
- `training_metadata.pkl` (~200B)

## Testing Setup

### Run All Tests
```bash
# Complete test suite
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Categories
```bash
# Data model tests
python -m pytest tests/test_models_simple.py -v

# ML service tests  
python -m pytest tests/test_ml_service.py -v

# API client tests
python -m pytest tests/test_unified_api_client.py -v

# Integration tests (requires database)
python -m pytest tests/test_integration.py -v
```

### Manual Application Testing
```bash
# Launch Streamlit app
streamlit run app.py

# Navigate to pages and test:
# - ML Predictions: Enter paper ID and generate predictions
# - Embedding Explorer: Visualize paper embeddings
# - Enhanced Visualizations: View network analysis
# - Results Interpretation: Explore contextual analysis
```

## Troubleshooting

### Database Connection Issues

**Error:** "Failed to initialize Neo4j connection"
```bash
# Check environment variables
python -c "
import os
print('URI:', os.getenv('NEO4J_URI'))
print('USER:', os.getenv('NEO4J_USER'))  
print('PASS:', bool(os.getenv('NEO4J_PASSWORD')))
"

# Test connection manually
python setup_database.py
```

**Error:** "Connection test failed"
- Verify Neo4j server is running
- Check firewall settings for port 7687
- Confirm credentials are correct
- For AuraDB, ensure you're using the correct URI format

### Model Loading Issues

**Error:** "Model file not found"
```bash
# Check model files exist
ls -la models/

# Verify with script
python verify_models.py
```

**Error:** "Health check failed"
- Ensure you have sufficient RAM (4GB+)
- Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`
- Verify CUDA setup if using GPU: `python -c "import torch; print(torch.cuda.is_available())"`

### API Client Issues

**Error:** Rate limit exceeded
- Add Semantic Scholar API key to `.env`
- Reduce batch sizes in API calls
- Check rate limiter settings in configuration

**Error:** SSL certificate issues
```bash
# For development/testing only:
export PYTHONHTTPSVERIFY=0
```

### Streamlit Issues

**Error:** "ModuleNotFoundError" 
- Ensure you're in the activated virtual environment
- Reinstall with `pip install -e ".[all]"`
- Check Python path: `python -c "import sys; print(sys.path)"`

**Error:** Page not loading
- Check console for JavaScript errors
- Try different browser
- Clear browser cache
- Restart Streamlit: `Ctrl+C` then `streamlit run app.py`

### Performance Issues

**Slow predictions:**
- Use smaller `top_k` values
- Enable prediction caching
- Consider using CPU vs GPU based on model size

**Memory errors:**
- Reduce batch sizes
- Clear caches periodically  
- Monitor memory usage: `htop` or Task Manager

## Development Setup

### Code Quality Tools
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
flake8 src/
pylint src/

# Format code
black src/
isort src/

# Type checking
mypy src/
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

### Adding New Dependencies

```bash
# Edit pyproject.toml dependencies
# Then reinstall in development mode
pip install -e ".[dev,all]"
```

## Production Deployment

### Environment Preparation
```bash
# Use production environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# Use managed database
NEO4J_URI=neo4j+s://production-database-url

# Enable monitoring
LOG_FILE=/var/log/citation-platform/app.log
```

### Docker Deployment
```bash
# Build container (Dockerfile not included, but recommended structure)
docker build -t citation-platform .

# Run with environment
docker run -p 8501:8501 --env-file .env citation-platform
```

### Health Monitoring
```bash
# Setup health check endpoints
python -c "
from src.services.ml_service import get_ml_service
from src.database.connection import Neo4jConnection

# Check all services
ml_health = get_ml_service().health_check()
db = Neo4jConnection()
db_health = db.test_connection()

print('ML Service:', ml_health['status'])
print('Database:', 'healthy' if db_health else 'unhealthy')
"
```

---

*This setup guide is updated as the system evolves. Last updated: August 2025*