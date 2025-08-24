# Environment Setup & Validation

Ensure your Academic Citation Platform environment is properly configured and ready for analysis.

## Database Initialization

Before running any analysis, you need to set up your Neo4j database with the proper schema and sample data.

### Step 1: Validate Database Connection

First, test that you can connect to your Neo4j database:

```bash
# Test database connection
python -c "
from src.database.connection import Neo4jConnection
conn = Neo4jConnection()
if conn.test_connection():
    print('✅ Neo4j connection successful')
    print(f'Connected to: {conn.uri}')
    print(f'Database: {conn.database}')
else:
    print('❌ Neo4j connection failed')
    exit(1)
"
```

### Step 2: Initialize Database Schema

Set up the required database schema and constraints:

```bash
# Initialize database schema
python setup_database.py --init-schema

# Verify schema setup
python setup_database.py --verify-schema
```

This creates:

- **Node constraints** for Papers, Authors, Venues, and Fields
- **Relationship types** for Citations, Authorship, and Publications  
- **Indexes** for efficient querying
- **Sample data** for testing (optional)

### Step 3: Load Sample Data (Optional)

For testing and learning, load sample citation data:

=== "🚀 Quick Sample Data"

    ```bash
    # Load minimal sample dataset (100 papers)
    python setup_database.py --load-sample --size=small
    ```

=== "📊 Medium Dataset"

    ```bash
    # Load medium dataset (1,000 papers)
    python setup_database.py --load-sample --size=medium
    ```

=== "🏢 Full Dataset"

    ```bash
    # Load full sample dataset (10,000 papers)
    python setup_database.py --load-sample --size=large
    ```

## Environment Validation

Run comprehensive validation to ensure everything is working:

### Automated Validation Script

```bash
# Run complete environment validation
python run_setup_validation.py
```

This script checks:

- ✅ Python dependencies and versions
- ✅ Database connectivity and schema
- ✅ ML model availability and loading
- ✅ Streamlit configuration
- ✅ API access and rate limits
- ✅ File system permissions

### Manual Validation Steps

If you prefer to validate manually:

#### 1. Test Core Services

```bash
# Test analytics service
python -c "
from src.services.analytics_service import get_analytics_service
analytics = get_analytics_service()
print('✅ Analytics service loaded')
"

# Test ML service (requires trained model)
python -c "
from src.services.ml_service import get_ml_service
ml = get_ml_service()
print('✅ ML service loaded')
"
```

#### 2. Test Database Operations

```bash
# Test database queries
python -c "
from src.data.unified_database import UnifiedDatabase
db = UnifiedDatabase()
papers = db.get_papers_sample(limit=5)
print(f'✅ Retrieved {len(papers)} papers from database')
db.close()
"
```

#### 3. Test Streamlit Interface

```bash
# Test Streamlit configuration
streamlit config show

# Verify Streamlit can start (dry run)
streamlit run app.py --server.headless true --server.port 8502 &
sleep 5
curl -f http://localhost:8502 > /dev/null && echo "✅ Streamlit interface accessible" || echo "❌ Streamlit interface failed"
pkill -f streamlit
```

## Performance Optimization

### System Resource Check

```bash
# Check system resources
python -c "
import psutil
print(f'CPU cores: {psutil.cpu_count()}')
print(f'Memory: {psutil.virtual_memory().total // (1024**3)} GB')
print(f'Available: {psutil.virtual_memory().available // (1024**3)} GB')
"
```

### Database Performance Tuning

For optimal performance, configure Neo4j settings:

```cypher
// Connect to Neo4j Browser and run these commands

// Check database statistics
CALL db.stats.retrieve('GRAPH COUNTS')

// Create performance indexes
CREATE INDEX papers_title IF NOT EXISTS FOR (p:Paper) ON (p.title)
CREATE INDEX authors_name IF NOT EXISTS FOR (a:Author) ON (a.name)
CREATE INDEX citations_year IF NOT EXISTS FOR ()-[c:CITES]-() ON (c.year)

// Verify indexes
SHOW INDEXES
```

### ML Model Setup

If you plan to use ML predictions, ensure models are available:

```bash
# Check for existing models
ls -la models/

# If no models exist, train initial model
jupyter notebook notebooks/02_model_training_pipeline.ipynb

# Verify model loading
python verify_models.py
```

## Development Environment Setup

For contributors and advanced users:

### Development Dependencies

```bash
# Install development tools
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run code quality checks
black src/
isort src/
flake8 src/
mypy src/
```

### Testing Environment

```bash
# Run full test suite
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_integration.py -v    # Integration tests
python -m pytest tests/test_analytics.py -v     # Analytics tests
python -m pytest tests/test_ml_service.py -v    # ML service tests

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting Setup Issues

### Common Setup Problems

??? question "Database connection refused"

    **Symptoms**: Connection timeout or "Connection refused" errors

    **Solutions**:
    ```bash
    # Check if Neo4j is running
    neo4j status

    # Start Neo4j if stopped
    neo4j start

    # Verify connection settings in .env
    grep NEO4J .env

    # Test with different URI formats
    NEO4J_URI=neo4j://localhost:7687  # For Neo4j 4.x+
    NEO4J_URI=bolt://localhost:7687    # For older versions
    ```

??? question "Import errors for src modules"

    **Symptoms**: `ImportError: No module named 'src'`

    **Solutions**:
    ```bash
    # Ensure editable installation
    pip install -e .

    # Check Python path
    python -c "import sys; print(sys.path)"

    # Add project root to PYTHONPATH
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    ```

??? question "Streamlit permission errors"

    **Symptoms**: Port binding or permission issues

    **Solutions**:
    ```bash
    # Use different port
    streamlit run app.py --server.port 8502

    # Check port availability
    netstat -an | grep 8501

    # Kill existing Streamlit processes
    pkill -f streamlit
    ```

??? question "Memory issues during setup"

    **Symptoms**: Out of memory errors during model loading or large dataset processing

    **Solutions**:
    ```bash
    # Reduce sample data size
    python setup_database.py --load-sample --size=small

    # Adjust ML batch sizes in .env
    ML_BATCH_SIZE=256
    MAX_WORKERS=2

    # Monitor memory usage
    python -c "
    import psutil
    print(f'Memory usage: {psutil.virtual_memory().percent:.1f}%')
    "
    ```

### Environment Validation Checklist

Use this checklist to verify your setup:

- [ ] Python 3.8+ installed and accessible
- [ ] Virtual environment activated  
- [ ] All dependencies installed (`pip install -e ".[all]"`)
- [ ] Neo4j database running and accessible
- [ ] Database schema initialized
- [ ] Sample data loaded (optional)
- [ ] Environment variables configured in `.env`
- [ ] Core services can be imported without errors
- [ ] Streamlit interface launches successfully
- [ ] ML models available or can be trained
- [ ] Test suite passes basic integration tests

### Getting Additional Help

If you continue to experience setup issues:

1. **Check the logs**: Look for detailed error messages in the console output
2. **Review configuration**: Double-check your `.env` file settings
3. **Consult documentation**: See [Troubleshooting Guide](../advanced/troubleshooting.md)
4. **Community support**: Join our discussions or file an issue on GitHub
5. **Professional support**: Contact [contact@citationcompass.com](mailto:contact@citationcompass.com)

## Next Steps

Once your environment is validated:

1. **[Quick Start →](quick-start.md)** - Run your first citation analysis
2. **[User Guide →](../user-guide/overview.md)** - Learn platform features
3. **[Notebooks →](../notebooks/overview.md)** - Explore analysis workflows

!!! success "Environment Ready!"
    
    Your Academic Citation Platform environment is now properly configured and validated. You're ready to start analyzing citation networks! 🎉