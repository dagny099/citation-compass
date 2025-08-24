# Configuration Guide

Configure your Academic Citation Platform for optimal performance.

## Basic Configuration

### Environment Setup

Create your configuration file:

```bash
# Copy the example configuration
cp .env.example .env

# Edit with your preferred editor
nano .env  # or vim, code, etc.
```

### Essential Settings

Configure these required settings in your `.env` file:

```env
# Database Configuration
NEO4J_URI=your-database-connection-uri
NEO4J_USER=your-database-username  
NEO4J_PASSWORD=your-secure-password
NEO4J_DATABASE=neo4j

# Application Settings
APP_NAME=Academic Citation Platform
DEBUG=false
LOG_LEVEL=INFO
```

## Database Configuration

### Connection Examples

Based on your database setup:

=== "☁️ Cloud Database"

    ```env
    # Example for cloud-hosted Neo4j
    NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your-generated-password
    NEO4J_DATABASE=neo4j
    ```

=== "🏠 Local Database"

    ```env
    # Example for local Neo4j installation
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your-password
    NEO4J_DATABASE=neo4j
    ```

=== "🐳 Docker Database"

    ```env
    # Example for Neo4j Docker container
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your-password
    NEO4J_DATABASE=neo4j
    ```

## API Configuration

### Academic Data Sources

Configure access to academic APIs:

```env
# Semantic Scholar API (recommended)
SEMANTIC_SCHOLAR_API_KEY=your-api-key-here  # Optional
SEMANTIC_SCHOLAR_RATE_LIMIT=100  # Requests per minute
SEMANTIC_SCHOLAR_TIMEOUT=30      # Request timeout
```

!!! tip "Getting API Keys"
    
    API keys provide higher rate limits and priority access:
    
    - **Semantic Scholar**: Free API keys available at [semanticscholar.org](https://www.semanticscholar.org/product/api)
    - **Benefits**: 1000 requests/minute vs 100, priority access, additional metadata

## Performance Configuration

### Resource Settings

Optimize for your system:

```env
# Processing Configuration
MAX_WORKERS=4                    # Adjust based on CPU cores
REQUEST_TIMEOUT=300             # Request timeout in seconds
CACHE_TTL=3600                  # Cache time-to-live in seconds

# ML Model Settings
ML_BATCH_SIZE=1024              # Reduce if memory constrained
ML_DEVICE=auto                  # 'auto', 'cpu', 'cuda', 'mps'
ML_EMBEDDING_DIM=128           # Embedding dimension
```

### Memory Optimization

For systems with limited resources:

```env
# Reduced resource usage
MAX_WORKERS=2
ML_BATCH_SIZE=512
MAX_CACHE_SIZE=500
ANALYTICS_SAMPLE_RATE=0.05  # Sample 5% of events
```

## Security Settings

!!! warning "Security Best Practices"
    
    - Never commit `.env` files to version control
    - Use strong, unique passwords
    - Enable database encryption in production
    - Regularly rotate credentials

### Security Configuration

```env
# Security Settings
SECRET_KEY=generate-a-secure-random-key-here
SECURE_COOKIES=true             # Enable in production
SESSION_TIMEOUT=3600           # Session timeout in seconds

# Database Security
NEO4J_ENCRYPTED=true           # Use encrypted connections
```

### Generate Secure Keys

Generate secure keys for your installation:

```bash
# Generate a secure secret key
python -c "import secrets; print(f'SECRET_KEY={secrets.token_hex(32)}')"
```

## Application Settings

### Streamlit Configuration

```env
# Streamlit Application
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_THEME_BASE=light      # 'light' or 'dark'
```

### Export Configuration

```env
# Export Settings
EXPORT_FORMATS=json,csv,latex   # Supported export formats
TEMP_DIR=outputs/temp          # Temporary files directory
MAX_EXPORT_SIZE=100MB         # Maximum export file size
```

## Environment-Specific Configuration

### Development Configuration

For development environments:

```env
# Development Settings
DEBUG=true
LOG_LEVEL=DEBUG
ANALYTICS_ENABLED=false       # Disable analytics in development
CACHE_ENABLED=false          # Disable caching for development

# Development database (optional)
NEO4J_TEST_URI=bolt://localhost:7688
```

### Production Configuration

For production deployments:

```env
# Production Settings
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=your-production-secret-key

# Production performance
MAX_WORKERS=8
CACHE_TTL=7200
ML_DEVICE=cuda  # If GPU available
```

## Configuration Validation

Test your configuration:

```bash
# Run configuration validation
python scripts/validate_environment.py

# Test database connection
python -c "
from src.database.connection import Neo4jConnection
conn = Neo4jConnection()
print('✅ Database connection successful' if conn.test_connection() else '❌ Connection failed')
"

# Test application startup
streamlit run app.py --check-config
```

## Troubleshooting

### Common Configuration Issues

??? question "Database connection timeouts"

    Increase timeout values:
    ```env
    NEO4J_TIMEOUT=60
    REQUEST_TIMEOUT=300
    ```

??? question "Memory issues with large datasets"

    Reduce resource usage:
    ```env
    ML_BATCH_SIZE=512         # Reduce batch size
    MAX_CACHE_SIZE=500        # Reduce cache size
    MAX_WORKERS=2             # Reduce worker threads
    ```

??? question "SSL/TLS certificate errors"

    For development environments:
    ```env
    NEO4J_ENCRYPTED=false
    ```

## Advanced Configuration

### Custom Model Paths

```env
# Custom ML model locations
ML_MODEL_PATH=models/custom_transe_model.pt
ML_ENTITY_MAPPING=models/entity_mapping.pkl
ML_TRAINING_METADATA=models/training_metadata.pkl
```

### Analytics Configuration

```env
# Analytics Settings
ANALYTICS_ENABLED=true
ANALYTICS_SAMPLE_RATE=0.1       # Sample 10% of events
PERFORMANCE_MONITORING=true     # Enable performance metrics
```

## Next Steps

After configuration:

1. **[Environment Setup →](environment-setup.md)** - Initialize your database and validate setup
2. **[Quick Start →](quick-start.md)** - Run your first citation analysis
3. **[User Guide →](../user-guide/overview.md)** - Learn about platform features

---

Need help? Check our **[troubleshooting guide](../developer-guide/troubleshooting.md)** or **[open an issue](https://github.com/your-username/academic-citation-platform/issues)**.