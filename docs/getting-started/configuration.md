# Configuration Guide

Configure your Academic Citation Platform for optimal performance and security.

## Environment Configuration

The platform uses environment variables for configuration. Start by creating your configuration file:

```bash
# Copy the example configuration
cp .env.example .env

# Edit with your preferred editor
nano .env  # or vim, code, etc.
```

## Database Configuration

### Neo4j Setup

The platform requires a Neo4j database instance. You can use:

- **Neo4j AuraDB** (cloud-hosted, recommended for beginners)
- **Local Neo4j installation**
- **Docker Neo4j container**

=== "🌤️ Neo4j AuraDB (Recommended)"

    1. Create a free account at [Neo4j Aura](https://console.neo4j.io/)
    2. Create a new database instance
    3. Copy the connection details to your `.env` file:

    ```env
    # Neo4j Database Configuration
    NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your-generated-password
    NEO4J_DATABASE=neo4j
    ```

=== "🏠 Local Installation"

    1. Download and install [Neo4j Desktop](https://neo4j.com/download/)
    2. Create a new database project
    3. Start your database instance
    4. Configure your `.env` file:

    ```env
    # Local Neo4j Configuration
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your-password
    NEO4J_DATABASE=neo4j
    ```

=== "🐳 Docker Container"

    ```bash
    # Run Neo4j in Docker
    docker run \
        --name neo4j-citation \
        -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/your-password \
        -v neo4j_data:/data \
        neo4j:5.0
    ```

    Then configure:
    ```env
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your-password
    NEO4J_DATABASE=neo4j
    ```

## API Configuration

### Semantic Scholar API

Configure access to the Semantic Scholar API for paper metadata:

```env
# Semantic Scholar API (optional but recommended)
SEMANTIC_SCHOLAR_API_KEY=your-api-key-here  # Optional, for rate limit increases
SEMANTIC_SCHOLAR_BASE_URL=https://api.semanticscholar.org/graph/v1
SEMANTIC_SCHOLAR_RATE_LIMIT=100  # Requests per minute
SEMANTIC_SCHOLAR_TIMEOUT=30      # Request timeout in seconds
```

!!! tip "API Key Benefits"
    While an API key is optional, having one provides:
    
    - Higher rate limits (1000 requests/minute vs 100)
    - Priority access during high usage
    - Access to additional paper metadata

## Application Configuration

### Core Settings

```env
# Application Settings
APP_NAME=Academic Citation Platform
APP_VERSION=0.1.0
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-here  # Generate with: python -c "import secrets; print(secrets.token_hex(32))"

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_THEME_BASE=light
```

### Performance Settings

```env
# Performance Configuration
MAX_WORKERS=4                    # Number of worker threads
REQUEST_TIMEOUT=300             # Request timeout in seconds
CACHE_TTL=3600                  # Cache time-to-live in seconds
MAX_CACHE_SIZE=1000             # Maximum number of cached items

# ML Model Configuration
ML_MODEL_PATH=models/transe_citation_model.pt
ML_BATCH_SIZE=1024
ML_DEVICE=auto                  # 'auto', 'cpu', 'cuda', 'mps'
ML_EMBEDDING_DIM=128
```

### Analytics Configuration

```env
# Analytics Settings
ANALYTICS_ENABLED=true
ANALYTICS_SAMPLE_RATE=0.1       # Sample 10% of events
EXPORT_FORMATS=json,csv,latex   # Supported export formats
TEMP_DIR=outputs/temp           # Temporary files directory
MAX_EXPORT_SIZE=100MB          # Maximum export file size
```

## Security Configuration

!!! warning "Security Best Practices"
    
    - Never commit `.env` files to version control
    - Use strong, unique passwords
    - Regularly rotate API keys and passwords
    - Enable database authentication and encryption

### Environment Variables Security

```env
# Security Settings
ALLOWED_HOSTS=localhost,127.0.0.1,citationcompass.barbhs.com
CORS_ENABLED=false
SECURE_COOKIES=true             # Enable in production
SESSION_TIMEOUT=3600           # Session timeout in seconds

# Database Security
NEO4J_ENCRYPTED=true           # Use encrypted connection
NEO4J_TRUST_STRATEGY=TRUST_ALL_CERTIFICATES  # For development only
```

## Development Configuration

For development and testing environments:

```env
# Development Settings
DEBUG=true
LOG_LEVEL=DEBUG
TESTING=false                  # Set to true when running tests

# Development Database (optional separate instance)
NEO4J_TEST_URI=bolt://localhost:7688
NEO4J_TEST_USER=neo4j
NEO4J_TEST_PASSWORD=test-password
NEO4J_TEST_DATABASE=test

# Development Analytics
ANALYTICS_ENABLED=false       # Disable analytics in development
CACHE_ENABLED=false          # Disable caching for development
```

## Configuration Validation

Validate your configuration after setup:

```bash
# Run configuration validation
python scripts/validate_environment.py

# Test database connection
python -c "
from src.database.connection import Neo4jConnection
conn = Neo4jConnection()
if conn.test_connection():
    print('✅ Database connection successful')
else:
    print('❌ Database connection failed')
"

# Test Streamlit configuration
streamlit run app.py --check-config
```

## Configuration Templates

### Production Configuration Template

```env title=".env.production"
# Production Configuration
APP_NAME=Academic Citation Platform
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=your-production-secret-key

# Production Database
NEO4J_URI=neo4j+s://production.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=strong-production-password
NEO4J_ENCRYPTED=true

# Production Security
ALLOWED_HOSTS=citationcompass.barbhs.com
SECURE_COOKIES=true
SESSION_TIMEOUT=1800

# Production Performance
MAX_WORKERS=8
CACHE_TTL=7200
ML_DEVICE=cuda
```

### Development Configuration Template

```env title=".env.development"
# Development Configuration
APP_NAME=Academic Citation Platform (Dev)
DEBUG=true
LOG_LEVEL=DEBUG

# Local Development Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=dev-password

# Development Settings
ANALYTICS_ENABLED=false
CACHE_ENABLED=false
MAX_WORKERS=2
```

## Next Steps

After configuration:

1. **[Environment Setup →](environment-setup.md)** - Validate your setup and initialize the database
2. **[Quick Start →](quick-start.md)** - Run your first citation analysis
3. **[User Guide →](../user-guide/overview.md)** - Learn about platform features

## Troubleshooting Configuration

### Common Configuration Issues

??? question "Database connection timeouts"

    Increase timeout values in your configuration:
    ```env
    NEO4J_TIMEOUT=60
    REQUEST_TIMEOUT=300
    ```

??? question "Memory issues with large datasets"

    Optimize memory settings:
    ```env
    ML_BATCH_SIZE=512         # Reduce batch size
    MAX_CACHE_SIZE=500        # Reduce cache size
    MAX_WORKERS=2             # Reduce worker threads
    ```

??? question "SSL/TLS certificate errors"

    For development environments:
    ```env
    NEO4J_ENCRYPTED=false
    NEO4J_TRUST_STRATEGY=TRUST_ALL_CERTIFICATES
    ```

### Configuration Best Practices

1. **Environment Separation**: Use different `.env` files for development, testing, and production
2. **Secret Management**: Use proper secret management tools in production
3. **Resource Limits**: Set appropriate limits based on your system capabilities
4. **Monitoring**: Enable logging and analytics to monitor configuration effectiveness
5. **Backup**: Keep secure backups of your configuration files