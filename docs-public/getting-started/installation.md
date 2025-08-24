# Installation Guide

Get the Academic Citation Platform up and running on your system.

## System Requirements

- **Python 3.8+** (Python 3.9+ recommended)
- **8GB RAM minimum** (16GB recommended for large datasets)
- **Graph database** (Neo4j recommended)

## Installation Options

Choose your preferred installation method:

=== "📦 Package Installation"

    ```bash
    # Install from PyPI (when available)
    pip install academic-citation-platform
    ```

=== "🛠️ Development Installation"

    ```bash
    # Clone the repository
    git clone https://github.com/your-username/academic-citation-platform.git
    cd academic-citation-platform
    
    # Install in development mode
    pip install -e ".[dev]"
    ```

=== "🐳 Docker Installation"

    ```bash
    # Pull and run the Docker image
    docker pull ghcr.io/your-username/academic-citation-platform:latest
    docker run -p 8501:8501 academic-citation-platform
    ```

## Database Setup

### Neo4j Database

The platform requires a Neo4j database instance:

=== "☁️ Cloud (Recommended)"

    1. Create a free account at [Neo4j Aura](https://console.neo4j.io/)
    2. Create a new database instance
    3. Note your connection details for configuration

=== "🏠 Local Installation"

    1. Download [Neo4j Desktop](https://neo4j.com/download/)
    2. Create a new database project
    3. Start your database instance
    4. Note the bolt connection details

=== "🐳 Docker Container"

    ```bash
    # Run Neo4j in Docker
    docker run \
        --name neo4j-citation \
        -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/your-secure-password \
        -v neo4j_data:/data \
        neo4j:5.0
    ```

## Verify Installation

Test your installation:

```bash
# Test Python package
python -c "import src; print('✅ Package imported successfully')"

# Test Streamlit app
streamlit run app.py --server.headless true --server.port 8502

# Run basic tests
python -m pytest tests/test_basic.py -v
```

## Common Installation Issues

??? question "Python version compatibility"

    Ensure you're using Python 3.8 or higher:
    ```bash
    python --version  # Should show 3.8+
    ```

??? question "Dependency conflicts"

    Use a virtual environment:
    ```bash
    python -m venv citation-env
    source citation-env/bin/activate  # Linux/Mac
    # or
    citation-env\Scripts\activate  # Windows
    ```

??? question "Memory issues"

    For systems with limited RAM:
    ```bash
    # Install with reduced dependencies
    pip install academic-citation-platform[minimal]
    ```

## Next Steps

After successful installation:

1. **[Configuration →](configuration.md)** - Set up your database and API connections
2. **[Quick Start →](quick-start.md)** - Run your first analysis
3. **[User Guide →](../user-guide/overview.md)** - Learn about platform features

## Optional Components

### Jupyter Extensions

For enhanced notebook experience:

```bash
# Install Jupyter Lab extensions
pip install jupyterlab
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### Performance Accelerators

For faster ML training:

```bash
# CUDA support (NVIDIA GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Metal Performance Shaders (Apple Silicon)
pip install torch torchvision torchaudio
```

### Development Tools

For contributors:

```bash
# Development dependencies
pip install -e ".[dev,test]"

# Pre-commit hooks
pre-commit install
```

---

Ready to configure? Continue to **[Configuration →](configuration.md)**