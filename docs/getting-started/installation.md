# Installation Guide

Get the Academic Citation Platform up and running on your system.

## Prerequisites

Before installing, ensure you have:

- **Python 3.8+** installed on your system
- **Neo4j database** (local or cloud instance)
- **Git** for cloning the repository
- **pip** for package management

!!! tip "Recommended Setup"
    We recommend using a **virtual environment** to isolate dependencies and avoid conflicts with other Python projects.

## Installation Options

Choose the installation method that best fits your needs:

=== "🚀 Quick Install (Recommended)"

    ```bash
    # Clone the repository
    git clone https://github.com/dagny099/citation-compass.git
    cd citation-compass
    
    # Install with all features
    pip install -e ".[all]"
    ```

=== "🔧 Custom Install"

    ```bash
    # Clone the repository
    git clone https://github.com/dagny099/citation-compass.git
    cd citation-compass
    
    # Install base dependencies only
    pip install -e .
    
    # Add optional feature sets as needed
    pip install -e ".[ml]"        # Machine learning features
    pip install -e ".[analytics]" # Advanced analytics
    pip install -e ".[viz]"       # Visualization tools
    pip install -e ".[web]"       # Streamlit interface
    pip install -e ".[dev]"       # Development tools
    ```

=== "🐳 Docker Setup"

    ```bash
    # Clone the repository
    git clone https://github.com/dagny099/citation-compass.git
    cd citation-compass
    
    # Build and run with Docker Compose
    docker-compose up -d
    ```

## Virtual Environment Setup

!!! warning "Important"
    Always use a virtual environment to avoid dependency conflicts:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install the platform
pip install -e ".[all]"
```

## Verify Installation

Test your installation to ensure everything is working correctly:

```bash
# Test basic imports
python -c "
from src.database.connection import Neo4jConnection
from src.services.analytics_service import get_analytics_service
print('✅ Basic installation successful')
"

# Run the test suite
python -m pytest tests/test_installation.py -v

# Verify Streamlit works
streamlit --version
```

## Next Steps

After installation, proceed to:

1. **[Configuration →](configuration.md)** - Set up your environment variables
2. **[Environment Setup →](environment-setup.md)** - Configure Neo4j and validate setup
3. **[Quick Start →](quick-start.md)** - Run your first analysis

## Troubleshooting

### Common Installation Issues

??? question "ImportError: No module named 'src'"

    This usually means the package wasn't installed in editable mode. Make sure to use:
    ```bash
    pip install -e ".[all]"
    ```
    Note the `-e` flag for editable installation.

??? question "Neo4j connection errors"

    Neo4j connection issues are usually configuration problems. See [Configuration Guide](configuration.md) for database setup.

??? question "PyTorch installation issues"

    If you encounter PyTorch installation problems:
    ```bash
    # Install PyTorch separately first
    pip install torch torchvision torchaudio
    # Then install the platform
    pip install -e ".[all]"
    ```

??? question "Permission denied errors"

    On some systems you may need to use `--user` flag:
    ```bash
    pip install --user -e ".[all]"
    ```

### Getting Help

If you encounter issues:

1. Check the installation steps above carefully
2. Search existing [GitHub Issues](https://github.com/dagny099/citation-compass/issues)
3. Join our community discussions
4. Create an issue on [GitHub](https://github.com/dagny099/citation-compass/issues)

## System Requirements

### Minimum Requirements

- **CPU**: 2+ cores
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for installation, additional space for data
- **Python**: 3.8, 3.9, 3.10, or 3.11

### Recommended for Large Datasets

- **CPU**: 4+ cores with good single-thread performance
- **RAM**: 16GB+ for large citation networks
- **Storage**: SSD recommended, 10GB+ free space
- **GPU**: CUDA-compatible GPU for ML training (optional)

### Supported Operating Systems

- **macOS**: 10.15+ (Catalina and later)
- **Linux**: Ubuntu 18.04+, CentOS 7+, other modern distributions
- **Windows**: Windows 10+ (Windows Subsystem for Linux recommended)
