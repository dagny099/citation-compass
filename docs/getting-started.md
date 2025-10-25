# Getting Started with Citation Compass

Welcome! Citation Compass is your toolkit for exploring academic citation networks through machine learning, graph analysis, and interactive visualization. Whether you're discovering research connections, analyzing citation patterns, or building reading lists, you're in the right place.

!!! tip "New here? Start with Demo Mode"
    No database setup required! Try Demo Mode first to explore features with curated datasets, then scale up to your own research collections.

---

## Quick Start (3 Steps)

### 1. Install

```bash
# Clone and install
git clone https://github.com/dagny099/citation-compass.git
cd citation-compass
pip install -e ".[all]"
```

!!! note "Virtual Environment Recommended"
    Using a virtual environment? Great idea! Run `python -m venv venv` and `source venv/bin/activate` first.

### 2. Configure (Optional for Demo Mode)

For demo mode, **skip this step** and go straight to launching!

For production use with your own data:

```bash
# Copy template and add your Neo4j credentials
cp .env.example .env

# Edit .env with your Neo4j database details:
# NEO4J_URI=neo4j+s://your-database-url
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=your-password
```

!!! tip "Free Neo4j Database"
    Don't have a database yet? [Neo4j AuraDB](https://neo4j.com/cloud/aura-free/) offers free cloud instances perfect for getting started. Note: free instances pause after 30 days of inactivity—see our [Neo4j Health Monitoring guide](resources/neo4j-health-monitoring.md) for keeping them alive!

### 3. Launch

```bash
# Start the interactive application
streamlit run app.py
```

Your browser will open to `http://localhost:8501` with the Citation Compass dashboard!

---

## Your First Analysis (Demo Mode)

The fastest way to understand what Citation Compass can do:

1. **Navigate to Demo Datasets** in the sidebar
2. **Select "complete_demo"** (13 high-impact papers across AI, neuroscience, physics)
3. **Click "Load Dataset"** and explore:
    - **ML Predictions**: Generate citation recommendations using synthetic embeddings
    - **Network Analysis**: Detect research communities with graph algorithms
    - **Interactive Visualizations**: Click nodes to explore paper details
    - **Export Results**: Generate reports in LaTeX, CSV, or JSON

Demo mode provides the full platform experience—no database required!

---

## What You Can Do

### 🧠 ML-Powered Citation Prediction
Discover hidden connections between papers using **TransE embeddings**. The model learns semantic relationships in citation networks: papers that cite similar work cluster together in embedding space. Generate predictions with confidence scores, then validate them against your research intuition.

### 🕸️ Network Analysis
Explore citation networks with:
- **Community detection** (Louvain, Label Propagation algorithms)
- **Centrality measures** (PageRank, betweenness, eigenvector)
- **Temporal analysis** (track citation trends over time)
- **Path analysis** (find connections between distant papers)

### 📊 Interactive Visualization
The Streamlit interface provides:
- **Clickable network graphs** with paper details on demand
- **Real-time progress tracking** for data imports
- **Embedding space explorer** for visualizing paper relationships
- **Multi-format export** (LaTeX tables, academic reports, CSV)

### 📓 Research Notebooks
Four comprehensive Jupyter notebooks guide you through:
1. **Comprehensive Exploration** - Data discovery and network analysis
2. **Model Training Pipeline** - Train custom TransE models on your data
3. **Prediction Evaluation** - Validate model performance with MRR, Hits@K metrics
4. **Narrative Presentation** - Generate publication-ready visualizations

---

## System Requirements

**Minimum**:
- Python 3.8+ (3.10+ recommended)
- 4GB RAM (8GB+ recommended)
- 2GB free disk space

**For Large Datasets**:
- 16GB+ RAM for networks with 100K+ papers
- SSD recommended for database operations
- Optional: CUDA-compatible GPU for faster model training

**Supported Platforms**: macOS, Linux, Windows (WSL recommended)

---

## Installation Options

Choose the profile that fits your needs:

=== "🎓 Researcher (Recommended)"
    **Everything you need for citation analysis**
    ```bash
    pip install -e ".[all]"
    ```
    Includes: ML models, analytics, web interface, notebook support

=== "🤖 ML Focus"
    **Just the machine learning components**
    ```bash
    pip install -e ".[ml]"
    ```
    Includes: TransE models, prediction engine, embeddings

=== "🌐 Web Interface Only"
    **Interactive dashboard without ML**
    ```bash
    pip install -e ".[web]"
    ```
    Includes: Streamlit app, network visualization, data import

=== "💻 Developer"
    **Full setup with development tools**
    ```bash
    pip install -e ".[dev,all]"
    ```
    Includes: Everything plus testing, linting, type checking

---

## Database Setup

### Option 1: Demo Mode (No Database)
**Perfect for learning and testing!**

No setup required—just launch `streamlit run app.py` and load a demo dataset. Full functionality with synthetic data.

### Option 2: Neo4j AuraDB (Cloud, Free Tier)
**Best for getting started with your own data**

1. Create account at [Neo4j AuraDB](https://neo4j.com/cloud/aura-free/)
2. Create a free database instance
3. Download credentials (URI, username, password)
4. Add to `.env` file:
   ```bash
   NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-generated-password
   ```
5. Run database setup:
   ```bash
   python setup_database.py
   ```

!!! warning "Free Tier Limits"
    AuraDB free instances pause after 30 days of inactivity. Check out our [Neo4j Health Monitoring guide](resources/neo4j-health-monitoring.md) for an automated solution!

### Option 3: Local Neo4j (Docker)
**For advanced users who want full control**

```bash
docker run \
  --name neo4j \
  -p7474:7474 -p7687:7687 \
  -d \
  -v $HOME/neo4j/data:/data \
  --env NEO4J_AUTH=neo4j/your-password \
  neo4j:latest
```

Update `.env` with `NEO4J_URI=neo4j://localhost:7687`

---

## Verify Your Setup

```bash
# Test basic functionality
python -c "
from src.services.ml_service import get_ml_service
from src.database.connection import Neo4jConnection

print('✅ ML Service:', get_ml_service().health_check()['status'])
print('✅ Database:', 'connected' if Neo4jConnection().test_connection() else 'check config')
"

# Run test suite
python -m pytest tests/ -v
```

---

## Common Workflows

### 🔍 Research Discovery
**Find related papers you might have missed**

1. Load your dataset (demo or imported)
2. Navigate to **ML Predictions**
3. Enter a paper ID or search by title
4. Generate predictions with confidence scores
5. Export recommended reading list

### 🕸️ Network Exploration
**Understand citation communities**

1. Go to **Enhanced Visualizations**
2. View interactive network graph
3. Run community detection (try Louvain algorithm)
4. Explore cross-field connections
5. Generate LaTeX report for publication

### 📈 Model Training
**Train custom embeddings on your data**

1. Import your citation network (via search or file upload)
2. Open Jupyter: `jupyter notebook notebooks/`
3. Run `02_model_training_pipeline.ipynb`
4. Evaluate with `03_prediction_evaluation.ipynb`
5. Use trained model in Streamlit app

---

## Troubleshooting

??? question "Import errors when running Python code"
    Ensure you installed in editable mode with `-e` flag:
    ```bash
    pip install -e ".[all]"
    ```
    And activate your virtual environment if using one.

??? question "Can't connect to Neo4j"
    Check your `.env` file has correct credentials, then test:
    ```bash
    python -c "import os; print('URI:', os.getenv('NEO4J_URI'))"
    ```
    For AuraDB, ensure URI starts with `neo4j+s://` (secure connection).

??? question "Streamlit won't start"
    Verify installation: `streamlit --version`

    If missing, reinstall: `pip install -e ".[web]"`

??? question "ML predictions show errors"
    Check model files exist:
    ```bash
    ls -la models/
    ```
    Should show `transe_citation_model.pt`, `entity_mapping.pkl`, `training_metadata.pkl`.

    If missing, train models using the notebook pipeline or use demo mode.

---

## Next Steps

**Explore the Interface**:
- [User Guide](user-guide/overview.md) - Complete walkthrough of all features
- [Demo Datasets](user-guide/demo-datasets.md) - Details on curated demo collections
- [Interactive Features](user-guide/interactive-features.md) - Clickable nodes, real-time progress

**Scale Up**:
- [Data Import](user-guide/data-import.md) - Import your research collections
- [Notebook Pipeline](user-guide/notebook-pipeline.md) - Advanced analysis workflows
- [ML Predictions](user-guide/ml-predictions.md) - Train custom models

**Extend & Customize**:
- [Developer Guide](architecture.md) - System architecture and design decisions
- [API Reference](api.md) - Programmatic access to all features
- [Resources](resources/neo4j-health-monitoring.md) - Helpful guides for common tasks

---

**Welcome to Citation Compass—happy exploring!** 🧭✨
