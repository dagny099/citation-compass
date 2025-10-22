# Citation Compass

Citation Compass helps you explore academic citation networks, analyze relationships, and generate citation predictions. It combines an interactive Streamlit app with analytics services, an optional Neo4j graph backend, and a small ML pipeline (TransE embeddings).

📚 [Documentation](https://docs.barbhs.com/citation-compass/)  
▶️ [Interactive Demo](https://cartography.barbhs.com/)  
🩺 [Neo4j Ping Playbook](docs/neo4j-ping-guide.md)

## What It Does

- Explore citation networks: interactive visualization of paper relationships
- Predict likely citations with TransE-based embeddings
- Analyze communities, temporal trends, and centrality metrics
- Export results as LaTeX tables, figures, and summaries

### Visual Overview

![System architecture](docs/assets/diagrams/system-architecture.png)

![Home dashboard](docs/assets/screenshots/01-home-dashboard.png)

## Get Started

### Option A — Try the Demo (no database)
```bash
streamlit run app.py
# Opens at http://localhost:8501/
```
In the app, go to “Demo Datasets” → load “complete_demo” to explore features with curated sample data.

### Option B — Full Setup (with Neo4j)

Prerequisites:
- Python 3.8+
- Git
- A Neo4j instance (local or cloud)

Install:
```bash
git clone https://github.com/dagny099/citation-compass.git
cd citation-compass
pip install -e ".[all]"
```

Configure environment:
```env
NEO4J_URI=neo4j+s://your-database-url
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

Optional tools:
```bash
# Notebooks
jupyter notebook notebooks/

# Local docs
mkdocs serve  # http://127.0.0.1:8000/
```

Jupyter kernel note (optional):
```bash
pip install ipykernel
python -m ipykernel install --user --name=myenv --display-name="Python (myenv)"
```

### Installation Variants
- All features (recommended): `pip install -e ".[all]"`
- Specific extras: `"[ml]"` (ML), `"[analytics]"` (analytics), `"[viz]"` (visualization), `"[web]"` (Streamlit)

## Features (Overview)

### Demo Mode & Sample Data
- Curated datasets across several fields
- Works offline without a Neo4j database
- Sample ML predictions using included embeddings

### Importing Data
- Search queries (by keywords)
- Paper ID lists (Semantic Scholar IDs)
- File upload (.txt or .csv)
- Progress tracking and batching with basic error handling and filters

### ML Predictions
- Predict citation relationships between papers using TransE embeddings
- Return top-K predictions with confidence scores
- Explore embeddings in vector space

### Network Analysis
- Interactive network views with filtering and zoom
- Community detection (e.g., Louvain) and centrality metrics
- Temporal trend analysis; export figures and tables

### Notebooks (Workflow)
Four notebooks support end-to-end analysis:
- 01_comprehensive_exploration.ipynb — network exploration, communities, temporal patterns
- 02_model_training_pipeline.ipynb — TransE training and evaluation
- 03_prediction_evaluation.ipynb — metrics (MRR, Hits@K, AUC) and predictions
- 04_narrative_presentation.ipynb — presentation and summary visuals

If you plan to train a model:
- Load data from Neo4j
- Train TransE
- Save artifacts under `models/`
- Review training visualizations and evaluation outputs

### Python API Usage
```python
# ML predictions (requires trained model)
from src.services.ml_service import get_ml_service
ml = get_ml_service()
predictions = ml.predict_citations('paper_id', top_k=10)

# Network analysis
from src.services.analytics_service import get_analytics_service
analytics = get_analytics_service()
communities = analytics.detect_communities('author_id')
```

### Command Line Interface
```bash
# Import papers by search query
python -m src.cli.import_data search "machine learning" --max-papers 100

# Import specific paper IDs from file
python -m src.cli.import_data ids --ids-file paper_ids.txt --batch-size 20

# Import with filtering options
python -m src.cli.import_data search "neural networks" --max-papers 100 --year-range 2020 2024
```

### Database Entities
- Papers — citation relationships
- Authors — linked to papers and institutions
- Venues — journals and conferences
- Fields — research domain classifications

### Verify Your Setup
```bash
# Basic environment check
python -c "from src.database.connection import Neo4jConnection; from src.services.analytics_service import get_analytics_service; print('✅ Basic installation successful')"

# Run tests
python -m pytest tests/ -v

# Services (no model needed for analytics)
python -c "from src.services.ml_service import get_ml_service; ml=get_ml_service(); print('✅ ML Service Ready')"
python -c "from src.services.analytics_service import get_analytics_service; print('✅ Analytics Ready')"
```

## Sample Workflows

### New Users
1. Launch Streamlit → Demo Datasets → load "complete_demo"
2. Upload a .txt/.csv list of paper IDs (optional)
3. Search and import papers by keywords
4. Explore predictions and networks; export results

### Analysis Tasks
1. Citation prediction: input a paper → review predictions → explore embeddings
2. Network analysis: load data → inspect communities and centrality → export figures
3. Temporal analysis: set date range → examine trends → export tables

## Configuration Notes

### APIs and Data
- Semantic Scholar API for metadata
- Basic rate limiting and caching
- Supports larger imports with batching

### Performance
- Streaming imports with progress tracking
- Adaptive batching and basic caching for analytics and predictions
- Resumable operations for longer imports
- Query efficiency improvements for Neo4j

## Documentation

See the full documentation site for installation, quick start, user and developer guides, and notebook workflows:
- Online: https://docs.barbhs.com/citation-compass/
- Local: `mkdocs serve` → http://127.0.0.1:8000/

Looking for technical architecture and APIs? Start with:
- `docs/architecture.md`
- `docs/api.md`
