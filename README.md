# Academic Citation Platform

Interactive platform for analyzing academic citation networks and predicting research connections using machine learning.

📚 **[View Complete Documentation](http://127.0.0.1:8000/)** (after running `mkdocs serve`)  
🎯 **[Try Interactive Demo](http://localhost:8501/)** (after running `streamlit run app.py`)

## 🚀 What This Does

- **Explore Citation Networks**: Interactive visualization of paper relationships
- **ML-Powered Predictions**: Find papers likely to cite each other using TransE embeddings
- **Research Analytics**: Community detection, temporal trends, and citation patterns
- **Export Results**: Generate reports, LaTeX tables, and research summaries

## 🎯 Quick Start

### Prerequisites
- **Python 3.8+** installed on your system
- **Git** for cloning the repository
- **Neo4j database** (optional - try demo mode first!)

### Installation & Setup

1. **Clone and Install**:
   ```bash
   git clone https://github.com/dagny099/citation-compass.git
   cd citation-compass
   pip install -e ".[all]"
   ```

2. **Try Demo Mode** (no database required):
   ```bash
   # Start interactive dashboard
   streamlit run app.py
   # Opens at http://localhost:8501/
   # Navigate to "Demo Datasets" to explore sample data instantly!
   ```

3. **For Production Use** (copy `.env.example` to `.env` and add your Neo4j database):
   ```env
   NEO4J_URI=neo4j+s://your-database-url
   NEO4J_USER=neo4j  
   NEO4J_PASSWORD=your-password
   ```

4. **Additional Options**:
   ```bash
   # Run Jupyter notebooks
   jupyter notebook notebooks/
   
   # View documentation
   mkdocs serve
   # Opens at http://127.0.0.1:8000/
   ```

### Installation Options
- **Quick Install**: `pip install -e ".[all]"` (recommended)
- **Custom Install**: Install specific components:
  - `pip install -e ".[ml]"` - Machine learning features
  - `pip install -e ".[analytics]"` - Advanced analytics
  - `pip install -e ".[viz]"` - Visualization tools
  - `pip install -e ".[web]"` - Streamlit interface

## 📱 Interactive Features

### 🎭 Demo Mode & Sample Data
Explore the platform instantly without any database setup:
- **Curated Datasets**: Academic papers from AI, NLP, physics, and computer science
- **Offline Mode**: Full functionality without Neo4j database
- **Quick Fixtures**: Fast-loading test data for development
- **Sample ML Predictions**: Pre-configured models with realistic results

### 📥 Data Import Pipeline
Populate your database with real academic data:
- **Semantic Scholar Integration**: Import papers by search query or ID list
- **Progress Tracking**: Resumable imports with real-time progress
- **Batch Processing**: Efficient handling of large datasets
- **Citation Networks**: Automatic relationship discovery and creation

### 🔮 ML Predictions
Predict citation relationships between papers using TransE embeddings:
- Input paper IDs or search by title
- Get top-K most likely citations with confidence scores
- Explore paper embeddings in vector space

### 📊 Network Analysis  
Analyze citation networks interactively:
- Community detection algorithms
- Centrality measures (betweenness, eigenvector, PageRank)
- Temporal citation trends
- Export results as academic reports

### 🧪 Research Notebooks
Run pre-built analysis workflows:
- Citation network analysis
- Author collaboration patterns  
- Field-specific research trends
- Configurable parameters and real-time results

### 📈 Results Interpretation
Get context for your findings:
- Academic benchmarking across domains
- Traffic light performance indicators
- Actionable research insights
- LaTeX export for papers

## 🛠️ For Data Scientists

### Train Your Own Models

**⚠️ No trained models yet? Follow the 4-notebook pipeline:**

1. **Start with exploration**: `jupyter notebook notebooks/01_comprehensive_exploration.ipynb`
2. **Train your model**: `jupyter notebook notebooks/02_model_training_pipeline.ipynb`
3. **Evaluate predictions**: `jupyter notebook notebooks/03_prediction_evaluation.ipynb`
4. **Create presentations**: `jupyter notebook notebooks/04_narrative_presentation.ipynb`

The training pipeline will:
- Load citation data from your Neo4j database
- Train a TransE model for citation prediction
- Save the trained model to `models/` directory
- Generate training visualizations and evaluations

**🚀 Quick Start (Without Models)**: You can explore network analysis and data preparation features even without trained models!

### Python API Usage
```python
# ML predictions (requires trained model)
from src.services.ml_service import get_ml_service
ml = get_ml_service()
# Note: Will raise FileNotFoundError if no model trained yet
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

### Database Schema
- **Papers**: Connected by citation relationships
- **Authors**: Linked to papers and institutions  
- **Venues**: Journals and conference publications
- **Fields**: Research domain classifications

### Testing Your Setup
```bash
# Test basic installation
python -c "from src.database.connection import Neo4jConnection; from src.services.analytics_service import get_analytics_service; print('✅ Basic installation successful')"

# Run all tests
python -m pytest tests/ -v

# Test ML service (will show warning if no trained model)
python -c "from src.services.ml_service import get_ml_service; ml=get_ml_service(); print('✅ ML Service Ready')"

# Test analytics
python -c "from src.services.analytics_service import get_analytics_service; print('✅ Analytics Ready')"
```

## 📊 Sample Workflows

### New User Experience:
1. **Try Demo Mode**: Launch Streamlit → Demo Datasets → Load sample data → Explore features
2. **Import Real Data**: Data Import → Search academic papers → Import with progress tracking
3. **Train Models**: Run training notebook → Train TransE model → Save to local models

### Analysis Workflows:
1. **Citation Prediction**: Input a paper → Get predicted citations → Validate with embeddings
2. **Network Analysis**: Select author/field → Detect communities → Export LaTeX summary
3. **Temporal Analysis**: Choose date range → Analyze citation trends → Generate insights
4. **Research Discovery**: Explore embeddings → Find similar papers → Build reading lists

## 🔧 Advanced Configuration

### API Integration
- Semantic Scholar API for paper metadata
- Rate limiting and caching built-in
- Handles large dataset imports

### Performance
- Intelligent caching for ML predictions
- Neo4j query optimization
- Background processing for large analyses

---

**Need the technical integration details?** See `README_INTEGRATION_SUMMARY.md`

## 📚 Documentation

The platform includes comprehensive documentation with:
- **Getting Started Guide**: Installation, configuration, and first steps
- **User Guide**: Interactive features and research workflows  
- **Developer Guide**: Architecture, API reference, and extending the platform
- **Notebook Documentation**: Complete pipeline from exploration to presentation

**Access Documentation**:
- **Online**: Run `mkdocs serve` and visit http://127.0.0.1:8000/
- **Features**: Search, dark/light theme, navigation, and code examples

**Getting Started?** 
1. Check the [installation guide](http://127.0.0.1:8000/getting-started/installation/) in the docs
2. Follow the [quick start](http://127.0.0.1:8000/getting-started/quick-start/) tutorial
3. Try the interactive demo: `streamlit run app.py`