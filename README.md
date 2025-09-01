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
- **Neo4j database** (local or cloud instance)
- **Git** for cloning the repository

### Installation & Setup

1. **Clone and Install**:
   ```bash
   git clone https://github.com/dagny099/citation-compass.git
   cd citation-compass
   pip install -e ".[all]"
   ```

2. **Configure Environment** (copy `.env.example` to `.env` and add your Neo4j database):
   ```env
   NEO4J_URI=neo4j+s://your-database-url
   NEO4J_USER=neo4j  
   NEO4J_PASSWORD=your-password
   ```

3. **Launch Platform**:
   ```bash
   # Start interactive dashboard
   streamlit run app.py
   # Opens at http://localhost:8501/
   
   # OR run Jupyter notebooks
   jupyter notebook notebooks/
   
   # OR view documentation
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

### Database Schema
- **Papers**: Connected by citation relationships
- **Authors**: Linked to papers and institutions  
- **Venues**: Journals and conference publications
- **Fields**: Research domain classifications

### Testing Your Setup
```bash
# Run all tests
python -m pytest tests/ -v

# Test ML service (will show warning if no trained model)
python -c "from src.services.ml_service import get_ml_service; ml=get_ml_service(); print('✅ ML Service Ready')"

# Test analytics
python -c "from src.services.analytics_service import get_analytics_service; print('✅ Analytics Ready')"
```

## 📊 Sample Workflows

1. **Model Training**: Run training notebook → Train TransE model → Save to local models
2. **Citation Prediction**: Input a paper → Get predicted citations → Validate with embeddings
3. **Network Analysis**: Select author/field → Detect communities → Export LaTeX summary
4. **Temporal Analysis**: Choose date range → Analyze citation trends → Generate insights
5. **Research Discovery**: Explore embeddings → Find similar papers → Build reading lists

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