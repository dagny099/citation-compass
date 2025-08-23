# Academic Citation Platform

Interactive platform for analyzing academic citation networks and predicting research connections using machine learning.

## 🚀 What This Does

- **Explore Citation Networks**: Interactive visualization of paper relationships
- **ML-Powered Predictions**: Find papers likely to cite each other using TransE embeddings
- **Research Analytics**: Community detection, temporal trends, and citation patterns
- **Export Results**: Generate reports, LaTeX tables, and research summaries

## 🎯 Quick Start

1. **Install**:
   ```bash
   pip install -e ".[all]"
   ```

2. **Configure** (copy `.env.example` to `.env` and add your Neo4j database):
   ```env
   NEO4J_URI=neo4j+s://your-database-url
   NEO4J_USER=neo4j  
   NEO4J_PASSWORD=your-password
   ```

3. **Launch Interactive App**:
   ```bash
   streamlit run app.py
   ```

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

### Python API Usage
```python
# ML predictions
from src.services.ml_service import get_ml_service
ml = get_ml_service()
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

# Test ML service
python -c "from src.services.ml_service import get_ml_service; print('✅ ML Ready')"

# Test analytics
python -c "from src.services.analytics_service import get_analytics_service; print('✅ Analytics Ready')"
```

## 📊 Sample Workflows

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

**Getting Started?** Just run `streamlit run app.py` and start exploring!