# 📚 Citation Compass - Streamlit App

A comprehensive web application for academic citation analysis powered by machine learning.

## 🚀 Features

### 🤖 ML Predictions
- **Citation Prediction**: Use our trained TransE model to predict which papers are most likely to cite a given paper
- **Confidence Scoring**: Get probability-like confidence scores for each prediction
- **Interactive Results**: Explore predicted papers with detailed information and export results
- **Paper Search**: Find papers by title, author, or direct paper ID

### 🧭 Embedding Explorer
- **Vector Space Exploration**: Dive deep into learned paper embeddings
- **Similarity Analysis**: Compare papers and find semantically similar research
- **Dimensionality Reduction**: Visualize embeddings in 2D/3D space using PCA and t-SNE
- **Embedding Statistics**: Analyze embedding properties and distributions

### 📊 Enhanced Visualizations
- **Network Visualization**: Interactive citation network graphs with prediction overlays
- **Advanced Charts**: Multi-dimensional analysis with customizable visualizations
- **Export Capabilities**: High-quality outputs in multiple formats (PNG, SVG, PDF)
- **Real-time Updates**: Dynamic visualization updates based on ML predictions

### 📔 Interactive Analytics Pipeline
- **Interactive Analysis**: Jupyter-style notebook execution within Streamlit
- **Advanced Analytics**: Network analysis, community detection, temporal trends
- **Batch Processing**: Large-scale citation analysis and reporting
- **Custom Workflows**: User-defined analytical pipelines with export capabilities

### 📈 Advanced Analytics (New)
- **Network Analysis**: Centrality measures, community detection, path analysis
- **Temporal Analysis**: Citation trends, growth patterns, impact over time
- **Author Analytics**: Collaboration networks, influence metrics, career trajectories
- **Performance Metrics**: System health, prediction accuracy, cache efficiency

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch (for ML models)
- Streamlit
- Required Python packages (see requirements)

### Quick Start

1. **Install Dependencies**:
   ```bash
   pip install streamlit torch plotly scikit-learn pandas numpy
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Open Browser**: Navigate to `http://localhost:8501`

### Configuration

The app automatically detects and loads:
- **TransE Model**: Locally trained model from `models/` directory
- **Entity Mapping**: Paper ID to model entity mappings
- **API Configuration**: Semantic Scholar API settings

## 🎯 How to Use

### ML Predictions Page

1. **Input Paper**:
   - Enter a paper ID directly
   - Search by title or keywords
   - Browse search results and select

2. **Configure Predictions**:
   - Set number of predictions (1-50)
   - Adjust confidence threshold
   - Check model health status

3. **View Results**:
   - Interactive results table with confidence scores
   - Confidence distribution charts
   - Detailed paper information
   - Export results as CSV

### Embedding Explorer Page

1. **Individual Embeddings**:
   - Enter paper ID to get embedding vector
   - View embedding statistics and distributions
   - Visualize embedding dimensions

2. **Compare Papers**:
   - Enter multiple paper IDs (one per line)
   - View cosine similarity matrix
   - Analyze pairwise relationships

3. **Visualization**:
   - Plot 3+ papers in reduced dimensional space
   - Choose PCA or t-SNE reduction
   - Explore in 2D or 3D

### Enhanced Visualizations Page

1. **Network Graphs**:
   - Interactive citation network visualization
   - Overlay ML predictions on network structure
   - Customize node sizes, colors, and layout algorithms
   - Export high-quality visualizations

2. **Advanced Charts**:
   - Multi-dimensional scatter plots with prediction confidence
   - Time-series analysis of citation patterns
   - Distribution analyses and statistical summaries

### Interactive Analytics Pipeline

1. **Interactive Analysis**:
   - Execute pre-built analytical notebooks
   - Customize parameters and data ranges
   - Real-time results with progress indicators
   
2. **Custom Workflows**:
   - Create custom analytical pipelines
   - Combine multiple analysis types
   - Export comprehensive reports
   
3. **Advanced Analytics**:
   - Network centrality analysis
   - Community detection in citation networks
   - Temporal trend analysis
   - Performance benchmarking

## 🧠 About the ML Model

### TransE Architecture
- **Model Type**: Translating Embeddings for Knowledge Graphs
- **Embedding Dimension**: 128
- **Training Data**: Academic citation networks
- **Entities**: 10,000+ computer science papers
- **Prediction Logic**: `source + relation ≈ target`

### Performance Metrics
- **Training Loss**: ~0.15
- **Prediction Speed**: <100ms per query
- **Cache Hit Rate**: 90%+ for repeated queries
- **Confidence Calibration**: Probability-like scores from distance metrics

## 🏗️ Architecture

### Service Layer
```
├── ML Service (src/services/ml_service.py)
│   ├── TransE Model Loading
│   ├── Prediction Generation
│   ├── Embedding Extraction
│   └── Intelligent Caching
│
├── API Client (src/data/unified_api_client.py)
│   ├── Semantic Scholar Integration
│   ├── Rate Limiting
│   ├── Response Caching
│   └── Error Handling
│
└── Data Models (src/models/)
    ├── ML Models (PaperEmbedding, CitationPrediction)
    ├── Network Models (NetworkNode, NetworkEdge)
    └── API Models (APIResponse, SearchRequest)
```

### Streamlit Pages
```
├── app.py (Main Application)
├── src/streamlit_app/pages/
│   ├── ML_Predictions.py         # Citation prediction interface
│   ├── Embedding_Explorer.py     # Vector space exploration
│   ├── Enhanced_Visualizations.py # Network graphs & charts
│   └── Notebook_Pipeline.py       # Interactive analytics pipeline
└── .streamlit/
    └── config.toml
```

### Advanced Analytics Architecture
```
├── src/analytics/ (New)
│   ├── __init__.py
│   ├── network_analysis.py       # Graph metrics & community detection
│   ├── temporal_analysis.py      # Time-series citation analysis
│   ├── performance_metrics.py    # System performance analysis
│   └── export_engine.py          # Multi-format export capabilities
│
├── src/services/
│   ├── ml_service.py             # Existing ML service
│   └── analytics_service.py      # New analytics orchestration
│
└── notebooks/ (New)
    ├── 01_network_exploration.ipynb
    ├── 02_citation_analysis.ipynb
    └── 03_performance_benchmarks.ipynb
```

## 🔧 Configuration

### Environment Variables
- `SEMANTIC_SCHOLAR_API_KEY`: Optional API key for higher rate limits
- `NEO4J_URI`: Neo4j database connection (if using database features)
- `NEO4J_USER`: Database username
- `NEO4J_PASSWORD`: Database password

### Streamlit Configuration
- **Port**: 8501 (default)
- **Theme**: Custom academic theme
- **Caching**: Enabled for ML models and API responses
- **Error Handling**: Detailed error messages in development

## 📊 Performance Optimizations

### Caching Strategy
- **Model Loading**: Models cached on first load
- **Predictions**: LRU cache with TTL expiration
- **API Responses**: Response caching with rate limiting
- **Embeddings**: In-memory caching of frequently accessed embeddings

### Scalability Features
- **Lazy Loading**: Components loaded on-demand
- **Batch Processing**: Efficient handling of multiple predictions
- **Memory Management**: Automatic cache eviction
- **Error Recovery**: Graceful handling of service failures

## 🐛 Troubleshooting

### Common Issues

1. **Model Not Found**:
   - Ensure `models/` directory contains the locally trained model files
   - Check file permissions and paths

2. **Paper Not in Model**:
   - Model trained on specific dataset (computer science papers)
   - Try papers from major CS venues (ICML, NeurIPS, etc.)

3. **Slow Performance**:
   - First prediction takes longer (model loading)
   - Subsequent predictions are cached
   - Consider GPU for large-scale usage

4. **API Rate Limits**:
   - Built-in rate limiting prevents 429 errors
   - Consider API key for higher limits

### Debug Mode
```bash
# Run with debug logging
STREAMLIT_LOGGER_LEVEL=debug streamlit run app.py
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

This project is part of Citation Compass and follows the same licensing terms.

---

**Built with ❤️ using Streamlit, PyTorch, and Machine Learning**
