# Academic Citation Platform

**ML-powered research discovery platform predicting citation relationships using TransE embeddings on 12K+ academic papers. Interactive visualization, community detection, and temporal analysis with Neo4j graph database backend.**

*Combines network analysis, machine learning, and data visualization to facilitate academic research discovery.*

An interactive platform for analyzing academic citation networks and predicting research connections using machine learning.

📚 **[View Complete Documentation](https://docs.barbhs.com/citation-compass/)** 
🎯 **[Try Interactive Demo](https://cartography.barbhs.com/)** 
🩺 **[Neo4j Ping Playbook](docs/neo4j-ping-guide.md)** 

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

2. **🎭 Try Demo Mode First!** (no database required):
   ```bash
   # Start interactive dashboard
   streamlit run app.py
   # Opens at http://localhost:8501/
   ```
   
   **Demo Mode Features**:
   - **🚀 Zero Setup** - Works instantly without any configuration!
   - **📊 Curated Academic Data** - 13 high-impact papers across AI, neuroscience, physics
   - **🤖 ML Predictions** - Citation prediction with synthetic embeddings (no training required!)
   - **🔗 Interactive Networks** - Clickable nodes, real-time filtering, enhanced visualizations
   - **📁 File Upload Testing** - Try the drag-and-drop interface with sample files
   - **📈 Full Analytics** - Community detection, temporal analysis, export capabilities
   
   Navigate to **"Demo Datasets"** → Load **"complete_demo"** → Explore all features!

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

   *A note about how to register your environment as a kernel in Jupyter:*. 
   *Pre-req: `pip install ipykernel`*   
   ```
   python -m ipykernel install --user --name=myenv --display-name="Python (myenv)"

   # The --user flag installs it for the current user, and --name is a unique internal name, while --display-name is what you'll see in the Jupyter UI. 
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

### 📁 File Upload for Research Collections (NEW!)
Import your paper collections effortlessly with drag-and-drop:
- **📂 Drag & Drop Interface** - Upload .txt or .csv files with paper IDs
- **📋 Multiple Formats** - Support for plain text lists and CSV with metadata
- **✅ Real-time Validation** - Instant feedback on file format and content
- **📊 Progress Tracking** - Monitor import with streaming updates and performance metrics
- **📄 Sample Files** - Download examples to get started quickly
- **🔄 Batch Processing** - Efficiently handle large research collections (1000+ papers)

### 📥 Enhanced Data Import Pipeline
Populate your database with real academic data using multiple methods:
- **🔍 Search Queries** - Import papers by academic search terms
- **🆔 Paper ID Lists** - Import specific papers by Semantic Scholar ID
- **📁 File Upload** - Bulk import from your research file collections
- **⚡ Streaming Performance** - 25x faster imports with real-time progress tracking
- **🛡️ Error Handling** - Graceful failure recovery with detailed reporting
- **🎯 Quality Filters** - Citation count, year range, and content filtering

### 🔮 ML Predictions
Predict citation relationships between papers using TransE embeddings:
- Input paper IDs or search by title
- Get top-K most likely citations with confidence scores
- Explore paper embeddings in vector space

### 📊 Enhanced Network Analysis
Analyze citation networks with powerful interactive features:
- **🖱️ Clickable Network Nodes** - Click any paper to view detailed information
- **🎨 Interactive Visualizations** - Real-time filtering, zoom, pan with smooth animations
- **🏘️ Community Detection** - Discover research clusters with multiple algorithms
- **📊 Centrality Analysis** - Betweenness, eigenvector, PageRank measures
- **⏰ Temporal Trends** - Track citation evolution over time
- **📄 Export Capabilities** - Generate academic reports, LaTeX tables, high-resolution graphics

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

### 🧪 Research Pipeline: Complete Analysis Workflow

Our 4-notebook pipeline tells a complete data science story from exploration to presentation:

#### **📓 01_comprehensive_exploration.ipynb** - *"Network Discovery & Foundation"*
- **What it does**: Analyzes citation network topology, community detection, temporal patterns
- **Key outputs**: Network statistics, centrality analysis, community structure insights
- **For portfolios**: Demonstrates systematic EDA and network analysis expertise
- **Status**: ✅ **Production ready** - Handles database fallbacks gracefully

#### **🤖 02_model_training_pipeline.ipynb** - *"TransE Model Development"*  
- **What it does**: Implements TransE from scratch, trains on citation data, validates performance
- **Key outputs**: Trained model (18.5MB), training visualizations, embedding analysis
- **For portfolios**: Shows ML engineering skills, custom model implementation
- **Status**: ✅ **Production ready** - Successfully trains models with early stopping

#### **📊 03_prediction_evaluation.ipynb** - *"Performance Validation & Discovery"*
- **What it does**: Comprehensive evaluation (MRR, Hits@K, AUC), generates citation predictions  
- **Key outputs**: Performance metrics, 1000+ novel citation predictions, evaluation dashboard
- **For portfolios**: Demonstrates proper ML evaluation and business value creation
- **Status**: ✅ **Production ready** - Robust evaluation with confidence analysis

#### **🎭 04_narrative_presentation.ipynb** - *"Scholarly Matchmaking Story"*
- **What it does**: Creates compelling 4-act narrative visualization of the complete project
- **Key outputs**: Portfolio-quality story visualizations, executive summaries
- **For portfolios**: Shows ability to communicate technical work to multiple audiences
- **Status**: ✅ **Production ready** - Handles both demo and actual data modes

**⚠️ No trained models yet? Follow the numbered sequence above.**

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

### New User Experience (Enhanced!):
1. **🎭 Start with Demo Mode**: Launch Streamlit → Demo Datasets → Load "complete_demo" → Explore all features with zero setup
2. **📁 Upload Your Research**: Data Import → File Upload → Drag-and-drop your .txt/.csv paper collections → Monitor real-time progress  
3. **🔍 Search & Import**: Search academic papers → Apply quality filters → Stream import with 25x faster performance
4. **🤖 Train Models**: Run training notebook → Train TransE model → Save to local models

### Enhanced Analysis Workflows:
1. **🎯 Citation Prediction**: Input paper → Get ML predictions (works in demo mode!) → Explore synthetic embeddings → Validate results
2. **🕸️ Interactive Network Analysis**: Load data → Click network nodes → Trace citation paths → Apply real-time filters → Export high-res visualizations
3. **📊 Community Detection**: Select research domain → Detect communities → Analyze cross-field connections → Generate LaTeX reports
4. **⏰ Temporal Analysis**: Choose date range → Track citation evolution → Identify trends → Export academic summaries
5. **🔍 Research Discovery**: Explore embeddings → Find similar papers → Build reading lists → Track confidence scores

## 🔧 Advanced Configuration

### API Integration
- Semantic Scholar API for paper metadata
- Rate limiting and caching built-in
- Handles large dataset imports

### Performance (Recently Enhanced!)
- **⚡ Streaming Data Import** - 25x faster imports with real-time progress tracking  
- **🚀 Intelligent Batching** - Adaptive batch sizing for optimal performance
- **🧠 Smart Caching** - Intelligent caching for ML predictions and analytics
- **📊 Real-time Updates** - Live progress monitoring without blocking UI
- **🔄 Resumable Operations** - Large imports continue where they left off
- **🎯 Query Optimization** - Enhanced Neo4j query performance

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
