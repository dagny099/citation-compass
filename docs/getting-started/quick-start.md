# Quick Start Guide

Get running with your first citation analysis in under 10 minutes!

## Before You Start

Choose your path:

**🎭 Demo Mode First!** (recommended for all users):
- ✅ [Installation](installation.md) - Platform installed with `pip install -e ".[all]"`
- ✅ **Zero setup required** - Use demo datasets to explore features instantly!
- ✅ **Learn all features** - Perfect for understanding capabilities before production

**🏢 Production Setup** (after mastering demo mode):
- ✅ [Installation](installation.md) - Platform installed with `pip install -e ".[all]"`
- ✅ [Configuration](configuration.md) - `.env` file configured with Neo4j credentials
- ✅ [Environment Setup](environment-setup.md) - Database connection validated
- ✅ [Demo Experience](demo-mode.md) - Understanding gained from hands-on exploration

## Your First Citation Analysis

### Step 1: Launch the Platform

Choose your preferred interface:

=== "🖥️ Interactive Dashboard"

    ```bash
    # Launch Streamlit interface
    streamlit run app.py
    ```

    Your browser will open to `http://localhost:8501` with the interactive dashboard.

=== "📓 Jupyter Notebooks"

    ```bash
    # Start Jupyter
    jupyter notebook notebooks/
    ```

    Open `01_comprehensive_exploration.ipynb` to begin analysis.

### Step 2: Choose Your Data Source

Once the platform is running, choose how you want to explore citation networks:

=== "🎭 Demo Mode (Recommended - Zero Setup!)"

    **Perfect for all users - start here!**

    1. **Navigate to Demo Datasets** in the sidebar
    2. **Browse curated datasets**: 
        - **complete_demo**: 13 high-impact papers across AI, neuroscience, physics
        - **minimal_demo_5papers**: Quick 5-paper network for fast testing
    3. **Click "Load Dataset"** for instant sample data (loads in 2-3 seconds)
    4. **Explore all features** with realistic academic data:
        - ML predictions with synthetic embeddings
        - Interactive network visualizations with clickable nodes
        - Community detection across research fields
        - Export capabilities for reports and analysis

    !!! success "Full Platform Experience"
        Demo mode provides complete functionality with curated academic papers spanning multiple research domains. Perfect for learning, testing, and demonstrating all platform capabilities!

=== "📁 File Upload (Your Research Collections)"

    **Import your own paper collections easily:**

    1. **Navigate to Data Import** → **Paper IDs** → **📁 File Upload**
    2. **Download sample files** to see the format (sample_paper_ids.txt/csv)
    3. **Upload your .txt/.csv files** with Semantic Scholar paper IDs
    4. **Monitor real-time progress** with streaming updates and performance metrics
    5. **Explore your imported data** using all platform features

    !!! tip "Start Small"
        Try with 10-50 papers first to learn the workflow, then scale up to larger collections!

=== "🔍 Search Import (Discover New Papers)"

    **Import papers by academic search:**

    1. **Navigate to Data Import** → **Search Query** 
    2. **Enter search terms**: "machine learning", "neural networks", etc.
    3. **Configure filters**: citation count, year range, quality settings
    4. **Start import** with real-time progress tracking
    5. **Analyze imported networks** immediately

=== "🏢 Production Database (Advanced)"

    **For large-scale production use:**

    1. **Complete demo experience first** to understand workflows
    2. **Configure Neo4j database** following [configuration guide](configuration.md)
    3. **Import data** using search or file upload methods
    4. **Train custom ML models** with your domain-specific data

#### In the Interactive Dashboard:

1. **Navigate to the Home page** - Overview of your citation network
2. **Check Network Analysis** - View basic statistics about your data:
   - Number of papers and citations
   - Network density and connectivity
   - Top-cited papers and influential authors

#### In Jupyter Notebooks:

Run the first few cells of `01_comprehensive_exploration.ipynb` to see:

```python
# Quick network overview
from src.services.analytics_service import get_analytics_service

analytics = get_analytics_service()
overview = analytics.get_network_overview()

print(f"📊 Network Overview:")
print(f"Papers: {overview.num_papers:,}")
print(f"Citations: {overview.num_citations:,}")
print(f"Authors: {overview.num_authors:,}")
print(f"Average citations per paper: {overview.avg_citations:.2f}")
```

### Step 3: Make Your First Citation Prediction

**New!** Citation predictions now work in demo mode with no setup required!

=== "🎭 Demo Mode Predictions (Recommended)"

    **Works immediately with demo datasets:**

    1. **Load a demo dataset** first (complete_demo recommended)
    2. Go to **ML Predictions** page  
    3. **Notice green status** - Demo ML service is ready!
    4. **Try a paper from your demo dataset**:
        - For complete_demo: Try "649def34f8be52c8b66281af98ae884c09aef38f9" (Attention Is All You Need)
        - Or search by title: "Attention"
    5. **Click Generate Predictions** 
    6. **Explore realistic results** with confidence scores based on:
        - Research field similarity (ML papers cite ML papers)  
        - Temporal patterns (newer papers cite foundational work)
        - Impact weighting (highly-cited papers get more predictions)

    !!! success "No Training Required!"
        Demo mode uses synthetic embeddings that cluster papers realistically by research field, providing educational ML prediction experience without model training!

=== "🏢 Production Mode Predictions"

    **For trained models with your data:**

    1. **Train model first** using [notebook pipeline](../user-guide/notebook-pipeline.md)
    2. Check **ML service status** (green = model loaded)
    3. Enter **paper ID from your database**
    4. Get **predictions based on your trained model**

=== "📓 Notebook Method"

    ```python
    # Works in both demo and production modes
    from src.services.ml_service import get_ml_service
    
    ml_service = get_ml_service()
    
    # Demo mode: Use papers from loaded demo dataset
    # Production: Use papers from your database
    paper_id = "649def34f8be52c8b66281af98ae884c09aef38f9"  # Attention paper in demo
    predictions = ml_service.predict_citations(paper_id, top_k=10)
    
    print(f"🤖 Predictions for paper: {paper_id}")
    for pred in predictions:
        print(f"📄 Target: {pred['target_id']}")
        print(f"   Confidence: {pred['confidence']:.3f}")
        print(f"   Field relationship: {pred.get('field_similarity', 'N/A')}")
        print()
    ```

### Step 4: Analyze Citation Communities

Discover research communities in your network:

=== "🖥️ Dashboard Method"

    1. Visit **Enhanced Visualizations** page
    2. **Explore interactive network** with clickable nodes!
        - Click any paper node to see detailed information
        - Trace citation paths visually
        - Filter by research field or publication year
    3. Try **Community Detection**:
        - Choose algorithm (Louvain recommended)
        - See research fields cluster together
        - Explore cross-field connections
    4. **Export visualizations** in high resolution

=== "📓 Notebook Method"

    ```python
    # Detect research communities
    communities = analytics.detect_communities(
        method='louvain',
        resolution=1.0
    )
    
    print(f"🏘️ Found {len(communities.communities)} research communities")
    
    # Show largest communities
    for i, community in enumerate(communities.communities[:5]):
        print(f"\nCommunity {i+1}: {len(community.papers)} papers")
        print(f"Top papers: {community.top_papers[:3]}")
    ```

### Step 5: Generate Your First Report

Export your analysis results:

=== "🖥️ Dashboard Method"

    1. Navigate to **Results Interpretation**
    2. Select the analysis results you want to export
    3. Choose export format (PDF, LaTeX, CSV)
    4. Click **Generate Report**

=== "📓 Notebook Method"

    ```python
    from src.analytics.export_engine import ExportEngine
    
    exporter = ExportEngine()
    
    # Generate comprehensive report
    report = exporter.generate_report(
        title="My First Citation Analysis",
        include_predictions=True,
        include_communities=True,
        format="latex"
    )
    
    print(f"📊 Report generated: {report.file_path}")
    ```

## Sample Workflows

Try these common analysis patterns:

### 🔍 Research Discovery Workflow

1. **Find a paper of interest** in your network
2. **Generate citation predictions** to find related work
3. **Explore the embedding space** to visualize paper relationships  
4. **Export reading list** with confidence scores

### 🕸️ Network Analysis Workflow

1. **Compute network statistics** (centrality, clustering)
2. **Detect research communities** using graph algorithms
3. **Analyze temporal trends** in citation patterns
4. **Generate LaTeX report** for publication

### 🤖 ML Pipeline Workflow

1. **Train custom TransE model** on your data
2. **Evaluate model performance** with standard metrics
3. **Generate predictions** for paper recommendation
4. **Validate results** against known citations

## Next Steps

Now that you've completed your first analysis:

### 📚 Learn More

**New User Path**:
- **[Demo Mode Guide](demo-mode.md)** - Master demo features and educational workflows
- **[Demo Datasets](../user-guide/demo-datasets.md)** - Explore all available demo datasets
- **[File Upload Guide](file-upload.md)** - Import your research collections easily

**Advanced Features**:
- **[Interactive Features](../user-guide/interactive-features.md)** - Clickable nodes, real-time progress, enhanced UI
- **[Data Import](../user-guide/data-import.md)** - Comprehensive import pipeline with streaming features  
- **[User Guide](../user-guide/overview.md)** - Complete feature walkthrough
- **[Notebook Pipeline](../user-guide/notebook-pipeline.md)** - Complete analysis workflows
- **[ML Predictions](../user-guide/ml-predictions.md)** - Advanced prediction techniques

### 🔧 Customize Your Setup

- **[Model Training Notebook](../notebooks/overview.md)** - Train models for your domain
- **[API Reference](../api/services.md)** - Scale for large datasets
- **[Developer Guide](../developer-guide/architecture.md)** - Connect with other tools

### 🤝 Get Help

- **[GitHub Issues](https://github.com/dagny099/citation-compass/issues)** - Common issues and solutions
- **[API Reference](../api/services.md)** - Complete API documentation
- **[GitHub Issues](https://github.com/dagny099/citation-compass/issues)** - Report bugs or request features

## Quick Reference

### Essential Commands

```bash
# Start interactive dashboard
streamlit run app.py

# Run complete analysis pipeline
jupyter notebook notebooks/01_comprehensive_exploration.ipynb

# Test your setup
python -m pytest tests/test_integration.py -v

# Validate configuration
python scripts/validate_environment.py

# Generate API documentation
mkdocs serve --watch-theme
```

### Key File Locations

- **Configuration**: `.env`
- **Models**: `models/`
- **Outputs**: `outputs/`
- **Notebooks**: `notebooks/`
- **Documentation**: `docs/`

### Important URLs

- **Interactive Dashboard**: `http://localhost:8501`
- **Jupyter Notebooks**: `http://localhost:8888`
- **Documentation**: `http://localhost:8000` (if running `mkdocs serve`)

!!! success "Congratulations!"
    
    You've completed your first citation analysis! The platform is now ready for advanced research workflows and custom analysis projects.

    **Happy researching!** 🔬✨