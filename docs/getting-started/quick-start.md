# Quick Start Guide

Get running with your first citation analysis in under 10 minutes!

## Before You Start

Make sure you've completed:

- ✅ [Installation](installation.md) - Platform installed with `pip install -e ".[all]"`
- ✅ [Configuration](configuration.md) - `.env` file configured with Neo4j credentials
- ✅ [Environment Setup](environment-setup.md) - Database connection validated

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

### Step 2: Explore Your Data

Once the platform is running, let's explore your citation network:

!!! tip "Demo Mode"
    If you don't have your own data yet, the platform includes sample data for testing all features.

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

!!! info "Model Training First?"
    Citation predictions require a trained model. If you haven't trained one yet, start with network analysis (Step 4) or follow the [notebook pipeline](../notebooks/overview.md) to train your model.

Let's predict potential citations between papers:

=== "🖥️ Dashboard Method"

    1. Go to **ML Predictions** page
    2. Check if model is loaded (green status)
    3. Enter a paper ID or search by title
    4. Click **Generate Predictions**
    5. Explore the results with confidence scores

=== "📓 Notebook Method"

    ```python
    # Load ML service
    from src.services.ml_service import get_ml_service
    
    ml_service = get_ml_service()
    
    # Get predictions for a paper
    paper_id = "your_paper_id_here"  # Replace with actual paper ID
    predictions = ml_service.predict_citations(paper_id, top_k=10)
    
    for pred in predictions:
        print(f"📄 {pred.target_title}")
        print(f"   Confidence: {pred.confidence:.3f}")
        print(f"   Score: {pred.score:.3f}")
        print()
    ```

### Step 4: Analyze Citation Communities

Discover research communities in your network:

=== "🖥️ Dashboard Method"

    1. Visit **Network Analysis** page
    2. Select **Community Detection**
    3. Choose algorithm (Louvain recommended)
    4. View interactive community visualization

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

- **[User Guide](../user-guide/overview.md)** - Comprehensive feature walkthrough
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