# Interactive Features Guide

Explore citation networks using the interactive Streamlit dashboard interface.

## Overview

The **Academic Citation Platform** provides a comprehensive interactive web interface built with **Streamlit**. This guide covers all the point-and-click features available through the dashboard.

## 🚀 Getting Started

### Launching the Dashboard

```bash
# Start the interactive dashboard
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501` with a multi-page interface.

## 📊 Dashboard Pages

### 1. Home Page

**Location**: Main landing page

**Features**:
- Platform overview and status
- Quick navigation to all features  
- System health indicators
- Recent activity summary

### 2. ML Predictions

**Location**: Machine Learning → ML Predictions

**Features**:
- Citation prediction interface
- Paper search and input
- Prediction results with confidence scores
- Interactive result exploration

!!! warning "Model Required"
    This page requires a trained TransE model. See [Model Training Guide](../notebooks/overview.md) for setup.

### 3. Embedding Explorer  

**Location**: Machine Learning → Embedding Explorer

**Features**:
- Paper embedding visualizations
- Interactive scatter plots
- Similarity exploration
- Embedding space navigation

### 4. Enhanced Visualizations

**Location**: Analysis → Enhanced Visualizations

**Features**:
- Interactive network plots
- Community detection visualization
- Temporal analysis charts
- Export capabilities

### 5. Results Interpretation

**Location**: Analysis → Results Interpretation

**Features**:
- Academic performance metrics
- Statistical interpretation
- Comparison benchmarking
- Report generation

### 6. Analysis Pipeline

**Location**: Analysis → Analysis Pipeline  

**Features**:
- Interactive notebook execution
- Parameter configuration
- Progress monitoring
- Result visualization

## 🎯 Key Interactive Features

### Search and Discovery

**Paper Search**:
- Search by title, keywords, or author
- Autocomplete suggestions
- Filter by publication year or venue
- Bulk paper selection

**Network Exploration**:
- Interactive network visualization
- Zoom and pan capabilities
- Node filtering and highlighting  
- Edge weight visualization

### Prediction Interface

**Input Methods**:
- Direct paper ID entry
- Title-based search
- Batch prediction upload
- API integration

**Result Exploration**:
- Sortable prediction tables
- Confidence score visualization
- Similar paper recommendations
- Export to various formats

### Visualization Controls

**Interactive Elements**:
- Dynamic filtering controls
- Color coding options
- Layout algorithm selection
- Animation controls

**Export Options**:
- PNG/SVG image export
- Interactive HTML files
- PDF report generation
- LaTeX table export

## 🔧 Customization Options

### Dashboard Configuration

Most features can be customized through the interface:

- **Display preferences**: Theme, layout, font sizes
- **Analysis parameters**: Algorithm settings, thresholds
- **Visualization options**: Colors, node sizes, edge styles
- **Export formats**: File types, quality settings

### Session Management

- **State persistence**: Settings saved between sessions
- **Progress tracking**: Analysis history and bookmarks
- **Data caching**: Improved performance for repeated queries
- **Export history**: Access previous exports

## 📈 Performance Tips

### Optimizing Interactive Performance

1. **Data Size**: Start with smaller datasets for exploration
2. **Caching**: Enable caching for repeated analyses
3. **Browser**: Use Chrome or Firefox for best performance
4. **Resources**: Close unused browser tabs

### Memory Management

- Monitor memory usage in large network visualizations
- Use filtering to reduce displayed data
- Clear cache periodically
- Restart session if performance degrades

## 🛠️ Troubleshooting

### Common Issues

**Dashboard Won't Load**:
- Check that `streamlit run app.py` completed successfully
- Verify port 8501 is available
- Try refreshing the browser

**Slow Performance**:
- Reduce visualization complexity
- Enable data sampling for large datasets
- Check available system memory

**Missing Features**:
- Verify all dependencies are installed
- Check that database connection is working
- Ensure trained models are available for ML features

### Getting Help

- Check the browser console for error messages
- Review Streamlit logs in the terminal
- Visit [GitHub Issues](https://github.com/dagny099/citation-compass/issues) for community support

## 🎨 Advanced Usage

### Custom Visualizations

The dashboard supports custom visualization parameters:

- **Network layouts**: Force-directed, circular, hierarchical
- **Color schemes**: Categorical, continuous, custom palettes
- **Node sizing**: By citation count, centrality, or custom metrics
- **Edge styling**: Thickness, opacity, color coding

### Integration with External Tools

- **Export compatibility**: Gephi, Cytoscape, NetworkX formats
- **API endpoints**: RESTful interface for external applications
- **Embedding integration**: Compatible with TensorBoard, UMAP

## 📚 Next Steps

Ready to dive deeper? Explore these related guides:

- **[Network Analysis](network-analysis.md)** - Advanced graph analysis features
- **[ML Predictions](ml-predictions.md)** - Machine learning capabilities  
- **[Notebook Pipeline](notebook-pipeline.md)** - Programmatic analysis workflows
- **[Results Interpretation](results-interpretation.md)** - Understanding your results

---

*Happy exploring! 🚀✨*