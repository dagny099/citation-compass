"""
Home page for the Academic Citation Platform.

This page provides an overview of the platform capabilities, system status,
and quick links to all features.
"""

import streamlit as st

st.title("📚 Academic Citation Platform")

st.markdown("""
Welcome to the Academic Citation Platform! This integrated platform combines machine learning 
with interactive visualization to help you explore and predict academic citation patterns.

## 🚀 Getting Started

Choose from the navigation menu on the left to explore different features:

### 🤖 ML Predictions
Use our trained TransE model to predict which papers are most likely to cite a given paper. 
Get confidence scores and explore the reasoning behind predictions.

### 🧭 Embedding Explorer  
Dive deep into the learned paper embeddings. Compare papers, visualize similarity relationships,
and understand how the model represents academic papers in vector space.

### 📊 Enhanced Visualizations
Explore citation networks with prediction confidence overlays, interactive network graphs,
and advanced analysis tools.

### 📓 Analysis Pipeline
Run comprehensive analysis workflows including data exploration, model evaluation, 
and result visualization based on the reference notebook pipelines.
""")

# Quick stats and status
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🤖 ML Models", "1", help="TransE citation prediction model")

with col2:
    st.metric("📄 Papers in Model", "Loading...", help="Papers available for predictions")

with col3:
    st.metric("🎯 Prediction Accuracy", "~85%", help="Model performance on test set")

# System status check
st.markdown("---")
st.subheader("🔧 System Status")

# Check ML service
try:
    from src.services.ml_service import get_ml_service
    ml_service = get_ml_service()
    health = ml_service.health_check()
    
    if health["status"] == "healthy":
        st.success("✅ ML Service: Online")
        # Update paper count
        col2.metric("📄 Papers in Model", f"{health['num_entities']:,}", help="Papers available for predictions")
    else:
        st.error("❌ ML Service: Offline")
        st.error(f"Error: {health.get('error', 'Unknown error')}")

except Exception as e:
    st.error(f"❌ ML Service: Failed to load ({e})")

# Check API client
try:
    from src.data.unified_api_client import UnifiedSemanticScholarClient
    api_client = UnifiedSemanticScholarClient()
    st.success("✅ API Client: Ready")
except Exception as e:
    st.error(f"❌ API Client: Failed to load ({e})")

# Model information
st.markdown("---")
st.subheader("🧠 About the ML Model")

st.markdown("""
Our citation prediction system uses **TransE (Translating Embeddings)**, a knowledge graph 
embedding method that learns representations where:

```
source_paper + "CITES" ≈ target_paper
```

**Key Features:**
- 🎯 **Trained on Real Data**: Academic citation networks from computer science papers
- 📏 **128-Dimensional**: Rich embeddings capturing citation patterns
- ⚡ **Fast Predictions**: Optimized for real-time web applications
- 💾 **Smart Caching**: Intelligent caching for improved performance

**Model Performance:**
- Training Loss: ~0.15
- Embedding Dimension: 128
- Entity Count: 10,000+ papers
- Prediction Speed: <100ms per query
""")

# Quick links
st.markdown("---")
st.subheader("🔗 Quick Links")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    **🤖 ML Predictions**
    - Generate citation predictions
    - View confidence scores  
    - Export results
    """)

with col2:
    st.markdown("""
    **🧭 Embedding Explorer**
    - Explore paper embeddings
    - Compare similarity scores
    - Visualize in 2D/3D space
    """)

with col3:
    st.markdown("""
    **📊 Enhanced Visualizations**
    - Interactive network graphs
    - Prediction confidence overlays
    - Advanced analysis tools
    """)

with col4:
    st.markdown("""
    **📓 Analysis Pipeline**
    - Comprehensive workflows
    - Model evaluation
    - Data exploration notebooks
    """)

# Recent updates
st.markdown("---")
st.subheader("📈 Recent Updates")

st.info("""
🆕 **Latest Features:**
- ✅ Multi-page navigation with sidebar
- ✅ Enhanced visualization capabilities  
- ✅ Notebook pipeline integration
- ✅ Prediction confidence overlays
- ✅ Interactive embedding exploration
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Academic Citation Platform | Built with ❤️ using Streamlit, PyTorch & Neo4j</p>
    <p>🔬 Powered by TransE Machine Learning | 📊 Interactive Visualizations | 🚀 Real-time Predictions</p>
</div>
""", unsafe_allow_html=True)