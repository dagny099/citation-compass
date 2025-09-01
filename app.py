"""
Main Streamlit application for the Academic Citation Platform.

This application provides an interactive web interface for exploring academic citations
with machine learning predictions powered by a trained TransE model.
"""

import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Academic Citation Platform",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/dagny099/citation-compass',
        'Report a bug': 'https://github.com/dagny099/citation-compass/issues',
        'About': '''
        # Academic Citation Platform
        
        An integrated platform for academic citation analysis powered by machine learning.
        
        **Features:**
        - 🤖 ML-powered citation predictions using TransE model
        - 🧭 Paper embedding exploration and visualization  
        - 📊 Interactive citation network analysis
        - 🔍 Paper and author search capabilities
        
        Built with Streamlit, PyTorch, and Neo4j.
        '''
    }
)

# Define pages for navigation
home_page = st.Page("pages/Home.py", title="Home", icon="🏠", default=True)
data_import_page = st.Page("src/streamlit_app/pages/Data_Import.py", title="Data Import", icon="📥")
demo_datasets_page = st.Page("src/streamlit_app/pages/Demo_Datasets.py", title="Demo Datasets", icon="🎭")
ml_predictions_page = st.Page("src/streamlit_app/pages/ML_Predictions.py", title="ML Predictions", icon="🤖")
embedding_explorer_page = st.Page("src/streamlit_app/pages/Embedding_Explorer.py", title="Embedding Explorer", icon="🧭")
visualization_page = st.Page("src/streamlit_app/pages/Enhanced_Visualizations.py", title="Enhanced Visualizations", icon="📊")
results_interpretation_page = st.Page("src/streamlit_app/pages/Results_Interpretation.py", title="Results Interpretation", icon="📋")
notebook_pipeline_page = st.Page("src/streamlit_app/pages/Notebook_Pipeline.py", title="Analysis Pipeline", icon="📓")

# Set up navigation
pg = st.navigation(
    {
        "Main": [home_page],
        "Data Management": [data_import_page, demo_datasets_page],
        "Machine Learning": [ml_predictions_page, embedding_explorer_page],
        "Analysis": [visualization_page, results_interpretation_page, notebook_pipeline_page],
    }
)

# Main application
def main():
    """Main application entry point."""
    # Run the selected page
    pg.run()


if __name__ == "__main__":
    main()