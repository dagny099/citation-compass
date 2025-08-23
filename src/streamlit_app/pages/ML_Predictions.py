"""
ML Predictions page for the Academic Citation Platform.

This page provides an interactive interface for users to:
- Enter a paper ID or search for papers
- Generate citation predictions using the TransE model
- View prediction results with confidence scores
- Explore predicted papers with detailed information
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional
import logging

from src.services.ml_service import get_ml_service, TransEModelService
from src.models.ml import CitationPrediction
from src.data.unified_api_client import UnifiedSemanticScholarClient
from src.data.api_config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="ML Citation Predictions",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ML Citation Predictions")
st.markdown("""
Use our trained TransE model to predict which papers are most likely to cite a given paper.
The model has been trained on academic citation networks to understand citation patterns.
""")

# Initialize services
@st.cache_resource
def get_services():
    """Initialize and cache ML service and API client."""
    try:
        ml_service = get_ml_service()
        api_client = UnifiedSemanticScholarClient()
        return ml_service, api_client
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        return None, None

ml_service, api_client = get_services()

if ml_service is None or api_client is None:
    st.error("❌ Services not available. Please check the configuration.")
    st.stop()

# Sidebar configuration
st.sidebar.header("🔧 Prediction Settings")

# Model health check
with st.sidebar:
    if st.button("🔍 Check Model Health"):
        with st.spinner("Checking model health..."):
            health = ml_service.health_check()
            
        if health["status"] == "healthy":
            st.success("✅ Model is healthy")
            st.json({
                "Entities": health["num_entities"],
                "Device": health["device"],
                "Cache": "Enabled" if health["cache_enabled"] else "Disabled"
            })
        else:
            st.error("❌ Model health check failed")
            st.json(health)

# Prediction parameters
top_k = st.sidebar.slider("Number of predictions", min_value=1, max_value=50, value=10)
score_threshold = st.sidebar.slider("Minimum confidence score", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

# Main interface
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📄 Input Paper")
    
    # Paper input methods
    input_method = st.radio(
        "How would you like to specify the paper?",
        ["Paper ID", "Search by Title", "Search by Author"]
    )
    
    paper_id = None
    paper_info = None
    
    if input_method == "Paper ID":
        paper_id_input = st.text_input(
            "Enter Paper ID",
            placeholder="e.g., 649def34f8be52c8b66281af98ae884c09aef38f9",
            help="Enter a Semantic Scholar paper ID"
        )
        
        if paper_id_input:
            paper_id = paper_id_input.strip()
            
            # Get paper details
            if st.button("📥 Load Paper Details"):
                with st.spinner("Loading paper details..."):
                    try:
                        paper_info = api_client.get_paper_details(paper_id)
                        if paper_info:
                            st.success("✅ Paper loaded successfully!")
                        else:
                            st.error("❌ Paper not found")
                    except Exception as e:
                        st.error(f"❌ Error loading paper: {e}")
    
    elif input_method == "Search by Title":
        title_query = st.text_input(
            "Enter paper title or keywords",
            placeholder="e.g., machine learning citation prediction"
        )
        
        if title_query and st.button("🔍 Search Papers"):
            with st.spinner("Searching papers..."):
                try:
                    search_results = api_client.search_papers(title_query, limit=10)
                    
                    if search_results.get("data"):
                        papers = search_results["data"]
                        
                        # Create selection interface
                        paper_options = []
                        for paper in papers:
                            title = paper.get("title", "No title")[:100]
                            authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])[:2]])
                            year = paper.get("year", "Unknown")
                            paper_options.append(f"{title} | {authors} | {year}")
                        
                        selected_idx = st.selectbox("Select a paper:", range(len(paper_options)), format_func=lambda x: paper_options[x])
                        
                        if selected_idx is not None:
                            selected_paper = papers[selected_idx]
                            paper_id = selected_paper.get("paperId")
                            paper_info = selected_paper
                            st.success(f"✅ Selected: {selected_paper.get('title', '')[:50]}...")
                    else:
                        st.warning("No papers found for this query")
                
                except Exception as e:
                    st.error(f"❌ Search error: {e}")
    
    elif input_method == "Search by Author":
        author_query = st.text_input(
            "Enter author name",
            placeholder="e.g., Geoffrey Hinton"
        )
        
        if author_query and st.button("🔍 Search by Author"):
            st.info("🚧 Author search functionality coming soon!")
    
    # Display paper information
    if paper_info:
        st.subheader("📋 Paper Information")
        
        # Basic info
        st.write(f"**Title:** {paper_info.get('title', 'No title')}")
        
        authors = paper_info.get("authors", [])
        if authors:
            author_names = ", ".join([a.get("name", "") for a in authors[:3]])
            if len(authors) > 3:
                author_names += f" (+{len(authors)-3} more)"
            st.write(f"**Authors:** {author_names}")
        
        if paper_info.get("year"):
            st.write(f"**Year:** {paper_info['year']}")
        
        if paper_info.get("citationCount"):
            st.write(f"**Citations:** {paper_info['citationCount']:,}")
        
        if paper_info.get("abstract"):
            with st.expander("📄 Abstract"):
                st.write(paper_info["abstract"])

with col2:
    st.header("🎯 Prediction Results")
    
    if paper_id and st.button("🚀 Generate Predictions", type="primary"):
        
        # Check if paper is in the model
        paper_embedding = ml_service.get_paper_embedding(paper_id)
        
        if not paper_embedding:
            st.warning(f"⚠️ Paper '{paper_id}' not found in the trained model. The model was trained on a specific dataset and may not include all papers.")
            st.info("💡 Try searching for papers from computer science venues published before 2023.")
        else:
            with st.spinner("🧠 Generating citation predictions..."):
                try:
                    # Generate predictions
                    predictions = ml_service.predict_citations(
                        source_paper_id=paper_id,
                        top_k=top_k,
                        score_threshold=score_threshold
                    )
                    
                    if predictions:
                        st.success(f"✅ Generated {len(predictions)} predictions!")
                        
                        # Create results DataFrame
                        results_data = []
                        for i, pred in enumerate(predictions):
                            results_data.append({
                                "Rank": i + 1,
                                "Paper ID": pred.target_paper_id,
                                "Confidence Score": f"{pred.prediction_score:.3f}",
                                "Confidence": pred.confidence_level.value,
                                "Raw Score": f"{pred.raw_score:.3f}" if pred.raw_score else "N/A"
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        
                        # Display results table
                        st.subheader("📊 Prediction Results")
                        st.dataframe(
                            results_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Confidence Score": st.column_config.ProgressColumn(
                                    "Confidence Score",
                                    help="Prediction confidence (higher = more likely)",
                                    min_value=0.0,
                                    max_value=1.0,
                                ),
                            }
                        )
                        
                        # Confidence distribution chart
                        st.subheader("📈 Confidence Distribution")
                        scores = [pred.prediction_score for pred in predictions]
                        
                        fig = px.bar(
                            x=list(range(1, len(scores) + 1)),
                            y=scores,
                            title="Prediction Confidence Scores",
                            labels={"x": "Prediction Rank", "y": "Confidence Score"},
                            color=scores,
                            color_continuous_scale="Viridis"
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed paper information
                        st.subheader("🔍 Explore Predicted Papers")
                        
                        # Select a paper to explore
                        paper_to_explore = st.selectbox(
                            "Select a paper to view details:",
                            options=range(len(predictions)),
                            format_func=lambda x: f"Rank {x+1}: {predictions[x].target_paper_id} (Score: {predictions[x].prediction_score:.3f})"
                        )
                        
                        if paper_to_explore is not None:
                            selected_prediction = predictions[paper_to_explore]
                            
                            # Get detailed information about the predicted paper
                            with st.spinner("Loading paper details..."):
                                try:
                                    predicted_paper_details = api_client.get_paper_details(selected_prediction.target_paper_id)
                                    
                                    if predicted_paper_details:
                                        col3, col4 = st.columns(2)
                                        
                                        with col3:
                                            st.write("**Prediction Info:**")
                                            st.write(f"- Confidence: {selected_prediction.prediction_score:.3f}")
                                            st.write(f"- Confidence Level: {selected_prediction.confidence_level.value}")
                                            st.write(f"- Raw Score: {selected_prediction.raw_score:.3f}" if selected_prediction.raw_score else "- Raw Score: N/A")
                                        
                                        with col4:
                                            st.write("**Paper Details:**")
                                            st.write(f"- Title: {predicted_paper_details.get('title', 'No title')}")
                                            
                                            if predicted_paper_details.get('authors'):
                                                authors = ", ".join([a.get('name', '') for a in predicted_paper_details['authors'][:2]])
                                                st.write(f"- Authors: {authors}")
                                            
                                            if predicted_paper_details.get('year'):
                                                st.write(f"- Year: {predicted_paper_details['year']}")
                                            
                                            if predicted_paper_details.get('citationCount'):
                                                st.write(f"- Citations: {predicted_paper_details['citationCount']:,}")
                                        
                                        if predicted_paper_details.get('abstract'):
                                            with st.expander("📄 Abstract"):
                                                st.write(predicted_paper_details['abstract'])
                                    
                                    else:
                                        st.warning("Could not load details for this paper")
                                
                                except Exception as e:
                                    st.error(f"Error loading paper details: {e}")
                        
                        # Export results
                        st.subheader("💾 Export Results")
                        
                        if st.button("📥 Download Results as CSV"):
                            csv_data = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv_data,
                                file_name=f"predictions_{paper_id[:12]}.csv",
                                mime="text/csv"
                            )
                    
                    else:
                        st.warning("No predictions generated. Try lowering the confidence threshold or check if the paper exists in the model.")
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
                    logger.error(f"Prediction error for paper {paper_id}: {e}")

# Footer information
st.markdown("---")
st.markdown("""
**About the Model:** This prediction system uses a TransE (Translating Embeddings) model trained on academic citation networks. 
The model learns embeddings where papers that cite each other are positioned close together in vector space.

**Confidence Scores:** Higher scores indicate stronger prediction confidence. Scores are normalized probability-like values derived from the model's distance calculations.
""")

# Model information in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("📊 Model Information")
    
    try:
        model_info = ml_service.get_model_info()
        st.write(f"**Model:** {model_info.model_name}")
        st.write(f"**Type:** {model_info.model_type.value}")
        st.write(f"**Embedding Dim:** {model_info.embedding_dim}")
        st.write(f"**Entities:** {model_info.num_entities:,}")
        
        if hasattr(model_info, 'training_metadata') and model_info.training_metadata:
            with st.expander("🔧 Training Details"):
                st.json(model_info.training_metadata)
    
    except Exception as e:
        st.write("Model info not available")