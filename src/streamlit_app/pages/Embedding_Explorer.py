"""
Embedding Explorer page for the Academic Citation Platform.

This page allows users to:
- Explore paper embeddings in vector space
- Find similar papers based on embedding distance
- Visualize embedding relationships
- Compare embeddings between papers
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging

from src.services.ml_service import get_ml_service
from src.data.unified_api_client import UnifiedSemanticScholarClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Embedding Explorer",
    page_icon="🧭",
    layout="wide"
)

st.title("🧭 Paper Embedding Explorer")
st.markdown("""
Explore the learned embeddings from our TransE model. Similar papers should be positioned 
close together in the embedding space, while dissimilar papers should be far apart.
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

if ml_service is None:
    st.error("❌ ML service not available.")
    st.stop()

# Sidebar controls
st.sidebar.header("🎛️ Explorer Controls")

# Embedding comparison
st.sidebar.subheader("📊 Embedding Comparison")
paper_ids_input = st.sidebar.text_area(
    "Enter paper IDs (one per line):",
    placeholder="paper_id_1\npaper_id_2\npaper_id_3",
    height=100
)

# Parse paper IDs
paper_ids = []
if paper_ids_input:
    paper_ids = [pid.strip() for pid in paper_ids_input.split('\n') if pid.strip()]

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Individual Embeddings", "📊 Embedding Comparison", "🗺️ Embedding Visualization"])

with tab1:
    st.header("🔍 Individual Paper Embeddings")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        paper_id = st.text_input(
            "Enter Paper ID:",
            placeholder="e.g., paper123",
            key="individual_paper_id"
        )
        
        if paper_id and st.button("🎯 Get Embedding"):
            with st.spinner("Retrieving embedding..."):
                embedding = ml_service.get_paper_embedding(paper_id)
                
                if embedding:
                    st.success("✅ Embedding retrieved!")
                    
                    # Store in session state for use across tabs
                    if 'embeddings' not in st.session_state:
                        st.session_state.embeddings = {}
                    st.session_state.embeddings[paper_id] = embedding.embedding
                    
                    # Display embedding info
                    st.write(f"**Paper ID:** {embedding.paper_id}")
                    st.write(f"**Model:** {embedding.model_name}")
                    st.write(f"**Dimensions:** {embedding.embedding_dim}")
                    st.write(f"**Created:** {embedding.created_at}")
                    
                    # Get paper details if possible
                    try:
                        paper_details = api_client.get_paper_details(paper_id)
                        if paper_details:
                            st.write(f"**Title:** {paper_details.get('title', 'N/A')}")
                            if paper_details.get('authors'):
                                authors = ", ".join([a.get('name', '') for a in paper_details['authors'][:3]])
                                st.write(f"**Authors:** {authors}")
                    except Exception as e:
                        logger.warning(f"Could not get paper details: {e}")
                
                else:
                    st.error("❌ Paper not found in model")
    
    with col2:
        if paper_id and paper_id in st.session_state.get('embeddings', {}):
            embedding_vector = st.session_state.embeddings[paper_id]
            
            # Embedding statistics
            st.subheader("📈 Embedding Statistics")
            
            col_stats1, col_stats2 = st.columns(2)
            
            with col_stats1:
                st.metric("Mean", f"{np.mean(embedding_vector):.4f}")
                st.metric("Std Dev", f"{np.std(embedding_vector):.4f}")
                st.metric("Min Value", f"{np.min(embedding_vector):.4f}")
            
            with col_stats2:
                st.metric("Max Value", f"{np.max(embedding_vector):.4f}")
                st.metric("L2 Norm", f"{np.linalg.norm(embedding_vector):.4f}")
                st.metric("Non-zero", f"{np.count_nonzero(embedding_vector)}")
            
            # Embedding visualization
            st.subheader("🎨 Embedding Visualization")
            
            # Histogram of embedding values
            fig_hist = px.histogram(
                x=embedding_vector,
                nbins=30,
                title="Distribution of Embedding Values",
                labels={"x": "Embedding Value", "y": "Count"}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Line plot of embedding dimensions
            fig_line = px.line(
                x=range(len(embedding_vector)),
                y=embedding_vector,
                title="Embedding Vector (by dimension)",
                labels={"x": "Dimension", "y": "Value"}
            )
            st.plotly_chart(fig_line, use_container_width=True)

with tab2:
    st.header("📊 Embedding Comparison")
    
    if len(paper_ids) >= 2:
        # Get embeddings for all papers
        embeddings_data = {}
        missing_papers = []
        
        with st.spinner("Retrieving embeddings..."):
            for pid in paper_ids:
                embedding = ml_service.get_paper_embedding(pid)
                if embedding:
                    embeddings_data[pid] = embedding.embedding
                else:
                    missing_papers.append(pid)
        
        if missing_papers:
            st.warning(f"⚠️ Could not find embeddings for: {', '.join(missing_papers)}")
        
        if len(embeddings_data) >= 2:
            st.success(f"✅ Retrieved embeddings for {len(embeddings_data)} papers")
            
            # Calculate similarity matrix
            paper_list = list(embeddings_data.keys())
            embedding_matrix = np.array([embeddings_data[pid] for pid in paper_list])
            
            # Cosine similarity
            similarity_matrix = cosine_similarity(embedding_matrix)
            
            # Create similarity heatmap
            st.subheader("🔥 Cosine Similarity Matrix")
            
            fig_heatmap = px.imshow(
                similarity_matrix,
                labels=dict(x="Papers", y="Papers", color="Cosine Similarity"),
                x=paper_list,
                y=paper_list,
                color_continuous_scale="RdYlBu_r",
                title="Pairwise Cosine Similarity"
            )
            fig_heatmap.update_layout(width=600, height=600)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Similarity table
            st.subheader("📋 Pairwise Similarities")
            similarity_data = []
            
            for i, paper1 in enumerate(paper_list):
                for j, paper2 in enumerate(paper_list):
                    if i < j:  # Only show upper triangle
                        similarity_data.append({
                            "Paper 1": paper1[:20] + "..." if len(paper1) > 20 else paper1,
                            "Paper 2": paper2[:20] + "..." if len(paper2) > 20 else paper2,
                            "Cosine Similarity": f"{similarity_matrix[i][j]:.4f}",
                            "Euclidean Distance": f"{np.linalg.norm(embedding_matrix[i] - embedding_matrix[j]):.4f}"
                        })
            
            similarity_df = pd.DataFrame(similarity_data)
            st.dataframe(similarity_df, use_container_width=True)
            
            # Paper details
            if api_client:
                st.subheader("📄 Paper Details")
                for pid in paper_list:
                    try:
                        details = api_client.get_paper_details(pid)
                        if details:
                            with st.expander(f"📄 {pid}"):
                                st.write(f"**Title:** {details.get('title', 'N/A')}")
                                if details.get('authors'):
                                    authors = ", ".join([a.get('name', '') for a in details['authors'][:3]])
                                    st.write(f"**Authors:** {authors}")
                                if details.get('year'):
                                    st.write(f"**Year:** {details['year']}")
                                if details.get('citationCount'):
                                    st.write(f"**Citations:** {details['citationCount']:,}")
                    except Exception as e:
                        logger.warning(f"Could not get details for {pid}: {e}")
        
        else:
            st.warning("Need at least 2 valid paper IDs for comparison")
    
    else:
        st.info("💡 Enter at least 2 paper IDs in the sidebar to compare their embeddings")

with tab3:
    st.header("🗺️ Embedding Space Visualization")
    
    if len(paper_ids) >= 3:
        # Get embeddings
        embeddings_data = {}
        with st.spinner("Preparing visualization..."):
            for pid in paper_ids:
                embedding = ml_service.get_paper_embedding(pid)
                if embedding:
                    embeddings_data[pid] = embedding.embedding
        
        if len(embeddings_data) >= 3:
            paper_list = list(embeddings_data.keys())
            embedding_matrix = np.array([embeddings_data[pid] for pid in paper_list])
            
            # Dimensionality reduction options
            reduction_method = st.selectbox(
                "Choose dimensionality reduction method:",
                ["PCA", "t-SNE"]
            )
            
            if reduction_method == "PCA":
                reducer = PCA(n_components=2)
                reduced_embeddings = reducer.fit_transform(embedding_matrix)
                variance_explained = reducer.explained_variance_ratio_
                
                st.write(f"**Variance explained:** PC1: {variance_explained[0]:.3f}, PC2: {variance_explained[1]:.3f}")
            
            elif reduction_method == "t-SNE":
                perplexity = st.slider("t-SNE Perplexity", min_value=2, max_value=min(30, len(paper_list)-1), value=min(5, len(paper_list)-1))
                reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                reduced_embeddings = reducer.fit_transform(embedding_matrix)
            
            # Create 2D scatter plot
            fig_scatter = px.scatter(
                x=reduced_embeddings[:, 0],
                y=reduced_embeddings[:, 1],
                text=[pid[:15] + "..." if len(pid) > 15 else pid for pid in paper_list],
                title=f"Embedding Space Visualization ({reduction_method})",
                labels={"x": f"{reduction_method} Component 1", "y": f"{reduction_method} Component 2"}
            )
            
            fig_scatter.update_traces(textposition="top center", marker=dict(size=10))
            fig_scatter.update_layout(height=600)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # 3D visualization if we have enough points
            if len(embeddings_data) >= 4:
                st.subheader("🌐 3D Visualization")
                
                if reduction_method == "PCA":
                    reducer_3d = PCA(n_components=3)
                    reduced_3d = reducer_3d.fit_transform(embedding_matrix)
                else:
                    reducer_3d = TSNE(n_components=3, perplexity=perplexity, random_state=42)
                    reduced_3d = reducer_3d.fit_transform(embedding_matrix)
                
                fig_3d = px.scatter_3d(
                    x=reduced_3d[:, 0],
                    y=reduced_3d[:, 1],
                    z=reduced_3d[:, 2],
                    text=[pid[:10] + "..." if len(pid) > 10 else pid for pid in paper_list],
                    title=f"3D Embedding Space ({reduction_method})",
                    labels={
                        "x": f"{reduction_method} Component 1",
                        "y": f"{reduction_method} Component 2", 
                        "z": f"{reduction_method} Component 3"
                    }
                )
                fig_3d.update_traces(marker=dict(size=8))
                st.plotly_chart(fig_3d, use_container_width=True)
        
        else:
            st.warning("Need valid embeddings for visualization")
    
    else:
        st.info("💡 Enter at least 3 paper IDs in the sidebar for visualization")

# Footer
st.markdown("---")
st.markdown("""
**About Embeddings:** Paper embeddings are dense vector representations learned by the TransE model. 
Papers with similar citation patterns should have similar embeddings (high cosine similarity, low Euclidean distance).

**Visualization:** PCA preserves global structure but may lose local relationships. t-SNE preserves local structure but may distort global relationships.
""")

# Sidebar info
with st.sidebar:
    st.markdown("---")
    st.subheader("💡 Tips")
    st.markdown("""
    - **Individual Embeddings**: Explore properties of single paper embeddings
    - **Comparison**: Compare 2+ papers to find similarities
    - **Visualization**: Plot 3+ papers in reduced dimensional space
    - **Similarity**: Higher cosine similarity = more related papers
    """)
    
    # Model info
    try:
        model_info = ml_service.get_model_info()
        st.markdown("---")
        st.subheader("🤖 Model Info")
        st.write(f"**Embedding Dim:** {model_info.embedding_dim}")
        st.write(f"**Papers in Model:** {model_info.num_entities:,}")
    except Exception:
        pass