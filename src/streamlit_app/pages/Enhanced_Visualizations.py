"""
Enhanced Visualizations page with prediction confidence overlays.

This page provides advanced visualization capabilities including:
- Citation network graphs with ML prediction overlays
- Confidence-based node and edge styling
- Interactive network exploration with prediction context
- Comparative analysis of actual vs predicted citations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from typing import List, Dict, Optional, Tuple
import logging

from src.services.ml_service import get_ml_service
from src.data.unified_api_client import UnifiedSemanticScholarClient
from src.models.ml import CitationPrediction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Enhanced Visualizations",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Enhanced Visualizations with Prediction Confidence")
st.markdown("""
Explore citation networks enhanced with machine learning prediction overlays. 
Visualize how our TransE model's predictions align with actual citation patterns.
""")

# Initialize services
@st.cache_resource
def get_services():
    """Initialize and cache services."""
    try:
        ml_service = get_ml_service()
        api_client = UnifiedSemanticScholarClient()
        return ml_service, api_client
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        return None, None

ml_service, api_client = get_services()

if ml_service is None:
    st.error("❌ Services not available.")
    st.stop()

# Sidebar configuration
st.sidebar.header("🎛️ Visualization Controls")

# Visualization type selection
viz_type = st.sidebar.selectbox(
    "Choose visualization type:",
    [
        "Citation Network with Predictions",
        "Prediction Confidence Heatmap", 
        "Citation vs Prediction Comparison",
        "Embedding Space Network",
        "Temporal Citation Analysis"
    ]
)

# Common parameters
st.sidebar.subheader("📋 Parameters")
max_papers = st.sidebar.slider("Maximum papers to analyze", 5, 50, 20)
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.05)
show_predictions = st.sidebar.checkbox("Show ML predictions", value=True)
show_actual_citations = st.sidebar.checkbox("Show actual citations", value=True)

# Paper input
st.sidebar.subheader("📄 Paper Selection")
paper_input_method = st.sidebar.radio(
    "Input method:",
    ["Manual Paper IDs", "Search and Select"]
)

center_papers = []

if paper_input_method == "Manual Paper IDs":
    paper_ids_text = st.sidebar.text_area(
        "Enter paper IDs (one per line):",
        placeholder="paper_id_1\npaper_id_2\npaper_id_3",
        height=100
    )
    
    if paper_ids_text:
        center_papers = [pid.strip() for pid in paper_ids_text.split('\n') if pid.strip()]

elif paper_input_method == "Search and Select":
    search_query = st.sidebar.text_input(
        "Search for papers:",
        placeholder="machine learning citation"
    )
    
    if search_query and st.sidebar.button("🔍 Search"):
        try:
            with st.spinner("Searching papers..."):
                search_results = api_client.search_papers(search_query, limit=10)
                
                if search_results.get("data"):
                    st.sidebar.success(f"Found {len(search_results['data'])} papers")
                    
                    # Store results in session state
                    st.session_state['search_results'] = search_results['data']
        except Exception as e:
            st.sidebar.error(f"Search failed: {e}")
    
    # Display search results for selection
    if 'search_results' in st.session_state:
        selected_indices = st.sidebar.multiselect(
            "Select papers for analysis:",
            range(len(st.session_state['search_results'])),
            format_func=lambda x: f"{st.session_state['search_results'][x].get('title', 'No title')[:50]}..."
        )
        
        center_papers = [st.session_state['search_results'][i]['paperId'] for i in selected_indices]

# Main content based on visualization type
if viz_type == "Citation Network with Predictions":
    st.header("🕸️ Citation Network with ML Prediction Overlay")
    
    if len(center_papers) >= 1:
        st.info(f"Analyzing network for {len(center_papers)} center paper(s)...")
        
        # Generate network data
        network_data = {}
        all_predictions = {}
        
        with st.spinner("Building citation network..."):
            for center_paper in center_papers:
                try:
                    # Get predictions
                    if show_predictions:
                        predictions = ml_service.predict_citations(
                            center_paper, 
                            top_k=min(max_papers, 20),
                            score_threshold=confidence_threshold
                        )
                        all_predictions[center_paper] = predictions
                    
                    # Get actual citations if available
                    if show_actual_citations and api_client:
                        try:
                            citations = api_client.get_paper_citations(
                                center_paper, 
                                limit=min(max_papers, 20)
                            )
                            network_data[center_paper] = citations
                        except Exception as e:
                            logger.warning(f"Could not get citations for {center_paper}: {e}")
                            network_data[center_paper] = []
                
                except Exception as e:
                    st.error(f"Error processing {center_paper}: {e}")
        
        # Create network visualization
        if all_predictions or network_data:
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add center papers
            for paper in center_papers:
                G.add_node(paper, node_type='center', color='red', size=30)
            
            # Add predicted citations
            prediction_edges = []
            if show_predictions:
                for center_paper, predictions in all_predictions.items():
                    for pred in predictions:
                        target_paper = pred.target_paper_id
                        G.add_node(target_paper, node_type='predicted', color='blue', size=20)
                        G.add_edge(center_paper, target_paper, 
                                 edge_type='predicted', 
                                 confidence=pred.prediction_score,
                                 color='blue',
                                 width=pred.prediction_score * 10)
                        prediction_edges.append((center_paper, target_paper, pred.prediction_score))
            
            # Add actual citations
            citation_edges = []
            if show_actual_citations:
                for center_paper, citations in network_data.items():
                    for citation in citations:
                        if isinstance(citation, dict):
                            target_paper = citation.get('paperId', citation.get('id', 'unknown'))
                        else:
                            target_paper = str(citation)
                        
                        if target_paper != 'unknown':
                            G.add_node(target_paper, node_type='cited', color='green', size=15)
                            G.add_edge(center_paper, target_paper, 
                                     edge_type='actual',
                                     color='green',
                                     width=5)
                            citation_edges.append((center_paper, target_paper))
            
            # Create Plotly network visualization
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Prepare node traces
            node_traces = {}
            
            # Center nodes
            center_x = [pos[node][0] for node in G.nodes() if G.nodes[node].get('node_type') == 'center']
            center_y = [pos[node][1] for node in G.nodes() if G.nodes[node].get('node_type') == 'center']
            
            # Predicted nodes
            pred_x = [pos[node][0] for node in G.nodes() if G.nodes[node].get('node_type') == 'predicted']
            pred_y = [pos[node][1] for node in G.nodes() if G.nodes[node].get('node_type') == 'predicted']
            
            # Cited nodes
            cited_x = [pos[node][0] for node in G.nodes() if G.nodes[node].get('node_type') == 'cited']
            cited_y = [pos[node][1] for node in G.nodes() if G.nodes[node].get('node_type') == 'cited']
            
            # Edge traces
            edge_traces = []
            
            # Predicted edges
            if show_predictions:
                pred_edge_x = []
                pred_edge_y = []
                pred_confidences = []
                
                for edge in G.edges(data=True):
                    if edge[2].get('edge_type') == 'predicted':
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        pred_edge_x.extend([x0, x1, None])
                        pred_edge_y.extend([y0, y1, None])
                        pred_confidences.append(edge[2].get('confidence', 0))
                
                edge_traces.append(go.Scatter(
                    x=pred_edge_x, y=pred_edge_y,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='ML Predictions',
                    hoverinfo='none',
                    opacity=0.7
                ))
            
            # Actual citation edges
            if show_actual_citations:
                cite_edge_x = []
                cite_edge_y = []
                
                for edge in G.edges(data=True):
                    if edge[2].get('edge_type') == 'actual':
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        cite_edge_x.extend([x0, x1, None])
                        cite_edge_y.extend([y0, y1, None])
                
                edge_traces.append(go.Scatter(
                    x=cite_edge_x, y=cite_edge_y,
                    mode='lines',
                    line=dict(color='green', width=3),
                    name='Actual Citations',
                    hoverinfo='none',
                    opacity=0.8
                ))
            
            # Create figure
            fig = go.Figure()
            
            # Add edge traces
            for trace in edge_traces:
                fig.add_trace(trace)
            
            # Add node traces
            if center_x:
                fig.add_trace(go.Scatter(
                    x=center_x, y=center_y,
                    mode='markers',
                    marker=dict(size=30, color='red'),
                    name='Center Papers',
                    text=[f"Center: {node[:15]}..." for node in G.nodes() if G.nodes[node].get('node_type') == 'center'],
                    hoverinfo='text'
                ))
            
            if pred_x:
                fig.add_trace(go.Scatter(
                    x=pred_x, y=pred_y,
                    mode='markers',
                    marker=dict(size=20, color='blue'),
                    name='ML Predictions',
                    text=[f"Predicted: {node[:15]}..." for node in G.nodes() if G.nodes[node].get('node_type') == 'predicted'],
                    hoverinfo='text'
                ))
            
            if cited_x:
                fig.add_trace(go.Scatter(
                    x=cited_x, y=cited_y,
                    mode='markers',
                    marker=dict(size=15, color='green'),
                    name='Actual Citations',
                    text=[f"Cited: {node[:15]}..." for node in G.nodes() if G.nodes[node].get('node_type') == 'cited'],
                    hoverinfo='text'
                ))
            
            fig.update_layout(
                title="Citation Network with ML Prediction Overlay",
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Red: Center papers | Blue: ML predictions | Green: Actual citations",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='gray', size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Network statistics
            st.subheader("📈 Network Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Nodes", len(G.nodes()))
            
            with col2:
                st.metric("Total Edges", len(G.edges()))
            
            with col3:
                predicted_edges = sum(1 for _, _, d in G.edges(data=True) if d.get('edge_type') == 'predicted')
                st.metric("Predicted Citations", predicted_edges)
            
            with col4:
                actual_edges = sum(1 for _, _, d in G.edges(data=True) if d.get('edge_type') == 'actual')
                st.metric("Actual Citations", actual_edges)
        
        else:
            st.warning("No network data available. Check that papers exist in the model.")
    
    else:
        st.info("👆 Please select at least one paper using the sidebar controls to visualize the citation network.")

elif viz_type == "Prediction Confidence Heatmap":
    st.header("🔥 Prediction Confidence Heatmap")
    
    if len(center_papers) >= 2:
        # Generate predictions for all pairs
        with st.spinner("Generating prediction matrix..."):
            prediction_matrix = np.zeros((len(center_papers), len(center_papers)))
            paper_labels = [f"{pid[:12]}..." for pid in center_papers]
            
            for i, source_paper in enumerate(center_papers):
                predictions = ml_service.predict_citations(
                    source_paper,
                    candidate_paper_ids=center_papers,
                    top_k=len(center_papers)
                )
                
                for pred in predictions:
                    if pred.target_paper_id in center_papers:
                        j = center_papers.index(pred.target_paper_id)
                        prediction_matrix[i][j] = pred.prediction_score
        
        # Create heatmap
        fig = px.imshow(
            prediction_matrix,
            labels=dict(x="Target Papers", y="Source Papers", color="Confidence"),
            x=paper_labels,
            y=paper_labels,
            color_continuous_scale="RdYlBu_r",
            title="Citation Prediction Confidence Matrix"
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.subheader("📊 Heatmap Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Confidence", f"{np.mean(prediction_matrix[prediction_matrix > 0]):.3f}")
        
        with col2:
            st.metric("Max Confidence", f"{np.max(prediction_matrix):.3f}")
        
        with col3:
            st.metric("Predictions Above Threshold", 
                     f"{np.sum(prediction_matrix > confidence_threshold)}")
    
    else:
        st.info("Please select at least 2 papers for confidence heatmap analysis.")

elif viz_type == "Citation vs Prediction Comparison":
    st.header("⚖️ Citation vs Prediction Comparison")
    st.info("🚧 This feature compares actual citations with ML predictions - coming soon!")

elif viz_type == "Embedding Space Network":
    st.header("🌐 Embedding Space Network")
    st.info("🚧 Visualizing citation networks in embedding space - coming soon!")

elif viz_type == "Temporal Citation Analysis":
    st.header("📅 Temporal Citation Analysis")
    st.info("🚧 Time-based citation and prediction analysis - coming soon!")

# Footer
st.markdown("---")
st.markdown("""
**Visualization Guide:**
- **Red nodes**: Center/source papers for analysis
- **Blue nodes**: ML predicted citations with confidence scores
- **Green nodes**: Actual citations from the academic literature
- **Edge thickness**: Represents prediction confidence (thicker = higher confidence)
- **Colors**: Distinguish between prediction types and confidence levels
""")

# Sidebar help
with st.sidebar:
    st.markdown("---")
    st.subheader("💡 Visualization Tips")
    st.markdown("""
    **Network View:**
    - Red = Center papers
    - Blue = ML predictions  
    - Green = Actual citations
    - Hover for details
    
    **Confidence Heatmap:**
    - Darker colors = Higher confidence
    - Diagonal shows self-citations
    - Use for pattern analysis
    """)
    
    # Model status
    try:
        model_info = ml_service.get_model_info()
        st.markdown("---")
        st.subheader("🤖 Model Status")
        st.write(f"**Papers:** {model_info.num_entities:,}")
        st.write(f"**Embedding Dim:** {model_info.embedding_dim}")
    except Exception:
        pass