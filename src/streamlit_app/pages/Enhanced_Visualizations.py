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
from src.analytics.contextual_explanations import ContextualExplanationEngine

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
        explanation_engine = ContextualExplanationEngine()
        return ml_service, api_client, explanation_engine
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        return None, None, None

ml_service, api_client, explanation_engine = get_services()

if api_client is None:
    st.error("❌ API client not available. Cannot retrieve citation data.")
    st.stop()

# Handle ML service gracefully
ml_available = ml_service is not None
if not ml_available:
    st.warning("⚠️ ML prediction service not available. Visualization will show actual citations only.")
    st.info("💡 To enable ML predictions: Train the TransE model using the Analysis Pipeline.")
else:
    try:
        model_info = ml_service.get_model_info()
        st.sidebar.success(f"✅ ML Model loaded ({model_info.num_entities:,} entities)")
    except Exception as e:
        st.sidebar.warning(f"⚠️ ML Model issue: {str(e)[:50]}...")
        ml_available = False

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
show_predictions = st.sidebar.checkbox("Show ML predictions", value=ml_available, disabled=not ml_available)
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
                    if show_predictions and ml_available:
                        try:
                            predictions = ml_service.predict_citations(
                                center_paper, 
                                top_k=min(max_papers, 20),
                                score_threshold=confidence_threshold
                            )
                            all_predictions[center_paper] = predictions
                        except Exception as pred_error:
                            st.warning(f"⚠️ Could not get predictions for {center_paper}: {pred_error}")
                            all_predictions[center_paper] = []
                    
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
            
            # Enhanced Network Statistics with Contextual Explanations
            st.subheader("📈 Network Analysis with Academic Context")
            
            # Calculate network metrics
            network_density = nx.density(G) if len(G.nodes()) > 1 else 0
            avg_confidence = np.mean([d.get('confidence', 0) for _, _, d in G.edges(data=True) if d.get('edge_type') == 'predicted']) if any(d.get('edge_type') == 'predicted' for _, _, d in G.edges(data=True)) else 0
            predicted_edges = sum(1 for _, _, d in G.edges(data=True) if d.get('edge_type') == 'predicted')
            actual_edges = sum(1 for _, _, d in G.edges(data=True) if d.get('edge_type') == 'actual')
            
            # Generate contextual explanations
            network_metrics = {
                "network_density": network_density,
                "hits_at_10": min(avg_confidence, 1.0) if avg_confidence > 0 else 0
            }
            
            explanations = explanation_engine.bulk_explain_metrics(
                network_metrics, 
                context={"num_entities": len(G.nodes()), "num_predictions": predicted_edges}
            )
            
            # Display metrics with traffic light indicators
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Nodes", len(G.nodes()))
                if len(G.nodes()) > 0:
                    st.caption("Papers in analysis network")
            
            with col2:
                density_explanation = explanations.get("network_density")
                if density_explanation:
                    st.metric(
                        f"{density_explanation.performance_icon} Network Density", 
                        f"{network_density:.4f}",
                        help=density_explanation.short_description
                    )
                    with st.expander("📖 What does this mean?"):
                        st.write(f"**Performance:** {density_explanation.performance_level.value.title()}")
                        st.write(density_explanation.detailed_explanation)
                        st.write(f"**Academic Context:** {density_explanation.academic_context}")
                        st.write(f"**Typical Range:** {density_explanation.typical_range_text}")
                else:
                    st.metric("Network Density", f"{network_density:.4f}")
            
            with col3:
                st.metric("Predicted Citations", predicted_edges)
                if predicted_edges > 0:
                    confidence_explanation = explanations.get("hits_at_10")
                    if confidence_explanation:
                        st.metric(
                            f"{confidence_explanation.performance_icon} Avg Confidence",
                            f"{avg_confidence:.3f}",
                            help="Average ML prediction confidence"
                        )
                        with st.expander("🎯 Prediction Quality"):
                            st.write(f"**Performance:** {confidence_explanation.performance_level.value.title()}")
                            st.write(confidence_explanation.detailed_explanation)
                            if confidence_explanation.suggested_actions:
                                st.write("**Suggested Actions:**")
                                for action in confidence_explanation.suggested_actions:
                                    st.write(f"• {action}")
                    else:
                        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            
            with col4:
                st.metric("Actual Citations", actual_edges)
                if actual_edges > 0 and predicted_edges > 0:
                    overlap_rate = len([e for e in G.edges(data=True) if e[2].get('edge_type') in ['predicted', 'actual']]) / max(predicted_edges, actual_edges)
                    st.metric("Prediction Overlap", f"{overlap_rate:.1%}")
            
            # Research Insights Section
            st.subheader("💡 Research Insights & Recommendations")
            
            # Generate research insights based on network characteristics
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                st.markdown("**🔍 Network Structure Analysis**")
                if network_density > 0.01:
                    st.success("🟢 Dense citation network detected - good for community analysis")
                    st.write("• High interconnectivity suggests well-established research field")
                    st.write("• Suitable for identifying influential papers and research clusters")
                elif network_density > 0.005:
                    st.info("🟡 Moderate density - typical for academic networks")
                    st.write("• Standard citation pattern for academic literature")
                    st.write("• Good balance of specialization and connectivity")
                else:
                    st.warning("🔴 Sparse network - may indicate emerging or specialized field")
                    st.write("• Low connectivity typical for new or highly specialized areas")
                    st.write("• Consider expanding search scope or timeframe")
            
            with insights_col2:
                st.markdown("**🤖 ML Prediction Analysis**")
                if avg_confidence > 0.7:
                    st.success("🟢 High confidence predictions - excellent for recommendations")
                    st.write("• Model shows strong predictive capability")
                    st.write("• Suitable for automated citation suggestions")
                elif avg_confidence > 0.4:
                    st.info("🟡 Moderate confidence - good for research exploration")
                    st.write("• Predictions provide useful research directions")
                    st.write("• Combine with domain expertise for best results")
                elif avg_confidence > 0:
                    st.warning("🔴 Lower confidence - use with caution")
                    st.write("• Predictions should be validated manually")
                    st.write("• Consider model retraining or parameter adjustment")
                else:
                    st.info("ℹ️ No ML predictions available for current selection")
            
            # Export Options
            st.subheader("📋 Export & Research Tools")
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                if st.button("📊 Generate LaTeX Table"):
                    latex_table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Citation Network Analysis Results}}
\\begin{{tabular}}{{|l|c|c|}}
\\hline
Metric & Value & Performance \\\\
\\hline
Network Density & {network_density:.4f} & {explanations.get('network_density', {}).get('performance_level', 'N/A')} \\\\
Total Nodes & {len(G.nodes())} & - \\\\
Predicted Citations & {predicted_edges} & - \\\\
Avg Confidence & {avg_confidence:.3f} & {explanations.get('hits_at_10', {}).get('performance_level', 'N/A')} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
                    st.code(latex_table, language="latex")
                    st.success("✅ LaTeX table generated - copy above code")
            
            with export_col2:
                if st.button("📝 Research Summary"):
                    summary = f"""
## Citation Network Analysis Summary

**Dataset:** {len(center_papers)} center paper(s), {len(G.nodes())} total papers analyzed

**Network Characteristics:**
- Density: {network_density:.4f} ({explanations.get('network_density', {}).get('performance_level', 'Unknown')} performance)
- Structure: {('Dense' if network_density > 0.01 else 'Moderate' if network_density > 0.005 else 'Sparse')} citation network

**ML Predictions:**
- {predicted_edges} predicted citations with {avg_confidence:.1%} average confidence
- Performance: {explanations.get('hits_at_10', {}).get('performance_level', 'Unknown')} prediction quality

**Research Implications:**
{explanations.get('network_density', {}).get('interpretation_guide', 'Network analysis complete')}

**Next Steps:**
{'• '.join(explanations.get('hits_at_10', {}).get('suggested_actions', ['Continue research analysis']))}
"""
                    st.markdown(summary)
                    st.success("✅ Research summary generated")
            
            with export_col3:
                if st.button("🎯 Action Items"):
                    action_items = []
                    
                    # Collect action items from explanations
                    for metric_name, explanation in explanations.items():
                        if hasattr(explanation, 'suggested_actions') and explanation.suggested_actions:
                            action_items.extend(explanation.suggested_actions)
                    
                    # Add network-specific actions
                    if network_density < 0.002:
                        action_items.append("Consider expanding the citation network scope")
                    if predicted_edges > 0 and avg_confidence < 0.5:
                        action_items.append("Review model parameters and training data quality")
                    if actual_edges == 0:
                        action_items.append("Verify paper IDs and check data connectivity")
                    
                    st.markdown("**🚀 Recommended Actions:**")
                    for i, action in enumerate(set(action_items[:6]), 1):  # Remove duplicates, limit to 6
                        st.write(f"{i}. {action}")
                    
                    st.success("✅ Action plan generated")
        
        else:
            st.warning("No network data available. Check that papers exist in the model.")
    
    else:
        st.info("👆 Please select at least one paper using the sidebar controls to visualize the citation network.")

elif viz_type == "Prediction Confidence Heatmap":
    st.header("🔥 Prediction Confidence Heatmap")
    
    if not ml_available:
        st.error("❌ ML predictions are required for confidence heatmap visualization.")
        st.info("💡 Train the TransE model using the Analysis Pipeline to enable this feature.")
    elif len(center_papers) >= 2:
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
        
        # Enhanced Statistics with Contextual Analysis
        st.subheader("📊 Heatmap Analysis with Academic Context")
        
        # Calculate metrics
        avg_conf = np.mean(prediction_matrix[prediction_matrix > 0])
        max_conf = np.max(prediction_matrix)
        above_threshold = np.sum(prediction_matrix > confidence_threshold)
        
        # Generate explanations for confidence metrics
        confidence_metrics = {
            "mrr": avg_conf,  # Use MRR as proxy for average confidence
            "hits_at_10": max_conf  # Use hits@10 for max confidence interpretation
        }
        
        conf_explanations = explanation_engine.bulk_explain_metrics(
            confidence_metrics,
            context={"num_entities": len(center_papers), "matrix_size": len(center_papers)**2}
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_explanation = conf_explanations.get("mrr")
            if avg_explanation:
                st.metric(
                    f"{avg_explanation.performance_icon} Average Confidence",
                    f"{avg_conf:.3f}",
                    help=avg_explanation.short_description
                )
                with st.expander("🎯 Average Performance Analysis"):
                    st.write(f"**Performance Level:** {avg_explanation.performance_level.value.title()}")
                    st.write(avg_explanation.detailed_explanation)
                    st.write(f"**Academic Context:** {avg_explanation.academic_context}")
            else:
                st.metric("Average Confidence", f"{avg_conf:.3f}")
        
        with col2:
            max_explanation = conf_explanations.get("hits_at_10")
            if max_explanation:
                st.metric(
                    f"{max_explanation.performance_icon} Max Confidence",
                    f"{max_conf:.3f}",
                    help="Highest prediction confidence in matrix"
                )
                with st.expander("🏆 Peak Performance Analysis"):
                    st.write(f"**Performance Level:** {max_explanation.performance_level.value.title()}")
                    st.write(f"Best prediction achieves {max_conf:.1%} confidence")
                    if max_conf > 0.8:
                        st.success("🟢 Excellent peak performance suggests strong model capability")
                    elif max_conf > 0.5:
                        st.info("🟡 Good peak performance with room for improvement")
                    else:
                        st.warning("🔴 Lower peak confidence may indicate training issues")
            else:
                st.metric("Max Confidence", f"{max_conf:.3f}")
        
        with col3:
            st.metric("Above Threshold", f"{above_threshold}")
            threshold_rate = above_threshold / (len(center_papers)**2 - len(center_papers)) if len(center_papers) > 1 else 0
            st.metric("Threshold Rate", f"{threshold_rate:.1%}")
        
        # Matrix Pattern Analysis
        st.subheader("🔍 Pattern Recognition & Research Insights")
        
        pattern_col1, pattern_col2 = st.columns(2)
        
        with pattern_col1:
            st.markdown("**📈 Citation Pattern Analysis**")
            
            # Analyze diagonal vs off-diagonal
            diagonal_avg = np.mean(np.diag(prediction_matrix)) if len(center_papers) > 1 else 0
            off_diagonal = prediction_matrix[~np.eye(prediction_matrix.shape[0], dtype=bool)]
            off_diagonal_avg = np.mean(off_diagonal[off_diagonal > 0]) if len(off_diagonal[off_diagonal > 0]) > 0 else 0
            
            if diagonal_avg > off_diagonal_avg * 1.5:
                st.warning("🔴 High self-citation bias detected")
                st.write("• Model may be overfitting to paper self-similarity")
                st.write("• Consider reviewing training data for bias")
            elif off_diagonal_avg > diagonal_avg:
                st.success("🟢 Healthy cross-citation patterns")
                st.write("• Model effectively identifies inter-paper relationships")
                st.write("• Good for citation recommendation tasks")
            else:
                st.info("🟡 Balanced citation prediction patterns")
            
            # Identify strongest connections
            max_idx = np.unravel_index(np.argmax(prediction_matrix), prediction_matrix.shape)
            if prediction_matrix[max_idx] > 0:
                st.write(f"**Strongest predicted connection:**")
                st.write(f"Paper {max_idx[0]+1} → Paper {max_idx[1]+1} ({prediction_matrix[max_idx]:.3f})")
        
        with pattern_col2:
            st.markdown("**🚀 Actionable Recommendations**")
            
            # Generate specific recommendations based on matrix patterns
            recommendations = []
            
            if avg_conf > 0.7:
                recommendations.extend([
                    "✅ High confidence predictions suitable for automated recommendations",
                    "📚 Results ready for academic publication or presentation"
                ])
            elif avg_conf > 0.4:
                recommendations.extend([
                    "🔄 Consider ensemble methods to boost confidence",
                    "🎯 Focus on high-confidence predictions for practical use"
                ])
            else:
                recommendations.extend([
                    "⚠️ Review model architecture and training parameters",
                    "📊 Analyze training data quality and completeness"
                ])
            
            if above_threshold < len(center_papers):
                recommendations.append("🔍 Consider lowering confidence threshold for broader exploration")
            
            if threshold_rate > 0.5:
                recommendations.append("🎉 Strong inter-paper connectivity detected - excellent for network analysis")
            
            for i, rec in enumerate(recommendations[:5], 1):
                st.write(f"{i}. {rec}")
        
        # Export heatmap analysis
        if st.button("📋 Export Heatmap Analysis"):
            heatmap_analysis = f"""
## Citation Confidence Heatmap Analysis

**Dataset:** {len(center_papers)} papers, {len(center_papers)**2} total predictions

**Performance Metrics:**
- Average Confidence: {avg_conf:.3f} ({conf_explanations.get('mrr', {}).get('performance_level', 'Unknown')} level)
- Peak Confidence: {max_conf:.3f} ({conf_explanations.get('hits_at_10', {}).get('performance_level', 'Unknown')} level)
- Above Threshold ({confidence_threshold}): {above_threshold}/{len(center_papers)**2} ({threshold_rate:.1%})

**Pattern Analysis:**
- Diagonal Average: {diagonal_avg:.3f}
- Off-diagonal Average: {off_diagonal_avg:.3f}
- Pattern Assessment: {'Self-citation bias' if diagonal_avg > off_diagonal_avg * 1.5 else 'Healthy patterns'}

**Research Implications:**
{conf_explanations.get('mrr', {}).get('interpretation_guide', 'Matrix analysis complete')}

**Recommended Next Steps:**
{chr(10).join(f'• {rec.split(maxsplit=1)[1] if len(rec.split(maxsplit=1)) > 1 else rec}' for rec in recommendations)}
"""
            
            st.code(heatmap_analysis, language="markdown")
            st.success("✅ Heatmap analysis exported - copy above text")
    
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
    st.markdown("---")
    st.subheader("🤖 Model Status")
    
    if ml_available:
        try:
            model_info = ml_service.get_model_info()
            st.write(f"**Papers:** {model_info.num_entities:,}")
            st.write(f"**Embedding Dim:** {model_info.embedding_dim}")
        except Exception:
            st.write("**Status:** Model issues detected")
    else:
        st.write("**Status:** ❌ Not available")
        st.write("**Working:** ✅ Actual citations")
        st.write("**Limited:** 🔥 Heatmaps, 🎯 Predictions")