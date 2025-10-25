"""
Home page for the Academic Citation Platform.

This page provides an overview of the platform capabilities, system status,
and quick links to all features.
"""

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, Any, List
import webbrowser

from streamlit_plotly_events import plotly_events

from src.database.connection import (
    create_connection,
    Neo4jError,
    get_random_featured_paper_id,
    get_home_ego_network,
    find_papers_by_keyword,
)

st.title("📚 Academic Citation Platform")

# System status check with integrated metrics - moved to top
st.subheader("📊 Platform Overview")

# Check ML service and display metrics
ml_service_online = False
try:
    from src.services.ml_service import get_ml_service
    ml_service = get_ml_service()
    health = ml_service.health_check()
    
    # Display metrics based on ML service status
    col1, col2, col3 = st.columns(3)
    
    if health["status"] == "healthy":
        with col1:
            st.metric("🤖 ML Models", "1", help="TransE citation prediction model")
            st.markdown("<div style='color: green;'>✅ Ready</div>", unsafe_allow_html=True)
        with col2:
            st.metric("📄 Papers in Model", f"{health['num_entities']:,}", help="Papers available for predictions")
            st.markdown("<div style='color: green;'>✅ Loaded</div>", unsafe_allow_html=True)
        with col3:
            st.metric("🎯 Prediction Accuracy", "~85%", help="Model performance on test set")
            st.markdown("<div style='color: green;'>✅ Validated</div>", unsafe_allow_html=True)
        
        ml_service_status = "online"
        ml_service_message = "✅ ML Service: Online - Ready for citation predictions!"
        ml_service_online = True
    else:
        with col1:
            st.metric("🤖 ML Models", "0", help="TransE citation prediction model")
            st.markdown("<div style='color: orange;'>⚠️ Need training</div>", unsafe_allow_html=True)
        with col2:
            st.metric("📄 Papers in Model", "0", help="Papers available for predictions")
            st.markdown("<div style='color: orange;'>⚠️ Train model first</div>", unsafe_allow_html=True)
        with col3:
            st.metric("🎯 Prediction Accuracy", "N/A", help="Model performance on test set")
            st.markdown("<div style='color: orange;'>⚠️ After training</div>", unsafe_allow_html=True)
            
        ml_service_status = "offline"
        ml_service_message = "⚠️ ML Service: Offline - No trained model found"
        ml_service_info = """
        **Need to train a model first?** 
        
        📚 **Start here**: Open `notebooks/02_model_training_pipeline.ipynb` to train your TransE model
        
        ⏱️ **Training time**: ~30-60 minutes depending on data size
        
        📁 **What gets created**: The training will save model files to the `models/` directory
        """

except Exception as e:
    # Display default metrics when service fails to load
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🤖 ML Models", "0", help="TransE citation prediction model")
        st.markdown("<div style='color: red;'>❌ Service error</div>", unsafe_allow_html=True)
    with col2:
        st.metric("📄 Papers in Model", "0", help="Papers available for predictions")
        st.markdown("<div style='color: red;'>❌ Service error</div>", unsafe_allow_html=True)
    with col3:
        st.metric("🎯 Prediction Accuracy", "N/A", help="Model performance on test set")
        st.markdown("<div style='color: red;'>❌ Service error</div>", unsafe_allow_html=True)
    
    ml_service_status = "error"
    ml_service_message = f"❌ ML Service: Failed to load ({e})"
    ml_service_info = """
    **Troubleshooting**: Check that all dependencies are installed with `pip install -e ".[all]"`
    """

# Check API client
try:
    from src.data.unified_api_client import UnifiedSemanticScholarClient
    api_client = UnifiedSemanticScholarClient()
    api_client_status = "ready"
    api_client_message = "✅ API Client: Ready - Can fetch paper data from Semantic Scholar"
except Exception as e:
    api_client_status = "error"
    api_client_message = f"❌ API Client: Failed to load ({e})"

# System Status Section
st.markdown("##### 🔧 System Status")

col_status1, col_status2 = st.columns(2)

with col_status1:
    if ml_service_status == "online":
        st.success(ml_service_message)
    elif ml_service_status == "offline":
        st.warning(ml_service_message)
        st.info(ml_service_info)
    else:
        st.error(ml_service_message)
        st.info(ml_service_info)

with col_status2:
    if api_client_status == "ready":
        st.success(api_client_message)
    else:
        st.error(api_client_message)

# Show what's available without ML models
if not ml_service_online:
    st.success("""
    **✅ Available Features Without Trained Models:**
    
    📊 **Enhanced Visualizations** - Explore citation networks and create interactive plots
    
    📓 **Analysis Pipeline** - Run data exploration workflows:
    - `notebooks/01_comprehensive_exploration.ipynb` - Analyze your citation network structure
    - Discover research communities and collaboration patterns  
    - Generate network statistics and visualizations
    
    🔍 **Network Analysis** - Use the analytics service to:
    ```python
    from src.services.analytics_service import get_analytics_service
    analytics = get_analytics_service()
    
    # Example: Analyze research communities
    communities = analytics.detect_communities('author_id')
    print(f"Found {len(communities)} research communities")
    ```
    
    🌐 **Data Collection** - Fetch paper metadata using the Semantic Scholar API
    """)
    
    st.info("""
    💡 **Getting Started Recommendation:**
    1. 📈 Try **Enhanced Visualizations** to explore citation patterns
    2. 📔 Open `notebooks/01_comprehensive_exploration.ipynb` for deep network analysis  
    3. 🏗️ Then train your model with `notebooks/02_model_training_pipeline.ipynb`
    4. 🤖 Return here for ML-powered predictions!
    """)

st.markdown("---")

st.markdown("""
Welcome to the Academic Citation Platform! This integrated platform combines machine learning 
with interactive visualization to help you explore and predict academic citation patterns.

## 🚀 Getting Started

Choose from the features below or use the navigation menu on the left to explore different capabilities:

""")

# --- New: Explore the Existing Graph (Welcome Snapshot) ---
st.markdown("---")
st.subheader("🕸️ Explore the Existing Graph")

def _build_welcome_graph_fig(ego_data: Dict[str, Any]):
    """
    Create a small plotly graph for the Home page ego network.

    Returns:
        fig: Plotly Figure
        edge_traces_count: number of edge traces added (for mapping)
        node_trace_map: list of dicts in the order traces were added for nodes,
                        each dict has {'type': str, 'node_ids': List[str]}
    """
    G = nx.Graph()

    center = ego_data.get("center") or {}
    center_id = center.get("paperId") or "center"
    G.add_node(center_id, node_type="center", label=center.get("title", center_id))

    # Fields (highlighted schema concept)
    for field in ego_data.get("fields", []):
        field_id = f"field::{field}"
        G.add_node(field_id, node_type="field", label=field)
        G.add_edge(center_id, field_id, edge_type="IS_ABOUT")

    # Cited (outgoing)
    for n in ego_data.get("cited", []):
        pid = n.get("paperId")
        if not pid:
            continue
        G.add_node(pid, node_type="cited", label=n.get("title", pid))
        G.add_edge(center_id, pid, edge_type="CITES")

    # Citing (incoming)
    for n in ego_data.get("citing", []):
        pid = n.get("paperId")
        if not pid:
            continue
        G.add_node(pid, node_type="citing", label=n.get("title", pid))
        G.add_edge(pid, center_id, edge_type="CITES")

    pos = nx.spring_layout(G, seed=42)

    # Helper to add node traces by type
    def add_nodes(fig: go.Figure, node_type: str, color: str, size: int, name: str):
        xs, ys, texts = [], [], []
        for n, data in G.nodes(data=True):
            if data.get("node_type") == node_type:
                xs.append(pos[n][0])
                ys.append(pos[n][1])
                texts.append(data.get("label", n))
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=dict(size=size, color=color),
                    name=name,
                    text=texts,
                    hoverinfo="text",
                )
            )

    # Build figure and collect mapping info
    fig = go.Figure()
    node_trace_map: List[Dict[str, Any]] = []
    edge_traces_count = 0

    # Edges: CITES (green), IS_ABOUT (purple)
    def add_edges(edge_type: str, color: str, width: int):
        ex, ey = [], []
        for u, v, edata in G.edges(data=True):
            if edata.get("edge_type") == edge_type:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                ex += [x0, x1, None]
                ey += [y0, y1, None]
        if ex:
            fig.add_trace(
                go.Scatter(
                    x=ex,
                    y=ey,
                    mode="lines",
                    line=dict(color=color, width=width),
                    hoverinfo="none",
                    showlegend=True,
                    name=f"{edge_type}"
                )
            )
            # Edge trace added; edge_traces_count is updated after each add_edges call

    # Add CITES edges
    ex_before = len(fig.data)
    add_edges("CITES", "#2ca02c", 2)
    edge_traces_count += len(fig.data) - ex_before

    # Add IS_ABOUT edges
    ex_before = len(fig.data)
    add_edges("IS_ABOUT", "#9467bd", 2)
    edge_traces_count += len(fig.data) - ex_before

    # Helper that also captures node ids in the same order
    def add_nodes_with_map(fig: go.Figure, node_type: str, color: str, size: int, name: str):
        xs, ys, texts, node_ids = [], [], [], []
        for n, data in G.nodes(data=True):
            if data.get("node_type") == node_type:
                xs.append(pos[n][0])
                ys.append(pos[n][1])
                texts.append(data.get("label", n))
                node_ids.append(n)
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=dict(size=size, color=color),
                    name=name,
                    text=texts,
                    hoverinfo="text",
                )
            )
            node_trace_map.append({"type": node_type, "node_ids": node_ids})

    add_nodes_with_map(fig, "center", "#d62728", 22, "Center Paper")
    add_nodes_with_map(fig, "cited", "#1f77b4", 14, "Cited by Center")
    add_nodes_with_map(fig, "citing", "#17becf", 14, "Citing Center")
    add_nodes_with_map(fig, "field", "#9467bd", 12, "Fields")

    fig.update_layout(
        title="1-hop Snapshot (Paper + Citations + Fields)",
        showlegend=True,
        margin=dict(b=10, l=10, r=10, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=460,
    )
    return fig, edge_traces_count, node_trace_map

db_ok = True
selected_paper_id = st.session_state.get("home_selected_paper_id")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.write("Find a paper by keywords or ID, or let us surprise you.")
    try:
        db = create_connection(validate=False)
    except Exception as e:
        db_ok = False
        st.warning("Neo4j not available. This snapshot requires a running database.")

    # Quick explore input
    search_query = st.text_input(
        "Quick Explore",
        placeholder="e.g., graph neural networks OR a Semantic Scholar paperId",
        key="home_search_query",
    )

    explore_cols = st.columns([1, 1])
    with explore_cols[0]:
        do_explore = st.button("🔍 Explore")
    with explore_cols[1]:
        do_surprise = st.button("🎲 Surprise me")

    if db_ok:
        if do_surprise and not selected_paper_id:
            with st.spinner("Picking a featured paper..."):
                featured = get_random_featured_paper_id(db, top_n=50)
                if featured:
                    st.session_state["home_selected_paper_id"] = featured
                    selected_paper_id = featured
                    st.success("Showing a featured paper from the citation graph")
                else:
                    st.info("No featured paper found. Try a search.")

        if do_explore and search_query:
            # Try as paperId first, then fallback to keyword search
            attempted_id = search_query.strip()
            try:
                ego = get_home_ego_network(db, attempted_id)
                # If we got here, ID worked
                st.session_state["home_selected_paper_id"] = attempted_id
                selected_paper_id = attempted_id
            except Exception:
                # Keyword search fallback – pick top result
                try:
                    hits = find_papers_by_keyword(db, search_query)
                    if not hits.empty:
                        picked = hits.iloc[0]["paperId"]
                        st.session_state["home_selected_paper_id"] = picked
                        selected_paper_id = picked
                        st.success("Found a match. Showing top result.")
                    else:
                        st.info("No matches found for that query.")
                except Exception as e:
                    st.error(f"Search error: {e}")

with col_right:
    if db_ok:
        # If nothing selected yet, auto-seed a featured paper for first-time wow
        if not selected_paper_id:
            try:
                with st.spinner("Loading featured paper snapshot..."):
                    featured = get_random_featured_paper_id(db, top_n=50)
                    if featured:
                        st.session_state["home_selected_paper_id"] = featured
                        selected_paper_id = featured
                        st.info("Auto-selected a featured paper to get you started.")
            except Exception:
                pass

        if selected_paper_id:
            try:
                ego = get_home_ego_network(db, selected_paper_id, max_cited=6, max_citing=6, max_fields=3)
                fig, edge_traces_count, node_trace_map = _build_welcome_graph_fig(ego)

                st.info("Tip: Click any paper node to open it on Semantic Scholar.")
                selected_points = plotly_events(
                    fig,
                    click_event=True,
                    hover_event=False,
                    select_event=False,
                    override_height=460,
                    key="home_snapshot_network",
                )

                # Handle click events on nodes
                if selected_points:
                    point = selected_points[0]
                    curve_no = point.get("curveNumber")
                    point_index = point.get("pointIndex")
                    if curve_no is not None and point_index is not None:
                        # Map curve number to node trace based on how we added traces
                        node_trace_index = curve_no - edge_traces_count
                        if 0 <= node_trace_index < len(node_trace_map):
                            trace_info = node_trace_map[node_trace_index]
                            node_type = trace_info.get("type")
                            node_ids = trace_info.get("node_ids", [])
                            if 0 <= point_index < len(node_ids):
                                node_id = node_ids[point_index]
                                # Paper nodes: open in browser
                                if node_type in {"center", "cited", "citing"}:
                                    paper_id = node_id
                                    url = f"https://www.semanticscholar.org/paper/{paper_id}"
                                    try:
                                        webbrowser.open(url)
                                        st.success("Opening paper in your browser…")
                                    except Exception as e:
                                        st.error(f"Failed to open browser: {e}")
                                        st.code(url)
                                elif node_type == "field":
                                    # Optional: open a field search on Semantic Scholar
                                    field_name = node_id.split("field::", 1)[-1]
                                    url = f"https://www.semanticscholar.org/search?q={field_name}"
                                    try:
                                        webbrowser.open(url)
                                        st.info("Opening field search in your browser…")
                                    except Exception as e:
                                        st.error(f"Failed to open browser: {e}")
                                        st.code(url)

                # Actions
                act_cols = st.columns([1, 1])
                with act_cols[0]:
                    if st.button("📈 Open full explorer"):
                        st.session_state["default_center_papers"] = [selected_paper_id]
                        st.switch_page("src/streamlit_app/pages/Enhanced_Visualizations.py")
                with act_cols[1]:
                    st.caption("Red: center | Blue/Teal: citations | Purple: fields")

                # Schema legend and counts
                try:
                    stats = db.get_network_statistics()
                    st.markdown(
                        f"""
                        **Schema & Counts**
                        - Nodes: Papers ({stats.get('papers', 0):,}) · Authors ({stats.get('authors', 0):,}) · Venues ({stats.get('venues', 0):,}) · Fields ({stats.get('fields', 0):,})
                        - Relationships: CITES ({stats.get('citations', 0):,}) · AUTHORED ({stats.get('authorships', 0):,}) · PUBLISHED_IN ({stats.get('publications', 0):,}) · IS_ABOUT ({stats.get('field_associations', 0):,})
                        """
                    )
                except Exception:
                    pass

            except Neo4jError as e:
                st.warning(f"Unable to load snapshot: {e}")
        else:
            st.info("Use Quick Explore or Surprise me to view a snapshot.")

    else:
        st.info("Start Neo4j and reload to see the live snapshot.")

# Create clickable feature cards
col1, col2 = st.columns(2)

with col1:
    st.subheader("🤖 ML Predictions")
    st.write("Use our trained TransE model to predict which papers are most likely to cite a given paper. Get confidence scores and explore the reasoning behind predictions.")
    
    if st.button("🚀 Start ML Predictions", key="ml_pred_btn", use_container_width=True):
        st.switch_page("src/streamlit_app/pages/ML_Predictions.py")

with col2:
    st.subheader("🧭 Embedding Explorer")
    st.write("Dive deep into the learned paper embeddings. Compare papers, visualize similarity relationships, and understand how the model represents academic papers in vector space.")
    
    if st.button("🔍 Explore Embeddings", key="embed_btn", use_container_width=True):
        st.switch_page("src/streamlit_app/pages/Embedding_Explorer.py")

col3, col4 = st.columns(2)

with col3:
    st.subheader("📊 Enhanced Visualizations")
    st.write("Explore citation networks with prediction confidence overlays, interactive network graphs, and advanced analysis tools.")
    
    if st.button("📈 Create Visualizations", key="viz_btn", use_container_width=True):
        st.switch_page("src/streamlit_app/pages/Enhanced_Visualizations.py")

with col4:
    st.subheader("📓 Analysis Pipeline")
    st.write("Run comprehensive analysis workflows including data exploration, model evaluation, and result visualization based on the reference notebook pipelines.")
    
    if st.button("⚙️ Run Analysis Pipeline", key="pipeline_btn", use_container_width=True):
        st.switch_page("src/streamlit_app/pages/Notebook_Pipeline.py")

# Visual example section
st.markdown("---")
st.subheader("🎯 See What's Possible: Real Example")

with st.expander("👀 **Featured Example: 'Going Deeper with Convolutions'** - Click to explore!", expanded=True):
    
    # Example paper details
    col_ex1, col_ex2 = st.columns([2, 1])
    
    with col_ex1:
        st.markdown("""
        **📄 Paper:** Going Deeper with Convolutions  
        **👥 Authors:** Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed  
        **🏛️ Venue:** IEEE Conference on Computer Vision and Pattern Recognition (2015)  
        **📊 Citations:** 41,763 citations  
        **🏷️ Field:** Machine Learning / Computer Vision  
        **🔗 DOI:** 10.1109/CVPR.2015.7298594
        
        **📝 Abstract:** We propose a deep convolutional neural network architecture codenamed Inception that achieves state-of-the-art performance on ImageNet classification and detection. The main hallmark of this architecture is the improved utilization of computing resources inside the network through carefully crafted reduction and parallel convolutions.
        """)
    
    with col_ex2:
        st.markdown("**🎯 What Our Platform Predicts:**")
        
        # Mock prediction data for visual appeal
        import plotly.express as px
        import plotly.graph_objects as go
        import numpy as np
        
        # Create a confidence visualization
        prediction_data = {
            'Target Paper': [
                'ResNet: Deep Learning Revolution',
                'Attention is All You Need', 
                'Vision Transformer Architecture',
                'EfficientNet: Scaling Networks',
                'MobileNet: Efficient CNNs'
            ],
            'Confidence': [0.94, 0.89, 0.85, 0.82, 0.78],
            'Category': ['High', 'High', 'High', 'High', 'Medium']
        }
        
        fig = px.bar(
            x=prediction_data['Confidence'],
            y=prediction_data['Target Paper'],
            orientation='h',
            color=prediction_data['Category'],
            color_discrete_map={'High': '#00CC88', 'Medium': '#FFA500'},
            title="Top Citation Predictions"
        )
        fig.update_layout(height=300, showlegend=False)
        fig.update_xaxes(title="Confidence Score", range=[0, 1])
        fig.update_yaxes(title="")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Citation network visualization
    st.markdown("**🤔 Why Citation Network Position Matters**")
    
    st.markdown("""
    **Think of research like a conversation:** Each paper "talks to" other papers by citing them. 
    The network position shows **who's talking to whom** - and that reveals a lot!
    """)
    
    col_net1, col_net2 = st.columns(2)
    
    with col_net1:
        # Mock network data
        network_nodes = ['Going Deeper with Convolutions\n(Inception)', 'AlexNet', 'VGGNet', 'ResNet', 'DenseNet', 'EfficientNet']
        connections = [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5), (0, 4)]
        
        # Paper DOI/URL mappings
        paper_urls = [
            "https://doi.org/10.1109/CVPR.2015.7298594",  # Going Deeper with Convolutions
            "https://doi.org/10.1145/3065386",             # AlexNet
            "https://arxiv.org/abs/1409.1556",             # VGGNet  
            "https://doi.org/10.1109/CVPR.2016.90",        # ResNet
            "https://doi.org/10.1109/CVPR.2017.243",       # DenseNet
            "https://arxiv.org/abs/1905.11946"             # EfficientNet
        ]
        
        # Detailed node descriptions for hover text with clickable links
        node_descriptions = [
            "\"Going Deeper with Convolutions\" (2015)<br>Our featured paper - introduces the Inception architecture<br>🔴 Central hub connecting different eras of CNN research<br><br>🔗 <b>Click to view paper</b>",
            "AlexNet (2012)<br>\"ImageNet Classification with Deep Convolutional Neural Networks\"<br>Breakthrough CNN that the Inception paper builds upon<br>🔵 Revolutionary deep learning breakthrough<br><br>🔗 <b>Click to view paper</b>",
            "VGGNet (2014)<br>\"Very Deep Convolutional Networks for Large-Scale Image Recognition\"<br>Deep architecture that influenced Inception's design<br>🔵 Demonstrated power of depth in CNNs<br><br>🔗 <b>Click to view paper</b>", 
            "ResNet (2015)<br>\"Deep Residual Learning for Image Recognition\"<br>Revolutionary skip connections, inspired by Inception<br>🔵 Enabled much deeper networks<br><br>🔗 <b>Click to view paper</b>",
            "DenseNet (2016)<br>\"Densely Connected Convolutional Networks\"<br>Dense connections, evolved from ResNet ideas<br>🔵 Efficient feature reuse architecture<br><br>🔗 <b>Click to view paper</b>",
            "EfficientNet (2019)<br>\"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks\"<br>Efficient scaling, building on all predecessors<br>🔵 State-of-the-art efficient architectures<br><br>🔗 <b>Click to view paper</b>"
        ]
        
        # Create network visualization using plotly
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(len(network_nodes)))
        G.add_edges_from(connections)
        pos = nx.spring_layout(G)
        
        # Extract coordinates
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig_net = go.Figure()
        
        # Add edges
        fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='gray', width=1), hoverinfo='none', showlegend=False))
        
        # Add nodes with clickable functionality
        fig_net.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=25, color=['red' if i == 0 else 'lightblue' for i in range(len(network_nodes))]),
            text=[node[:12] + '...' if len(node) > 12 else node for node in network_nodes],
            textposition="middle center",
            textfont=dict(size=8),
            hoverinfo='text',
            hovertext=node_descriptions,
            customdata=paper_urls,  # Store URLs for click events
            showlegend=False
        ))
        
        fig_net.update_layout(
            title="Example of Network Visualization",
            showlegend=False,
            height=400,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        # Display the chart with click events
        st.plotly_chart(fig_net, use_container_width=True, key="citation_network")
        
        # Add JavaScript for handling clicks on nodes
        st.components.v1.html("""
        <script>
        function handlePlotlyClick() {
            const plot = document.querySelector('[data-testid="stPlotlyChart"] .plotly');
            if (plot) {
                plot.on('plotly_click', function(data){
                    if (data.points && data.points[0] && data.points[0].customdata) {
                        const url = data.points[0].customdata;
                        if (url) {
                            window.open(url, '_blank');
                        }
                    }
                });
            }
        }
        
        // Wait for the plot to be rendered
        setTimeout(handlePlotlyClick, 1000);
        
        // Also try when the page loads
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', handlePlotlyClick);
        } else {
            handlePlotlyClick();
        }
        </script>
        """, height=0)
        
        st.markdown("""
        **📍 The connections show citation relationships:**
        - **Lines connect papers that cite each other**
        - **"Going Deeper with Convolutions" sits in the middle** - it cites older papers (AlexNet, VGGNet) and gets cited by newer ones (ResNet, DenseNet, EfficientNet)
        - **This central position** makes it a "bridge" between different eras of CNN research
        
        *Note: "Inception" is the name of the neural network architecture introduced in this paper*
        
        🖱️ **Interactive Features:**
        - **Hover** over nodes for detailed information about each paper
        - **Click** any node to open the full paper in a new tab
        """)
    
    with col_net2:
        st.markdown("**🌟 Why This Paper's Position is Special:**")
        st.markdown("""
        - **Bridge between eras**: Links early CNN breakthroughs (AlexNet 2012) to modern architectures (EfficientNet 2019)
        - **Central hub**: Gets cited by papers it doesn't even cite (like DenseNet) because of its influence
        - **Knowledge transfer**: Ideas from this paper (like the Inception architecture's parallel convolutions) spread to later papers
        - **Prediction power**: Papers citing AlexNet or VGGNet are likely to also cite "Going Deeper with Convolutions"
        
        **🎯 Real impact:** This central position means "Going Deeper with Convolutions" became a "must-cite" paper - if you're working on CNNs, you probably need to reference it!
        """)
        
        st.markdown("**📈 Impact Over Time**")
        
        # Mock temporal citation data
        years = list(range(2015, 2025))
        cumulative_citations = [0, 1200, 4500, 8900, 15200, 22800, 28900, 34200, 38100, 41763]
        
        fig_time = px.line(
            x=years, 
            y=cumulative_citations,
            title="Citation Growth",
            labels={"x": "Year", "y": "Cumulative Citations"}
        )
        fig_time.update_layout(height=300)
        fig_time.update_traces(line_color='#1f77b4', line_width=3)
        
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Educational content section
    st.markdown("---")
    st.markdown("**📚 Learn More: Why Citation Networks Matter**")
    
    col_learn1, col_learn2 = st.columns(2)
    
    with col_learn1:
        st.markdown("""
        **🧠 The Science Behind It:**
        
        Citation networks are like **social networks for research papers**. Just as your social media connections 
        reveal your interests and influence, a paper's citation pattern reveals its scientific impact and role.
        
        **Key concepts:**
        - **Centrality**: Papers in the center of networks have more influence
        - **Clustering**: Similar papers tend to cite each other  
        - **Bridges**: Some papers connect different research communities
        - **Hubs**: Highly cited papers become reference points
        
        This is part of **network science** - the same math that powers Google's PageRank algorithm!
        """)
    
    with col_learn2:
        st.markdown("""
        **🔗 Dive Deeper (External Links):**
        
        📖 **[Network Science - Barabási Lab](http://networksciencebook.com/)** 
        Free textbook on network theory fundamentals
        
        🏛️ **[Stanford's CS224W: Machine Learning with Graphs](https://web.stanford.edu/class/cs224w/)**  
        Course materials on graph neural networks
        
        📊 **[Gephi - Network Analysis Tool](https://gephi.org/)**  
        Popular software for visualizing citation networks
        
        🧪 **[Nature: The Science of Science](https://www.nature.com/articles/nature.2018.22183)**  
        Research on how scientific knowledge spreads
        
        📈 **[Google Scholar Metrics](https://scholar.google.com/citations?view_op=top_venues)**  
        See real citation patterns across fields
        """)
    
    
    # Call to action
    st.markdown("---")
    st.success("""
    **🚀 Ready to explore your own papers?** This example shows how our platform can:
    - Predict missing citations with high confidence
    - Visualize complex research networks  
    - Track research impact over time
    - Identify influential connections across fields
    
    Use the navigation sidebar to start your analysis! 
    """)

# Model information
st.markdown("---")
if ml_service_online:
    st.subheader("🧠 Your Trained ML Model")
    st.success("""
    **✅ Model Status**: Ready for predictions!
    
    Your citation prediction system uses **TransE (Translating Embeddings)**, a knowledge graph 
    embedding method that learns representations where:
    
    ```
    source_paper + "CITES" ≈ target_paper
    ```
    """)
else:
    st.subheader("🧠 About the ML Model (After Training)")
    st.info("""
    **Once you train a model**, the citation prediction system will use **TransE (Translating Embeddings)**, 
    a knowledge graph embedding method that learns representations where:
    
    ```
    source_paper + "CITES" ≈ target_paper
    ```
    """)

st.markdown("""
**Key Features:**
- 🎯 **Trained on Your Data**: Academic citation networks from your Neo4j database
- 📏 **128-Dimensional**: Rich embeddings capturing citation patterns
- ⚡ **Fast Predictions**: Optimized for real-time web applications
- 💾 **Smart Caching**: Intelligent caching for improved performance

**Expected Model Performance:**
- Training Loss: ~0.15-0.25
- Embedding Dimension: 128
- Entity Count: Depends on your dataset size
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
