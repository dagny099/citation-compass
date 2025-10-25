"""
Home page for the Academic Citation Platform.

Streamlined landing page focused on exploring an existing graph snippet
and quick navigation.
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
st.caption("Explore my existing Neo4j knowledge graph. Use Search or Surprise me, then dive deeper in Enhanced Visualizations.")

st.markdown("---")
st.subheader("🧭 My Knowledge Graph at a Glance")

# Above-the-fold stats + schema
db = None
db_ok = True
stats = {}
try:
    db = create_connection(validate=False)
    stats = db.get_network_statistics()
except Exception:
    db_ok = False

top_left, top_right = st.columns([2, 1])
with top_left:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("📄 Papers", f"{stats.get('papers', 0):,}")
    with c2:
        st.metric("👤 Authors", f"{stats.get('authors', 0):,}")
    with c3:
        st.metric("🏛️ Venues", f"{stats.get('venues', 0):,}")
    with c4:
        st.metric("🏷️ Fields", f"{stats.get('fields', 0):,}")
    with c5:
        st.metric("🔗 Citations", f"{stats.get('citations', 0):,}")

    st.caption("Legend: Red = center paper • Blue = references (outgoing) • Teal = papers that cite the center (incoming) • Purple = fields (topics). Edges: green = cites, purple = is about.")

with top_right:
    show_schema = st.checkbox("Show schema diagram", value=True)
    if show_schema:
        try:
            st.image("docs/assets/diagrams/database-schema.png", caption="Schema overview", use_column_width=True)
        except Exception:
            st.info("Schema diagram not found. Add it at docs/assets/diagrams/database-schema.png")

st.markdown("---")
st.subheader("🚀 Explore an Existing Graph Snippet")


def _build_welcome_graph_fig(ego_data: Dict[str, Any]):
    G = nx.DiGraph()

    center = ego_data.get("center") or {}
    center_id = center.get("paperId") or "center"
    G.add_node(center_id, node_type="center", label=center.get("title", center_id))

    for field in ego_data.get("fields", []):
        fid = f"field::{field}"
        G.add_node(fid, node_type="field", label=field)
        G.add_edge(center_id, fid, edge_type="IS_ABOUT")

    for n in ego_data.get("cited", []):
        pid = n.get("paperId")
        if not pid:
            continue
        G.add_node(pid, node_type="cited", label=n.get("title", pid))
        G.add_edge(center_id, pid, edge_type="CITES")

    for n in ego_data.get("citing", []):
        pid = n.get("paperId")
        if not pid:
            continue
        G.add_node(pid, node_type="citing", label=n.get("title", pid))
        G.add_edge(pid, center_id, edge_type="CITES")

    pos = nx.spring_layout(G, seed=42)

    fig = go.Figure()
    node_trace_map: List[Dict[str, Any]] = []
    edge_traces_count = 0

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
                go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=color, width=2), hoverinfo="none", name=edge_type)
            )

    before = len(fig.data)
    add_edges("CITES", "#2ca02c", 2)
    edge_traces_count += len(fig.data) - before
    before = len(fig.data)
    add_edges("IS_ABOUT", "#9467bd", 2)
    edge_traces_count += len(fig.data) - before

    def add_nodes_with_map(node_type: str, color: str, size: int, name: str):
        xs, ys, texts, node_ids = [], [], [], []
        for n, data in G.nodes(data=True):
            if data.get("node_type") == node_type:
                xs.append(pos[n][0]); ys.append(pos[n][1])
                texts.append(data.get("label", n)); node_ids.append(n)
        if xs:
            fig.add_trace(
                go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=size, color=color), name=name, text=texts, hoverinfo="text")
            )
            node_trace_map.append({"type": node_type, "node_ids": node_ids})

    add_nodes_with_map("center", "#d62728", 22, "Center paper")
    add_nodes_with_map("cited", "#1f77b4", 14, "References (cited by center)")
    add_nodes_with_map("citing", "#17becf", 14, "Cites center (incoming)")
    add_nodes_with_map("field", "#9467bd", 12, "Fields")

    fig.update_layout(
        title="1-hop Snapshot (Paper + Citations + Fields)",
        showlegend=True,
        margin=dict(b=10, l=10, r=10, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=460,
    )
    return fig, edge_traces_count, node_trace_map


selected_paper_id = st.session_state.get("home_selected_paper_id")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.write("Find a paper by keywords or ID, or let me surprise you.")
    if not db_ok:
        st.warning("Neo4j not available. This snapshot requires a running database.")

    search_query = st.text_input(
        "Quick Explore",
        placeholder="e.g., graph neural networks OR a Semantic Scholar paperId",
        key="home_search_query",
    )
    b1, b2 = st.columns([1, 1])
    with b1:
        do_explore = st.button("🔍 Explore")
    with b2:
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
            attempted_id = search_query.strip()
            try:
                _ = get_home_ego_network(db, attempted_id)
                st.session_state["home_selected_paper_id"] = attempted_id
                selected_paper_id = attempted_id
            except Exception:
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

                if selected_points:
                    point = selected_points[0]
                    curve_no = point.get("curveNumber")
                    point_index = point.get("pointIndex")
                    if curve_no is not None and point_index is not None:
                        node_trace_index = curve_no - edge_traces_count
                        if 0 <= node_trace_index < len(node_trace_map):
                            trace_info = node_trace_map[node_trace_index]
                            node_type = trace_info.get("type")
                            node_ids = trace_info.get("node_ids", [])
                            if 0 <= point_index < len(node_ids):
                                node_id = node_ids[point_index]
                                if node_type in {"center", "cited", "citing"}:
                                    paper_id = node_id
                                    url = f"https://www.semanticscholar.org/paper/{paper_id}"
                                    try:
                                        webbrowser.open(url)
                                        st.success("Opening paper in a new tab…")
                                    except Exception as e:
                                        st.error(f"Failed to open browser: {e}")
                                        st.code(url)
                                elif node_type == "field":
                                    field_name = node_id.split("field::", 1)[-1]
                                    url = f"https://www.semanticscholar.org/search?q={field_name}"
                                    try:
                                        webbrowser.open(url)
                                        st.info("Opening field search in a new tab…")
                                    except Exception as e:
                                        st.error(f"Failed to open browser: {e}")
                                        st.code(url)

                # plotly_events renders the chart; no second render needed

                act_cols = st.columns([1, 1])
                with act_cols[0]:
                    if st.button("📈 Open full explorer"):
                        st.session_state["default_center_papers"] = [selected_paper_id]
                        st.switch_page("src/streamlit_app/pages/Enhanced_Visualizations.py")
                with act_cols[1]:
                    st.caption("Legend: Red = center paper • Blue = references (outgoing) • Teal = papers that cite the center (incoming) • Purple = fields (topics). Edges: green = cites, purple = is about.")

                # (Schema & Counts now lives in the top section)

            except Neo4jError as e:
                st.warning(f"Unable to load snapshot: {e}")
        else:
            st.info("Use Quick Explore or Surprise me to view a snapshot.")
    else:
        st.info("Start Neo4j and reload to see the live snapshot.")

# Quick Links
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
