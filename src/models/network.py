"""
Network visualization models for the Academic Citation Platform.

This module provides data models for network visualization, supporting:
- Interactive network exploration (from knowledge-cartography)
- Multi-backend visualization (NetworkX, Pyvis, Plotly)
- Citation network analysis and display
- Node and edge attribute management

These models enable seamless integration between database queries,
ML predictions, and visualization systems.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import pandas as pd


class NodeType(str, Enum):
    """Types of nodes in the citation network."""
    PAPER = "Paper"
    AUTHOR = "Author"
    VENUE = "PubVenue"
    FIELD = "Field"
    YEAR = "PubYear"


class EdgeType(str, Enum):
    """Types of edges in the citation network."""
    CITES = "CITES"
    AUTHORED = "AUTHORED"
    PUBLISHED_IN = "PUBLISHED_IN"
    IS_ABOUT = "IS_ABOUT"
    PUB_YEAR = "PUB_YEAR"
    CO_AUTHORED = "CO_AUTHORED"  # Derived relationship


class VisualizationBackend(str, Enum):
    """Supported visualization backends."""
    NETWORKX = "networkx"
    PYVIS = "pyvis"
    PLOTLY = "plotly"
    STREAMLIT_AGRAPH = "streamlit_agraph"


class LayoutAlgorithm(str, Enum):
    """Network layout algorithms."""
    SPRING = "spring"
    KAMADA_KAWAI = "kamada_kawai"
    CIRCULAR = "circular"
    RANDOM = "random"
    FRUCHTERMAN_REINGOLD = "fruchterman_reingold"
    HIERARCHICAL = "hierarchical"


class NodeSize(str, Enum):
    """Node sizing strategies."""
    UNIFORM = "uniform"
    CITATION_COUNT = "citation_count"
    DEGREE = "degree"
    BETWEENNESS = "betweenness"
    PAGERANK = "pagerank"


class EdgeWidth(str, Enum):
    """Edge width strategies."""
    UNIFORM = "uniform"
    WEIGHT = "weight"
    CONFIDENCE = "confidence"
    CITATION_COUNT = "citation_count"


class NetworkNode(BaseModel):
    """
    Represents a node in the citation network.
    
    Supports all node types and provides flexible attribute storage
    for different visualization backends and analysis needs.
    """
    
    # Core identification
    id: str = Field(..., description="Unique node identifier")
    label: str = Field(..., description="Human-readable node label")
    node_type: NodeType = Field(..., description="Type of node")
    
    # Visual properties
    size: Optional[float] = Field(None, ge=0.0, description="Node size for visualization")
    color: Optional[str] = Field(None, description="Node color (hex, rgb, or named)")
    shape: Optional[str] = Field(None, description="Node shape for visualization")
    
    # Node-specific attributes
    title: Optional[str] = Field(None, description="Full title (for papers)")
    citation_count: Optional[int] = Field(None, ge=0, description="Number of citations")
    year: Optional[int] = Field(None, description="Publication year")
    authors: Optional[List[str]] = Field(default_factory=list, description="Author names")
    venues: Optional[List[str]] = Field(default_factory=list, description="Publication venues")
    fields: Optional[List[str]] = Field(default_factory=list, description="Research fields")
    
    # Network metrics (computed)
    degree: Optional[int] = Field(None, ge=0, description="Node degree")
    betweenness_centrality: Optional[float] = Field(None, ge=0.0, le=1.0, description="Betweenness centrality")
    closeness_centrality: Optional[float] = Field(None, ge=0.0, le=1.0, description="Closeness centrality")
    pagerank: Optional[float] = Field(None, ge=0.0, description="PageRank score")
    
    # ML-related attributes
    embedding: Optional[List[float]] = Field(None, description="Node embedding vector")
    prediction_scores: Optional[Dict[str, float]] = Field(default_factory=dict, description="Prediction scores")
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional node attributes")
    
    def get_display_label(self, max_length: int = 50) -> str:
        """Get truncated label for display."""
        if len(self.label) <= max_length:
            return self.label
        return self.label[:max_length-3] + "..."
    
    def get_tooltip_text(self) -> str:
        """Generate tooltip text for interactive visualizations."""
        tooltip_parts = [f"<b>{self.label}</b>"]
        
        if self.node_type == NodeType.PAPER:
            if self.citation_count is not None:
                tooltip_parts.append(f"Citations: {self.citation_count}")
            if self.year:
                tooltip_parts.append(f"Year: {self.year}")
            if self.authors:
                authors_str = ", ".join(self.authors[:3])
                if len(self.authors) > 3:
                    authors_str += f" (+{len(self.authors)-3} more)"
                tooltip_parts.append(f"Authors: {authors_str}")
        
        elif self.node_type == NodeType.AUTHOR:
            if self.citation_count is not None:
                tooltip_parts.append(f"Total Citations: {self.citation_count}")
        
        if self.degree is not None:
            tooltip_parts.append(f"Connections: {self.degree}")
        
        return "<br>".join(tooltip_parts)
    
    def to_networkx_data(self) -> Dict[str, Any]:
        """Convert to dictionary for NetworkX graph."""
        data = {
            'label': self.label,
            'node_type': self.node_type.value,
            'size': self.size or 10,
            'color': self.color or 'blue'
        }
        
        # Add all non-None attributes
        for field_name, field_value in self.dict(exclude={'id'}).items():
            if field_value is not None:
                data[field_name] = field_value
        
        return data
    
    def to_pyvis_data(self) -> Dict[str, Any]:
        """Convert to dictionary for Pyvis visualization."""
        return {
            'id': self.id,
            'label': self.get_display_label(30),
            'title': self.get_tooltip_text(),
            'size': self.size or 10,
            'color': self.color or 'lightblue',
            'shape': self.shape or 'dot'
        }
    
    def to_plotly_data(self) -> Dict[str, Any]:
        """Convert to dictionary for Plotly visualization."""
        return {
            'id': self.id,
            'label': self.label,
            'hover_text': self.get_tooltip_text().replace('<br>', '\n').replace('<b>', '').replace('</b>', ''),
            'size': self.size or 10,
            'color': self.color or 'lightblue'
        }


class NetworkEdge(BaseModel):
    """
    Represents an edge in the citation network.
    
    Supports different edge types and provides attributes for
    visualization and analysis across multiple backends.
    """
    
    # Core identification
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    edge_type: EdgeType = Field(..., description="Type of edge")
    
    # Visual properties
    width: Optional[float] = Field(None, ge=0.0, description="Edge width for visualization")
    color: Optional[str] = Field(None, description="Edge color")
    style: Optional[str] = Field(None, description="Edge style (solid, dashed, dotted)")
    
    # Edge attributes
    weight: Optional[float] = Field(default=1.0, ge=0.0, description="Edge weight")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Prediction confidence")
    created_at: Optional[datetime] = Field(None, description="When relationship was created")
    
    # Labels for display
    source_label: Optional[str] = Field(None, description="Human-readable source label")
    target_label: Optional[str] = Field(None, description="Human-readable target label")
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional edge attributes")
    
    def get_edge_id(self) -> str:
        """Generate unique edge identifier."""
        return f"{self.source}_{self.edge_type.value}_{self.target}"
    
    def get_tooltip_text(self) -> str:
        """Generate tooltip text for interactive visualizations."""
        source_name = self.source_label or self.source
        target_name = self.target_label or self.target
        
        tooltip_parts = [f"<b>{source_name} → {target_name}</b>"]
        tooltip_parts.append(f"Relationship: {self.edge_type.value}")
        
        if self.confidence is not None:
            tooltip_parts.append(f"Confidence: {self.confidence:.2f}")
        if self.weight != 1.0:
            tooltip_parts.append(f"Weight: {self.weight:.2f}")
        
        return "<br>".join(tooltip_parts)
    
    def to_networkx_data(self) -> Dict[str, Any]:
        """Convert to dictionary for NetworkX graph."""
        data = {
            'edge_type': self.edge_type.value,
            'weight': self.weight,
            'width': self.width or 1.0,
            'color': self.color or 'gray'
        }
        
        # Add all non-None attributes
        for field_name, field_value in self.dict(exclude={'source', 'target'}).items():
            if field_value is not None:
                data[field_name] = field_value
        
        return data
    
    def to_pyvis_data(self) -> Dict[str, Any]:
        """Convert to dictionary for Pyvis visualization."""
        return {
            'from': self.source,
            'to': self.target,
            'title': self.get_tooltip_text(),
            'width': self.width or 1.0,
            'color': self.color or 'gray'
        }
    
    def to_plotly_data(self) -> Tuple[str, str]:
        """Convert to source-target tuple for Plotly visualization."""
        return (self.source, self.target)


class NetworkGraph(BaseModel):
    """
    Complete network graph representation.
    
    Aggregates nodes and edges with metadata for visualization
    and analysis across different backends.
    """
    
    # Graph components
    nodes: List[NetworkNode] = Field(default_factory=list, description="Network nodes")
    edges: List[NetworkEdge] = Field(default_factory=list, description="Network edges")
    
    # Graph metadata
    name: Optional[str] = Field(None, description="Graph name or identifier")
    description: Optional[str] = Field(None, description="Graph description")
    created_at: datetime = Field(default_factory=datetime.now, description="Graph creation time")
    
    # Visual configuration
    layout_algorithm: LayoutAlgorithm = Field(default=LayoutAlgorithm.SPRING, description="Layout algorithm")
    node_sizing: NodeSize = Field(default=NodeSize.CITATION_COUNT, description="Node sizing strategy")
    edge_width_strategy: EdgeWidth = Field(default=EdgeWidth.UNIFORM, description="Edge width strategy")
    
    # Statistics (computed)
    num_nodes: Optional[int] = Field(None, description="Number of nodes")
    num_edges: Optional[int] = Field(None, description="Number of edges")
    density: Optional[float] = Field(None, description="Graph density")
    
    @validator('nodes')
    def update_node_count(cls, v, values):
        """Update node count when nodes are set."""
        values['num_nodes'] = len(v)
        return v
    
    @validator('edges')
    def update_edge_count(cls, v, values):
        """Update edge count when edges are set."""
        values['num_edges'] = len(v)
        # Compute density if we have nodes
        if 'num_nodes' in values and values['num_nodes'] > 1:
            max_edges = values['num_nodes'] * (values['num_nodes'] - 1)
            values['density'] = len(v) / max_edges if max_edges > 0 else 0.0
        return v
    
    def get_node_by_id(self, node_id: str) -> Optional[NetworkNode]:
        """Get node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_edges_for_node(self, node_id: str) -> List[NetworkEdge]:
        """Get all edges connected to a node."""
        return [edge for edge in self.edges 
                if edge.source == node_id or edge.target == node_id]
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighbor node IDs for a given node."""
        neighbors = set()
        for edge in self.edges:
            if edge.source == node_id:
                neighbors.add(edge.target)
            elif edge.target == node_id:
                neighbors.add(edge.source)
        return list(neighbors)
    
    def filter_by_node_type(self, node_types: List[NodeType]) -> NetworkGraph:
        """Create subgraph with only specified node types."""
        filtered_nodes = [node for node in self.nodes if node.node_type in node_types]
        node_ids = {node.id for node in filtered_nodes}
        
        filtered_edges = [edge for edge in self.edges 
                         if edge.source in node_ids and edge.target in node_ids]
        
        return NetworkGraph(
            nodes=filtered_nodes,
            edges=filtered_edges,
            name=f"{self.name}_filtered" if self.name else "filtered",
            description=f"Filtered subgraph with node types: {[t.value for t in node_types]}",
            layout_algorithm=self.layout_algorithm,
            node_sizing=self.node_sizing,
            edge_width_strategy=self.edge_width_strategy
        )
    
    def to_pandas_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert to pandas DataFrames for analysis."""
        # Nodes DataFrame
        nodes_data = []
        for node in self.nodes:
            node_dict = node.dict()
            node_dict['node_id'] = node_dict.pop('id')  # Rename for clarity
            nodes_data.append(node_dict)
        
        nodes_df = pd.DataFrame(nodes_data)
        
        # Edges DataFrame
        edges_data = []
        for edge in self.edges:
            edge_dict = edge.dict()
            edges_data.append(edge_dict)
        
        edges_df = pd.DataFrame(edges_data)
        
        return nodes_df, edges_df
    
    def to_networkx_format(self) -> Dict[str, Any]:
        """Convert to format suitable for NetworkX."""
        return {
            'nodes': [(node.id, node.to_networkx_data()) for node in self.nodes],
            'edges': [(edge.source, edge.target, edge.to_networkx_data()) for edge in self.edges]
        }
    
    def to_pyvis_format(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert to format suitable for Pyvis."""
        return {
            'nodes': [node.to_pyvis_data() for node in self.nodes],
            'edges': [edge.to_pyvis_data() for edge in self.edges]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        node_types = {}
        edge_types = {}
        
        for node in self.nodes:
            node_types[node.node_type.value] = node_types.get(node.node_type.value, 0) + 1
        
        for edge in self.edges:
            edge_types[edge.edge_type.value] = edge_types.get(edge.edge_type.value, 0) + 1
        
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'density': self.density or 0.0,
            'node_types': node_types,
            'edge_types': edge_types,
            'average_degree': (2 * len(self.edges)) / len(self.nodes) if self.nodes else 0
        }


class VisualizationConfig(BaseModel):
    """
    Configuration for network visualizations.
    
    Provides settings for customizing visualization appearance
    and behavior across different backends.
    """
    
    # Backend selection
    backend: VisualizationBackend = Field(default=VisualizationBackend.PYVIS, description="Visualization backend")
    
    # Layout settings
    layout_algorithm: LayoutAlgorithm = Field(default=LayoutAlgorithm.SPRING, description="Layout algorithm")
    layout_iterations: int = Field(default=50, ge=1, description="Number of layout iterations")
    
    # Visual settings
    width: int = Field(default=800, ge=100, description="Visualization width in pixels")
    height: int = Field(default=600, ge=100, description="Visualization height in pixels")
    background_color: str = Field(default="white", description="Background color")
    
    # Node settings
    node_sizing: NodeSize = Field(default=NodeSize.CITATION_COUNT, description="Node sizing strategy")
    min_node_size: float = Field(default=5.0, ge=1.0, description="Minimum node size")
    max_node_size: float = Field(default=50.0, ge=1.0, description="Maximum node size")
    node_color_scheme: str = Field(default="category10", description="Node color scheme")
    
    # Edge settings
    edge_width_strategy: EdgeWidth = Field(default=EdgeWidth.UNIFORM, description="Edge width strategy")
    min_edge_width: float = Field(default=1.0, ge=0.1, description="Minimum edge width")
    max_edge_width: float = Field(default=10.0, ge=0.1, description="Maximum edge width")
    show_edge_labels: bool = Field(default=False, description="Show edge labels")
    
    # Interactivity
    enable_physics: bool = Field(default=True, description="Enable physics simulation")
    enable_drag: bool = Field(default=True, description="Enable node dragging")
    enable_zoom: bool = Field(default=True, description="Enable zoom")
    enable_hover: bool = Field(default=True, description="Enable hover tooltips")
    
    # Filtering
    max_nodes: int = Field(default=500, ge=1, description="Maximum nodes to display")
    max_edges: int = Field(default=1000, ge=1, description="Maximum edges to display")
    min_citation_count: int = Field(default=0, ge=0, description="Minimum citation count for nodes")
    
    @validator('max_node_size')
    def validate_node_sizes(cls, v, values):
        """Ensure max node size >= min node size."""
        if 'min_node_size' in values and v < values['min_node_size']:
            raise ValueError("Maximum node size must be >= minimum node size")
        return v
    
    @validator('max_edge_width')
    def validate_edge_widths(cls, v, values):
        """Ensure max edge width >= min edge width."""
        if 'min_edge_width' in values and v < values['min_edge_width']:
            raise ValueError("Maximum edge width must be >= minimum edge width")
        return v


class NetworkAnalysis(BaseModel):
    """
    Results of network analysis calculations.
    
    Stores computed network metrics and statistics for display
    and further analysis.
    """
    
    # Basic statistics
    num_nodes: int = Field(..., description="Number of nodes")
    num_edges: int = Field(..., description="Number of edges")
    density: float = Field(..., description="Graph density")
    average_degree: float = Field(..., description="Average node degree")
    
    # Centrality measures
    degree_centrality: Optional[Dict[str, float]] = Field(None, description="Degree centrality scores")
    betweenness_centrality: Optional[Dict[str, float]] = Field(None, description="Betweenness centrality scores")
    closeness_centrality: Optional[Dict[str, float]] = Field(None, description="Closeness centrality scores")
    pagerank: Optional[Dict[str, float]] = Field(None, description="PageRank scores")
    
    # Community detection
    communities: Optional[Dict[str, int]] = Field(None, description="Community assignments")
    modularity: Optional[float] = Field(None, description="Modularity score")
    
    # Top nodes
    most_cited_papers: Optional[List[str]] = Field(None, description="Most cited paper IDs")
    most_connected_authors: Optional[List[str]] = Field(None, description="Most connected author IDs")
    central_nodes: Optional[List[str]] = Field(None, description="Most central node IDs")
    
    # Analysis metadata
    analyzed_at: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    analysis_duration: Optional[float] = Field(None, description="Analysis duration in seconds")
    
    def get_top_nodes_by_metric(self, metric_name: str, k: int = 10) -> List[Tuple[str, float]]:
        """Get top K nodes by specified metric."""
        metric_dict = getattr(self, metric_name, None)
        if metric_dict is None:
            return []
        
        sorted_nodes = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:k]
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary of key network statistics."""
        summary = {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'density': round(self.density, 4),
            'average_degree': round(self.average_degree, 2)
        }
        
        if self.modularity is not None:
            summary['modularity'] = round(self.modularity, 4)
        
        if self.communities:
            summary['num_communities'] = len(set(self.communities.values()))
        
        return summary