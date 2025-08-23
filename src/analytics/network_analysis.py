"""
Network analysis module for citation graph analysis.

Provides comprehensive network analysis capabilities including centrality measures,
community detection, path analysis, and structural properties analysis.
"""

import logging
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

from ..models.network import NetworkNode, NetworkEdge, NetworkAnalysis
from ..data.unified_database import UnifiedDatabaseManager


@dataclass
class CentralityMetrics:
    """Container for centrality metrics of a node."""
    paper_id: str
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    eigenvector_centrality: float
    pagerank: float


@dataclass
class CommunityInfo:
    """Information about a detected community."""
    community_id: int
    size: int
    members: List[str]
    modularity: float
    internal_edges: int
    external_edges: int
    conductance: float


@dataclass
class NetworkMetrics:
    """Comprehensive network-wide metrics."""
    num_nodes: int
    num_edges: int
    density: float
    average_degree: float
    clustering_coefficient: float
    diameter: Optional[int]
    average_path_length: Optional[float]
    num_components: int
    largest_component_size: int
    modularity: float
    assortativity: Optional[float]


class NetworkAnalyzer:
    """
    Advanced network analyzer for citation graphs.
    
    Provides comprehensive analysis capabilities including centrality measures,
    structural properties, and network-wide statistics.
    """
    
    def __init__(self, database: Optional[UnifiedDatabaseManager] = None):
        """
        Initialize network analyzer.
        
        Args:
            database: Database connection for loading network data
        """
        self.database = database
        self.logger = logging.getLogger(__name__)
        self._graph_cache: Dict[str, nx.Graph] = {}
    
    def load_citation_network(self, 
                            paper_ids: Optional[List[str]] = None,
                            max_papers: Optional[int] = None) -> nx.DiGraph:
        """
        Load citation network from database.
        
        Args:
            paper_ids: Optional list of specific papers to include
            max_papers: Maximum number of papers to load
            
        Returns:
            NetworkX directed graph representing citation network
        """
        if self.database is None:
            raise ValueError("Database connection required for loading network")
        
        self.logger.info(f"Loading citation network (max_papers={max_papers})")
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Mock implementation - in real scenario, query database
        # For now, create a sample network for testing
        if paper_ids is None:
            paper_ids = [f"paper_{i}" for i in range(min(1000, max_papers or 1000))]
        
        # Add nodes
        for paper_id in paper_ids:
            G.add_node(paper_id, type="paper")
        
        # Add sample edges (mock citation relationships)
        import random
        random.seed(42)
        num_edges = min(len(paper_ids) * 2, 2000)
        
        for _ in range(num_edges):
            source = random.choice(paper_ids)
            target = random.choice(paper_ids)
            if source != target and not G.has_edge(source, target):
                G.add_edge(source, target, relation="cites")
        
        self.logger.info(f"Loaded network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def calculate_centrality_metrics(self, graph: nx.Graph) -> Dict[str, CentralityMetrics]:
        """
        Calculate comprehensive centrality metrics for all nodes.
        
        Args:
            graph: NetworkX graph to analyze
            
        Returns:
            Dictionary mapping node IDs to centrality metrics
        """
        self.logger.info("Calculating centrality metrics")
        
        # Convert to undirected for some metrics
        undirected = graph.to_undirected() if graph.is_directed() else graph
        
        # Calculate various centrality measures
        degree_cent = nx.degree_centrality(undirected)
        betweenness_cent = nx.betweenness_centrality(undirected, k=min(100, len(graph.nodes())))
        closeness_cent = nx.closeness_centrality(undirected)
        
        # Handle disconnected graphs for eigenvector centrality
        try:
            eigenvector_cent = nx.eigenvector_centrality(undirected, max_iter=1000)
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            self.logger.warning("Eigenvector centrality failed, using degree centrality as fallback")
            eigenvector_cent = {node: 0.0 for node in graph.nodes()}
        
        # PageRank (works with directed graphs)
        pagerank = nx.pagerank(graph, max_iter=1000)
        
        # Create centrality metrics objects
        metrics = {}
        for node in graph.nodes():
            metrics[node] = CentralityMetrics(
                paper_id=node,
                degree_centrality=degree_cent.get(node, 0.0),
                betweenness_centrality=betweenness_cent.get(node, 0.0),
                closeness_centrality=closeness_cent.get(node, 0.0),
                eigenvector_centrality=eigenvector_cent.get(node, 0.0),
                pagerank=pagerank.get(node, 0.0)
            )
        
        self.logger.info(f"Calculated centrality metrics for {len(metrics)} nodes")
        return metrics
    
    def analyze_network_structure(self, graph: nx.Graph) -> NetworkMetrics:
        """
        Analyze overall network structure and properties.
        
        Args:
            graph: NetworkX graph to analyze
            
        Returns:
            NetworkMetrics object with comprehensive statistics
        """
        self.logger.info("Analyzing network structure")
        
        # Basic metrics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        if num_nodes <= 1:
            return NetworkMetrics(
                num_nodes=num_nodes,
                num_edges=num_edges,
                density=0.0,
                average_degree=0.0,
                clustering_coefficient=0.0,
                diameter=None,
                average_path_length=None,
                num_components=num_nodes,
                largest_component_size=1 if num_nodes == 1 else 0,
                modularity=0.0,
                assortativity=None
            )
        
        # Convert to undirected for structural analysis
        undirected = graph.to_undirected() if graph.is_directed() else graph
        
        # Density and degree statistics
        density = nx.density(undirected)
        degrees = [d for n, d in undirected.degree()]
        average_degree = sum(degrees) / len(degrees) if degrees else 0.0
        
        # Clustering coefficient
        clustering_coeff = nx.average_clustering(undirected)
        
        # Connected components
        components = list(nx.connected_components(undirected))
        num_components = len(components)
        largest_component_size = len(max(components, key=len)) if components else 0
        
        # Diameter and average path length (only for largest component if disconnected)
        diameter = None
        avg_path_length = None
        
        if num_components == 1 and num_nodes > 1:
            try:
                diameter = nx.diameter(undirected)
                avg_path_length = nx.average_shortest_path_length(undirected)
            except nx.NetworkXError:
                pass
        elif num_components > 1 and largest_component_size > 1:
            # Analyze largest component
            largest_component = max(components, key=len)
            largest_subgraph = undirected.subgraph(largest_component)
            try:
                diameter = nx.diameter(largest_subgraph)
                avg_path_length = nx.average_shortest_path_length(largest_subgraph)
            except nx.NetworkXError:
                pass
        
        # Modularity (using simple community detection)
        modularity = 0.0
        try:
            communities = nx.community.greedy_modularity_communities(undirected)
            modularity = nx.community.modularity(undirected, communities)
        except:
            pass
        
        # Assortativity
        assortativity = None
        try:
            assortativity = nx.degree_assortativity_coefficient(undirected)
        except:
            pass
        
        return NetworkMetrics(
            num_nodes=num_nodes,
            num_edges=num_edges,
            density=density,
            average_degree=average_degree,
            clustering_coefficient=clustering_coeff,
            diameter=diameter,
            average_path_length=avg_path_length,
            num_components=num_components,
            largest_component_size=largest_component_size,
            modularity=modularity,
            assortativity=assortativity
        )
    
    def find_shortest_paths(self, 
                           graph: nx.Graph, 
                           source: str, 
                           targets: List[str],
                           max_path_length: int = 6) -> Dict[str, List[str]]:
        """
        Find shortest paths from source to multiple targets.
        
        Args:
            graph: NetworkX graph
            source: Source node ID
            targets: List of target node IDs
            max_path_length: Maximum path length to consider
            
        Returns:
            Dictionary mapping target nodes to shortest path (as list of nodes)
        """
        self.logger.info(f"Finding paths from {source} to {len(targets)} targets")
        
        paths = {}
        
        try:
            # Use single-source shortest path algorithm
            if graph.is_directed():
                path_lengths = nx.single_source_shortest_path_length(
                    graph, source, cutoff=max_path_length
                )
                shortest_paths = nx.single_source_shortest_path(
                    graph, source, cutoff=max_path_length
                )
            else:
                path_lengths = nx.single_source_shortest_path_length(
                    graph, source, cutoff=max_path_length
                )
                shortest_paths = nx.single_source_shortest_path(
                    graph, source, cutoff=max_path_length
                )
            
            # Extract paths to requested targets
            for target in targets:
                if target in shortest_paths:
                    paths[target] = shortest_paths[target]
                    
        except nx.NetworkXError as e:
            self.logger.warning(f"Error finding paths: {e}")
        
        return paths
    
    def identify_influential_papers(self, 
                                  centrality_metrics: Dict[str, CentralityMetrics],
                                  top_k: int = 20) -> Dict[str, List[str]]:
        """
        Identify most influential papers by different centrality measures.
        
        Args:
            centrality_metrics: Dictionary of centrality metrics
            top_k: Number of top papers to return for each metric
            
        Returns:
            Dictionary mapping metric names to lists of top paper IDs
        """
        self.logger.info(f"Identifying top {top_k} influential papers")
        
        influential = {}
        
        # Sort by each centrality measure
        metrics_to_sort = [
            ('degree_centrality', lambda x: x.degree_centrality),
            ('betweenness_centrality', lambda x: x.betweenness_centrality),
            ('closeness_centrality', lambda x: x.closeness_centrality),
            ('eigenvector_centrality', lambda x: x.eigenvector_centrality),
            ('pagerank', lambda x: x.pagerank)
        ]
        
        for metric_name, key_func in metrics_to_sort:
            sorted_papers = sorted(
                centrality_metrics.values(),
                key=key_func,
                reverse=True
            )
            influential[metric_name] = [paper.paper_id for paper in sorted_papers[:top_k]]
        
        return influential


class CommunityDetector:
    """
    Community detection algorithms for citation networks.
    
    Implements multiple community detection methods to identify clusters
    of related papers in citation networks.
    """
    
    def __init__(self):
        """Initialize community detector."""
        self.logger = logging.getLogger(__name__)
    
    def detect_communities_louvain(self, graph: nx.Graph) -> List[CommunityInfo]:
        """
        Detect communities using the Louvain method.
        
        Args:
            graph: NetworkX graph to analyze
            
        Returns:
            List of CommunityInfo objects
        """
        self.logger.info("Detecting communities using Louvain method")
        
        # Convert to undirected
        undirected = graph.to_undirected() if graph.is_directed() else graph
        
        try:
            # Use NetworkX community detection
            communities = nx.community.louvain_communities(undirected)
            modularity = nx.community.modularity(undirected, communities)
            
            community_info = []
            
            for i, community in enumerate(communities):
                members = list(community)
                subgraph = undirected.subgraph(members)
                
                # Calculate community metrics
                internal_edges = subgraph.number_of_edges()
                external_edges = 0
                
                # Count edges going out of the community
                for node in members:
                    for neighbor in undirected.neighbors(node):
                        if neighbor not in community:
                            external_edges += 1
                
                # Conductance: external edges / (internal + external)
                total_edges = internal_edges + external_edges
                conductance = external_edges / total_edges if total_edges > 0 else 0.0
                
                community_info.append(CommunityInfo(
                    community_id=i,
                    size=len(members),
                    members=members,
                    modularity=modularity,  # Global modularity
                    internal_edges=internal_edges,
                    external_edges=external_edges,
                    conductance=conductance
                ))
            
            self.logger.info(f"Detected {len(community_info)} communities")
            return community_info
            
        except Exception as e:
            self.logger.error(f"Community detection failed: {e}")
            return []
    
    def detect_communities_greedy_modularity(self, graph: nx.Graph) -> List[CommunityInfo]:
        """
        Detect communities using greedy modularity optimization.
        
        Args:
            graph: NetworkX graph to analyze
            
        Returns:
            List of CommunityInfo objects
        """
        self.logger.info("Detecting communities using greedy modularity")
        
        # Convert to undirected
        undirected = graph.to_undirected() if graph.is_directed() else graph
        
        try:
            communities = nx.community.greedy_modularity_communities(undirected)
            modularity = nx.community.modularity(undirected, communities)
            
            community_info = []
            
            for i, community in enumerate(communities):
                members = list(community)
                subgraph = undirected.subgraph(members)
                
                # Calculate community metrics
                internal_edges = subgraph.number_of_edges()
                external_edges = sum(
                    1 for node in members 
                    for neighbor in undirected.neighbors(node)
                    if neighbor not in community
                )
                
                total_edges = internal_edges + external_edges
                conductance = external_edges / total_edges if total_edges > 0 else 0.0
                
                community_info.append(CommunityInfo(
                    community_id=i,
                    size=len(members),
                    members=members,
                    modularity=modularity,
                    internal_edges=internal_edges,
                    external_edges=external_edges,
                    conductance=conductance
                ))
            
            self.logger.info(f"Detected {len(community_info)} communities")
            return community_info
            
        except Exception as e:
            self.logger.error(f"Community detection failed: {e}")
            return []
    
    def analyze_community_structure(self, 
                                  communities: List[CommunityInfo],
                                  graph: nx.Graph) -> Dict[str, Any]:
        """
        Analyze overall community structure properties.
        
        Args:
            communities: List of detected communities
            graph: Original graph
            
        Returns:
            Dictionary with community structure analysis
        """
        if not communities:
            return {}
        
        total_nodes = graph.number_of_nodes()
        community_sizes = [c.size for c in communities]
        
        analysis = {
            'num_communities': len(communities),
            'total_nodes': total_nodes,
            'coverage': sum(community_sizes) / total_nodes,
            'largest_community_size': max(community_sizes),
            'smallest_community_size': min(community_sizes),
            'average_community_size': np.mean(community_sizes),
            'community_size_std': np.std(community_sizes),
            'modularity': communities[0].modularity if communities else 0.0,
            'average_conductance': np.mean([c.conductance for c in communities])
        }
        
        return analysis