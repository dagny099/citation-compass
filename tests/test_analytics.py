"""
Tests for the analytics module components.

Comprehensive test suite covering network analysis, temporal analysis,
performance metrics, and export capabilities.
"""

import pytest
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

from src.analytics.network_analysis import (
    NetworkAnalyzer, 
    CommunityDetector, 
    CentralityMetrics, 
    CommunityInfo,
    NetworkMetrics
)
from src.analytics.temporal_analysis import (
    TemporalAnalyzer, 
    TrendAnalyzer, 
    CitationGrowthMetrics,
    TrendAnalysis,
    TimeSeriesPoint
)
from src.analytics.performance_metrics import (
    PerformanceAnalyzer,
    SystemHealthMonitor,
    BenchmarkResult,
    HealthStatus
)
from src.analytics.export_engine import (
    ExportEngine,
    ReportGenerator,
    ExportConfiguration,
    ExportResult
)
from src.services.analytics_service import AnalyticsService
from src.models.paper import Paper
from src.models.citation import Citation


class TestNetworkAnalyzer:
    """Test suite for NetworkAnalyzer component."""
    
    @pytest.fixture
    def network_analyzer(self):
        """Create NetworkAnalyzer instance for testing."""
        return NetworkAnalyzer()
    
    @pytest.fixture
    def sample_graph(self):
        """Create sample graph for testing."""
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(10):
            G.add_node(f"paper_{i}", type="paper")
        
        # Add edges (citations)
        edges = [
            ("paper_0", "paper_1"), ("paper_0", "paper_2"),
            ("paper_1", "paper_3"), ("paper_1", "paper_4"),
            ("paper_2", "paper_4"), ("paper_2", "paper_5"),
            ("paper_3", "paper_6"), ("paper_4", "paper_7"),
            ("paper_5", "paper_8"), ("paper_6", "paper_9")
        ]
        
        for source, target in edges:
            G.add_edge(source, target, relation="cites")
        
        return G
    
    @patch('src.analytics.network_analysis.UnifiedDatabaseManager')
    def test_load_citation_network(self, mock_db_manager, network_analyzer):
        """Test citation network loading."""
        # Mock database
        mock_db = Mock()
        network_analyzer.database = mock_db
        
        # Mock query results
        mock_db.query_citation_network.return_value = pd.DataFrame({
            'source_id': ['paper_1', 'paper_2'],
            'target_id': ['paper_2', 'paper_3'],
            'source_title': ['Title 1', 'Title 2'],
            'target_title': ['Title 2', 'Title 3']
        })
        
        graph = network_analyzer.load_citation_network(
            paper_ids=["paper_1", "paper_2", "paper_3"],
            max_papers=100
        )
        
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() >= 0
        assert graph.number_of_edges() >= 0
    
    def test_calculate_centrality_metrics(self, network_analyzer, sample_graph):
        """Test centrality metrics calculation."""
        centrality_metrics = network_analyzer.calculate_centrality_metrics(sample_graph)
        
        assert len(centrality_metrics) == sample_graph.number_of_nodes()
        
        # Check that all nodes have metrics
        for node in sample_graph.nodes():
            assert node in centrality_metrics
            metrics = centrality_metrics[node]
            
            assert isinstance(metrics, CentralityMetrics)
            assert hasattr(metrics, 'degree_centrality')
            assert hasattr(metrics, 'betweenness_centrality')
            assert hasattr(metrics, 'pagerank')
            
            # Metrics should be non-negative
            assert metrics.degree_centrality >= 0
            assert metrics.betweenness_centrality >= 0
            assert metrics.pagerank >= 0
    
    def test_analyze_network_structure(self, network_analyzer, sample_graph):
        """Test network structure analysis."""
        network_metrics = network_analyzer.analyze_network_structure(sample_graph)
        
        assert isinstance(network_metrics, NetworkMetrics)
        assert network_metrics.num_nodes == sample_graph.number_of_nodes()
        assert network_metrics.num_edges == sample_graph.number_of_edges()
        assert 0 <= network_metrics.density <= 1
        assert network_metrics.average_degree >= 0
        assert network_metrics.clustering_coefficient >= 0
        assert network_metrics.num_components >= 1
        assert network_metrics.largest_component_size >= 1
    
    def test_find_shortest_paths(self, network_analyzer, sample_graph):
        """Test shortest path finding."""
        source = "paper_0"
        targets = ["paper_3", "paper_5", "paper_9"]
        
        paths = network_analyzer.find_shortest_paths(
            sample_graph, source, targets, max_path_length=5
        )
        
        assert isinstance(paths, dict)
        
        # Check that paths are valid
        for target, path in paths.items():
            assert isinstance(path, list)
            assert path[0] == source
            assert path[-1] == target
    
    def test_identify_influential_papers(self, network_analyzer, sample_graph):
        """Test influential papers identification."""
        centrality_metrics = network_analyzer.calculate_centrality_metrics(sample_graph)
        influential = network_analyzer.identify_influential_papers(
            centrality_metrics, top_k=5
        )
        
        assert isinstance(influential, dict)
        assert 'degree_centrality' in influential
        assert 'pagerank' in influential
        
        # Check that we get the requested number of papers
        for metric_name, papers in influential.items():
            assert len(papers) <= 5
            assert all(paper in sample_graph.nodes() for paper in papers)


class TestCommunityDetector:
    """Test suite for CommunityDetector component."""
    
    @pytest.fixture
    def community_detector(self):
        """Create CommunityDetector instance for testing."""
        return CommunityDetector()
    
    @pytest.fixture
    def sample_community_graph(self):
        """Create graph with clear community structure."""
        G = nx.Graph()
        
        # Community 1
        for i in range(5):
            for j in range(i+1, 5):
                G.add_edge(f"c1_node_{i}", f"c1_node_{j}")
        
        # Community 2
        for i in range(5):
            for j in range(i+1, 5):
                G.add_edge(f"c2_node_{i}", f"c2_node_{j}")
        
        # Few inter-community edges
        G.add_edge("c1_node_0", "c2_node_0")
        G.add_edge("c1_node_1", "c2_node_1")
        
        return G
    
    def test_detect_communities_louvain(self, community_detector, sample_community_graph):
        """Test Louvain community detection."""
        communities = community_detector.detect_communities_louvain(sample_community_graph)
        
        assert isinstance(communities, list)
        assert len(communities) > 0
        
        for community in communities:
            assert isinstance(community, CommunityInfo)
            assert community.size > 0
            assert len(community.members) == community.size
            assert community.community_id >= 0
            assert community.internal_edges >= 0
            assert 0 <= community.conductance <= 1
    
    def test_detect_communities_greedy_modularity(self, community_detector, sample_community_graph):
        """Test greedy modularity community detection."""
        communities = community_detector.detect_communities_greedy_modularity(sample_community_graph)
        
        assert isinstance(communities, list)
        assert len(communities) > 0
        
        for community in communities:
            assert isinstance(community, CommunityInfo)
            assert community.size > 0
    
    def test_analyze_community_structure(self, community_detector, sample_community_graph):
        """Test community structure analysis."""
        communities = community_detector.detect_communities_louvain(sample_community_graph)
        analysis = community_detector.analyze_community_structure(
            communities, sample_community_graph
        )
        
        assert isinstance(analysis, dict)
        assert 'num_communities' in analysis
        assert 'total_nodes' in analysis
        assert 'coverage' in analysis
        assert 'modularity' in analysis
        
        assert analysis['num_communities'] == len(communities)
        assert 0 <= analysis['coverage'] <= 1


class TestTemporalAnalyzer:
    """Test suite for TemporalAnalyzer component."""
    
    @pytest.fixture
    def temporal_analyzer(self):
        """Create TemporalAnalyzer instance for testing."""
        return TemporalAnalyzer()
    
    @pytest.fixture
    def sample_papers(self):
        """Create sample papers for testing."""
        papers = []
        for i in range(10):
            paper = Paper(
                paper_id=f"paper_{i}",
                title=f"Test Paper {i}",
                abstract=f"Abstract for paper {i}",
                publication_year=2020 + (i % 4),  # 2020-2023
                authors=[f"Author_{i}"],
                venue=f"Venue_{i % 3}"
            )
            papers.append(paper)
        
        return papers
    
    @pytest.fixture
    def sample_citations(self):
        """Create sample citations for testing."""
        citations = []
        base_date = datetime(2020, 1, 1)
        
        for i in range(50):
            citation_date = base_date + timedelta(days=i*30)  # Monthly citations
            citation = Citation(
                citation_id=f"citation_{i}",
                source_paper_id=f"source_{i}",
                target_paper_id=f"paper_{i % 10}",
                citation_date=citation_date
            )
            citations.append(citation)
        
        return citations
    
    def test_analyze_citation_growth(self, temporal_analyzer, sample_papers, sample_citations):
        """Test citation growth analysis."""
        growth_metrics = temporal_analyzer.analyze_citation_growth(
            sample_papers, sample_citations
        )
        
        assert isinstance(growth_metrics, list)
        assert len(growth_metrics) == len(sample_papers)
        
        for metrics in growth_metrics:
            assert isinstance(metrics, CitationGrowthMetrics)
            assert metrics.total_citations >= 0
            assert metrics.publication_year >= 2020
            assert metrics.years_to_peak >= 0
            assert metrics.impact_factor >= 0
    
    def test_create_citation_time_series(self, temporal_analyzer, sample_citations):
        """Test time series creation."""
        time_series = temporal_analyzer.create_citation_time_series(sample_citations, freq='M')
        
        assert isinstance(time_series, list)
        assert len(time_series) > 0
        
        for point in time_series:
            assert isinstance(point, TimeSeriesPoint)
            assert isinstance(point.timestamp, datetime)
            assert point.value >= 0
    
    def test_analyze_seasonal_patterns(self, temporal_analyzer, sample_citations):
        """Test seasonal pattern analysis."""
        time_series = temporal_analyzer.create_citation_time_series(sample_citations)
        seasonal_analysis = temporal_analyzer.analyze_seasonal_patterns(time_series)
        
        assert isinstance(seasonal_analysis, dict)
        
        # Should have analysis results
        if 'error' not in seasonal_analysis:
            assert 'seasonal_variation_coefficient' in seasonal_analysis
            assert 'has_strong_seasonality' in seasonal_analysis
            assert seasonal_analysis['seasonal_variation_coefficient'] >= 0
    
    def test_detect_citation_bursts(self, temporal_analyzer, sample_citations):
        """Test citation burst detection."""
        time_series = temporal_analyzer.create_citation_time_series(sample_citations)
        bursts = temporal_analyzer.detect_citation_bursts(time_series)
        
        assert isinstance(bursts, list)
        
        for burst in bursts:
            assert isinstance(burst, dict)
            assert 'start_time' in burst
            assert 'end_time' in burst
            assert 'intensity' in burst
            assert burst['intensity'] > 1.0  # Should be above baseline


class TestTrendAnalyzer:
    """Test suite for TrendAnalyzer component."""
    
    @pytest.fixture
    def trend_analyzer(self):
        """Create TrendAnalyzer instance for testing."""
        return TrendAnalyzer()
    
    @pytest.fixture
    def upward_trend_data(self):
        """Create time series with upward trend."""
        timestamps = [datetime(2020, 1, 1) + timedelta(days=i*30) for i in range(12)]
        values = [10 + i*2 + np.random.normal(0, 1) for i in range(12)]  # Upward trend
        
        return [TimeSeriesPoint(ts, max(0, val)) for ts, val in zip(timestamps, values)]
    
    def test_analyze_trend(self, trend_analyzer, upward_trend_data):
        """Test trend analysis."""
        trend_analysis = trend_analyzer.analyze_trend(upward_trend_data)
        
        assert isinstance(trend_analysis, TrendAnalysis)
        assert trend_analysis.trend_direction in ['increasing', 'decreasing', 'stable']
        assert 0 <= trend_analysis.trend_strength <= 1
        assert isinstance(trend_analysis.growth_rate, float)
    
    def test_compare_growth_rates(self, trend_analyzer):
        """Test growth rate comparison."""
        # Create sample growth metrics
        growth_metrics = []
        for i in range(10):
            metrics = CitationGrowthMetrics(
                paper_id=f"paper_{i}",
                publication_year=2020,
                total_citations=i*5 + 10,
                citations_per_year={2020: i*2, 2021: i*3},
                peak_citation_year=2021,
                years_to_peak=1,
                half_life=2.0,
                impact_factor=i*0.5 + 1.0
            )
            growth_metrics.append(metrics)
        
        comparison = trend_analyzer.compare_growth_rates(growth_metrics)
        
        assert isinstance(comparison, dict)
        assert 'total_papers' in comparison
        assert 'impact_factor_stats' in comparison
        assert 'top_impact_papers' in comparison
        
        assert comparison['total_papers'] == len(growth_metrics)
    
    def test_project_future_citations(self, trend_analyzer):
        """Test future citation projection."""
        # Create sample growth metrics with citation history
        growth_metrics = CitationGrowthMetrics(
            paper_id="test_paper",
            publication_year=2020,
            total_citations=50,
            citations_per_year={2020: 10, 2021: 15, 2022: 20, 2023: 5},
            peak_citation_year=2022,
            years_to_peak=2,
            half_life=3.0,
            impact_factor=12.5
        )
        
        projections = trend_analyzer.project_future_citations(growth_metrics, years_ahead=3)
        
        assert isinstance(projections, dict)
        assert len(projections) == 3
        
        for year, projection in projections.items():
            assert year > 2023
            assert projection >= 0


class TestPerformanceAnalyzer:
    """Test suite for PerformanceAnalyzer component."""
    
    @pytest.fixture
    def performance_analyzer(self):
        """Create PerformanceAnalyzer instance for testing."""
        mock_ml_service = Mock()
        return PerformanceAnalyzer(mock_ml_service)
    
    def test_benchmark_ml_predictions(self, performance_analyzer):
        """Test ML prediction benchmarking."""
        # Mock ML service predictions
        performance_analyzer.ml_service.predict_citations.return_value = [
            Mock(prediction_score=0.8),
            Mock(prediction_score=0.6)
        ]
        
        result = performance_analyzer.benchmark_ml_predictions(
            paper_ids=["paper_1", "paper_2"],
            num_iterations=2,
            top_k=5
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "ML Predictions"
        assert 0 <= result.success_rate <= 1
        assert result.throughput >= 0
        assert result.execution_time >= 0
        assert isinstance(result.detailed_metrics, dict)
    
    def test_analyze_memory_usage(self, performance_analyzer):
        """Test memory usage analysis."""
        analysis = performance_analyzer.analyze_memory_usage()
        
        assert isinstance(analysis, dict)
        assert 'system_memory' in analysis
        assert 'process_memory' in analysis
        assert 'recommendations' in analysis
        
        # Check system memory info
        sys_mem = analysis['system_memory']
        assert 'total' in sys_mem
        assert 'available' in sys_mem
        assert 'percent_used' in sys_mem


class TestSystemHealthMonitor:
    """Test suite for SystemHealthMonitor component."""
    
    @pytest.fixture
    def health_monitor(self):
        """Create SystemHealthMonitor instance for testing."""
        mock_ml_service = Mock()
        mock_api_client = Mock()
        return SystemHealthMonitor(mock_ml_service, mock_api_client)
    
    def test_check_system_health(self, health_monitor):
        """Test system health check."""
        # Mock ML service health
        health_monitor.ml_service.health_check.return_value = {
            'status': 'healthy',
            'model_loaded': True,
            'prediction_test': True
        }
        
        health_status = health_monitor.check_system_health()
        
        assert isinstance(health_status, HealthStatus)
        assert health_status.status in ['healthy', 'warning', 'critical']
        assert 0 <= health_status.score <= 100
        assert isinstance(health_status.checks, dict)
        assert isinstance(health_status.metrics, dict)
        assert isinstance(health_status.recommendations, list)


class TestExportEngine:
    """Test suite for ExportEngine component."""
    
    @pytest.fixture
    def export_engine(self):
        """Create ExportEngine instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ExportEngine(Path(temp_dir))
    
    @pytest.fixture
    def sample_network_data(self):
        """Create sample network data for export testing."""
        network_metrics = NetworkMetrics(
            num_nodes=10,
            num_edges=15,
            density=0.15,
            average_degree=3.0,
            clustering_coefficient=0.2,
            diameter=4,
            average_path_length=2.5,
            num_components=1,
            largest_component_size=10,
            modularity=0.3,
            assortativity=0.1
        )
        
        centrality_metrics = {
            "paper_1": CentralityMetrics(
                paper_id="paper_1",
                degree_centrality=0.5,
                betweenness_centrality=0.3,
                closeness_centrality=0.7,
                eigenvector_centrality=0.4,
                pagerank=0.15
            )
        }
        
        communities = [
            CommunityInfo(
                community_id=0,
                size=5,
                members=["paper_1", "paper_2", "paper_3", "paper_4", "paper_5"],
                modularity=0.3,
                internal_edges=8,
                external_edges=2,
                conductance=0.2
            )
        ]
        
        return network_metrics, centrality_metrics, communities
    
    def test_export_network_analysis_json(self, export_engine, sample_network_data):
        """Test network analysis export to JSON."""
        network_metrics, centrality_metrics, communities = sample_network_data
        
        config = ExportConfiguration(format='json')
        result = export_engine.export_network_analysis(
            network_metrics, centrality_metrics, communities, config
        )
        
        assert result.success
        assert result.file_path is not None
        assert result.format == 'json'
        assert Path(result.file_path).exists()
        
        # Verify JSON content
        with open(result.file_path, 'r') as f:
            data = json.load(f)
        
        assert 'network_metrics' in data
        assert 'centrality_metrics' in data
        assert 'communities' in data
    
    def test_export_network_analysis_html(self, export_engine, sample_network_data):
        """Test network analysis export to HTML."""
        network_metrics, centrality_metrics, communities = sample_network_data
        
        config = ExportConfiguration(format='html', include_visualizations=True)
        result = export_engine.export_network_analysis(
            network_metrics, centrality_metrics, communities, config
        )
        
        assert result.success
        assert result.file_path is not None
        assert result.format == 'html'
        assert Path(result.file_path).exists()
        
        # Verify HTML content contains expected elements
        with open(result.file_path, 'r') as f:
            content = f.read()
        
        assert '<html' in content
        assert 'Network Analysis Report' in content
        assert str(network_metrics.num_nodes) in content


class TestAnalyticsService:
    """Test suite for AnalyticsService component."""
    
    @pytest.fixture
    def analytics_service(self):
        """Create AnalyticsService instance for testing."""
        mock_ml_service = Mock()
        return AnalyticsService(ml_service=mock_ml_service)
    
    def test_analyze_citation_network(self, analytics_service):
        """Test citation network analysis."""
        result = analytics_service.analyze_citation_network(
            max_papers=100,
            include_communities=True,
            include_centrality=True
        )
        
        assert isinstance(result, dict)
        assert 'analysis_timestamp' in result
        assert 'graph_info' in result
        
        # Should not have error if successful
        if 'error' not in result:
            assert 'network_metrics' in result
    
    def test_get_system_health(self, analytics_service):
        """Test system health check."""
        health = analytics_service.get_system_health()
        
        assert isinstance(health, dict)
        assert 'overall_health' in health
        assert 'service_info' in health
        assert 'component_health' in health
    
    def test_create_analytics_workflow(self, analytics_service):
        """Test analytics workflow creation."""
        tasks = [
            {'type': 'network_analysis', 'parameters': {'max_papers': 100}},
            {'type': 'performance_benchmark', 'parameters': {'benchmark_types': ['ml']}}
        ]
        
        workflow = analytics_service.create_analytics_workflow(
            "Test Workflow",
            "Test workflow description",
            tasks
        )
        
        assert workflow.workflow_id is not None
        assert workflow.name == "Test Workflow"
        assert len(workflow.tasks) == 2
        assert workflow.status == 'pending'


class TestIntegration:
    """Integration tests for analytics components."""
    
    def test_end_to_end_network_analysis(self):
        """Test complete network analysis pipeline."""
        # Create sample data
        papers = [Paper(
            paper_id=f"paper_{i}",
            title=f"Paper {i}",
            abstract=f"Abstract {i}",
            publication_year=2020,
            authors=[f"Author_{i}"],
            venue="Test Venue"
        ) for i in range(10)]
        
        citations = [Citation(
            citation_id=f"citation_{i}",
            source_paper_id=f"paper_{i}",
            target_paper_id=f"paper_{(i+1) % 10}",
            citation_date=datetime.now()
        ) for i in range(10)]
        
        # Initialize components
        network_analyzer = NetworkAnalyzer()
        community_detector = CommunityDetector()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_engine = ExportEngine(Path(temp_dir))
            
            # Perform analysis
            graph = network_analyzer.load_citation_network(
                paper_ids=[p.paper_id for p in papers[:5]],
                max_papers=5
            )
            
            network_metrics = network_analyzer.analyze_network_structure(graph)
            centrality_metrics = network_analyzer.calculate_centrality_metrics(graph)
            communities = community_detector.detect_communities_louvain(graph)
            
            # Export results
            config = ExportConfiguration(format='json')
            export_result = export_engine.export_network_analysis(
                network_metrics, centrality_metrics, communities, config
            )
            
            assert export_result.success
            assert Path(export_result.file_path).exists()
    
    def test_performance_benchmark_integration(self):
        """Test performance benchmarking integration."""
        mock_ml_service = Mock()
        mock_ml_service.predict_citations.return_value = [Mock(prediction_score=0.8)]
        
        performance_analyzer = PerformanceAnalyzer(mock_ml_service)
        health_monitor = SystemHealthMonitor(mock_ml_service)
        
        # Run benchmarks
        benchmark_result = performance_analyzer.benchmark_ml_predictions(
            ["paper_1", "paper_2"], num_iterations=2
        )
        
        health_status = health_monitor.check_system_health()
        
        # Generate performance summary
        summary = health_monitor.get_performance_summary([benchmark_result])
        
        assert isinstance(summary, dict)
        assert 'total_benchmarks' in summary
        assert summary['total_benchmarks'] == 1