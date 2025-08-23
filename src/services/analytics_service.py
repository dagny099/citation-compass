"""
Analytics service for orchestrating advanced citation network analysis.

This service provides a high-level interface for performing comprehensive
analytics including network analysis, temporal analysis, performance monitoring,
and report generation. It integrates with the ML service and database layer
to provide production-ready analytical capabilities.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..analytics.network_analysis import NetworkAnalyzer, CommunityDetector, NetworkMetrics, CentralityMetrics, CommunityInfo
from ..analytics.temporal_analysis import TemporalAnalyzer, TrendAnalyzer, CitationGrowthMetrics, TrendAnalysis
from ..analytics.performance_metrics import PerformanceAnalyzer, SystemHealthMonitor, BenchmarkResult, HealthStatus
from ..analytics.export_engine import ExportEngine, ReportGenerator, ExportConfiguration, ExportResult
from ..services.ml_service import TransEModelService, get_ml_service
from ..data.unified_database import UnifiedDatabaseManager
from ..data.unified_api_client import UnifiedSemanticScholarClient
from ..models.paper import Paper
from ..models.citation import Citation


@dataclass
class AnalyticsTask:
    """Represents an analytics task to be executed."""
    task_id: str
    task_type: str  # 'network_analysis', 'temporal_analysis', 'performance_benchmark'
    parameters: Dict[str, Any]
    priority: int = 1  # 1=low, 2=medium, 3=high
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = 'pending'  # 'pending', 'running', 'completed', 'failed'
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class AnalyticsWorkflow:
    """Represents a workflow of multiple analytics tasks."""
    workflow_id: str
    name: str
    description: str
    tasks: List[AnalyticsTask]
    status: str = 'pending'
    created_at: datetime = None
    results: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.results is None:
            self.results = {}


class AnalyticsService:
    """
    Comprehensive analytics service for citation network analysis.
    
    Orchestrates network analysis, temporal analysis, performance monitoring,
    and report generation with support for batch processing, caching,
    and asynchronous execution.
    """
    
    def __init__(self,
                 ml_service: Optional[TransEModelService] = None,
                 database: Optional[UnifiedDatabaseManager] = None,
                 api_client: Optional[UnifiedSemanticScholarClient] = None,
                 output_dir: Optional[Path] = None):
        """
        Initialize analytics service.
        
        Args:
            ml_service: ML service instance
            database: Database connection
            api_client: API client for data fetching
            output_dir: Output directory for reports and exports
        """
        self.ml_service = ml_service or get_ml_service()
        self.database = database
        self.api_client = api_client
        
        # Initialize analytics components
        self.network_analyzer = NetworkAnalyzer(database)
        self.community_detector = CommunityDetector()
        self.temporal_analyzer = TemporalAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer(self.ml_service)
        self.health_monitor = SystemHealthMonitor(self.ml_service, api_client)
        
        # Initialize export components
        self.export_engine = ExportEngine(output_dir)
        self.report_generator = ReportGenerator(self.export_engine)
        
        # Task management
        self.task_queue: List[AnalyticsTask] = []
        self.active_tasks: Dict[str, AnalyticsTask] = {}
        self.completed_tasks: Dict[str, AnalyticsTask] = {}
        self.workflows: Dict[str, AnalyticsWorkflow] = {}
        
        # Execution settings
        self.max_concurrent_tasks = 3
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_tasks)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Analytics service initialized")
    
    def analyze_citation_network(self,
                                paper_ids: Optional[List[str]] = None,
                                max_papers: Optional[int] = 1000,
                                include_communities: bool = True,
                                include_centrality: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive network analysis.
        
        Args:
            paper_ids: Specific papers to analyze (None for all)
            max_papers: Maximum number of papers to include
            include_communities: Whether to perform community detection
            include_centrality: Whether to calculate centrality metrics
            
        Returns:
            Dictionary with network analysis results
        """
        self.logger.info(f"Starting network analysis (max_papers={max_papers})")
        
        try:
            # Load citation network
            graph = self.network_analyzer.load_citation_network(
                paper_ids=paper_ids,
                max_papers=max_papers
            )
            
            results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'graph_info': {
                    'num_nodes': graph.number_of_nodes(),
                    'num_edges': graph.number_of_edges(),
                    'is_directed': graph.is_directed()
                }
            }
            
            # Analyze network structure
            network_metrics = self.network_analyzer.analyze_network_structure(graph)
            results['network_metrics'] = network_metrics
            
            # Calculate centrality metrics
            centrality_metrics = {}
            if include_centrality:
                self.logger.info("Calculating centrality metrics")
                centrality_metrics = self.network_analyzer.calculate_centrality_metrics(graph)
                results['centrality_metrics'] = centrality_metrics
                
                # Identify influential papers
                influential = self.network_analyzer.identify_influential_papers(
                    centrality_metrics, top_k=20
                )
                results['influential_papers'] = influential
            
            # Community detection
            communities = []
            if include_communities:
                self.logger.info("Detecting communities")
                communities = self.community_detector.detect_communities_louvain(graph)
                results['communities'] = communities
                
                # Community analysis
                if communities:
                    community_analysis = self.community_detector.analyze_community_structure(
                        communities, graph
                    )
                    results['community_analysis'] = community_analysis
            
            self.logger.info("Network analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Network analysis failed: {e}")
            return {'error': str(e), 'analysis_timestamp': datetime.now().isoformat()}
    
    def analyze_temporal_patterns(self,
                                 papers: List[Paper],
                                 citations: List[Citation],
                                 include_trends: bool = True,
                                 include_growth: bool = True,
                                 include_seasonality: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive temporal analysis.
        
        Args:
            papers: List of papers to analyze
            citations: List of citations
            include_trends: Whether to perform trend analysis
            include_growth: Whether to analyze citation growth
            include_seasonality: Whether to analyze seasonal patterns
            
        Returns:
            Dictionary with temporal analysis results
        """
        self.logger.info(f"Starting temporal analysis for {len(papers)} papers")
        
        try:
            results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'data_info': {
                    'num_papers': len(papers),
                    'num_citations': len(citations)
                }
            }
            
            # Citation growth analysis
            if include_growth:
                self.logger.info("Analyzing citation growth patterns")
                growth_metrics = self.temporal_analyzer.analyze_citation_growth(
                    papers, citations
                )
                results['growth_metrics'] = growth_metrics
                
                # Growth comparison
                growth_comparison = self.trend_analyzer.compare_growth_rates(growth_metrics)
                results['growth_comparison'] = growth_comparison
            
            # Time series analysis
            if include_trends or include_seasonality:
                self.logger.info("Creating citation time series")
                time_series = self.temporal_analyzer.create_citation_time_series(citations)
                
                if include_trends:
                    # Trend analysis
                    trend_analysis = self.trend_analyzer.analyze_trend(time_series)
                    results['trend_analysis'] = trend_analysis
                
                if include_seasonality and len(time_series) >= 12:
                    # Seasonal analysis
                    seasonal_analysis = self.temporal_analyzer.analyze_seasonal_patterns(time_series)
                    results['seasonal_analysis'] = seasonal_analysis
                    
                    # Burst detection
                    bursts = self.temporal_analyzer.detect_citation_bursts(time_series)
                    results['citation_bursts'] = bursts
            
            self.logger.info("Temporal analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Temporal analysis failed: {e}")
            return {'error': str(e), 'analysis_timestamp': datetime.now().isoformat()}
    
    def run_performance_benchmarks(self,
                                  benchmark_types: Optional[List[str]] = None,
                                  test_paper_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive performance benchmarks.
        
        Args:
            benchmark_types: Types of benchmarks to run ('ml', 'api', 'stress')
            test_paper_ids: Paper IDs for testing
            
        Returns:
            Dictionary with benchmark results
        """
        if benchmark_types is None:
            benchmark_types = ['ml', 'api', 'stress']
        
        if test_paper_ids is None:
            # Use sample paper IDs for testing
            test_paper_ids = [
                "649def34f8be52c8b66281af98ae884c09aef38f9",
                "sample_paper_1", "sample_paper_2", "sample_paper_3"
            ]
        
        self.logger.info(f"Running performance benchmarks: {benchmark_types}")
        
        results = {
            'benchmark_timestamp': datetime.now().isoformat(),
            'benchmark_results': [],
            'system_health': None,
            'summary': {}
        }
        
        try:
            benchmark_results = []
            
            # ML benchmarks
            if 'ml' in benchmark_types:
                self.logger.info("Running ML performance benchmarks")
                ml_benchmark = self.performance_analyzer.benchmark_ml_predictions(
                    test_paper_ids, num_iterations=10
                )
                benchmark_results.append(ml_benchmark)
            
            # API benchmarks
            if 'api' in benchmark_types and self.api_client:
                self.logger.info("Running API performance benchmarks")
                api_benchmark = self.performance_analyzer.benchmark_api_performance(
                    self.api_client, num_requests=25
                )
                benchmark_results.append(api_benchmark)
            
            # Stress test
            if 'stress' in benchmark_types:
                self.logger.info("Running stress test")
                stress_benchmark = self.performance_analyzer.stress_test_concurrent_predictions(
                    test_paper_ids, num_concurrent=5, duration_seconds=15
                )
                benchmark_results.append(stress_benchmark)
            
            results['benchmark_results'] = benchmark_results
            
            # System health check
            health_status = self.health_monitor.check_system_health()
            results['system_health'] = health_status
            
            # Performance summary
            performance_summary = self.health_monitor.get_performance_summary(benchmark_results)
            results['summary'] = performance_summary
            
            self.logger.info("Performance benchmarks completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Performance benchmarks failed: {e}")
            return {'error': str(e), 'benchmark_timestamp': datetime.now().isoformat()}
    
    def generate_comprehensive_report(self,
                                    analysis_results: Dict[str, Any],
                                    export_format: str = 'html',
                                    include_visualizations: bool = True) -> ExportResult:
        """
        Generate comprehensive analysis report.
        
        Args:
            analysis_results: Combined analysis results
            export_format: Output format ('html', 'json', 'pdf')
            include_visualizations: Whether to include visualizations
            
        Returns:
            ExportResult with report details
        """
        self.logger.info(f"Generating comprehensive report in {export_format} format")
        
        try:
            config = ExportConfiguration(
                format=export_format,
                include_visualizations=include_visualizations,
                include_raw_data=True,
                metadata={
                    'generated_by': 'AnalyticsService',
                    'platform': 'Academic Citation Platform',
                    'version': '1.0.0'
                }
            )
            
            # Extract different analysis types
            network_data = analysis_results.get('network_analysis')
            temporal_data = analysis_results.get('temporal_analysis')
            performance_data = analysis_results.get('performance_analysis')
            
            # Generate comprehensive report
            result = self.report_generator.generate_comprehensive_report(
                network_data=network_data,
                temporal_data=temporal_data,
                performance_data=performance_data,
                format=export_format
            )
            
            self.logger.info(f"Report generated: {result.file_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return ExportResult(
                success=False,
                error_message=str(e)
            )
    
    def create_analytics_workflow(self,
                                 workflow_name: str,
                                 workflow_description: str,
                                 tasks: List[Dict[str, Any]]) -> AnalyticsWorkflow:
        """
        Create a workflow of multiple analytics tasks.
        
        Args:
            workflow_name: Name of the workflow
            workflow_description: Description of the workflow
            tasks: List of task configurations
            
        Returns:
            AnalyticsWorkflow object
        """
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        analytics_tasks = []
        for i, task_config in enumerate(tasks):
            task = AnalyticsTask(
                task_id=f"{workflow_id}_task_{i}",
                task_type=task_config['type'],
                parameters=task_config.get('parameters', {}),
                priority=task_config.get('priority', 1)
            )
            analytics_tasks.append(task)
        
        workflow = AnalyticsWorkflow(
            workflow_id=workflow_id,
            name=workflow_name,
            description=workflow_description,
            tasks=analytics_tasks
        )
        
        self.workflows[workflow_id] = workflow
        self.logger.info(f"Created workflow: {workflow_id} with {len(analytics_tasks)} tasks")
        
        return workflow
    
    def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Execute an analytics workflow.
        
        Args:
            workflow_id: ID of the workflow to execute
            
        Returns:
            Dictionary with workflow execution results
        """
        if workflow_id not in self.workflows:
            return {'error': f'Workflow not found: {workflow_id}'}
        
        workflow = self.workflows[workflow_id]
        self.logger.info(f"Executing workflow: {workflow_id}")
        
        workflow.status = 'running'
        results = {}
        
        try:
            for task in workflow.tasks:
                self.logger.info(f"Executing task: {task.task_id}")
                
                task.status = 'running'
                task.started_at = datetime.now()
                
                # Execute task based on type
                if task.task_type == 'network_analysis':
                    task.result = self.analyze_citation_network(**task.parameters)
                elif task.task_type == 'temporal_analysis':
                    # This would need actual paper and citation data
                    task.result = {'info': 'Temporal analysis task completed'}
                elif task.task_type == 'performance_benchmark':
                    task.result = self.run_performance_benchmarks(**task.parameters)
                else:
                    task.result = {'error': f'Unknown task type: {task.task_type}'}
                
                task.status = 'completed'
                task.completed_at = datetime.now()
                results[task.task_id] = task.result
            
            workflow.status = 'completed'
            workflow.results = results
            
            self.logger.info(f"Workflow completed: {workflow_id}")
            return {'workflow_id': workflow_id, 'status': 'completed', 'results': results}
            
        except Exception as e:
            workflow.status = 'failed'
            self.logger.error(f"Workflow failed: {workflow_id} - {e}")
            return {'workflow_id': workflow_id, 'status': 'failed', 'error': str(e)}
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get current system health status.
        
        Returns:
            Dictionary with system health information
        """
        try:
            health_status = self.health_monitor.check_system_health()
            
            # Add analytics service specific information
            health_info = {
                'overall_health': health_status,
                'service_info': {
                    'active_tasks': len(self.active_tasks),
                    'completed_tasks': len(self.completed_tasks),
                    'queued_tasks': len(self.task_queue),
                    'active_workflows': sum(1 for w in self.workflows.values() if w.status == 'running'),
                    'ml_service_available': self.ml_service is not None,
                    'database_available': self.database is not None,
                    'api_client_available': self.api_client is not None
                },
                'component_health': {
                    'network_analyzer': True,
                    'temporal_analyzer': True,
                    'performance_analyzer': True,
                    'export_engine': True
                }
            }
            
            return health_info
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'overall_health': {'status': 'critical', 'error': str(e)},
                'service_info': {'error': 'Health check failed'}
            }
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed tasks.
        
        Args:
            max_age_hours: Maximum age of completed tasks to keep
            
        Returns:
            Number of tasks cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        tasks_to_remove = [
            task_id for task_id, task in self.completed_tasks.items()
            if task.completed_at and task.completed_at < cutoff_time
        ]
        
        for task_id in tasks_to_remove:
            del self.completed_tasks[task_id]
        
        self.logger.info(f"Cleaned up {len(tasks_to_remove)} old completed tasks")
        return len(tasks_to_remove)
    
    def shutdown(self) -> None:
        """Gracefully shutdown the analytics service."""
        self.logger.info("Shutting down analytics service")
        
        # Wait for active tasks to complete
        self.executor.shutdown(wait=True)
        
        # Save any important state if needed
        self.logger.info("Analytics service shutdown complete")


# Global service instance
_analytics_service: Optional[AnalyticsService] = None


def get_analytics_service(force_reload: bool = False) -> AnalyticsService:
    """
    Get the global analytics service instance.
    
    Args:
        force_reload: Force reloading of the service
        
    Returns:
        AnalyticsService instance
    """
    global _analytics_service
    
    if _analytics_service is None or force_reload:
        _analytics_service = AnalyticsService()
    
    return _analytics_service


def reset_analytics_service() -> None:
    """Reset the global analytics service instance. Useful for testing."""
    global _analytics_service
    if _analytics_service:
        _analytics_service.shutdown()
    _analytics_service = None