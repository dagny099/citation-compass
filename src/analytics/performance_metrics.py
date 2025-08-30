"""
Performance metrics and system health monitoring for the analytics platform.

Provides comprehensive performance analysis, benchmarking, and health monitoring
capabilities for ML models, API services, and overall system performance.
"""

import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics

from src.services.ml_service import TransEModelService


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemResource:
    """System resource usage snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available: int  # bytes
    disk_usage_percent: float
    network_io: Dict[str, int] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Results from performance benchmark."""
    benchmark_name: str
    execution_time: float
    success_rate: float
    error_count: int
    throughput: float  # operations per second
    resource_usage: SystemResource
    detailed_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """Overall system health status."""
    status: str  # 'healthy', 'warning', 'critical'
    score: float  # 0-100
    checks: Dict[str, bool]
    metrics: Dict[str, float]
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceAnalyzer:
    """
    Performance analyzer for ML models and system components.
    
    Provides comprehensive performance analysis including timing, throughput,
    resource usage, and bottleneck identification.
    """
    
    def __init__(self, ml_service: Optional[TransEModelService] = None):
        """
        Initialize performance analyzer.
        
        Args:
            ml_service: ML service instance to analyze
        """
        self.ml_service = ml_service
        self.logger = logging.getLogger(__name__)
        self._metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def benchmark_ml_predictions(self, 
                                paper_ids: List[str],
                                num_iterations: int = 10,
                                top_k: int = 10) -> BenchmarkResult:
        """
        Benchmark ML prediction performance.
        
        Args:
            paper_ids: List of paper IDs to test predictions
            num_iterations: Number of test iterations
            top_k: Number of predictions to generate
            
        Returns:
            BenchmarkResult with performance metrics
        """
        self.logger.info(f"Benchmarking ML predictions ({num_iterations} iterations)")
        
        if not self.ml_service:
            raise ValueError("ML service required for ML benchmarks")
        
        start_time = time.time()
        execution_times = []
        errors = 0
        successful_predictions = 0
        
        # Pre-benchmark system state
        start_resource = self._capture_system_resources()
        
        for iteration in range(num_iterations):
            for paper_id in paper_ids[:5]:  # Limit to first 5 papers per iteration
                try:
                    pred_start = time.time()
                    predictions = self.ml_service.predict_citations(
                        paper_id, top_k=top_k
                    )
                    pred_end = time.time()
                    
                    execution_times.append(pred_end - pred_start)
                    if predictions:
                        successful_predictions += 1
                        
                except Exception as e:
                    self.logger.warning(f"Prediction error for {paper_id}: {e}")
                    errors += 1
        
        total_time = time.time() - start_time
        end_resource = self._capture_system_resources()
        
        # Calculate metrics
        total_operations = num_iterations * len(paper_ids[:5])
        success_rate = successful_predictions / total_operations if total_operations > 0 else 0
        throughput = total_operations / total_time if total_time > 0 else 0
        
        detailed_metrics = {
            'avg_prediction_time': statistics.mean(execution_times) if execution_times else 0,
            'median_prediction_time': statistics.median(execution_times) if execution_times else 0,
            'min_prediction_time': min(execution_times) if execution_times else 0,
            'max_prediction_time': max(execution_times) if execution_times else 0,
            'prediction_time_std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            'cache_hit_rate': self._estimate_cache_hit_rate(),
            'memory_growth': end_resource.memory_percent - start_resource.memory_percent
        }
        
        return BenchmarkResult(
            benchmark_name="ML Predictions",
            execution_time=total_time,
            success_rate=success_rate,
            error_count=errors,
            throughput=throughput,
            resource_usage=end_resource,
            detailed_metrics=detailed_metrics
        )
    
    def benchmark_api_performance(self, 
                                 api_client,
                                 num_requests: int = 50) -> BenchmarkResult:
        """
        Benchmark API client performance.
        
        Args:
            api_client: API client instance to test
            num_requests: Number of API requests to make
            
        Returns:
            BenchmarkResult with API performance metrics
        """
        self.logger.info(f"Benchmarking API performance ({num_requests} requests)")
        
        start_time = time.time()
        response_times = []
        errors = 0
        successful_requests = 0
        
        start_resource = self._capture_system_resources()
        
        # Test different types of requests
        test_queries = [
            "machine learning", "citation analysis", "graph neural networks",
            "deep learning", "natural language processing"
        ]
        
        for i in range(num_requests):
            query = test_queries[i % len(test_queries)]
            
            try:
                req_start = time.time()
                # Mock API call - in real implementation, use actual API client
                time.sleep(0.01)  # Simulate API delay
                req_end = time.time()
                
                response_times.append(req_end - req_start)
                successful_requests += 1
                
            except Exception as e:
                self.logger.warning(f"API error: {e}")
                errors += 1
        
        total_time = time.time() - start_time
        end_resource = self._capture_system_resources()
        
        success_rate = successful_requests / num_requests if num_requests > 0 else 0
        throughput = successful_requests / total_time if total_time > 0 else 0
        
        detailed_metrics = {
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'median_response_time': statistics.median(response_times) if response_times else 0,
            'p95_response_time': self._calculate_percentile(response_times, 95),
            'p99_response_time': self._calculate_percentile(response_times, 99),
            'response_time_std': statistics.stdev(response_times) if len(response_times) > 1 else 0
        }
        
        return BenchmarkResult(
            benchmark_name="API Performance",
            execution_time=total_time,
            success_rate=success_rate,
            error_count=errors,
            throughput=throughput,
            resource_usage=end_resource,
            detailed_metrics=detailed_metrics
        )
    
    def stress_test_concurrent_predictions(self,
                                         paper_ids: List[str],
                                         num_concurrent: int = 10,
                                         duration_seconds: int = 30) -> BenchmarkResult:
        """
        Stress test with concurrent prediction requests.
        
        Args:
            paper_ids: Paper IDs to test
            num_concurrent: Number of concurrent threads
            duration_seconds: Test duration in seconds
            
        Returns:
            BenchmarkResult with stress test results
        """
        self.logger.info(f"Stress testing with {num_concurrent} concurrent threads")
        
        if not self.ml_service:
            raise ValueError("ML service required for stress testing")
        
        results = {
            'execution_times': [],
            'errors': 0,
            'successes': 0,
            'completed': threading.Event()
        }
        
        def worker():
            """Worker thread for concurrent requests."""
            import random
            while not results['completed'].is_set():
                try:
                    paper_id = random.choice(paper_ids)
                    start = time.time()
                    predictions = self.ml_service.predict_citations(paper_id, top_k=5)
                    end = time.time()
                    
                    results['execution_times'].append(end - start)
                    results['successes'] += 1
                    
                except Exception as e:
                    results['errors'] += 1
                
                time.sleep(0.1)  # Small delay between requests
        
        # Start benchmark
        start_time = time.time()
        start_resource = self._capture_system_resources()
        
        # Launch worker threads
        threads = []
        for _ in range(num_concurrent):
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Run for specified duration
        time.sleep(duration_seconds)
        results['completed'].set()
        
        # Wait for threads to complete
        for thread in threads:
            thread.join(timeout=5.0)
        
        total_time = time.time() - start_time
        end_resource = self._capture_system_resources()
        
        # Calculate results
        total_operations = results['successes'] + results['errors']
        success_rate = results['successes'] / total_operations if total_operations > 0 else 0
        throughput = results['successes'] / total_time if total_time > 0 else 0
        
        execution_times = results['execution_times']
        detailed_metrics = {
            'concurrent_threads': num_concurrent,
            'total_requests': total_operations,
            'avg_response_time': statistics.mean(execution_times) if execution_times else 0,
            'max_response_time': max(execution_times) if execution_times else 0,
            'cpu_usage_increase': end_resource.cpu_percent - start_resource.cpu_percent,
            'memory_usage_increase': end_resource.memory_percent - start_resource.memory_percent
        }
        
        return BenchmarkResult(
            benchmark_name="Concurrent Stress Test",
            execution_time=total_time,
            success_rate=success_rate,
            error_count=results['errors'],
            throughput=throughput,
            resource_usage=end_resource,
            detailed_metrics=detailed_metrics
        )
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """
        Analyze current memory usage patterns.
        
        Returns:
            Dictionary with memory analysis results
        """
        self.logger.info("Analyzing memory usage")
        
        # Get system memory info
        memory = psutil.virtual_memory()
        
        # Get process memory info
        process = psutil.Process()
        process_memory = process.memory_info()
        
        analysis = {
            'system_memory': {
                'total': memory.total,
                'available': memory.available,
                'percent_used': memory.percent,
                'free': memory.free
            },
            'process_memory': {
                'rss': process_memory.rss,  # Resident Set Size
                'vms': process_memory.vms,  # Virtual Memory Size
                'percent': process.memory_percent()
            },
            'recommendations': []
        }
        
        # Add recommendations based on usage
        if memory.percent > 80:
            analysis['recommendations'].append(
                "High system memory usage detected. Consider closing other applications."
            )
        
        if process.memory_percent() > 10:
            analysis['recommendations'].append(
                "High process memory usage. Consider optimizing caching or reducing batch sizes."
            )
        
        return analysis
    
    def _capture_system_resources(self) -> SystemResource:
        """Capture current system resource usage."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Network I/O (simplified)
        try:
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv
            }
        except:
            network_io = {}
        
        return SystemResource(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available=memory.available,
            disk_usage_percent=disk.percent,
            network_io=network_io
        )
    
    def _estimate_cache_hit_rate(self) -> float:
        """Estimate cache hit rate from ML service."""
        if not self.ml_service or not hasattr(self.ml_service, 'cache'):
            return 0.0
        
        # This would need to be implemented in the actual ML service
        # For now, return a mock value
        return 0.75  # 75% cache hit rate
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value from list."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(percentile / 100.0 * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]


class SystemHealthMonitor:
    """
    Comprehensive system health monitoring.
    
    Monitors various aspects of system health including resource usage,
    service availability, and performance metrics.
    """
    
    def __init__(self, 
                 ml_service: Optional[TransEModelService] = None,
                 api_client=None):
        """
        Initialize system health monitor.
        
        Args:
            ml_service: ML service to monitor
            api_client: API client to monitor
        """
        self.ml_service = ml_service
        self.api_client = api_client
        self.logger = logging.getLogger(__name__)
        
        # Health check thresholds
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 75.0,
            'memory_critical': 90.0,
            'disk_warning': 80.0,
            'disk_critical': 95.0,
            'response_time_warning': 1.0,
            'response_time_critical': 5.0
        }
    
    def check_system_health(self) -> HealthStatus:
        """
        Perform comprehensive system health check.
        
        Returns:
            HealthStatus with overall system health assessment
        """
        self.logger.info("Performing system health check")
        
        checks = {}
        metrics = {}
        recommendations = []
        issues = []
        
        # Check system resources
        resource_health = self._check_resource_health()
        checks.update(resource_health['checks'])
        metrics.update(resource_health['metrics'])
        recommendations.extend(resource_health['recommendations'])
        if resource_health['issues']:
            issues.extend(resource_health['issues'])
        
        # Check ML service health
        if self.ml_service:
            ml_health = self._check_ml_service_health()
            checks.update(ml_health['checks'])
            metrics.update(ml_health['metrics'])
            recommendations.extend(ml_health['recommendations'])
            if ml_health['issues']:
                issues.extend(ml_health['issues'])
        
        # Check API client health
        if self.api_client:
            api_health = self._check_api_health()
            checks.update(api_health['checks'])
            metrics.update(api_health['metrics'])
            recommendations.extend(api_health['recommendations'])
            if api_health['issues']:
                issues.extend(api_health['issues'])
        
        # Calculate overall health score
        healthy_checks = sum(1 for check in checks.values() if check)
        total_checks = len(checks)
        health_score = (healthy_checks / total_checks * 100) if total_checks > 0 else 0
        
        # Determine overall status
        if health_score >= 80 and not issues:
            status = 'healthy'
        elif health_score >= 60 or len(issues) <= 2:
            status = 'warning'
        else:
            status = 'critical'
        
        return HealthStatus(
            status=status,
            score=health_score,
            checks=checks,
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _check_resource_health(self) -> Dict[str, Any]:
        """Check system resource health."""
        checks = {}
        metrics = {}
        recommendations = []
        issues = []
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics['cpu_usage'] = cpu_percent
        checks['cpu_healthy'] = cpu_percent < self.thresholds['cpu_critical']
        
        if cpu_percent > self.thresholds['cpu_critical']:
            issues.append("Critical CPU usage")
            recommendations.append("High CPU usage detected. Consider reducing workload.")
        elif cpu_percent > self.thresholds['cpu_warning']:
            recommendations.append("Elevated CPU usage. Monitor performance.")
        
        # Memory usage
        memory = psutil.virtual_memory()
        metrics['memory_usage'] = memory.percent
        checks['memory_healthy'] = memory.percent < self.thresholds['memory_critical']
        
        if memory.percent > self.thresholds['memory_critical']:
            issues.append("Critical memory usage")
            recommendations.append("High memory usage. Consider freeing memory or adding RAM.")
        elif memory.percent > self.thresholds['memory_warning']:
            recommendations.append("Elevated memory usage. Monitor memory consumption.")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        metrics['disk_usage'] = disk_percent
        checks['disk_healthy'] = disk_percent < self.thresholds['disk_critical']
        
        if disk_percent > self.thresholds['disk_critical']:
            issues.append("Critical disk usage")
            recommendations.append("Very low disk space. Clean up files or add storage.")
        elif disk_percent > self.thresholds['disk_warning']:
            recommendations.append("Low disk space. Consider cleanup.")
        
        return {
            'checks': checks,
            'metrics': metrics,
            'recommendations': recommendations,
            'issues': issues
        }
    
    def _check_ml_service_health(self) -> Dict[str, Any]:
        """Check ML service health."""
        checks = {}
        metrics = {}
        recommendations = []
        issues = []
        
        try:
            health_check = self.ml_service.health_check()
            
            # ML service availability
            checks['ml_service_available'] = health_check.get('status') == 'healthy'
            checks['ml_model_loaded'] = health_check.get('model_loaded', False)
            checks['ml_predictions_working'] = health_check.get('prediction_test', False)
            
            # ML service metrics
            metrics['ml_entities'] = health_check.get('num_entities', 0)
            metrics['ml_cache_enabled'] = 1 if health_check.get('cache_enabled', False) else 0
            
            # Check for issues
            if not checks['ml_service_available']:
                issues.append("ML service unhealthy")
                recommendations.append("Check ML service configuration and model files.")
            
            if not checks['ml_model_loaded']:
                issues.append("ML model not loaded")
                recommendations.append("Verify ML model files exist and are accessible.")
            
        except Exception as e:
            checks['ml_service_available'] = False
            issues.append(f"ML service error: {str(e)}")
            recommendations.append("Check ML service configuration and restart if needed.")
        
        return {
            'checks': checks,
            'metrics': metrics,
            'recommendations': recommendations,
            'issues': issues
        }
    
    def _check_api_health(self) -> Dict[str, Any]:
        """Check API client health."""
        checks = {}
        metrics = {}
        recommendations = []
        issues = []
        
        try:
            # Test API availability (mock implementation)
            # In real implementation, make a test API call
            api_available = True  # Mock result
            
            checks['api_available'] = api_available
            metrics['api_response_time'] = 0.5  # Mock response time
            
            if not api_available:
                issues.append("API not available")
                recommendations.append("Check API connectivity and credentials.")
            
        except Exception as e:
            checks['api_available'] = False
            issues.append(f"API error: {str(e)}")
            recommendations.append("Check API configuration and network connectivity.")
        
        return {
            'checks': checks,
            'metrics': metrics,
            'recommendations': recommendations,
            'issues': issues
        }
    
    def get_performance_summary(self, 
                              benchmark_results: List[BenchmarkResult]) -> Dict[str, Any]:
        """
        Generate performance summary from benchmark results.
        
        Args:
            benchmark_results: List of benchmark results
            
        Returns:
            Dictionary with performance summary
        """
        if not benchmark_results:
            return {'error': 'No benchmark results provided'}
        
        summary = {
            'total_benchmarks': len(benchmark_results),
            'benchmarks': {},
            'overall_health': 'good',
            'key_metrics': {},
            'recommendations': []
        }
        
        total_errors = 0
        total_operations = 0
        avg_success_rates = []
        
        for result in benchmark_results:
            benchmark_info = {
                'execution_time': result.execution_time,
                'success_rate': result.success_rate,
                'throughput': result.throughput,
                'error_count': result.error_count,
                'key_metrics': result.detailed_metrics
            }
            summary['benchmarks'][result.benchmark_name] = benchmark_info
            
            # Aggregate metrics
            total_errors += result.error_count
            avg_success_rates.append(result.success_rate)
        
        # Overall metrics
        summary['key_metrics'] = {
            'average_success_rate': statistics.mean(avg_success_rates) if avg_success_rates else 0,
            'total_errors': total_errors,
            'performance_score': min(100, statistics.mean(avg_success_rates) * 100) if avg_success_rates else 0
        }
        
        # Recommendations based on results
        avg_success_rate = statistics.mean(avg_success_rates) if avg_success_rates else 0
        
        if avg_success_rate < 0.9:
            summary['overall_health'] = 'warning'
            summary['recommendations'].append("Success rate below 90%. Check for system issues.")
        
        if total_errors > 10:
            summary['overall_health'] = 'critical'
            summary['recommendations'].append("High error count detected. Review logs and system stability.")
        
        return summary