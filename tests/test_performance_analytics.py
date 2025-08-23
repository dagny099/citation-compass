"""
Performance and stress testing for analytics components.

Comprehensive performance testing including load testing, memory analysis,
concurrent access testing, and scalability validation.
"""

import pytest
import time
import threading
import psutil
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime, timedelta

from src.analytics.performance_metrics import (
    PerformanceAnalyzer,
    SystemHealthMonitor,
    BenchmarkResult,
    HealthStatus,
    PerformanceMetric
)
from src.services.analytics_service import AnalyticsService
from src.analytics.network_analysis import NetworkAnalyzer
from src.analytics.export_engine import ExportEngine


class TestPerformanceStress:
    """Stress testing for analytics components."""
    
    @pytest.fixture
    def mock_ml_service(self):
        """Create mock ML service for testing."""
        mock_service = Mock()
        mock_service.predict_citations.return_value = [
            Mock(prediction_score=0.8 + np.random.random()*0.2)
            for _ in range(10)
        ]
        mock_service.health_check.return_value = {
            'status': 'healthy',
            'model_loaded': True,
            'prediction_test': True
        }
        return mock_service
    
    @pytest.fixture
    def performance_analyzer(self, mock_ml_service):
        """Create PerformanceAnalyzer for testing."""
        return PerformanceAnalyzer(mock_ml_service)
    
    @pytest.mark.slow
    def test_concurrent_ml_predictions(self, performance_analyzer):
        """Test ML prediction performance under concurrent load."""
        test_papers = [f"paper_{i}" for i in range(20)]
        
        def predict_worker(paper_id):
            """Worker function for concurrent predictions."""
            start_time = time.time()
            try:
                result = performance_analyzer.benchmark_ml_predictions(
                    [paper_id], num_iterations=5, top_k=5
                )
                end_time = time.time()
                return {
                    'success': True,
                    'duration': end_time - start_time,
                    'throughput': result.throughput,
                    'success_rate': result.success_rate
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'duration': time.time() - start_time
                }
        
        # Test with multiple concurrent threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(predict_worker, paper) for paper in test_papers[:10]]
            results = [future.result() for future in futures]
        
        # Analyze results
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        assert len(successful_results) > 0, "No successful concurrent predictions"
        
        # Performance metrics
        avg_duration = np.mean([r['duration'] for r in successful_results])
        avg_throughput = np.mean([r['throughput'] for r in successful_results])
        success_rate = len(successful_results) / len(results)
        
        print(f"\n🔍 Concurrent Performance Results:")
        print(f"   Success Rate: {success_rate:.2%}")
        print(f"   Average Duration: {avg_duration:.2f}s")
        print(f"   Average Throughput: {avg_throughput:.2f} ops/sec")
        print(f"   Failed Requests: {len(failed_results)}")
        
        # Performance assertions
        assert success_rate > 0.8, f"Low success rate: {success_rate:.2%}"
        assert avg_duration < 10.0, f"High average duration: {avg_duration:.2f}s"
    
    @pytest.mark.slow
    def test_memory_usage_under_load(self, performance_analyzer):
        """Test memory usage patterns under sustained load."""
        initial_memory = psutil.virtual_memory().percent
        process = psutil.Process()
        initial_process_memory = process.memory_info().rss
        
        print(f"\n💾 Initial Memory State:")
        print(f"   System Memory: {initial_memory:.1f}%")
        print(f"   Process Memory: {initial_process_memory / 1024 / 1024:.1f} MB")
        
        # Simulate sustained load
        memory_samples = []
        test_papers = [f"paper_{i}" for i in range(50)]
        
        for i in range(10):  # 10 iterations of load
            # Run benchmark
            performance_analyzer.benchmark_ml_predictions(
                test_papers[:5], num_iterations=3, top_k=10
            )
            
            # Sample memory
            current_memory = psutil.virtual_memory().percent
            current_process_memory = process.memory_info().rss
            memory_samples.append({
                'iteration': i,
                'system_memory': current_memory,
                'process_memory': current_process_memory,
                'timestamp': time.time()
            })
            
            time.sleep(0.5)  # Brief pause between iterations
        
        final_memory = psutil.virtual_memory().percent
        final_process_memory = process.memory_info().rss
        
        print(f"\n💾 Final Memory State:")
        print(f"   System Memory: {final_memory:.1f}%")
        print(f"   Process Memory: {final_process_memory / 1024 / 1024:.1f} MB")
        
        # Analyze memory growth
        memory_growth = final_memory - initial_memory
        process_memory_growth = (final_process_memory - initial_process_memory) / 1024 / 1024
        
        print(f"\n📈 Memory Growth:")
        print(f"   System Memory Growth: {memory_growth:.1f}%")
        print(f"   Process Memory Growth: {process_memory_growth:.1f} MB")
        
        # Memory leak detection
        assert memory_growth < 10.0, f"Excessive system memory growth: {memory_growth:.1f}%"
        assert process_memory_growth < 100.0, f"Excessive process memory growth: {process_memory_growth:.1f} MB"
    
    def test_stress_test_concurrent_predictions(self, performance_analyzer):
        """Test the built-in stress testing functionality."""
        test_papers = [f"paper_{i}" for i in range(10)]
        
        result = performance_analyzer.stress_test_concurrent_predictions(
            test_papers, num_concurrent=3, duration_seconds=5
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "Concurrent Stress Test"
        assert result.execution_time > 0
        assert 0 <= result.success_rate <= 1
        assert result.throughput >= 0
        
        # Check detailed metrics
        detailed = result.detailed_metrics
        assert 'concurrent_threads' in detailed
        assert 'total_requests' in detailed
        assert detailed['concurrent_threads'] == 3
        
        print(f"\n🔥 Stress Test Results:")
        print(f"   Success Rate: {result.success_rate:.2%}")
        print(f"   Throughput: {result.throughput:.2f} ops/sec")
        print(f"   Total Requests: {detailed['total_requests']}")
        print(f"   Average Response Time: {detailed['avg_response_time']:.4f}s")
    
    @pytest.mark.slow
    def test_scalability_analysis(self, performance_analyzer):
        """Test how performance scales with increasing load."""
        thread_counts = [1, 2, 4, 8]
        test_papers = [f"paper_{i}" for i in range(20)]
        scalability_results = []
        
        for num_threads in thread_counts:
            print(f"\n📊 Testing with {num_threads} threads...")
            
            start_time = time.time()
            
            # Run concurrent predictions
            def worker():
                return performance_analyzer.benchmark_ml_predictions(
                    test_papers[:2], num_iterations=2, top_k=5
                )
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker) for _ in range(num_threads * 2)]
                results = [future.result() for future in futures]
            
            end_time = time.time()
            
            # Calculate metrics
            successful_results = [r for r in results if r.success_rate > 0.5]
            total_throughput = sum(r.throughput for r in successful_results)
            avg_response_time = np.mean([r.execution_time for r in successful_results])
            
            scalability_results.append({
                'threads': num_threads,
                'total_throughput': total_throughput,
                'avg_response_time': avg_response_time,
                'success_count': len(successful_results),
                'total_time': end_time - start_time
            })
            
            print(f"   Total Throughput: {total_throughput:.2f} ops/sec")
            print(f"   Avg Response Time: {avg_response_time:.4f}s")
            print(f"   Successful Workers: {len(successful_results)}/{len(results)}")
        
        # Analyze scalability
        print(f"\n📈 Scalability Analysis:")
        for result in scalability_results:
            efficiency = result['total_throughput'] / result['threads']
            print(f"   {result['threads']} threads: {result['total_throughput']:.1f} ops/sec "
                  f"(efficiency: {efficiency:.1f} ops/sec/thread)")
        
        # Basic scalability assertions
        assert scalability_results[1]['total_throughput'] > scalability_results[0]['total_throughput']
        assert all(r['success_count'] > 0 for r in scalability_results)


class TestResourceMonitoring:
    """Tests for system resource monitoring capabilities."""
    
    @pytest.fixture
    def health_monitor(self):
        """Create SystemHealthMonitor for testing."""
        return SystemHealthMonitor()
    
    def test_system_resource_monitoring(self, health_monitor):
        """Test continuous system resource monitoring."""
        # Monitor resources over time
        monitoring_duration = 10  # seconds
        sample_interval = 1  # second
        
        resource_samples = []
        start_time = time.time()
        
        while time.time() - start_time < monitoring_duration:
            # Get current resource usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            sample = {
                'timestamp': time.time() - start_time,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available,
                'disk_percent': disk.percent,
            }
            resource_samples.append(sample)
            
            time.sleep(sample_interval)
        
        # Analyze resource patterns
        cpu_values = [s['cpu_percent'] for s in resource_samples]
        memory_values = [s['memory_percent'] for s in resource_samples]
        
        print(f"\n🖥️  Resource Monitoring Results ({monitoring_duration}s):")
        print(f"   CPU Usage - Avg: {np.mean(cpu_values):.1f}%, "
              f"Max: {np.max(cpu_values):.1f}%, Std: {np.std(cpu_values):.1f}%")
        print(f"   Memory Usage - Avg: {np.mean(memory_values):.1f}%, "
              f"Max: {np.max(memory_values):.1f}%, Std: {np.std(memory_values):.1f}%")
        
        # Resource stability checks
        cpu_stability = np.std(cpu_values) < 20  # CPU variation should be reasonable
        memory_stability = np.std(memory_values) < 5  # Memory should be relatively stable
        
        assert len(resource_samples) > 5, "Insufficient resource samples collected"
        assert max(memory_values) < 95, f"Memory usage too high: {max(memory_values):.1f}%"
        print(f"   CPU Stability: {'✅' if cpu_stability else '⚠️'}")
        print(f"   Memory Stability: {'✅' if memory_stability else '⚠️'}")
    
    def test_health_check_performance(self, health_monitor):
        """Test performance of health check operations."""
        # Measure health check performance
        check_times = []
        
        for _ in range(10):
            start_time = time.time()
            health_status = health_monitor.check_system_health()
            end_time = time.time()
            
            check_duration = end_time - start_time
            check_times.append(check_duration)
            
            assert isinstance(health_status, HealthStatus)
            assert health_status.status in ['healthy', 'warning', 'critical']
        
        avg_check_time = np.mean(check_times)
        max_check_time = np.max(check_times)
        
        print(f"\n🏥 Health Check Performance:")
        print(f"   Average Time: {avg_check_time:.4f}s")
        print(f"   Maximum Time: {max_check_time:.4f}s")
        print(f"   Standard Deviation: {np.std(check_times):.4f}s")
        
        # Performance assertions
        assert avg_check_time < 1.0, f"Health checks too slow: {avg_check_time:.4f}s"
        assert max_check_time < 2.0, f"Maximum health check too slow: {max_check_time:.4f}s"


class TestAnalyticsServicePerformance:
    """Performance tests for the main AnalyticsService."""
    
    @pytest.fixture
    def analytics_service(self):
        """Create AnalyticsService for testing."""
        mock_ml_service = Mock()
        mock_ml_service.health_check.return_value = {'status': 'healthy'}
        return AnalyticsService(ml_service=mock_ml_service)
    
    def test_network_analysis_performance(self, analytics_service):
        """Test network analysis performance with different sizes."""
        test_sizes = [10, 50, 100]
        
        for size in test_sizes:
            print(f"\n🏗️ Testing network analysis with {size} papers...")
            
            start_time = time.time()
            result = analytics_service.analyze_citation_network(
                max_papers=size,
                include_communities=True,
                include_centrality=True
            )
            end_time = time.time()
            
            analysis_time = end_time - start_time
            
            print(f"   Analysis Time: {analysis_time:.2f}s")
            
            if 'error' not in result:
                graph_info = result['graph_info']
                print(f"   Processed: {graph_info['num_nodes']} nodes, {graph_info['num_edges']} edges")
                print(f"   Performance: {graph_info['num_nodes'] / analysis_time:.1f} nodes/sec")
            
            # Performance assertions based on size
            if size <= 50:
                assert analysis_time < 5.0, f"Small network analysis too slow: {analysis_time:.2f}s"
            elif size <= 100:
                assert analysis_time < 15.0, f"Medium network analysis too slow: {analysis_time:.2f}s"
    
    def test_concurrent_analytics_operations(self, analytics_service):
        """Test concurrent analytics operations."""
        def network_analysis_worker():
            """Worker for network analysis."""
            return analytics_service.analyze_citation_network(
                max_papers=20,
                include_communities=False,  # Faster without communities
                include_centrality=False
            )
        
        def health_check_worker():
            """Worker for health checks."""
            return analytics_service.get_system_health()
        
        # Run concurrent operations
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit mixed workload
            futures = []
            futures.extend([executor.submit(network_analysis_worker) for _ in range(2)])
            futures.extend([executor.submit(health_check_worker) for _ in range(4)])
            
            results = [future.result() for future in futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        network_results = results[:2]
        health_results = results[2:]
        
        successful_network = sum(1 for r in network_results if 'error' not in r)
        successful_health = sum(1 for r in health_results if 'overall_health' in r)
        
        print(f"\n🔄 Concurrent Operations Results:")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Network Analyses: {successful_network}/{len(network_results)} successful")
        print(f"   Health Checks: {successful_health}/{len(health_results)} successful")
        print(f"   Overall Success Rate: {(successful_network + successful_health) / len(results):.2%}")
        
        # Assertions
        assert successful_network > 0, "No successful network analyses"
        assert successful_health > 0, "No successful health checks"
        assert total_time < 30.0, f"Concurrent operations too slow: {total_time:.2f}s"
    
    @pytest.mark.slow
    def test_workflow_performance(self, analytics_service):
        """Test analytics workflow performance."""
        # Create test workflow
        tasks = [
            {'type': 'network_analysis', 'parameters': {'max_papers': 50}},
            {'type': 'performance_benchmark', 'parameters': {'benchmark_types': ['ml']}}
        ]
        
        workflow = analytics_service.create_analytics_workflow(
            "Performance Test Workflow",
            "Workflow for performance testing",
            tasks
        )
        
        # Execute workflow
        start_time = time.time()
        result = analytics_service.execute_workflow(workflow.workflow_id)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        print(f"\n⚙️ Workflow Performance:")
        print(f"   Workflow ID: {workflow.workflow_id}")
        print(f"   Execution Time: {execution_time:.2f}s")
        print(f"   Status: {result['status']}")
        print(f"   Tasks: {len(workflow.tasks)}")
        
        # Assertions
        assert result['status'] in ['completed', 'failed']
        assert execution_time < 60.0, f"Workflow execution too slow: {execution_time:.2f}s"
        
        if result['status'] == 'completed':
            assert 'results' in result
            assert len(result['results']) == len(workflow.tasks)


class TestExportPerformance:
    """Performance tests for export engine."""
    
    @pytest.fixture
    def export_engine(self, tmp_path):
        """Create ExportEngine for testing."""
        return ExportEngine(tmp_path)
    
    def test_large_export_performance(self, export_engine):
        """Test export performance with large datasets."""
        # Create large dataset
        from src.analytics.network_analysis import NetworkMetrics, CentralityMetrics, CommunityInfo
        
        # Large network metrics
        network_metrics = NetworkMetrics(
            num_nodes=10000,
            num_edges=50000,
            density=0.001,
            average_degree=10.0,
            clustering_coefficient=0.15,
            diameter=8,
            average_path_length=4.2,
            num_components=1,
            largest_component_size=10000,
            modularity=0.45,
            assortativity=0.2
        )
        
        # Large centrality dataset
        centrality_metrics = {}
        for i in range(1000):  # 1000 papers
            centrality_metrics[f"paper_{i}"] = CentralityMetrics(
                paper_id=f"paper_{i}",
                degree_centrality=np.random.random(),
                betweenness_centrality=np.random.random(),
                closeness_centrality=np.random.random(),
                eigenvector_centrality=np.random.random(),
                pagerank=np.random.random() / 100
            )
        
        # Many communities
        communities = []
        for i in range(100):  # 100 communities
            size = np.random.randint(10, 100)
            members = [f"paper_{j}" for j in range(i*10, i*10 + size)]
            communities.append(CommunityInfo(
                community_id=i,
                size=size,
                members=members,
                modularity=0.3,
                internal_edges=size * 2,
                external_edges=size // 2,
                conductance=0.2
            ))
        
        # Test different export formats
        formats_to_test = ['json', 'html', 'csv']
        
        for fmt in formats_to_test:
            print(f"\n📁 Testing {fmt.upper()} export...")
            
            config = ExportConfiguration(format=fmt, include_visualizations=False)
            
            start_time = time.time()
            result = export_engine.export_network_analysis(
                network_metrics, centrality_metrics, communities, config
            )
            end_time = time.time()
            
            export_time = end_time - start_time
            
            print(f"   Export Time: {export_time:.2f}s")
            
            if result.success:
                file_size = result.file_size / 1024 / 1024  # MB
                print(f"   File Size: {file_size:.2f} MB")
                print(f"   Export Rate: {file_size / export_time:.2f} MB/s")
                
                # Performance assertions
                assert export_time < 30.0, f"{fmt} export too slow: {export_time:.2f}s"
                assert file_size < 100.0, f"{fmt} file too large: {file_size:.2f} MB"
            else:
                pytest.fail(f"Export failed for {fmt}: {result.error_message}")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])