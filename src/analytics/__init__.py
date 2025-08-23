"""
Advanced analytics module for the Academic Citation Platform.

This module provides sophisticated analytical capabilities including:
- Network analysis with centrality measures and community detection
- Temporal analysis of citation patterns and trends
- Performance metrics and system health monitoring
- Multi-format export capabilities for reports and visualizations

The analytics framework is designed to work seamlessly with the existing
ML service and data pipeline, providing production-ready analytical
capabilities for citation network research.
"""

from .network_analysis import NetworkAnalyzer, CommunityDetector
from .temporal_analysis import TemporalAnalyzer, TrendAnalyzer
from .performance_metrics import PerformanceAnalyzer, SystemHealthMonitor
from .export_engine import ExportEngine, ReportGenerator

__version__ = "1.0.0"

__all__ = [
    "NetworkAnalyzer",
    "CommunityDetector", 
    "TemporalAnalyzer",
    "TrendAnalyzer",
    "PerformanceAnalyzer",
    "SystemHealthMonitor",
    "ExportEngine",
    "ReportGenerator"
]