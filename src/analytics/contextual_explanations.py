"""
Contextual explanations system for research insights and analytics interpretation.

Provides academic context, benchmarking, and interpretations for all metrics
displayed in the Academic Citation Platform.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics


class MetricCategory(Enum):
    """Categories of metrics for contextual explanations."""
    NETWORK_STRUCTURE = "network_structure"
    CITATION_PREDICTION = "citation_prediction"
    EMBEDDING_QUALITY = "embedding_quality"
    SYSTEM_PERFORMANCE = "system_performance"
    ACADEMIC_IMPACT = "academic_impact"


class PerformanceLevel(Enum):
    """Performance levels for traffic light system."""
    EXCELLENT = "excellent"  # 🟢
    GOOD = "good"           # 🟢  
    FAIR = "fair"           # 🟡
    POOR = "poor"           # 🔴


@dataclass
class MetricBenchmark:
    """Academic benchmark thresholds for a specific metric."""
    metric_name: str
    domain: str  # e.g., "computer_science", "biology", "physics"
    excellent_threshold: float
    good_threshold: float
    fair_threshold: float
    poor_threshold: float
    unit: str = ""
    higher_is_better: bool = True
    academic_sources: List[str] = field(default_factory=list)


@dataclass
class MetricExplanation:
    """Complete contextual explanation for a metric."""
    metric_name: str
    value: float
    performance_level: PerformanceLevel
    performance_icon: str
    short_description: str
    detailed_explanation: str
    academic_context: str
    typical_range_text: str
    interpretation_guide: str
    suggested_actions: List[str] = field(default_factory=list)
    confidence_interval: Optional[Tuple[float, float]] = None
    benchmark_comparison: Optional[str] = None


class ContextualExplanationEngine:
    """
    Engine for generating contextual explanations and academic benchmarking.
    
    Core component that transforms raw metrics into actionable research insights
    with academic context and performance benchmarking.
    """
    
    def __init__(self):
        """Initialize the contextual explanation engine."""
        self.logger = logging.getLogger(__name__)
        self._init_benchmarks()
    
    def _init_benchmarks(self):
        """Initialize academic benchmarks from literature."""
        # Citation Prediction Benchmarks (from academic literature)
        self.benchmarks = {
            # Computer Science domain
            "hits_at_10_cs": MetricBenchmark(
                metric_name="hits_at_10",
                domain="computer_science",
                excellent_threshold=0.35,
                good_threshold=0.25,
                fair_threshold=0.15,
                poor_threshold=0.10,
                unit="proportion",
                academic_sources=[
                    "Knowledge Graph Embedding: A Survey of Approaches and Applications (2017)",
                    "Citation Recommendation with Neural Networks (2019)"
                ]
            ),
            
            "mrr_cs": MetricBenchmark(
                metric_name="mrr",
                domain="computer_science", 
                excellent_threshold=0.20,
                good_threshold=0.15,
                fair_threshold=0.10,
                poor_threshold=0.05,
                unit="proportion",
                academic_sources=[
                    "Evaluating Knowledge Graph Embeddings (2018)",
                    "Neural Citation Recommendation Systems (2020)"
                ]
            ),
            
            "auc_cs": MetricBenchmark(
                metric_name="auc",
                domain="computer_science",
                excellent_threshold=0.95,
                good_threshold=0.90,
                fair_threshold=0.85,
                poor_threshold=0.80,
                unit="proportion",
                academic_sources=[
                    "Machine Learning for Citation Prediction (2019)"
                ]
            ),
            
            # Network Analysis Benchmarks
            "modularity_cs": MetricBenchmark(
                metric_name="modularity",
                domain="computer_science",
                excellent_threshold=0.70,
                good_threshold=0.50,
                fair_threshold=0.30,
                poor_threshold=0.10,
                unit="modularity score",
                academic_sources=[
                    "Community Detection in Academic Citation Networks (2018)",
                    "Modularity and Network Structure Analysis (2016)"
                ]
            ),
            
            "network_density_cs": MetricBenchmark(
                metric_name="network_density",
                domain="computer_science",
                excellent_threshold=0.010,
                good_threshold=0.005,
                fair_threshold=0.002,
                poor_threshold=0.001,
                unit="proportion",
                academic_sources=[
                    "Citation Network Topology in Academic Fields (2020)"
                ]
            ),
            
            # Biology domain benchmarks
            "hits_at_10_bio": MetricBenchmark(
                metric_name="hits_at_10",
                domain="biology",
                excellent_threshold=0.30,
                good_threshold=0.22,
                fair_threshold=0.14,
                poor_threshold=0.08,
                unit="proportion"
            ),
            
            # General benchmarks
            "response_time": MetricBenchmark(
                metric_name="response_time",
                domain="general",
                excellent_threshold=0.5,
                good_threshold=1.0,
                fair_threshold=2.0,
                poor_threshold=5.0,
                unit="seconds",
                higher_is_better=False
            )
        }
        
        # Metric categories and base explanations
        self.metric_descriptions = {
            "hits_at_10": {
                "category": MetricCategory.CITATION_PREDICTION,
                "short": "Proportion of correct predictions in top-10 results",
                "detailed": "Hits@10 measures how often the true citation appears in the top-10 predictions. A score of 0.25 means the correct citation is found in the top-10 about 25% of the time.",
                "interpretation": "Higher values indicate better prediction accuracy. This metric is crucial for recommendation systems where users typically only consider top results."
            },
            
            "mrr": {
                "category": MetricCategory.CITATION_PREDICTION,
                "short": "Mean Reciprocal Rank - average rank quality of correct predictions",
                "detailed": "MRR computes the average of reciprocal ranks of correct predictions. If the true citation ranks 1st, MRR gets 1.0; if 2nd, it gets 0.5; if 3rd, it gets 0.33, etc.",
                "interpretation": "Higher MRR indicates predictions are not just accurate but also highly ranked. Essential for systems where rank order matters significantly."
            },
            
            "auc": {
                "category": MetricCategory.CITATION_PREDICTION,
                "short": "Area Under ROC Curve - overall classification performance",
                "detailed": "AUC measures the model's ability to distinguish between positive (actual citations) and negative (non-citations) examples across all classification thresholds.",
                "interpretation": "Values closer to 1.0 indicate excellent discrimination ability. AUC = 0.5 means random performance, while AUC > 0.9 is considered excellent."
            },
            
            "modularity": {
                "category": MetricCategory.NETWORK_STRUCTURE,
                "short": "Community structure strength in citation networks",
                "detailed": "Modularity measures how well a network divides into distinct communities. High modularity indicates strong internal connections within communities and weak connections between them.",
                "interpretation": "Values > 0.7 suggest well-defined research communities. Essential for understanding field structure and identifying research clusters."
            },
            
            "network_density": {
                "category": MetricCategory.NETWORK_STRUCTURE,
                "short": "Proportion of actual connections vs. possible connections",
                "detailed": "Network density = actual edges / possible edges. In citation networks, this indicates how interconnected papers are within the analyzed corpus.",
                "interpretation": "Academic networks are typically sparse (density < 0.01) because papers cite only a small fraction of available literature."
            },
            
            "response_time": {
                "category": MetricCategory.SYSTEM_PERFORMANCE,
                "short": "Time taken to process requests",
                "detailed": "Average time from request initiation to response completion, including model inference and data retrieval.",
                "interpretation": "Lower response times improve user experience. Sub-second responses are excellent for interactive applications."
            }
        }
    
    def explain_metric(self, 
                      metric_name: str, 
                      value: float, 
                      domain: str = "computer_science",
                      context: Dict[str, Any] = None) -> MetricExplanation:
        """
        Generate comprehensive contextual explanation for a metric.
        
        Args:
            metric_name: Name of the metric to explain
            value: Actual metric value
            domain: Academic domain (e.g., "computer_science", "biology")
            context: Additional context information
            
        Returns:
            MetricExplanation with full contextual information
        """
        context = context or {}
        
        # Get benchmark for this metric and domain
        benchmark_key = f"{metric_name}_{domain}" if f"{metric_name}_{domain}" in self.benchmarks else metric_name
        benchmark = self.benchmarks.get(benchmark_key)
        
        # Get base description
        description = self.metric_descriptions.get(metric_name, {
            "short": f"Metric: {metric_name}",
            "detailed": f"Analysis result for {metric_name}",
            "interpretation": "No specific interpretation available."
        })
        
        # Determine performance level
        performance_level, performance_icon = self._assess_performance(value, benchmark)
        
        # Generate detailed explanation
        detailed_explanation = self._generate_detailed_explanation(
            metric_name, value, benchmark, description, context
        )
        
        # Generate academic context
        academic_context = self._generate_academic_context(metric_name, benchmark, domain)
        
        # Generate typical range text
        typical_range_text = self._generate_range_text(benchmark, domain)
        
        # Generate interpretation guide
        interpretation_guide = self._generate_interpretation_guide(
            metric_name, value, performance_level, benchmark
        )
        
        # Generate suggested actions
        suggested_actions = self._generate_suggested_actions(
            metric_name, value, performance_level, context
        )
        
        # Generate benchmark comparison
        benchmark_comparison = self._generate_benchmark_comparison(value, benchmark, domain)
        
        return MetricExplanation(
            metric_name=metric_name,
            value=value,
            performance_level=performance_level,
            performance_icon=performance_icon,
            short_description=description["short"],
            detailed_explanation=detailed_explanation,
            academic_context=academic_context,
            typical_range_text=typical_range_text,
            interpretation_guide=interpretation_guide,
            suggested_actions=suggested_actions,
            benchmark_comparison=benchmark_comparison
        )
    
    def _assess_performance(self, value: float, benchmark: Optional[MetricBenchmark]) -> Tuple[PerformanceLevel, str]:
        """Assess performance level and return appropriate icon."""
        if not benchmark:
            return PerformanceLevel.FAIR, "🟡"
        
        if benchmark.higher_is_better:
            if value >= benchmark.excellent_threshold:
                return PerformanceLevel.EXCELLENT, "🟢"
            elif value >= benchmark.good_threshold:
                return PerformanceLevel.GOOD, "🟢"
            elif value >= benchmark.fair_threshold:
                return PerformanceLevel.FAIR, "🟡"
            else:
                return PerformanceLevel.POOR, "🔴"
        else:
            # Lower is better (e.g., response time)
            if value <= benchmark.excellent_threshold:
                return PerformanceLevel.EXCELLENT, "🟢"
            elif value <= benchmark.good_threshold:
                return PerformanceLevel.GOOD, "🟢"
            elif value <= benchmark.fair_threshold:
                return PerformanceLevel.FAIR, "🟡"
            else:
                return PerformanceLevel.POOR, "🔴"
    
    def _generate_detailed_explanation(self, 
                                     metric_name: str, 
                                     value: float,
                                     benchmark: Optional[MetricBenchmark],
                                     description: Dict[str, str],
                                     context: Dict[str, Any]) -> str:
        """Generate detailed explanation combining value and context."""
        base_explanation = description.get("detailed", "")
        
        # Add value-specific context
        if metric_name == "hits_at_10":
            percentage = value * 100
            explanation = f"Your model achieves {percentage:.1f}% Hits@10, meaning it correctly identifies the true citation in the top-10 predictions about {int(percentage)} times out of 100."
            
        elif metric_name == "mrr":
            explanation = f"Your MRR score of {value:.3f} indicates that on average, correct citations appear at rank {1/value:.1f} in your prediction lists."
            
        elif metric_name == "auc":
            percentage = value * 100
            explanation = f"Your AUC score of {value:.3f} ({percentage:.1f}%) shows {'excellent' if value > 0.9 else 'good' if value > 0.8 else 'fair'} discrimination between actual and non-citations."
            
        elif metric_name == "modularity":
            explanation = f"Modularity of {value:.3f} indicates {'strong' if value > 0.7 else 'moderate' if value > 0.5 else 'weak'} community structure in your citation network."
            
        elif metric_name == "network_density":
            percentage = value * 100
            explanation = f"Network density of {value:.3f} ({percentage:.1f}%) means papers cite {percentage:.1f}% of available literature - {'typical' if value < 0.01 else 'high'} for academic networks."
            
        else:
            explanation = f"{base_explanation} Current value: {value}"
        
        return explanation
    
    def _generate_academic_context(self, 
                                  metric_name: str, 
                                  benchmark: Optional[MetricBenchmark],
                                  domain: str) -> str:
        """Generate academic context and literature grounding."""
        if not benchmark or not benchmark.academic_sources:
            return f"This metric is commonly used in {domain.replace('_', ' ')} citation analysis."
        
        sources_text = "; ".join(benchmark.academic_sources[:2])  # Limit to first 2 sources
        
        context = f"Based on research in {domain.replace('_', ' ')}, this metric is well-established for citation analysis. "
        context += f"Key references: {sources_text}"
        
        return context
    
    def _generate_range_text(self, benchmark: Optional[MetricBenchmark], domain: str) -> str:
        """Generate text describing typical ranges for this metric."""
        if not benchmark:
            return f"Typical ranges vary by domain and dataset in {domain.replace('_', ' ')}."
        
        range_text = f"Typical {domain.replace('_', ' ')} ranges: "
        range_text += f"Excellent (>{benchmark.excellent_threshold:.2f}), "
        range_text += f"Good (>{benchmark.good_threshold:.2f}), "
        range_text += f"Fair (>{benchmark.fair_threshold:.2f}), "
        range_text += f"Poor (<{benchmark.fair_threshold:.2f})"
        
        if benchmark.unit:
            range_text += f" [{benchmark.unit}]"
        
        return range_text
    
    def _generate_interpretation_guide(self,
                                     metric_name: str,
                                     value: float,
                                     performance_level: PerformanceLevel,
                                     benchmark: Optional[MetricBenchmark]) -> str:
        """Generate interpretation guide for the current result."""
        level_descriptions = {
            PerformanceLevel.EXCELLENT: "Outstanding performance - significantly above typical academic standards",
            PerformanceLevel.GOOD: "Good performance - meets or exceeds typical academic standards", 
            PerformanceLevel.FAIR: "Fair performance - below typical standards but acceptable for some applications",
            PerformanceLevel.POOR: "Poor performance - significantly below academic standards, improvement needed"
        }
        
        base_interpretation = level_descriptions[performance_level]
        
        # Add metric-specific interpretation
        if metric_name == "hits_at_10" and performance_level == PerformanceLevel.GOOD:
            specific = " This level of accuracy is suitable for citation recommendation systems."
        elif metric_name == "modularity" and performance_level == PerformanceLevel.EXCELLENT:
            specific = " Strong community structure enables effective research area analysis."
        elif metric_name == "response_time" and performance_level == PerformanceLevel.EXCELLENT:
            specific = " Excellent responsiveness for interactive research applications."
        else:
            specific = ""
        
        return base_interpretation + specific
    
    def _generate_suggested_actions(self,
                                  metric_name: str,
                                  value: float,
                                  performance_level: PerformanceLevel,
                                  context: Dict[str, Any]) -> List[str]:
        """Generate actionable suggestions based on performance."""
        actions = []
        
        if performance_level == PerformanceLevel.POOR:
            if metric_name in ["hits_at_10", "mrr", "auc"]:
                actions.extend([
                    "Consider retraining the model with more data",
                    "Experiment with different embedding dimensions", 
                    "Review data preprocessing and feature engineering",
                    "Compare with simpler baseline models"
                ])
            elif metric_name == "modularity":
                actions.extend([
                    "Consider different community detection algorithms",
                    "Check if the network has sufficient structure for analysis",
                    "Verify data quality and completeness"
                ])
        
        elif performance_level == PerformanceLevel.FAIR:
            if metric_name in ["hits_at_10", "mrr"]:
                actions.extend([
                    "Fine-tune hyperparameters for better performance",
                    "Consider ensemble methods",
                    "Evaluate on domain-specific test sets"
                ])
        
        elif performance_level in [PerformanceLevel.GOOD, PerformanceLevel.EXCELLENT]:
            actions.extend([
                "Results are ready for academic use",
                "Consider publishing methodology and results",
                "Use insights for research planning and collaboration"
            ])
        
        # Add context-specific actions
        if context.get("num_entities", 0) < 1000:
            actions.append("Consider expanding dataset size for more robust results")
        
        return actions[:4]  # Limit to top 4 actions
    
    def _generate_benchmark_comparison(self,
                                     value: float,
                                     benchmark: Optional[MetricBenchmark],
                                     domain: str) -> Optional[str]:
        """Generate comparison against academic benchmarks."""
        if not benchmark:
            return None
        
        if benchmark.higher_is_better:
            if value >= benchmark.excellent_threshold:
                percentile = "top 10%"
            elif value >= benchmark.good_threshold:
                percentile = "top 25%"
            elif value >= benchmark.fair_threshold:
                percentile = "median"
            else:
                percentile = "bottom 25%"
        else:
            # Lower is better
            if value <= benchmark.excellent_threshold:
                percentile = "top 10%"
            elif value <= benchmark.good_threshold:
                percentile = "top 25%"  
            elif value <= benchmark.fair_threshold:
                percentile = "median"
            else:
                percentile = "bottom 25%"
        
        return f"Your result places in the {percentile} of typical {domain.replace('_', ' ')} studies"

    def get_metric_categories(self) -> Dict[MetricCategory, List[str]]:
        """Get metrics organized by category."""
        categories = {}
        for metric_name, info in self.metric_descriptions.items():
            category = info.get("category", MetricCategory.SYSTEM_PERFORMANCE)
            if category not in categories:
                categories[category] = []
            categories[category].append(metric_name)
        return categories
    
    def bulk_explain_metrics(self, 
                           metrics: Dict[str, float],
                           domain: str = "computer_science",
                           context: Dict[str, Any] = None) -> Dict[str, MetricExplanation]:
        """Generate explanations for multiple metrics at once."""
        explanations = {}
        for metric_name, value in metrics.items():
            try:
                explanations[metric_name] = self.explain_metric(
                    metric_name, value, domain, context
                )
            except Exception as e:
                self.logger.warning(f"Failed to explain metric {metric_name}: {e}")
        
        return explanations