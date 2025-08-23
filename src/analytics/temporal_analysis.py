"""
Temporal analysis module for citation pattern analysis over time.

Provides comprehensive temporal analysis capabilities including citation trends,
growth patterns, impact evolution, and time-series analysis of citation networks.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from ..models.paper import Paper
from ..models.citation import Citation


@dataclass
class TimeSeriesPoint:
    """Single point in a time series."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = None


@dataclass
class TrendAnalysis:
    """Results of trend analysis."""
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0-1, strength of the trend
    growth_rate: float  # Annual growth rate
    seasonal_component: Optional[List[float]] = None
    trend_component: Optional[List[float]] = None


@dataclass
class CitationGrowthMetrics:
    """Metrics for citation growth analysis."""
    paper_id: str
    publication_year: int
    total_citations: int
    citations_per_year: Dict[int, int]
    peak_citation_year: int
    years_to_peak: int
    half_life: Optional[float]  # Years to half maximum citations
    impact_factor: float  # Citations per year since publication


@dataclass
class TemporalNetworkMetrics:
    """Network metrics over time."""
    timestamp: datetime
    num_nodes: int
    num_edges: int
    density: float
    average_degree: float
    clustering_coefficient: float
    largest_component_size: int


class TemporalAnalyzer:
    """
    Temporal analysis for citation patterns and trends.
    
    Analyzes how citation patterns evolve over time, including growth trends,
    seasonality, and temporal network properties.
    """
    
    def __init__(self):
        """Initialize temporal analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_citation_growth(self, 
                               papers: List[Paper],
                               citations: List[Citation],
                               current_year: Optional[int] = None) -> List[CitationGrowthMetrics]:
        """
        Analyze citation growth patterns for papers.
        
        Args:
            papers: List of papers to analyze
            citations: List of citations
            current_year: Current year (default: current year)
            
        Returns:
            List of CitationGrowthMetrics for each paper
        """
        if current_year is None:
            current_year = datetime.now().year
        
        self.logger.info(f"Analyzing citation growth for {len(papers)} papers")
        
        # Build citation mapping
        paper_citations = defaultdict(list)
        for citation in citations:
            paper_citations[citation.target_paper_id].append(citation)
        
        growth_metrics = []
        
        for paper in papers:
            paper_id = paper.paper_id
            pub_year = getattr(paper, 'publication_year', current_year)
            
            if isinstance(pub_year, str):
                try:
                    pub_year = int(pub_year)
                except ValueError:
                    pub_year = current_year
            
            paper_citation_list = paper_citations.get(paper_id, [])
            
            # Count citations by year
            citations_by_year = defaultdict(int)
            for citation in paper_citation_list:
                # Try to extract year from citation
                citation_year = current_year  # Default
                
                if hasattr(citation, 'citation_date') and citation.citation_date:
                    if isinstance(citation.citation_date, datetime):
                        citation_year = citation.citation_date.year
                    elif isinstance(citation.citation_date, str):
                        try:
                            citation_year = datetime.fromisoformat(citation.citation_date.replace('Z', '+00:00')).year
                        except:
                            pass
                
                citations_by_year[citation_year] += 1
            
            # Calculate metrics
            total_citations = len(paper_citation_list)
            peak_year = max(citations_by_year.keys()) if citations_by_year else pub_year
            peak_citations = max(citations_by_year.values()) if citations_by_year else 0
            years_to_peak = max(0, peak_year - pub_year)
            
            # Calculate impact factor (citations per year since publication)
            years_since_pub = max(1, current_year - pub_year + 1)
            impact_factor = total_citations / years_since_pub
            
            # Estimate half-life (simplified calculation)
            half_life = None
            if peak_citations > 0:
                half_target = peak_citations / 2
                for year in sorted(citations_by_year.keys(), reverse=True):
                    if citations_by_year[year] <= half_target:
                        half_life = max(0.1, year - peak_year)
                        break
            
            growth_metrics.append(CitationGrowthMetrics(
                paper_id=paper_id,
                publication_year=pub_year,
                total_citations=total_citations,
                citations_per_year=dict(citations_by_year),
                peak_citation_year=peak_year,
                years_to_peak=years_to_peak,
                half_life=half_life,
                impact_factor=impact_factor
            ))
        
        self.logger.info(f"Calculated growth metrics for {len(growth_metrics)} papers")
        return growth_metrics
    
    def create_citation_time_series(self,
                                   citations: List[Citation],
                                   freq: str = 'M') -> List[TimeSeriesPoint]:
        """
        Create time series of citation counts.
        
        Args:
            citations: List of citations
            freq: Frequency for aggregation ('D', 'W', 'M', 'Y')
            
        Returns:
            List of time series points
        """
        self.logger.info(f"Creating citation time series with frequency {freq}")
        
        # Extract timestamps from citations
        timestamps = []
        for citation in citations:
            if hasattr(citation, 'citation_date') and citation.citation_date:
                if isinstance(citation.citation_date, datetime):
                    timestamps.append(citation.citation_date)
                elif isinstance(citation.citation_date, str):
                    try:
                        ts = datetime.fromisoformat(citation.citation_date.replace('Z', '+00:00'))
                        timestamps.append(ts)
                    except:
                        timestamps.append(datetime.now())
                else:
                    timestamps.append(datetime.now())
            else:
                timestamps.append(datetime.now())
        
        if not timestamps:
            return []
        
        # Create DataFrame for easier aggregation
        df = pd.DataFrame({'timestamp': timestamps, 'count': 1})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Aggregate by frequency
        if freq == 'D':
            df_grouped = df.groupby(df['timestamp'].dt.date)['count'].sum()
        elif freq == 'W':
            df_grouped = df.groupby(df['timestamp'].dt.to_period('W'))['count'].sum()
        elif freq == 'M':
            df_grouped = df.groupby(df['timestamp'].dt.to_period('M'))['count'].sum()
        elif freq == 'Y':
            df_grouped = df.groupby(df['timestamp'].dt.to_period('Y'))['count'].sum()
        else:
            df_grouped = df.groupby(df['timestamp'].dt.date)['count'].sum()
        
        # Convert to TimeSeriesPoint objects
        time_series = []
        for period, count in df_grouped.items():
            if hasattr(period, 'start_time'):
                timestamp = period.start_time
            elif isinstance(period, str):
                timestamp = pd.to_datetime(period)
            else:
                timestamp = pd.to_datetime(str(period))
            
            time_series.append(TimeSeriesPoint(
                timestamp=timestamp,
                value=float(count)
            ))
        
        # Sort by timestamp
        time_series.sort(key=lambda x: x.timestamp)
        
        self.logger.info(f"Created time series with {len(time_series)} points")
        return time_series
    
    def analyze_seasonal_patterns(self, 
                                 time_series: List[TimeSeriesPoint]) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in citation data.
        
        Args:
            time_series: Time series data
            
        Returns:
            Dictionary with seasonal analysis results
        """
        self.logger.info("Analyzing seasonal patterns")
        
        if len(time_series) < 12:  # Need at least a year of data
            return {"error": "Insufficient data for seasonal analysis"}
        
        # Convert to pandas series
        timestamps = [point.timestamp for point in time_series]
        values = [point.value for point in time_series]
        
        df = pd.DataFrame({'timestamp': timestamps, 'value': values})
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        
        # Monthly aggregation for seasonal analysis
        monthly = df.resample('M').sum()
        
        # Calculate seasonal statistics
        monthly_stats = {}
        for month in range(1, 13):
            month_data = monthly[monthly.index.month == month]
            if len(month_data) > 0:
                monthly_stats[month] = {
                    'mean': month_data['value'].mean(),
                    'std': month_data['value'].std(),
                    'median': month_data['value'].median()
                }
        
        # Identify peak months
        peak_month = max(monthly_stats.items(), 
                        key=lambda x: x[1]['mean'])[0] if monthly_stats else None
        
        # Calculate seasonal variation coefficient
        if monthly_stats:
            monthly_means = [stats['mean'] for stats in monthly_stats.values()]
            seasonal_variation = np.std(monthly_means) / np.mean(monthly_means)
        else:
            seasonal_variation = 0.0
        
        return {
            'seasonal_variation_coefficient': seasonal_variation,
            'peak_month': peak_month,
            'monthly_statistics': monthly_stats,
            'has_strong_seasonality': seasonal_variation > 0.3
        }
    
    def detect_citation_bursts(self,
                              time_series: List[TimeSeriesPoint],
                              threshold_multiplier: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect sudden bursts in citation activity.
        
        Args:
            time_series: Time series data
            threshold_multiplier: Multiplier for burst detection threshold
            
        Returns:
            List of detected bursts with start/end times and intensity
        """
        self.logger.info("Detecting citation bursts")
        
        if len(time_series) < 5:
            return []
        
        values = [point.value for point in time_series]
        timestamps = [point.timestamp for point in time_series]
        
        # Calculate rolling statistics
        window_size = min(7, len(values) // 3)
        
        bursts = []
        
        for i in range(window_size, len(values)):
            # Calculate baseline (average of previous window)
            baseline = np.mean(values[i-window_size:i])
            current_value = values[i]
            
            # Detect burst
            if baseline > 0 and current_value > baseline * threshold_multiplier:
                # Find burst duration
                burst_start = i
                burst_end = i
                
                # Extend burst while values remain high
                for j in range(i+1, len(values)):
                    if values[j] > baseline * 1.5:  # Lower threshold for continuation
                        burst_end = j
                    else:
                        break
                
                burst_intensity = np.mean(values[burst_start:burst_end+1]) / baseline
                
                bursts.append({
                    'start_time': timestamps[burst_start],
                    'end_time': timestamps[burst_end],
                    'duration_points': burst_end - burst_start + 1,
                    'intensity': burst_intensity,
                    'peak_value': max(values[burst_start:burst_end+1]),
                    'baseline': baseline
                })
        
        # Remove overlapping bursts (keep the most intense)
        filtered_bursts = []
        for burst in sorted(bursts, key=lambda x: x['intensity'], reverse=True):
            # Check for overlap with existing bursts
            overlaps = False
            for existing in filtered_bursts:
                if (burst['start_time'] <= existing['end_time'] and 
                    burst['end_time'] >= existing['start_time']):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_bursts.append(burst)
        
        self.logger.info(f"Detected {len(filtered_bursts)} citation bursts")
        return filtered_bursts


class TrendAnalyzer:
    """
    Advanced trend analysis for citation data.
    
    Provides sophisticated trend detection and analysis capabilities
    including linear and non-linear trend fitting.
    """
    
    def __init__(self):
        """Initialize trend analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_trend(self, 
                     time_series: List[TimeSeriesPoint]) -> TrendAnalysis:
        """
        Analyze overall trend in time series data.
        
        Args:
            time_series: Time series data
            
        Returns:
            TrendAnalysis object with trend characteristics
        """
        self.logger.info("Analyzing trend in time series")
        
        if len(time_series) < 3:
            return TrendAnalysis(
                trend_direction='stable',
                trend_strength=0.0,
                growth_rate=0.0
            )
        
        # Convert to numerical data
        values = np.array([point.value for point in time_series])
        timestamps = [point.timestamp for point in time_series]
        
        # Convert timestamps to days since first point
        first_time = timestamps[0]
        x_days = np.array([(ts - first_time).total_seconds() / (24 * 3600) 
                          for ts in timestamps])
        
        # Linear regression for trend
        if len(x_days) > 1:
            slope, intercept = np.polyfit(x_days, values, 1)
            
            # Calculate correlation coefficient (trend strength)
            correlation = np.corrcoef(x_days, values)[0, 1]
            trend_strength = abs(correlation) if not np.isnan(correlation) else 0.0
            
            # Determine trend direction
            if abs(slope) < 0.01 * np.mean(values):  # Very small slope
                trend_direction = 'stable'
            elif slope > 0:
                trend_direction = 'increasing'
            else:
                trend_direction = 'decreasing'
            
            # Calculate annual growth rate
            if len(x_days) > 0 and x_days[-1] > 0:
                days_span = x_days[-1]
                years_span = days_span / 365.25
                if years_span > 0 and intercept > 0:
                    final_value = slope * x_days[-1] + intercept
                    growth_rate = ((final_value / intercept) ** (1/years_span)) - 1
                else:
                    growth_rate = 0.0
            else:
                growth_rate = 0.0
        else:
            trend_direction = 'stable'
            trend_strength = 0.0
            growth_rate = 0.0
        
        return TrendAnalysis(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            growth_rate=growth_rate
        )
    
    def compare_growth_rates(self, 
                           paper_metrics: List[CitationGrowthMetrics]) -> Dict[str, Any]:
        """
        Compare growth rates across different papers.
        
        Args:
            paper_metrics: List of citation growth metrics
            
        Returns:
            Dictionary with comparative growth analysis
        """
        self.logger.info(f"Comparing growth rates for {len(paper_metrics)} papers")
        
        if not paper_metrics:
            return {}
        
        # Extract impact factors
        impact_factors = [metric.impact_factor for metric in paper_metrics]
        years_to_peak = [metric.years_to_peak for metric in paper_metrics 
                        if metric.years_to_peak is not None]
        
        # Calculate statistics
        analysis = {
            'total_papers': len(paper_metrics),
            'impact_factor_stats': {
                'mean': np.mean(impact_factors),
                'median': np.median(impact_factors),
                'std': np.std(impact_factors),
                'min': np.min(impact_factors),
                'max': np.max(impact_factors)
            }
        }
        
        if years_to_peak:
            analysis['years_to_peak_stats'] = {
                'mean': np.mean(years_to_peak),
                'median': np.median(years_to_peak),
                'std': np.std(years_to_peak),
                'min': np.min(years_to_peak),
                'max': np.max(years_to_peak)
            }
        
        # Identify top performers
        top_impact = sorted(paper_metrics, 
                          key=lambda x: x.impact_factor, 
                          reverse=True)[:10]
        
        analysis['top_impact_papers'] = [
            {
                'paper_id': metric.paper_id,
                'impact_factor': metric.impact_factor,
                'total_citations': metric.total_citations,
                'publication_year': metric.publication_year
            }
            for metric in top_impact
        ]
        
        return analysis
    
    def project_future_citations(self,
                                growth_metrics: CitationGrowthMetrics,
                                years_ahead: int = 5) -> Dict[int, float]:
        """
        Project future citation counts based on historical patterns.
        
        Args:
            growth_metrics: Historical growth data
            years_ahead: Number of years to project
            
        Returns:
            Dictionary mapping future years to projected citation counts
        """
        self.logger.info(f"Projecting citations {years_ahead} years ahead")
        
        citations_by_year = growth_metrics.citations_per_year
        current_year = datetime.now().year
        
        if len(citations_by_year) < 2:
            # Not enough data for projection
            return {current_year + i: 0.0 for i in range(1, years_ahead + 1)}
        
        # Get recent years for trend calculation
        recent_years = sorted(citations_by_year.keys())[-3:]  # Last 3 years
        recent_citations = [citations_by_year[year] for year in recent_years]
        
        # Simple linear projection based on recent trend
        if len(recent_citations) >= 2:
            x = np.array(range(len(recent_citations)))
            y = np.array(recent_citations)
            slope, intercept = np.polyfit(x, y, 1)
            
            projections = {}
            for i in range(1, years_ahead + 1):
                future_year = current_year + i
                # Project based on trend
                projected_value = slope * (len(recent_citations) + i - 1) + intercept
                # Ensure non-negative
                projected_value = max(0, projected_value)
                projections[future_year] = projected_value
        else:
            # Use average of recent years
            avg_recent = np.mean(recent_citations)
            projections = {current_year + i: avg_recent for i in range(1, years_ahead + 1)}
        
        return projections