"""
Results Interpretation Dashboard

This page provides multi-level exploration of analysis results with academic context,
benchmarking, and actionable insights for research applications.

Features:
- Summary View: High-level metrics with context
- Detailed View: Drill-down into specific aspects
- Comparative View: Against benchmarks and baselines  
- Export View: Formatted for presentations/papers
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
import logging

from src.services.ml_service import get_ml_service
from src.data.unified_api_client import UnifiedSemanticScholarClient
from src.analytics.contextual_explanations import ContextualExplanationEngine, MetricCategory
from src.models.ml import CitationPrediction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Results Interpretation Dashboard",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Results Interpretation Dashboard")
st.markdown("""
**From Data to Understanding** - Transform your analysis results into actionable research insights 
with academic context, performance benchmarking, and export-ready summaries.
""")

# Initialize services
@st.cache_resource
def get_services():
    """Initialize and cache services."""
    try:
        ml_service = get_ml_service()
        api_client = UnifiedSemanticScholarClient()
        explanation_engine = ContextualExplanationEngine()
        return ml_service, api_client, explanation_engine
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        return None, None, None

ml_service, api_client, explanation_engine = get_services()

if ml_service is None:
    st.error("❌ Services not available.")
    st.stop()

# Sidebar - Configuration
st.sidebar.header("🎛️ Configuration")

# Analysis view selection
view_mode = st.sidebar.selectbox(
    "View Mode:",
    [
        "📊 Quick Overview",
        "🔍 Detailed Analysis"
    ]
)

# Domain selection for benchmarking
domain = st.sidebar.selectbox(
    "Academic Domain:",
    ["computer_science", "biology", "physics", "general"],
    help="Select domain for benchmarking"
)

# Sample data toggle
use_sample_data = st.sidebar.checkbox("Use Demo Data", value=True, help="Toggle between sample and real data")

if use_sample_data:
    # Generate realistic sample metrics
    sample_metrics = {
        "hits_at_10": np.random.beta(5, 15),  # Typically 0.1-0.4 range
        "mrr": np.random.beta(3, 20),         # Typically 0.05-0.2 range  
        "auc": 0.8 + np.random.beta(2, 2) * 0.15,  # 0.8-0.95 range
        "modularity": 0.3 + np.random.beta(2, 2) * 0.5,  # 0.3-0.8 range
        "network_density": np.random.exponential(0.003),  # Typical sparse network
        "response_time": np.random.gamma(2, 0.5),  # 0.5-3 second range
        "num_entities": np.random.randint(500, 5000),
        "num_predictions": np.random.randint(100, 1000)
    }
    
    st.sidebar.success(f"✅ Demo data: {sample_metrics['num_entities']} entities")
else:
    st.sidebar.info("ℹ️ Real data integration coming soon")

# Main content based on view mode
if view_mode == "📊 Quick Overview":
    st.header("📊 Quick Overview")
    st.markdown("Get an immediate understanding of your analysis results with performance indicators and key insights.")
    
    if use_sample_data:
        # Generate explanations for all metrics
        explanations = explanation_engine.bulk_explain_metrics(
            sample_metrics, 
            domain=domain,
            context={"num_entities": sample_metrics["num_entities"]}
        )
        
        # Performance Dashboard
        st.subheader("🎯 Performance Dashboard")
        
        # Overall assessment first
        performance_counts = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        for explanation in explanations.values():
            if hasattr(explanation, 'performance_level'):
                performance_counts[explanation.performance_level.value] += 1
        
        overall_score = sum(performance_counts[level] * weight for level, weight in 
                          [("excellent", 4), ("good", 3), ("fair", 2), ("poor", 1)])
        max_score = len(explanations) * 4
        overall_percentage = (overall_score / max_score) * 100 if max_score > 0 else 0
        
        # Display overall status
        if overall_percentage >= 80:
            st.success(f"🟢 **Excellent Overall Performance** ({overall_percentage:.0f}%)")
        elif overall_percentage >= 60:
            st.info(f"🟡 **Good Overall Performance** ({overall_percentage:.0f}%)")
        else:
            st.warning(f"🔴 **Performance Needs Improvement** ({overall_percentage:.0f}%)")
        
        # Top 3 Key Metrics (simplified)
        st.subheader("📊 Top Metrics")
        
        key_metrics = ["hits_at_10", "mrr", "auc"]
        
        for metric in key_metrics:
            explanation = explanations.get(metric)
            if explanation:
                st.metric(
                    f"{explanation.performance_icon} {metric.replace('_', ' ').title()}",
                    f"{sample_metrics[metric]:.3f}",
                    help=explanation.short_description
                )
        
        # Quick Actions
        st.subheader("⚡ Immediate Actions")
        all_actions = []
        for explanation in explanations.values():
            if hasattr(explanation, 'suggested_actions') and explanation.suggested_actions:
                all_actions.extend(explanation.suggested_actions)
        
        # Get unique top 3 actions
        unique_actions = list(dict.fromkeys(all_actions))[:3]
        
        for i, action in enumerate(unique_actions, 1):
            st.write(f"{i}. {action}")
            
        # Simple summary info
        st.subheader("📊 Summary")
        st.write(f"**Dataset:** {sample_metrics['num_entities']:,} entities")
        st.write(f"**Domain:** {domain.replace('_', ' ').title()}")
        
        performance_text = f"**Performance:** {performance_counts['excellent']} excellent, {performance_counts['good']} good, {performance_counts['fair']} fair, {performance_counts['poor']} poor metrics"
        st.write(performance_text)

elif view_mode == "🔍 Detailed Analysis":
    st.header("🔍 Detailed Analysis")
    st.markdown("Deep dive into specific metrics with detailed context, benchmarking, and export options.")
    
    if use_sample_data:
        # Category-based detailed analysis
        categories = explanation_engine.get_metric_categories()
        
        selected_category = st.selectbox(
            "Choose metric category for detailed analysis:",
            list(categories.keys()),
            format_func=lambda x: x.value.replace("_", " ").title()
        )
        
        st.subheader(f"📊 {selected_category.value.replace('_', ' ').title()} - Detailed Analysis")
        
        # Get metrics for selected category
        category_metrics = categories[selected_category]
        available_metrics = {k: v for k, v in sample_metrics.items() if k in category_metrics}
        
        if available_metrics:
            # Generate detailed explanations
            detailed_explanations = explanation_engine.bulk_explain_metrics(
                available_metrics,
                domain=domain,
                context={"num_entities": sample_metrics["num_entities"], "analysis_type": "detailed"}
            )
            
            # Simplified metric analysis
            for metric_name, value in available_metrics.items():
                explanation = detailed_explanations.get(metric_name)
                if explanation:
                    # Simple metric header with performance
                    st.metric(
                        f"{explanation.performance_icon} {metric_name.replace('_', ' ').title()}",
                        f"{value:.4f}",
                        help=explanation.short_description
                    )
                    
                    # Key information in expandable section
                    with st.expander("📝 View Details"):
                        st.write(f"**Explanation:** {explanation.detailed_explanation}")
                        st.write(f"**Performance:** {explanation.performance_level.value.title()}")
                        
                        if explanation.suggested_actions:
                            st.write("🚀 **Top Actions:**")
                            for action in explanation.suggested_actions[:2]:
                                st.write(f"• {action}")
                    
                    st.markdown("---")
        
        # Add simple export section
        st.subheader("📎 Export Options")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("📊 Generate Summary Report"):
                # Generate simplified summary
                performance_counts = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
                for explanation in detailed_explanations.values():
                    if hasattr(explanation, 'performance_level'):
                        performance_counts[explanation.performance_level.value] += 1
                
                summary_report = f"""# Analysis Results Summary

**Dataset:** {sample_metrics['num_entities']:,} entities  
**Domain:** {domain.replace('_', ' ').title()}

## Performance Overview
- Excellent: {performance_counts['excellent']} metrics
- Good: {performance_counts['good']} metrics  
- Fair: {performance_counts['fair']} metrics
- Poor: {performance_counts['poor']} metrics

## Key Metrics ({selected_category.value.replace('_', ' ').title()})
"""
                
                for metric_name, value in available_metrics.items():
                    explanation = detailed_explanations.get(metric_name)
                    if explanation:
                        summary_report += f"\n**{metric_name.replace('_', ' ').title()}:** {value:.4f} ({explanation.performance_level.value})"
                
                summary_report += "\n\n*Generated by Academic Citation Platform*"
                
                st.text_area("Summary Report (copy to clipboard):", summary_report, height=300)
                
        with export_col2:
            if st.button("📊 Generate Data Table"):
                # Generate simple data table
                table_data = []
                for metric_name, value in available_metrics.items():
                    explanation = detailed_explanations.get(metric_name)
                    if explanation:
                        table_data.append({
                            "Metric": metric_name.replace('_', ' ').title(),
                            "Value": f"{value:.4f}",
                            "Performance": explanation.performance_level.value.title()
                        })
                
                if table_data:
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Convert to CSV for easy copying
                    csv = df.to_csv(index=False)
                    st.text_area("CSV Data (copy to clipboard):", csv, height=150)
        
    else:
        st.info(f"No metrics available for {selected_category.value.replace('_', ' ').title()} category.")



# Footer with tips and model info

# Sidebar model status
with st.sidebar:
    st.markdown("---")
    st.subheader("🤖 Model Status")
    try:
        if ml_service:
            model_info = ml_service.get_model_info()
            st.success("✅ ML Service Online")
            st.write(f"**Entities:** {model_info.num_entities:,}")
            st.write(f"**Embedding Dim:** {model_info.embedding_dim}")
        else:
            st.warning("⚠️ ML Service Offline")
    except Exception as e:
        st.error(f"❌ Model Error: {e}")
    
    st.markdown("---")
    st.subheader("📚 Quick Guide")
    st.markdown("""
    **🎯 Overview:** Performance dashboard with key insights
    
    **🔍 Detailed:** Metric-by-metric analysis with export options
    """)