"""
Results Interpretation Dashboard - Phase 4 Implementation

This page provides multi-level exploration of analysis results with academic context,
benchmarking, and actionable insights as described in Phase 4 of the platform roadmap.

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

# Sidebar - Analysis Configuration
st.sidebar.header("🎛️ Analysis Configuration")

# Multi-level exploration tabs
exploration_level = st.sidebar.selectbox(
    "Choose exploration level:",
    [
        "📊 Summary View",
        "🔍 Detailed Analysis", 
        "⚖️ Comparative Benchmarks",
        "📋 Export Ready"
    ]
)

# Domain selection for benchmarking
domain = st.sidebar.selectbox(
    "Academic domain:",
    ["computer_science", "biology", "physics", "general"],
    help="Select domain for appropriate academic benchmarking"
)

# Analysis type selection
analysis_type = st.sidebar.selectbox(
    "Analysis type:",
    [
        "Citation Prediction Performance",
        "Network Structure Analysis",
        "Embedding Quality Assessment",
        "System Performance Metrics"
    ]
)

# Sample data generation for demonstration
st.sidebar.subheader("📊 Analysis Data")
use_sample_data = st.sidebar.checkbox("Use sample data for demonstration", value=True)

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
    
    st.sidebar.success(f"✅ Using sample data: {sample_metrics['num_entities']} entities")
else:
    st.sidebar.info("Real data integration - coming soon")

# Main content based on exploration level
if exploration_level == "📊 Summary View":
    st.header("📊 Summary View - High-Level Results with Context")
    st.markdown("Get an immediate understanding of your analysis results with academic context and performance indicators.")
    
    if use_sample_data:
        # Generate explanations for all metrics
        explanations = explanation_engine.bulk_explain_metrics(
            sample_metrics, 
            domain=domain,
            context={"num_entities": sample_metrics["num_entities"]}
        )
        
        # Performance overview
        st.subheader("🎯 Performance Overview")
        
        # Create performance summary
        performance_counts = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        for explanation in explanations.values():
            if hasattr(explanation, 'performance_level'):
                performance_counts[explanation.performance_level.value] += 1
        
        # Display performance distribution
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("🟢 Excellent", performance_counts["excellent"])
        with perf_col2:
            st.metric("🟢 Good", performance_counts["good"])
        with perf_col3:
            st.metric("🟡 Fair", performance_counts["fair"])
        with perf_col4:
            st.metric("🔴 Poor", performance_counts["poor"])
        
        # Key metrics with traffic lights
        st.subheader("🚦 Key Metrics Analysis")
        
        key_metrics = ["hits_at_10", "mrr", "auc", "modularity"]
        
        for i, metric in enumerate(key_metrics):
            if i % 2 == 0:
                col1, col2 = st.columns(2)
                cols = [col1, col2]
            
            explanation = explanations.get(metric)
            if explanation:
                with cols[i % 2]:
                    st.metric(
                        f"{explanation.performance_icon} {metric.replace('_', ' ').title()}",
                        f"{sample_metrics[metric]:.3f}",
                        help=explanation.short_description
                    )
                    
                    with st.expander(f"📖 What does {metric} mean?"):
                        st.write(f"**Performance:** {explanation.performance_level.value.title()}")
                        st.write(explanation.detailed_explanation)
                        st.write(f"**Academic Context:** {explanation.academic_context}")
                        
                        if explanation.suggested_actions:
                            st.write("**Immediate Actions:**")
                            for action in explanation.suggested_actions[:2]:
                                st.write(f"• {action}")
        
        # Overall assessment
        st.subheader("🎓 Overall Assessment")
        
        overall_score = sum(performance_counts[level] * weight for level, weight in 
                          [("excellent", 4), ("good", 3), ("fair", 2), ("poor", 1)])
        max_score = len(explanations) * 4
        overall_percentage = (overall_score / max_score) * 100 if max_score > 0 else 0
        
        if overall_percentage >= 80:
            st.success(f"🟢 **Excellent Overall Performance** ({overall_percentage:.0f}%)")
            st.write("Your results meet or exceed academic standards across most metrics. Ready for publication or practical application.")
        elif overall_percentage >= 60:
            st.info(f"🟡 **Good Overall Performance** ({overall_percentage:.0f}%)")
            st.write("Solid results with some areas for improvement. Suitable for most research applications.")
        else:
            st.warning(f"🔴 **Performance Needs Improvement** ({overall_percentage:.0f}%)")
            st.write("Several metrics below academic standards. Consider model refinement or data quality review.")
        
        # Quick action summary
        st.subheader("⚡ Quick Action Summary")
        all_actions = []
        for explanation in explanations.values():
            if hasattr(explanation, 'suggested_actions') and explanation.suggested_actions:
                all_actions.extend(explanation.suggested_actions)
        
        # Get unique top actions
        unique_actions = list(dict.fromkeys(all_actions))[:5]
        
        for i, action in enumerate(unique_actions, 1):
            st.write(f"{i}. {action}")

elif exploration_level == "🔍 Detailed Analysis":
    st.header("🔍 Detailed Analysis - Drill-Down Exploration")
    st.markdown("Deep dive into specific aspects of your results with detailed academic context and interpretation.")
    
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
            
            # Create detailed metric analysis
            for metric_name, value in available_metrics.items():
                explanation = detailed_explanations.get(metric_name)
                if explanation:
                    st.markdown(f"### {explanation.performance_icon} {metric_name.replace('_', ' ').title()}")
                    
                    # Create detailed metric card
                    metric_col1, metric_col2 = st.columns([1, 2])
                    
                    with metric_col1:
                        st.metric(
                            "Current Value",
                            f"{value:.4f}",
                            help=explanation.short_description
                        )
                        
                        # Performance level with color coding
                        level_colors = {
                            "excellent": "🟢", "good": "🟢", 
                            "fair": "🟡", "poor": "🔴"
                        }
                        st.markdown(f"**Performance Level:** {level_colors.get(explanation.performance_level.value, '⚪')} {explanation.performance_level.value.title()}")
                    
                    with metric_col2:
                        st.markdown("**Detailed Explanation:**")
                        st.write(explanation.detailed_explanation)
                        
                        st.markdown("**Academic Context:**")
                        st.write(explanation.academic_context)
                        
                        if explanation.typical_range_text:
                            st.markdown("**Typical Ranges:**")
                            st.write(explanation.typical_range_text)
                    
                    # Interpretation guide
                    st.markdown("**Interpretation Guide:**")
                    st.info(explanation.interpretation_guide)
                    
                    # Actionable recommendations
                    if explanation.suggested_actions:
                        st.markdown("**🚀 Actionable Recommendations:**")
                        for i, action in enumerate(explanation.suggested_actions, 1):
                            st.write(f"{i}. {action}")
                    
                    # Benchmark comparison
                    if explanation.benchmark_comparison:
                        st.markdown("**📊 Benchmark Comparison:**")
                        st.write(explanation.benchmark_comparison)
                    
                    st.markdown("---")
        else:
            st.info(f"No metrics available for {selected_category.value.replace('_', ' ').title()} category.")

elif exploration_level == "⚖️ Comparative Benchmarks":
    st.header("⚖️ Comparative Benchmarks - Against Academic Standards")
    st.markdown("Compare your results against published academic benchmarks and industry standards.")
    
    if use_sample_data:
        # Generate explanations for comparison
        explanations = explanation_engine.bulk_explain_metrics(
            sample_metrics, 
            domain=domain,
            context={"num_entities": sample_metrics["num_entities"]}
        )
        
        st.subheader(f"📚 Benchmarking Against {domain.replace('_', ' ').title()} Standards")
        
        # Create comparison visualization
        metrics_for_viz = ["hits_at_10", "mrr", "auc", "modularity"]
        available_viz_metrics = {k: v for k, v in sample_metrics.items() if k in metrics_for_viz}
        
        if available_viz_metrics:
            # Prepare data for radar chart
            categories = []
            values = []
            benchmarks = []
            
            for metric_name, value in available_viz_metrics.items():
                explanation = explanations.get(metric_name)
                categories.append(metric_name.replace('_', ' ').title())
                values.append(value)
                
                # Get benchmark thresholds
                benchmark_key = f"{metric_name}_{domain}"
                benchmark = explanation_engine.benchmarks.get(benchmark_key) or explanation_engine.benchmarks.get(metric_name)
                if benchmark:
                    benchmarks.append(benchmark.good_threshold)
                else:
                    benchmarks.append(value * 0.8)  # Fallback
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Your Results',
                line_color='blue'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=benchmarks,
                theta=categories,
                fill='toself',
                name='Academic Benchmark',
                line_color='red',
                opacity=0.3
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(max(values), max(benchmarks)) * 1.1]
                    )),
                showlegend=True,
                title="Performance vs Academic Benchmarks",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed benchmark comparison table
        st.subheader("📋 Detailed Benchmark Comparison")
        
        comparison_data = []
        for metric_name, value in sample_metrics.items():
            explanation = explanations.get(metric_name)
            if explanation:
                benchmark_key = f"{metric_name}_{domain}"
                benchmark = explanation_engine.benchmarks.get(benchmark_key) or explanation_engine.benchmarks.get(metric_name)
                
                if benchmark:
                    comparison_data.append({
                        "Metric": metric_name.replace('_', ' ').title(),
                        "Your Value": f"{value:.4f}",
                        "Performance": explanation.performance_icon + " " + explanation.performance_level.value.title(),
                        "Good Threshold": f"{benchmark.good_threshold:.4f}",
                        "Excellent Threshold": f"{benchmark.excellent_threshold:.4f}",
                        "Domain": domain.replace('_', ' ').title(),
                        "Academic Sources": len(benchmark.academic_sources) if benchmark.academic_sources else 0
                    })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
        
        # Literature grounding
        st.subheader("📖 Literature Grounding & References")
        
        all_sources = set()
        for explanation in explanations.values():
            if hasattr(explanation, 'metric_name'):
                benchmark_key = f"{explanation.metric_name}_{domain}"
                benchmark = explanation_engine.benchmarks.get(benchmark_key) or explanation_engine.benchmarks.get(explanation.metric_name)
                if benchmark and benchmark.academic_sources:
                    all_sources.update(benchmark.academic_sources)
        
        if all_sources:
            st.markdown("**Key Academic References:**")
            for i, source in enumerate(sorted(all_sources), 1):
                st.write(f"{i}. {source}")
        else:
            st.info("Academic references available in detailed metric explanations.")

elif exploration_level == "📋 Export Ready":
    st.header("📋 Export Ready - Publication & Presentation Formats")
    st.markdown("Generate publication-ready tables, summaries, and presentations from your analysis results.")
    
    if use_sample_data:
        # Generate explanations for export
        explanations = explanation_engine.bulk_explain_metrics(
            sample_metrics, 
            domain=domain,
            context={"num_entities": sample_metrics["num_entities"]}
        )
        
        # Export format selection
        export_format = st.selectbox(
            "Choose export format:",
            ["LaTeX Table", "Academic Summary", "PowerPoint Outline", "Research Proposal Template"]
        )
        
        if export_format == "LaTeX Table":
            st.subheader("📊 LaTeX Table for Academic Papers")
            
            latex_table = """\\begin{table}[h]
\\centering
\\caption{Citation Analysis Results with Academic Benchmarking}
\\begin{tabular}{|l|c|c|c|c|}
\\hline
\\textbf{Metric} & \\textbf{Value} & \\textbf{Performance} & \\textbf{Benchmark} & \\textbf{Domain} \\\\
\\hline
"""
            
            for metric_name, value in sample_metrics.items():
                if metric_name in ["hits_at_10", "mrr", "auc", "modularity", "network_density"]:
                    explanation = explanations.get(metric_name)
                    if explanation:
                        benchmark_key = f"{metric_name}_{domain}"
                        benchmark = explanation_engine.benchmarks.get(benchmark_key) or explanation_engine.benchmarks.get(metric_name)
                        benchmark_val = benchmark.good_threshold if benchmark else "N/A"
                        
                        latex_table += f"{metric_name.replace('_', ' ').title()} & {value:.4f} & {explanation.performance_level.value.title()} & {benchmark_val} & {domain.replace('_', ' ').title()} \\\\\n"
                        latex_table += "\\hline\n"
            
            latex_table += """\\end{tabular}
\\label{tab:citation_analysis}
\\end{table}"""
            
            st.code(latex_table, language="latex")
            st.success("✅ LaTeX table ready for copy-paste into academic papers")
        
        elif export_format == "Academic Summary":
            st.subheader("📝 Academic Summary for Publications")
            
            # Calculate overall performance
            performance_counts = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
            for explanation in explanations.values():
                if hasattr(explanation, 'performance_level'):
                    performance_counts[explanation.performance_level.value] += 1
            
            academic_summary = f"""# Citation Analysis Results Summary

## Methodology
This analysis evaluated citation prediction and network structure using established academic metrics in the {domain.replace('_', ' ')} domain.

## Dataset Characteristics
- **Entities Analyzed:** {sample_metrics['num_entities']:,} papers
- **Prediction Coverage:** {sample_metrics.get('num_predictions', 'N/A')} citation predictions
- **Domain:** {domain.replace('_', ' ').title()}

## Performance Results

### Citation Prediction Metrics
- **Hits@10:** {sample_metrics.get('hits_at_10', 0):.4f} ({explanations.get('hits_at_10', {}).get('performance_level', {}).get('value', 'N/A')} performance)
- **Mean Reciprocal Rank:** {sample_metrics.get('mrr', 0):.4f} ({explanations.get('mrr', {}).get('performance_level', {}).get('value', 'N/A')} performance)
- **Area Under Curve:** {sample_metrics.get('auc', 0):.4f} ({explanations.get('auc', {}).get('performance_level', {}).get('value', 'N/A')} performance)

### Network Analysis Metrics
- **Modularity:** {sample_metrics.get('modularity', 0):.4f} ({explanations.get('modularity', {}).get('performance_level', {}).get('value', 'N/A')} performance)
- **Network Density:** {sample_metrics.get('network_density', 0):.4f} ({explanations.get('network_density', {}).get('performance_level', {}).get('value', 'N/A')} performance)

## Academic Context
Results were benchmarked against published standards in {domain.replace('_', ' ')} citation analysis. Performance levels indicate comparison to typical academic baselines.

## Research Implications
{explanations.get('hits_at_10', {}).get('interpretation_guide', 'Analysis demonstrates model effectiveness for citation prediction tasks.')}

## Limitations
- Sample size: {sample_metrics['num_entities']} entities
- Domain specificity: Results specific to {domain.replace('_', ' ')}
- Temporal scope: Analysis snapshot at time of evaluation

## Recommended Future Work
1. Expand dataset for broader generalizability
2. Cross-domain validation of model performance
3. Temporal analysis of citation prediction accuracy
"""
            
            st.markdown(academic_summary)
            st.success("✅ Academic summary ready for research papers")
        
        elif export_format == "PowerPoint Outline":
            st.subheader("🎯 PowerPoint Presentation Outline")
            
            ppt_outline = f"""# Citation Analysis Presentation

## Slide 1: Title Slide
- **Title:** Academic Citation Analysis Results
- **Subtitle:** {domain.replace('_', ' ').title()} Domain Study
- **Data:** {sample_metrics['num_entities']:,} papers analyzed

## Slide 2: Research Question
- **Objective:** Evaluate citation prediction model performance
- **Domain:** {domain.replace('_', ' ').title()}
- **Metrics:** Hits@10, MRR, AUC, Network Analysis

## Slide 3: Key Results Overview
- **Overall Performance:** {sum(1 for e in explanations.values() if hasattr(e, 'performance_level') and e.performance_level.value in ['excellent', 'good'])} of {len(explanations)} metrics meet/exceed standards
- **Best Performance:** {max(explanations.items(), key=lambda x: getattr(x[1], 'value', 0) if hasattr(x[1], 'value') else 0)[0] if explanations else 'N/A'}
- **Dataset Size:** {sample_metrics['num_entities']:,} entities

## Slide 4: Citation Prediction Results
- **Hits@10:** {sample_metrics.get('hits_at_10', 0):.1%} accuracy
- **MRR:** {sample_metrics.get('mrr', 0):.3f} average rank quality
- **AUC:** {sample_metrics.get('auc', 0):.3f} classification performance

## Slide 5: Network Structure Analysis
- **Modularity:** {sample_metrics.get('modularity', 0):.3f} community strength
- **Density:** {sample_metrics.get('network_density', 0):.4f} connectivity
- **Interpretation:** {explanations.get('modularity', {}).get('interpretation_guide', 'Network analysis complete')[:100]}...

## Slide 6: Academic Benchmarking
- **Performance vs Standards:** [Radar chart comparing your results to academic benchmarks]
- **Domain Context:** Compared against {domain.replace('_', ' ')} literature
- **Benchmark Sources:** Academic publications and established datasets

## Slide 7: Research Implications
- **Practical Applications:** {explanations.get('hits_at_10', {}).get('suggested_actions', ['Citation recommendation', 'Research discovery'])[0] if explanations.get('hits_at_10', {}).get('suggested_actions') else 'Citation applications'}
- **Academic Contribution:** Model performance suitable for research applications
- **Publication Readiness:** Results meet academic standards

## Slide 8: Next Steps & Future Work
- **Immediate Actions:** [List top 3 recommended actions]
- **Long-term Research:** Cross-domain validation, temporal analysis
- **Collaboration Opportunities:** Based on network analysis insights
"""
            
            st.code(ppt_outline, language="markdown")
            st.success("✅ PowerPoint outline ready for presentation development")
        
        elif export_format == "Research Proposal Template":
            st.subheader("📋 Research Proposal Template")
            
            proposal = f"""# Research Proposal: Advanced Citation Analysis in {domain.replace('_', ' ').title()}

## Abstract
This proposal outlines research directions based on citation analysis results from {sample_metrics['num_entities']:,} academic papers. Our preliminary analysis achieved {sample_metrics.get('hits_at_10', 0):.1%} prediction accuracy (Hits@10) and identified network modularity of {sample_metrics.get('modularity', 0):.3f}, suggesting {explanations.get('modularity', {}).get('interpretation_guide', 'structured research communities')[:100]}...

## 1. Research Motivation
Current analysis demonstrates {explanations.get('hits_at_10', {}).get('performance_level', {}).get('value', 'good')} performance in citation prediction, indicating potential for:
- Automated research discovery systems
- Citation recommendation engines  
- Research trend analysis platforms

## 2. Research Questions
1. How can we improve upon current {sample_metrics.get('hits_at_10', 0):.1%} prediction accuracy?
2. What network patterns drive the observed {sample_metrics.get('modularity', 0):.3f} modularity score?
3. How do results generalize beyond {domain.replace('_', ' ')} domain?

## 3. Methodology
### 3.1 Dataset Expansion
- Current: {sample_metrics['num_entities']:,} papers
- Proposed: Scale to 10x current size
- Multi-domain validation

### 3.2 Model Enhancement
- Address current performance gaps: {', '.join([k for k, v in explanations.items() if hasattr(v, 'performance_level') and v.performance_level.value == 'poor'])}
- Investigate ensemble methods
- Temporal prediction capabilities

### 3.3 Evaluation Framework
- Academic benchmarking against published baselines
- Cross-domain performance validation
- Real-world application testing

## 4. Expected Outcomes
- Improved citation prediction accuracy (target: >{sample_metrics.get('hits_at_10', 0)*1.3:.1%})
- Enhanced network analysis capabilities
- Published research contributions
- Open-source research tools

## 5. Research Impact
### Academic Impact
- Publications in top-tier venues
- Methodological contributions to citation analysis
- Dataset contributions to research community

### Practical Impact  
- Research discovery tools
- Academic search enhancement
- Collaboration recommendation systems

## 6. Budget & Timeline
- **Phase 1 (6 months):** Dataset expansion and model development
- **Phase 2 (6 months):** Cross-domain validation and optimization
- **Phase 3 (6 months):** Application development and evaluation
- **Total Duration:** 18 months
- **Estimated Budget:** [To be determined based on computational and personnel needs]

## 7. Success Metrics
- Technical: Achieve >30% improvement in prediction metrics
- Academic: 3+ publications in top venues
- Impact: 100+ citations within 2 years post-publication

## References
[Academic sources from benchmark analysis would be listed here]
"""
            
            st.markdown(proposal)
            st.success("✅ Research proposal template ready for grant applications")

# Footer with tips and model info
st.markdown("---")
st.markdown("### 💡 Phase 4 Features Implemented")

feature_col1, feature_col2, feature_col3 = st.columns(3)

with feature_col1:
    st.markdown("""
    **✅ Contextual Explanations**
    - Traffic light performance indicators
    - Academic benchmarking
    - Domain-specific interpretation
    """)

with feature_col2:
    st.markdown("""
    **✅ Multi-Level Exploration**
    - Summary to detailed drill-down
    - Comparative benchmarking
    - Export-ready formats
    """)

with feature_col3:
    st.markdown("""
    **✅ Actionable Insights**  
    - Research recommendations
    - Publication-ready outputs
    - Future work suggestions
    """)

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
    st.subheader("📚 Phase 4 Guide")
    st.markdown("""
    **🎯 Summary View:** Quick overview with performance indicators
    
    **🔍 Detailed Analysis:** Deep dive into specific metrics
    
    **⚖️ Benchmarks:** Compare against academic standards
    
    **📋 Export Ready:** Publication and presentation formats
    """)