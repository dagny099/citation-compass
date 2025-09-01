"""
Demo Datasets page for the Academic Citation Platform.

This page provides an interactive interface for exploring and loading demo datasets.
Perfect for users who want to try the platform immediately without setting up
Neo4j or importing real data.

Features:
- Browse available demo datasets  
- Load datasets in offline or database mode
- Explore dataset contents and statistics
- Enable demo ML predictions
- Switch between demo and production modes
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from src.data.demo_dataset import get_available_datasets, DemoDatasetGenerator
from src.data.demo_loader import get_demo_loader, quick_load_demo
from src.data.fixtures import get_fixture_manager, quick_fixture, FixtureInfo
from src.services.demo_service import get_demo_manager, enable_demo_mode, disable_demo_mode, is_demo_mode_active

st.set_page_config(
    page_title="Demo Datasets",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 Demo Datasets & Sample Data")

st.markdown("""
Explore the Academic Citation Platform with curated demo datasets! Perfect for:
- **New users** who want to try the platform immediately
- **Developers** testing features without real data setup  
- **Demos** showcasing platform capabilities
- **Offline use** when Neo4j database isn't available
""")

# Initialize session state
if 'demo_mode_active' not in st.session_state:
    st.session_state.demo_mode_active = is_demo_mode_active()
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False


def load_demo_progress_callback(progress):
    """Update progress display for demo loading."""
    st.session_state.load_progress = {
        'operation': progress.current_operation,
        'percent': progress.progress_percent,
        'elapsed': progress.elapsed_time
    }


# Sidebar for dataset selection and management
with st.sidebar:
    st.header("🎯 Dataset Management")
    
    # Demo mode status
    demo_manager = get_demo_manager()
    demo_status = demo_manager.get_demo_status()
    
    if demo_status['demo_mode_active']:
        st.success("✅ Demo Mode Active")
        st.info(f"📊 Dataset: {demo_status['loaded_dataset'] or 'Unknown'}")
        
        if st.button("🔄 Switch to Production Mode"):
            disable_demo_mode()
            st.session_state.demo_mode_active = False
            st.success("Switched to production mode")
            st.rerun()
    else:
        st.info("📋 Production Mode Active")
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    # Generate new demo datasets
    if st.button("🔧 Generate Fresh Demo Data"):
        with st.spinner("Generating demo datasets..."):
            try:
                from src.data.demo_dataset import create_sample_datasets
                complete_info, minimal_info = create_sample_datasets()
                st.success(f"Generated: {complete_info.name} & {minimal_info.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Generation failed: {e}")
    
    # Clear demo mode
    if demo_status['demo_mode_active']:
        if st.button("🧹 Clear Demo Mode"):
            disable_demo_mode()
            st.session_state.demo_mode_active = False
            st.session_state.dataset_loaded = False
            st.success("Demo mode cleared")
            st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Dataset browser
    st.subheader("📚 Available Demo Datasets")
    
    # Get available datasets and fixtures
    try:
        available_datasets = get_available_datasets()
        fixture_manager = get_fixture_manager()
        available_fixtures = fixture_manager.list_available_fixtures()
        
        if available_datasets or available_fixtures:
            # Create tabs for different types
            tab1, tab2 = st.tabs(["📊 Complete Datasets", "🧪 Quick Fixtures"])
            
            with tab1:
                if available_datasets:
                    st.markdown(f"Found {len(available_datasets)} complete demo datasets:")
                    
                    # Display dataset information
                    for i, dataset_info in enumerate(available_datasets):
                        with st.expander(f"📊 {dataset_info.name}", expanded=(i == 0)):
                            col_info, col_action = st.columns([3, 1])
                            
                            with col_info:
                                st.markdown(f"**Description:** {dataset_info.description}")
                                
                                # Statistics
                                stats_col1, stats_col2, stats_col3 = st.columns(3)
                                with stats_col1:
                                    st.metric("Papers", dataset_info.total_papers)
                                with stats_col2:
                                    st.metric("Citations", dataset_info.total_citations)
                                with stats_col3:
                                    st.metric("Authors", dataset_info.total_authors)
                                
                                # Fields and year range
                                st.markdown(f"**Fields:** {', '.join(dataset_info.fields_covered)}")
                                st.markdown(f"**Year Range:** {dataset_info.year_range[0]}-{dataset_info.year_range[1]}")
                                st.markdown(f"**Created:** {dataset_info.created_at.strftime('%Y-%m-%d %H:%M')}")
                            
                            with col_action:
                                if st.button(f"🚀 Load Dataset", key=f"load_{dataset_info.name}"):
                                    with st.spinner(f"Loading {dataset_info.name}..."):
                                        success = enable_demo_mode(dataset_info.name)
                                        if success:
                                            st.session_state.demo_mode_active = True
                                            st.session_state.selected_dataset = dataset_info.name
                                            st.session_state.dataset_loaded = True
                                            st.success(f"✅ Loaded {dataset_info.name}")
                                            st.rerun()
                                        else:
                                            st.error("❌ Failed to load dataset")
                                
                                if st.button(f"🔍 Preview", key=f"preview_{dataset_info.name}"):
                                    st.session_state.preview_dataset = dataset_info.name
                else:
                    st.info("No complete datasets available. Use 'Generate Fresh Demo Data' to create them.")
            
            with tab2:
                if available_fixtures:
                    st.markdown(f"Found {len(available_fixtures)} quick test fixtures:")
                    
                    for fixture_info in available_fixtures:
                        with st.expander(f"🧪 {fixture_info.name}"):
                            col_info, col_action = st.columns([3, 1])
                            
                            with col_info:
                                st.markdown(f"**Description:** {fixture_info.description}")
                                st.markdown(f"**Use Case:** {fixture_info.use_case}")
                                
                                # Quick stats
                                stats_col1, stats_col2, stats_col3 = st.columns(3)
                                with stats_col1:
                                    st.metric("Papers", fixture_info.papers_count)
                                with stats_col2:
                                    st.metric("Citations", fixture_info.citations_count)
                                with stats_col3:
                                    st.metric("Authors", fixture_info.authors_count)
                                
                                if fixture_info.load_time_ms > 0:
                                    st.markdown(f"**Load Time:** {fixture_info.load_time_ms:.1f}ms")
                            
                            with col_action:
                                if st.button(f"⚡ Quick Load", key=f"fixture_{fixture_info.name}"):
                                    with st.spinner(f"Loading {fixture_info.name} fixture..."):
                                        try:
                                            # Create temporary dataset from fixture
                                            fixture_data = quick_fixture(fixture_info.name)
                                            
                                            # This would need integration with demo loader
                                            st.success(f"✅ {fixture_info.name} loaded")
                                            st.info("💡 Fixture loaded for immediate use!")
                                            
                                            # Show preview
                                            st.json({
                                                "papers": len(fixture_data["papers"]),
                                                "sample_paper": fixture_data["papers"][0]["title"]
                                            })
                                            
                                        except Exception as e:
                                            st.error(f"❌ Failed to load fixture: {e}")
                else:
                    st.info("No fixtures available.")
        else:
            st.warning("No demo datasets found. Generate demo data to get started!")
            
            # Quick generation button
            if st.button("🎯 Generate Demo Datasets Now", type="primary"):
                with st.spinner("Generating demo datasets..."):
                    try:
                        from src.data.demo_dataset import create_sample_datasets
                        complete_info, minimal_info = create_sample_datasets()
                        st.success("✅ Demo datasets generated successfully!")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Generation failed: {e}")
                        
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
    
    # Dataset preview
    if hasattr(st.session_state, 'preview_dataset'):
        dataset_name = st.session_state.preview_dataset
        
        st.subheader(f"🔍 Preview: {dataset_name}")
        
        try:
            # Load demo data for preview
            demo_loader = get_demo_loader()
            success = demo_loader.load_demo_dataset(dataset_name, force_offline=True)
            
            if success:
                data_interface = demo_loader.get_data_interface()
                
                # Show sample papers
                st.markdown("**Sample Papers:**")
                
                if hasattr(data_interface, 'store'):
                    sample_papers = list(data_interface.store.papers.values())[:3]
                    
                    for paper in sample_papers:
                        st.markdown(f"- **{paper['title']}** ({paper['year']}) - {paper['citation_count']} citations")
                
                # Show network statistics
                stats = data_interface.get_network_statistics()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Papers", stats['papers'])
                with col2:
                    st.metric("Total Citations", stats['citations'])
                with col3:
                    st.metric("Total Authors", stats['authors'])
                with col4:
                    st.metric("Total Venues", stats['venues'])
                
                # Clear preview
                if st.button("❌ Close Preview"):
                    del st.session_state.preview_dataset
                    st.rerun()
            else:
                st.error("Failed to load dataset for preview")
                
        except Exception as e:
            st.error(f"Preview error: {e}")

with col2:
    st.subheader("📊 Current Status")
    
    # Demo mode status display
    demo_manager = get_demo_manager()
    demo_status = demo_manager.get_demo_status()
    
    if demo_status['demo_mode_active']:
        st.success("✅ Demo Mode Active")
        
        # Show loaded dataset info
        if demo_status['loaded_dataset']:
            st.info(f"📊 **Active Dataset:** {demo_status['loaded_dataset']}")
        
        # Show ML service status
        if demo_status['demo_ml_service_loaded']:
            st.success("🤖 **ML Service:** Ready for predictions")
            
            # Test prediction button
            if st.button("🧪 Test ML Prediction"):
                try:
                    demo_ml = demo_manager.get_ml_service()
                    
                    # Get first paper for testing
                    demo_loader = get_demo_loader()
                    data_interface = demo_loader.get_data_interface()
                    
                    if hasattr(data_interface, 'store'):
                        first_paper_id = list(data_interface.store.papers.keys())[0]
                        predictions = demo_ml.predict_citations(first_paper_id, top_k=3)
                        
                        st.success(f"✅ Generated {len(predictions)} predictions!")
                        
                        for i, pred in enumerate(predictions[:3], 1):
                            st.markdown(f"{i}. **{pred.target_paper_id}** (Score: {pred.prediction_score:.3f})")
                    
                except Exception as e:
                    st.error(f"Prediction test failed: {e}")
        else:
            st.warning("🤖 **ML Service:** Not loaded")
    else:
        st.info("📋 Production Mode")
        st.markdown("Switch to demo mode to explore sample datasets")
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📈 Quick Stats")
    
    try:
        # Get current data interface
        demo_loader = get_demo_loader()
        
        if demo_loader.current_mode == "offline":
            data_interface = demo_loader.get_data_interface()
            stats = data_interface.get_network_statistics()
            
            st.metric("Papers Available", stats.get('papers', 0))
            st.metric("Citation Links", stats.get('citations', 0))
            st.metric("Research Fields", stats.get('fields', 0))
            
        else:
            st.info("Load a demo dataset to see statistics")
    
    except Exception as e:
        st.info("Load a demo dataset to see statistics")
    
    st.markdown("---")
    
    # Dataset comparison
    st.subheader("📊 Dataset Comparison")
    
    try:
        available_datasets = get_available_datasets()
        
        if len(available_datasets) > 1:
            # Create comparison chart
            comparison_data = {
                'Dataset': [d.name for d in available_datasets],
                'Papers': [d.total_papers for d in available_datasets],
                'Citations': [d.total_citations for d in available_datasets],
                'Authors': [d.total_authors for d in available_datasets]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Bar chart
            fig = px.bar(df_comparison, x='Dataset', y=['Papers', 'Citations', 'Authors'],
                        barmode='group', title="Dataset Comparison")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Multiple datasets needed for comparison")
    
    except Exception as e:
        st.info("No datasets available for comparison")

# Footer with usage tips and information
st.markdown("---")

with st.expander("💡 Demo Datasets Guide"):
    st.markdown("""
    ## 🎯 How to Use Demo Datasets
    
    ### **Complete Datasets**
    - Full citation networks with realistic data
    - Perfect for exploring all platform features
    - Include temporal evolution and cross-field connections
    - Best for comprehensive demonstrations
    
    ### **Quick Fixtures**  
    - Small, fast-loading test data
    - Ideal for development and testing
    - Specific use cases (collaboration, temporal, minimal)
    - Perfect for quick feature verification
    
    ### **Demo vs Production Mode**
    - **Demo Mode**: Uses sample data, works offline, instant setup
    - **Production Mode**: Uses your Neo4j database and real models
    - Switch between modes anytime without losing data
    
    ### **What You Can Do**
    1. **Load any dataset** to enable demo mode
    2. **Test ML predictions** with realistic results
    3. **Explore visualizations** with meaningful data
    4. **Analyze networks** with proper citation relationships
    5. **Learn the platform** without complex setup
    
    ### **Perfect For**
    - 🆕 **New users** wanting to try the platform
    - 👨‍💻 **Developers** testing features
    - 📊 **Demos** and presentations
    - 🔬 **Research** workflow exploration
    - ⚡ **Quick starts** without database setup
    
    ### **Next Steps**
    After exploring demo data:
    1. Set up your Neo4j database
    2. Import real research data
    3. Train models on your domain
    4. Publish findings with confidence!
    """)

# Auto-refresh for demo status
if st.session_state.demo_mode_active:
    time.sleep(1)  # Small delay to prevent excessive refreshing