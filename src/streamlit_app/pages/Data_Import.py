"""
Data Import page for the Academic Citation Platform.

This page provides an interactive interface for importing academic papers and citations
from Semantic Scholar using the new data import pipeline. Features include:
- Search-based paper import
- Import by specific paper IDs  
- Progress tracking with real-time updates
- Configuration management
- Import status monitoring
- Data validation and quality checks
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import threading
from typing import List, Optional
import json
from datetime import datetime

from src.data.import_pipeline import (
    ImportConfiguration,
    ImportProgress, 
    ImportStatus,
    DataImportPipeline,
    create_sample_import_config,
    quick_import_by_search,
    quick_import_by_ids
)
from src.utils.validation import validate_import_configuration
from src.data.unified_database import get_database

st.set_page_config(
    page_title="Data Import Pipeline",
    page_icon="📥",
    layout="wide"
)

st.title("📥 Data Import Pipeline")

st.markdown("""
Import academic papers and citation data from Semantic Scholar into your Neo4j database.
This pipeline supports batch processing, progress tracking, and resumable imports.
""")

# Initialize session state
if 'import_progress' not in st.session_state:
    st.session_state.import_progress = None
if 'import_config' not in st.session_state:
    st.session_state.import_config = None
if 'import_running' not in st.session_state:
    st.session_state.import_running = False


def update_progress_display(progress: ImportProgress):
    """Update the Streamlit display with current progress."""
    st.session_state.import_progress = progress


import asyncio

def run_import_in_background(config: ImportConfiguration, import_type: str, **kwargs):
    """Run import operation in a separate thread with async support."""
    try:
        pipeline = DataImportPipeline(config)
        
        # Create enhanced progress callback with real-time updates
        def enhanced_progress_callback(progress: ImportProgress):
            """Enhanced callback with more detailed progress tracking."""
            update_progress_display(progress)
            # Force Streamlit to update more frequently for real-time feedback
            if hasattr(st, 'session_state') and 'last_progress_update' in st.session_state:
                import time
                now = time.time()
                if now - st.session_state.get('last_progress_update', 0) > 0.5:  # Update every 0.5 seconds
                    st.session_state.last_progress_update = now
                    try:
                        # This will trigger a rerun only if UI is waiting
                        if st.session_state.get('import_running', False):
                            st.rerun()
                    except:
                        pass  # Ignore rerun errors in background thread
        
        pipeline.add_progress_callback(enhanced_progress_callback)
        
        # Initialize progress immediately
        initial_progress = ImportProgress()
        initial_progress.status = ImportStatus.IN_PROGRESS
        initial_progress.start_time = datetime.now()
        st.session_state.import_progress = initial_progress
        
        if import_type == "search":
            progress = pipeline.import_papers_by_search(
                kwargs.get('search_query'),
                kwargs.get('max_results')
            )
        elif import_type == "ids":
            progress = pipeline.import_papers_by_ids(kwargs.get('paper_ids'))
        else:
            raise ValueError(f"Unknown import type: {import_type}")
            
        st.session_state.import_progress = progress
        
    except Exception as e:
        # Create error progress
        error_progress = ImportProgress()
        error_progress.status = ImportStatus.FAILED
        error_progress.errors.append(str(e))
        error_progress.end_time = datetime.now()
        st.session_state.import_progress = error_progress
        
        # Log the error for debugging
        import logging
        logging.error(f"Import failed: {e}", exc_info=True)
    
    finally:
        st.session_state.import_running = False


# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Import Configuration")
    
    # Import method selection
    import_method = st.selectbox(
        "Import Method",
        ["Search Query", "Paper IDs", "Sample Configuration"],
        help="Choose how to specify papers to import"
    )
    
    # Basic configuration
    st.subheader("Basic Settings")
    
    max_papers = st.number_input(
        "Maximum Papers",
        min_value=1,
        max_value=10000,
        value=100,
        help="Maximum number of papers to import"
    )
    
    batch_size = st.number_input(
        "Batch Size",
        min_value=1,
        max_value=1000,
        value=50,
        help="Number of papers to process in each batch"
    )
    
    # Advanced configuration
    with st.expander("🔧 Advanced Settings"):
        include_citations = st.checkbox("Include Citations", value=True)
        include_references = st.checkbox("Include References", value=True)
        include_authors = st.checkbox("Include Authors", value=True)
        include_venues = st.checkbox("Include Venues", value=True)
        
        min_citation_count = st.number_input(
            "Minimum Citation Count",
            min_value=0,
            max_value=1000,
            value=0,
            help="Filter papers with fewer citations"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            min_year = st.number_input(
                "Min Year",
                min_value=1900,
                max_value=2030,
                value=2000,
                help="Earliest publication year"
            )
        with col2:
            max_year = st.number_input(
                "Max Year", 
                min_value=1900,
                max_value=2030,
                value=2024,
                help="Latest publication year"
            )
        
        api_delay = st.slider(
            "API Delay (seconds)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Delay between API requests"
        )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Import method specific inputs
    if import_method == "Search Query":
        st.subheader("🔍 Search-Based Import")
        
        search_query = st.text_input(
            "Search Query",
            value="machine learning",
            help="Search terms for finding papers"
        )
        
        st.markdown("""
        **Examples:**
        - `machine learning`
        - `deep learning neural networks`
        - `computer vision transformers`
        - `natural language processing`
        """)
        
        import_params = {
            'import_type': 'search',
            'search_query': search_query,
            'max_results': max_papers
        }
        
    elif import_method == "Paper IDs":
        st.subheader("📋 Import by Paper IDs")
        
        paper_ids_text = st.text_area(
            "Paper IDs (one per line)",
            height=150,
            help="Enter Semantic Scholar paper IDs, one per line"
        )
        
        paper_ids = [id.strip() for id in paper_ids_text.split('\n') if id.strip()]
        
        if paper_ids:
            st.info(f"Found {len(paper_ids)} paper IDs")
            if len(paper_ids) > 10:
                st.warning(f"Large number of paper IDs may take significant time to process")
        
        st.markdown("""
        **Example Paper IDs:**
        ```
        649def34f8be52c8b66281af98ae884c09aef38f9
        204e3073870fae3d05bcbc2f6a8e263d9b72e776
        ```
        """)
        
        import_params = {
            'import_type': 'ids',
            'paper_ids': paper_ids
        }
        
    else:  # Sample Configuration
        st.subheader("🎯 Sample Import Configuration")
        
        sample_config = create_sample_import_config()
        
        st.code(f"""
Search Query: {sample_config.search_query}
Max Papers: {sample_config.max_papers}
Batch Size: {sample_config.batch_size}
Min Citation Count: {sample_config.min_citation_count}
Year Range: {sample_config.year_range}
        """)
        
        st.info("This will import papers using the predefined sample configuration.")
        
        import_params = {
            'import_type': 'search',
            'search_query': sample_config.search_query,
            'max_results': sample_config.max_papers
        }

    # Create configuration
    config = ImportConfiguration(
        search_query=import_params.get('search_query') if import_method != "Paper IDs" else None,
        paper_ids=import_params.get('paper_ids') if import_method == "Paper IDs" else None,
        max_papers=max_papers,
        batch_size=batch_size,
        include_citations=include_citations,
        include_references=include_references,
        include_authors=include_authors,
        include_venues=include_venues,
        min_citation_count=min_citation_count,
        year_range=(min_year, max_year) if min_year <= max_year else None,
        api_delay=api_delay,
        save_progress=True
    )
    
    # Validate configuration
    validation = validate_import_configuration(config)
    
    if not validation['is_valid']:
        st.error("❌ Configuration Invalid")
        for error in validation['errors']:
            st.error(f"• {error}")
    else:
        if validation['warnings']:
            st.warning("⚠️ Configuration Warnings")
            for warning in validation['warnings']:
                st.warning(f"• {warning}")

    # Import controls
    st.subheader("🚀 Import Controls")
    
    col_start, col_stop = st.columns(2)
    
    with col_start:
        start_import = st.button(
            "▶️ Start Import",
            disabled=st.session_state.import_running or not validation['is_valid'],
            help="Begin the import process"
        )
    
    with col_stop:
        stop_import = st.button(
            "⏹️ Stop Import",
            disabled=not st.session_state.import_running,
            help="Cancel the ongoing import"
        )

    # Handle import controls
    if start_import and not st.session_state.import_running:
        st.session_state.import_running = True
        st.session_state.import_config = config
        
        # Start import in background thread
        import_thread = threading.Thread(
            target=run_import_in_background,
            args=(config, import_params['import_type']),
            kwargs=import_params
        )
        import_thread.daemon = True
        import_thread.start()
        
        st.success("🚀 Import started!")
        st.rerun()

    # Progress display
    if st.session_state.import_progress:
        progress = st.session_state.import_progress
        
        st.subheader("📊 Import Progress")
        
        # Status indicator with enhanced information
        status_colors = {
            ImportStatus.PENDING: "🟡",
            ImportStatus.IN_PROGRESS: "🔵", 
            ImportStatus.COMPLETED: "🟢",
            ImportStatus.FAILED: "🔴",
            ImportStatus.CANCELLED: "🟠",
            ImportStatus.PAUSED: "🟤"
        }
        
        # Create two columns for status and elapsed time
        status_col1, status_col2 = st.columns([2, 1])
        
        with status_col1:
            st.markdown(f"**Status:** {status_colors.get(progress.status, '⚪')} {progress.status.value.title()}")
        
        with status_col2:
            # Elapsed time with live updates
            if progress.elapsed_time:
                elapsed_str = str(progress.elapsed_time).split('.')[0]
                st.markdown(f"**Elapsed:** {elapsed_str}")
            elif progress.start_time and progress.status == ImportStatus.IN_PROGRESS:
                # Calculate live elapsed time for in-progress imports
                from datetime import datetime
                live_elapsed = datetime.now() - progress.start_time
                elapsed_str = str(live_elapsed).split('.')[0]
                st.markdown(f"**Elapsed:** {elapsed_str}")
        
        # Enhanced progress bars with better visualization
        progress_container = st.container()
        
        with progress_container:
            # Papers progress
            if progress.total_papers > 0:
                papers_progress = progress.papers_progress_percent / 100.0
                st.progress(
                    papers_progress, 
                    text=f"📄 Papers: {progress.processed_papers:,}/{progress.total_papers:,} ({progress.papers_progress_percent:.1f}%)"
                )
            elif progress.processed_papers > 0:
                # Show indeterminate progress for unknown total
                st.progress(0.0, text=f"📄 Papers: {progress.processed_papers:,} processed (searching...)")
            
            # Citations progress
            if progress.total_citations > 0:
                citations_progress = progress.citations_progress_percent / 100.0
                st.progress(
                    citations_progress, 
                    text=f"🔗 Citations: {progress.processed_citations:,}/{progress.total_citations:,} ({progress.citations_progress_percent:.1f}%)"
                )
            elif progress.processed_citations > 0:
                st.progress(0.0, text=f"🔗 Citations: {progress.processed_citations:,} processed")
            
            # Overall progress (if we have totals for both)
            if progress.total_papers > 0 and progress.total_citations > 0:
                overall_progress = progress.overall_progress_percent / 100.0
                st.progress(
                    overall_progress,
                    text=f"🎯 Overall Progress: {progress.overall_progress_percent:.1f}%"
                )
        
        # Enhanced statistics with delta indicators
        st.subheader("📈 Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Show rate if import is in progress
            if progress.status == ImportStatus.IN_PROGRESS and progress.elapsed_time:
                papers_per_sec = progress.papers_created / progress.elapsed_time.total_seconds()
                st.metric(
                    "Papers Created", 
                    f"{progress.papers_created:,}",
                    delta=f"{papers_per_sec:.1f}/sec" if papers_per_sec > 0 else None
                )
            else:
                st.metric("Papers Created", f"{progress.papers_created:,}")
        
        with col2:
            if progress.status == ImportStatus.IN_PROGRESS and progress.elapsed_time:
                citations_per_sec = progress.citations_created / progress.elapsed_time.total_seconds()
                st.metric(
                    "Citations Created", 
                    f"{progress.citations_created:,}",
                    delta=f"{citations_per_sec:.1f}/sec" if citations_per_sec > 0 else None
                )
            else:
                st.metric("Citations Created", f"{progress.citations_created:,}")
        
        with col3:
            st.metric("Authors Created", f"{progress.authors_created:,}")
        
        with col4:
            st.metric("Venues Created", f"{progress.venues_created:,}")
        
        # Performance indicators
        if progress.status == ImportStatus.IN_PROGRESS and progress.elapsed_time:
            st.subheader("⚡ Performance")
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            total_seconds = progress.elapsed_time.total_seconds()
            if total_seconds > 0:
                with perf_col1:
                    items_per_sec = (progress.papers_created + progress.citations_created) / total_seconds
                    st.metric("Items/Second", f"{items_per_sec:.1f}")
                
                with perf_col2:
                    if progress.total_papers > 0 and progress.papers_progress_percent > 0:
                        estimated_total_time = total_seconds * (100 / progress.papers_progress_percent)
                        eta_seconds = estimated_total_time - total_seconds
                        eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s"
                        st.metric("ETA", eta_str)
                
                with perf_col3:
                    completion_rate = progress.overall_progress_percent if progress.overall_progress_percent > 0 else progress.papers_progress_percent
                    st.metric("Completion", f"{completion_rate:.1f}%")
        
        # Errors and warnings with better formatting
        error_warning_container = st.container()
        
        with error_warning_container:
            if progress.errors:
                with st.expander(f"❌ Errors ({len(progress.errors)})", expanded=len(progress.errors) <= 3):
                    for i, error in enumerate(progress.errors[-5:], 1):  # Show last 5 errors
                        st.error(f"**Error {len(progress.errors)-5+i}:** {error}")
                    if len(progress.errors) > 5:
                        st.info(f"... and {len(progress.errors) - 5} more errors (check logs for details)")
            
            if progress.warnings:
                with st.expander(f"⚠️ Warnings ({len(progress.warnings)})"):
                    for i, warning in enumerate(progress.warnings[-10:], 1):  # Show last 10 warnings
                        st.warning(f"**Warning {len(progress.warnings)-10+i}:** {warning}")
                    if len(progress.warnings) > 10:
                        st.info(f"... and {len(progress.warnings) - 10} more warnings")

with col2:
    st.subheader("💾 Database Status")
    
    # Database connection check
    try:
        db = get_database()
        connection_ok = db.test_connection()
        
        if connection_ok:
            st.success("✅ Database Connected")
            
            # Get database statistics
            try:
                stats = db.get_network_statistics()
                
                st.metric("Papers in Database", stats.get('papers', 0))
                st.metric("Authors in Database", stats.get('authors', 0))
                st.metric("Citations in Database", stats.get('citations', 0))
                st.metric("Venues in Database", stats.get('venues', 0))
                
                # Database schema info
                with st.expander("🔧 Schema Information"):
                    schema_info = db.get_schema_info()
                    if 'error' not in schema_info:
                        st.json(schema_info['statistics'])
                    else:
                        st.error(f"Schema error: {schema_info['error']}")
                        
            except Exception as e:
                st.warning(f"Could not fetch database statistics: {e}")
        else:
            st.error("❌ Database Connection Failed")
            st.info("Please check your Neo4j configuration in .env file")
            
    except Exception as e:
        st.error(f"Database Error: {e}")
    
    # Import history placeholder
    st.subheader("📈 Import History")
    st.info("Import history will be displayed here in future updates")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    if st.button("🎯 Load Sample Config"):
        st.session_state.sample_config_loaded = True
        st.success("Sample configuration loaded!")
    
    if st.button("🧹 Clear Progress"):
        st.session_state.import_progress = None
        st.success("Progress cleared!")
    
    # Configuration export/import
    with st.expander("💾 Config Management"):
        if st.button("📤 Export Config"):
            config_dict = {
                'search_query': config.search_query,
                'max_papers': config.max_papers,
                'batch_size': config.batch_size,
                'include_citations': config.include_citations,
                'include_authors': config.include_authors,
                'min_citation_count': config.min_citation_count,
                'year_range': config.year_range,
                'api_delay': config.api_delay
            }
            st.download_button(
                "⬇️ Download Config",
                data=json.dumps(config_dict, indent=2),
                file_name=f"import_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Auto-refresh for active imports with more responsive updates
if st.session_state.get('import_running', False):
    # Initialize last update time if not exists
    if 'last_progress_update' not in st.session_state:
        st.session_state.last_progress_update = time.time()
    
    # Check if we should update (every 1 second instead of 2)
    now = time.time()
    time_since_update = now - st.session_state.last_progress_update
    
    if time_since_update >= 1.0:  # Update every 1 second
        st.session_state.last_progress_update = now
        
        # Add a status indicator for real-time updates
        if st.session_state.get('import_progress'):
            progress = st.session_state.import_progress
            if progress.status == ImportStatus.IN_PROGRESS:
                # Show a small "live" indicator
                st.sidebar.success("🔴 Live Updates Active")
        
        st.rerun()
    else:
        # Sleep for remaining time to next update
        time.sleep(max(0.1, 1.0 - time_since_update))
        st.rerun()

# Footer with tips
st.markdown("---")

with st.expander("💡 Tips and Best Practices"):
    st.markdown("""
    **Performance Tips:**
    - Use smaller batch sizes (10-50) for initial testing
    - Increase API delay if you encounter rate limiting
    - Monitor database performance during large imports
    
    **Data Quality:**
    - Set minimum citation count to filter low-quality papers
    - Use specific search queries for better targeted results
    - Year range filtering helps focus on recent research
    
    **Troubleshooting:**
    - Check Neo4j connection if import fails to start
    - Review errors tab for specific issues
    - Smaller test imports help identify configuration problems
    
    **Import Strategy:**
    - Start with small imports (10-100 papers) to test configuration
    - Gradually increase size for production imports
    - Use resumable imports for very large datasets
    """)