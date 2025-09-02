# Data Import - Comprehensive Import Pipeline

Build your academic citation database using the platform's sophisticated data import pipeline. Import papers from Semantic Scholar using search queries or paper ID lists, with real-time progress tracking and intelligent error handling.

## 🚀 Import Pipeline Overview

The Academic Citation Platform provides **multiple import methods** with advanced features for building comprehensive academic databases:

### Key Capabilities
- **🔍 Search-based import** - Find and import papers using academic search queries
- **📋 ID-based import** - Import specific papers using Semantic Scholar IDs  
- **📁 File upload import** - Bulk import from .txt/.csv files with paper ID lists
- **🔄 Real-time progress** - Monitor imports with detailed progress tracking
- **🛡️ Error handling** - Graceful failure handling with detailed error reporting
- **⚡ Performance optimization** - Streaming pagination and intelligent batching
- **🎯 Quality filtering** - Citation count, year range, and content quality filters

### Import Methods

=== "🔍 Search Queries"
    **Import papers by academic search terms**:
    - Natural language queries: "machine learning", "neural networks"
    - Field-specific searches: "computer vision transformers"
    - Author-focused queries: "Geoffrey Hinton deep learning"
    - Venue-specific searches: "NIPS 2023 reinforcement learning"

=== "📋 Paper IDs"
    **Import specific papers by Semantic Scholar ID**:
    - Direct paper ID lists from research collections
    - Systematic literature review paper sets
    - Citation network seed papers for expansion
    - Curated high-impact paper collections

=== "📁 File Upload"
    **Bulk import from research files**:
    - Upload .txt files with paper IDs (one per line)
    - Import .csv files with paper metadata  
    - Process bibliographic exports from reference managers
    - Handle large collections (1000+ papers) efficiently

## 🎯 Getting Started

### Quick Import via Web Interface

#### Step 1: Access Data Import
```bash
streamlit run app.py
```
1. **Open sidebar** and navigate to **"Data Management"**
2. **Select "Data Import"** from the menu
3. **Choose your import method** from the available options

#### Step 2: Configure Import
=== "Search Query Import"
    1. **Enter search terms**: e.g., "machine learning transformers"
    2. **Set max papers**: Start with 100-500 papers
    3. **Configure filters**: Citation count, year range, quality settings
    4. **Choose content options**: Citations, authors, venues, references

=== "Paper ID Import" 
    1. **Enter paper IDs**: Paste Semantic Scholar IDs (one per line)
    2. **Set batch size**: 25-50 papers per batch (recommended)
    3. **Configure processing**: API delays, retry settings
    4. **Select data types**: Choose what to import with each paper

=== "File Upload Import"
    1. **Upload your file**: Drag-and-drop or browse for .txt/.csv files
    2. **Preview paper IDs**: Review first 10 IDs for validation
    3. **Adjust settings**: Batch size, API timing, content options  
    4. **Validate format**: Ensure paper IDs meet format requirements

#### Step 3: Monitor Progress
- **Real-time progress bars** for overall and batch progress
- **Performance metrics**: Papers/second, success rate, error count
- **Status indicators**: 🟡 Pending → 🔵 In Progress → 🟢 Complete → 🔴 Failed
- **Detailed statistics**: Papers, citations, authors, venues imported

### Command Line Interface

#### Search-based Import
```bash
# Basic search import
python -m src.cli.import_data search "machine learning" --max-papers 100

# Advanced search with filtering
python -m src.cli.import_data search "neural networks" \
    --max-papers 500 \
    --batch-size 50 \
    --min-citations 10 \
    --year-range 2020 2024 \
    --api-delay 1.5
```

#### Paper ID Import
```bash
# Import specific paper IDs
python -m src.cli.import_data ids \
    649def34f8be52c8b66281af98ae884c09aef38f9 \
    204e3073870fae3d05bcbc2f6a8e263d9b72e776 \
    --batch-size 25

# Import from file
python -m src.cli.import_data ids --ids-file paper_ids.txt \
    --batch-size 50 \
    --include-citations \
    --include-authors
```

#### Configuration Options
```bash
# Full configuration example
python -m src.cli.import_data search "deep learning" \
    --max-papers 1000 \
    --batch-size 100 \
    --api-delay 1.0 \
    --min-citations 5 \
    --max-year 2024 \
    --include-citations \
    --include-authors \
    --include-venues \
    --save-progress \
    --verbose
```

### Python API Integration

#### Basic Import
```python
from src.data.import_pipeline import (
    ImportConfiguration, 
    DataImportPipeline,
    quick_import_by_search
)

# Quick search import
progress = quick_import_by_search(
    search_query="computer vision",
    max_papers=200,
    progress_callback=lambda p: print(f"Progress: {p.overall_progress_percent:.1f}%")
)

print(f"Import completed: {progress.processed_papers} papers imported")
```

#### Advanced Configuration
```python
from src.data.import_pipeline import ImportConfiguration, DataImportPipeline

# Create detailed configuration
config = ImportConfiguration(
    search_query="natural language processing",
    max_papers=1000,
    batch_size=75,
    include_citations=True,
    include_authors=True,
    include_venues=True,
    min_citation_count=10,
    year_range=(2018, 2024),
    api_delay=1.2,
    save_progress=True,
    max_workers=4
)

# Execute import with configuration
pipeline = DataImportPipeline(config)
progress = pipeline.import_papers_by_search(
    search_query=config.search_query,
    max_papers=config.max_papers
)

# Monitor results
print(f"Status: {progress.status}")
print(f"Papers: {progress.processed_papers}/{progress.total_papers}")
print(f"Citations: {progress.citations_created}")
print(f"Authors: {progress.authors_created}")
```

#### Progress Monitoring
```python
def detailed_progress_callback(progress):
    """Comprehensive progress monitoring"""
    print(f"Status: {progress.status.value}")
    print(f"Overall Progress: {progress.overall_progress_percent:.1f}%")
    print(f"Current Batch: {progress.current_batch_progress_percent:.1f}%")
    print(f"Performance: {progress.papers_per_second:.2f} papers/sec")
    print(f"Errors: {progress.error_count}")
    if progress.errors:
        print("Recent errors:")
        for error in progress.errors[-3:]:  # Show last 3 errors
            print(f"  - {error}")
    print("---")

# Use callback for detailed monitoring
progress = quick_import_by_search(
    search_query="reinforcement learning",
    max_papers=500,
    progress_callback=detailed_progress_callback
)
```

## 🔧 Configuration Reference

### Core Settings

| Parameter | Description | Range | Recommended |
|-----------|-------------|--------|-------------|
| **max_papers** | Maximum papers to import | 1-100,000 | 100-1,000 |
| **batch_size** | Papers per processing batch | 1-1,000 | 50-100 |
| **api_delay** | Delay between API requests (seconds) | 0.1-10.0 | 1.0-2.0 |

### Content Options

| Option | Description | Impact | Default |
|--------|-------------|--------|---------|
| **include_citations** | Import citation relationships | High network value | ✅ True |
| **include_authors** | Import author information | Collaboration analysis | ✅ True |
| **include_venues** | Import publication venues | Publication analysis | ✅ True |
| **include_references** | Import reference relationships | Bidirectional networks | ✅ True |

### Quality Filters

| Filter | Purpose | Range | Usage |
|--------|---------|--------|--------|
| **min_citation_count** | Filter low-impact papers | 0-10,000 | 5-20 for quality |
| **year_range** | Publication year filtering | (start_year, end_year) | Recent: (2020, 2024) |
| **min_year/max_year** | Individual year limits | 1900-2024 | Flexible filtering |

### Performance Options

| Setting | Purpose | Range | Recommendation |
|---------|---------|--------|----------------|
| **max_workers** | Concurrent processing threads | 1-8 | 2-4 for stability |
| **retry_attempts** | Failed operation retries | 1-10 | 3 for reliability |
| **save_progress** | Enable resumable imports | Boolean | ✅ True for large imports |
| **progress_file** | Custom progress file path | String | Auto-generated recommended |

## 📊 Progress Tracking

### Status Indicators

The import pipeline provides comprehensive status tracking:

| Status | Icon | Meaning | Actions Available |
|--------|------|---------|-------------------|
| **PENDING** | 🟡 | Import queued, not started | Start, Configure |
| **IN_PROGRESS** | 🔵 | Import currently running | Monitor, Pause |
| **COMPLETED** | 🟢 | Import finished successfully | Review results |
| **FAILED** | 🔴 | Import encountered errors | Review errors, Retry |
| **CANCELLED** | 🟠 | Import cancelled by user | Restart if needed |
| **PAUSED** | 🟤 | Import temporarily paused | Resume, Cancel |

### Metrics Tracking

**Real-time Statistics**:
- **Papers**: Total found, processed, successfully imported
- **Citations**: Citation relationships created in database
- **Authors**: Author records created or updated
- **Venues**: Publication venue records created
- **Performance**: Processing speed (papers/second)
- **Quality**: Success rate, error percentage
- **Time**: Elapsed time, estimated completion time

**Example Progress Output**:
```
Status: IN_PROGRESS (🔵)
Progress: 67.3% (673/1000 papers)
Current Batch: 84% (42/50 papers)
Performance: 8.2 papers/second
Success Rate: 94.7%
Citations Created: 2,347
Authors Processed: 1,891
Elapsed Time: 2:14
Estimated Remaining: 1:23
```

### Error Tracking

**Comprehensive Error Handling**:
- **Paper-level errors**: Individual paper processing failures
- **API errors**: Rate limiting, network issues, authentication
- **Database errors**: Connection issues, constraint violations
- **Validation errors**: Data format issues, missing fields

**Error Categories**:
```python
# Example error tracking
progress = pipeline.get_current_progress()

print(f"Total errors: {progress.error_count}")
print(f"Error rate: {progress.error_rate:.2%}")

# Review recent errors
for error in progress.errors[-5:]:
    print(f"Error: {error.message}")
    print(f"Type: {error.error_type}")
    print(f"Paper ID: {error.paper_id}")
    print(f"Timestamp: {error.timestamp}")
    print("---")
```

## 🎛️ Advanced Import Features

### Streaming Pagination

The platform uses **advanced streaming pagination** for improved performance:

**Benefits**:
- **25x faster** than traditional pagination for large imports
- **Real-time progress** updates during data fetching
- **Memory efficient** processing of large result sets
- **Resumable operations** with state preservation

**Technical Implementation**:
```python
from src.data.import_pipeline import DataImportPipeline

# Streaming is automatically enabled for search imports
pipeline = DataImportPipeline(config)

# Monitor streaming progress
def streaming_callback(progress):
    print(f"Fetching: {progress.current_fetch_progress:.1f}%")
    print(f"Processing: {progress.current_batch_progress:.1f}%")

pipeline.add_progress_callback(streaming_callback)
```

### Resumable Imports

**State Management** for large imports:

**Features**:
- **Automatic checkpointing** every batch
- **Progress file persistence** across application restarts
- **Intelligent resumption** from last successful batch
- **Error recovery** with retry mechanisms

**Usage**:
```python
config = ImportConfiguration(
    search_query="large scale import",
    max_papers=5000,
    save_progress=True,
    progress_file="large_import_progress.json"  # Optional custom path
)

# If import is interrupted, restart with same configuration
# Pipeline automatically resumes from last checkpoint
pipeline = DataImportPipeline(config)
progress = pipeline.resume_import()  # Resumes if checkpoint exists
```

### Intelligent Batching

**Adaptive Batch Processing** optimizes performance:

**Dynamic Adjustments**:
- **API response time monitoring** adjusts batch sizes
- **Error rate tracking** modifies retry strategies  
- **Memory usage optimization** prevents system overload
- **Network condition adaptation** adjusts API delays

**Configuration**:
```python
config = ImportConfiguration(
    max_papers=2000,
    batch_size=100,  # Starting batch size
    adaptive_batching=True,  # Enable intelligent adjustments
    max_batch_size=200,      # Upper limit for batch adjustments
    min_batch_size=25        # Lower limit for error recovery
)
```

### Quality Assurance

**Data Validation Pipeline** ensures import quality:

**Validation Stages**:
1. **Input validation** - Search queries and paper ID format checks
2. **API response validation** - Complete paper metadata verification
3. **Database constraint validation** - Foreign key and uniqueness checks
4. **Post-import validation** - Citation network integrity verification

**Quality Metrics**:
```python
# Access quality metrics after import
quality_report = pipeline.get_quality_report()

print(f"Data completeness: {quality_report.completeness_score:.1%}")
print(f"Citation coverage: {quality_report.citation_coverage:.1%}")
print(f"Author match rate: {quality_report.author_match_rate:.1%}")
print(f"Venue match rate: {quality_report.venue_match_rate:.1%}")
```

## 🛠️ Integration Patterns

### Workflow Integration

**Seamless Integration** with other platform features:

#### Import → ML Pipeline
```python
from src.data.import_pipeline import quick_import_by_search
from src.services.ml_service import get_ml_service

# Import data
progress = quick_import_by_search("machine learning", max_papers=500)

# Immediately use for ML predictions
ml_service = get_ml_service()
if progress.status == ImportStatus.COMPLETED:
    # Train model with new data
    model = ml_service.train_model()
    
    # Generate predictions
    predictions = ml_service.predict_citations(
        paper_id="some_imported_paper_id",
        top_k=10
    )
```

#### Import → Analytics Pipeline
```python
from src.data.import_pipeline import quick_import_by_search
from src.services.analytics_service import get_analytics_service

# Import domain-specific papers
progress = quick_import_by_search("computer vision", max_papers=1000)

# Analyze imported network
analytics = get_analytics_service()
if progress.status == ImportStatus.COMPLETED:
    # Community detection
    communities = analytics.detect_communities()
    
    # Network metrics
    metrics = analytics.compute_network_metrics()
    
    # Export analysis
    report = analytics.generate_network_report()
```

### Custom Workflows

**Extensible Architecture** supports custom import workflows:

#### Multi-Query Import
```python
from src.data.import_pipeline import DataImportPipeline, ImportConfiguration

queries = [
    "deep learning computer vision",
    "machine learning natural language processing", 
    "reinforcement learning robotics"
]

all_results = []
for query in queries:
    config = ImportConfiguration(
        search_query=query,
        max_papers=300,
        min_citation_count=15
    )
    
    pipeline = DataImportPipeline(config)
    progress = pipeline.import_papers_by_search(query, 300)
    all_results.append(progress)

# Combine results for unified analysis
total_papers = sum(p.processed_papers for p in all_results)
print(f"Multi-query import completed: {total_papers} papers total")
```

#### Incremental Updates
```python
from src.data.import_pipeline import DataImportPipeline
from datetime import datetime, timedelta

# Import recent papers only
recent_date = datetime.now() - timedelta(days=30)
config = ImportConfiguration(
    search_query="latest research",
    min_year=recent_date.year,
    max_papers=200,
    include_citations=True
)

# Regular update workflow
pipeline = DataImportPipeline(config)
progress = pipeline.import_papers_by_search("latest research", 200)

# Update existing ML models with new data
if progress.processed_papers > 0:
    ml_service.update_model_with_new_data()
```

## 🚨 Troubleshooting

### Common Issues

=== "Import Failures"
    **Import won't start**:
    - ✅ Check Neo4j database connection in `.env` file
    - ✅ Verify network connectivity to Semantic Scholar API
    - ✅ Ensure sufficient disk space for progress files
    - ✅ Check system memory availability (>2GB recommended)

    **Import stops unexpectedly**:
    - ✅ Review error logs in import progress file
    - ✅ Check API rate limiting messages
    - ✅ Monitor system resource usage
    - ✅ Verify database constraints and foreign keys

=== "Performance Issues"
    **Slow import speed**:
    - ✅ Increase `api_delay` to avoid rate limiting (try 2-3 seconds)
    - ✅ Reduce `batch_size` to 25-50 papers per batch
    - ✅ Add Semantic Scholar API key to `.env` for higher limits
    - ✅ Check network connectivity and DNS resolution

    **Memory issues during import**:
    - ✅ Reduce `batch_size` to 10-25 papers
    - ✅ Disable concurrent processing (`max_workers=1`)
    - ✅ Close other applications to free system memory
    - ✅ Consider incremental imports instead of large batches

=== "Data Quality Issues"
    **High error rates (>10%)**:
    - ✅ Check search query specificity (avoid overly broad terms)
    - ✅ Review paper ID format for ID-based imports
    - ✅ Verify file encoding for file uploads (use UTF-8)
    - ✅ Monitor API response codes for authentication issues

    **Missing citations/authors**:
    - ✅ Enable `include_citations=True` and `include_authors=True`
    - ✅ Check API limits haven't been exceeded
    - ✅ Verify paper quality (some papers have incomplete metadata)
    - ✅ Review database schema constraints

### Debug Mode

**Enable Verbose Logging** for detailed troubleshooting:

```bash
# CLI with detailed logging
python -m src.cli.import_data search "debug test" \
    --max-papers 10 \
    --verbose \
    --log-level DEBUG

# Check specific log files
tail -f logs/import.log
tail -f logs/app.log
```

**Python API Debugging**:
```python
import logging
from src.data.import_pipeline import DataImportPipeline

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Create pipeline with debug options
config = ImportConfiguration(
    search_query="debug import",
    max_papers=5,
    debug_mode=True,
    verbose_errors=True
)

pipeline = DataImportPipeline(config)

# Monitor detailed progress
def debug_callback(progress):
    print(f"DEBUG: {progress.debug_info}")
    if progress.current_error:
        print(f"Current error: {progress.current_error}")

progress = pipeline.import_papers_by_search("debug", 5, debug_callback)
```

### Performance Optimization

**Optimal Settings** for different scenarios:

=== "Small Imports (<500 papers)"
    ```python
    config = ImportConfiguration(
        max_papers=500,
        batch_size=50,
        api_delay=1.0,
        max_workers=2,
        include_citations=True,
        include_authors=True
    )
    ```

=== "Medium Imports (500-2000 papers)"
    ```python
    config = ImportConfiguration(
        max_papers=2000,
        batch_size=75,
        api_delay=1.5,
        max_workers=4,
        save_progress=True,
        adaptive_batching=True
    )
    ```

=== "Large Imports (2000+ papers)"
    ```python
    config = ImportConfiguration(
        max_papers=5000,
        batch_size=100,
        api_delay=2.0,
        max_workers=6,
        save_progress=True,
        progress_file="large_import.json",
        adaptive_batching=True,
        retry_attempts=5
    )
    ```

## 📈 Performance Benchmarks

### Typical Import Speeds

| Papers | Time Range | Factors |
|--------|------------|---------|
| **10-100** | 1-5 minutes | Network speed, API delays |
| **100-500** | 5-20 minutes | Batch size, citation inclusion |
| **500-2000** | 20-60 minutes | Database performance, system resources |
| **2000+** | 1+ hours | All factors, resumable imports recommended |

### Optimization Impact

| Optimization | Speed Improvement | Trade-offs |
|-------------|------------------|------------|
| **API Key** | 2-3x faster | Requires registration |
| **Larger Batches** | 20-30% faster | Higher memory usage |
| **Fewer Inclusions** | 30-50% faster | Less comprehensive data |
| **Higher API Delay** | Slower but stable | Avoids rate limiting |

## 🔗 Next Steps

After successful data import:

1. **[Train ML Models](ml-predictions.md)** - Use imported data for citation prediction
2. **[Analyze Networks](network-analysis.md)** - Explore citation and collaboration networks
3. **[Interactive Exploration](interactive-features.md)** - Visualize and interact with your data
4. **[Generate Reports](results-interpretation.md)** - Create publication-ready analysis
5. **[API Integration](../developer-guide/architecture.md)** - Build custom applications

---

## 🔗 **Related Guides**

**Getting Started**:
- **[Demo Datasets](demo-datasets.md)** - Try import features with sample data first!
- **[File Upload Guide](../getting-started/file-upload.md)** - Step-by-step file upload tutorial
- **[Demo Mode](../getting-started/demo-mode.md)** - Zero-setup exploration

**Using Your Imported Data**:
- **[ML Predictions](ml-predictions.md)** - Citation prediction with your data
- **[Network Analysis](network-analysis.md)** - Explore citation and collaboration networks  
- **[Interactive Features](interactive-features.md)** - Visualize and interact with imported data
- **[Results Interpretation](results-interpretation.md)** - Generate reports and analysis

**Technical Guides**:
- **[Configuration](../getting-started/configuration.md)** - Database and API setup
- **[Quick Start](../getting-started/quick-start.md)** - Complete workflow guide
- **[Notebook Pipeline](notebook-pipeline.md)** - Programmatic analysis workflows

**Ready to build your research database?** Start with a [demo dataset](demo-datasets.md) to learn the workflow, then use [file upload](../getting-started/file-upload.md) for your research collections! 🚀