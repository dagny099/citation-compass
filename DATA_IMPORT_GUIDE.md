# Data Import Pipeline Guide

The Academic Citation Platform now includes a comprehensive data import pipeline for importing papers and citations from Semantic Scholar into your Neo4j database.

## ✨ New Features

### 🔄 Bulk Data Import
- **Search-based import**: Import papers using search queries
- **ID-based import**: Import specific papers by their Semantic Scholar IDs
- **Citation network building**: Automatically create citation relationships
- **Author and venue processing**: Import complete bibliographic data

### 📊 Progress Tracking
- **Real-time progress updates**: Monitor import status with progress bars
- **Comprehensive metrics**: Track papers, citations, authors, and venues created
- **Error handling**: Detailed error reporting and graceful failure handling
- **Resumable imports**: State management for interrupted operations

### 🎛️ Flexible Configuration
- **Batch processing**: Configurable batch sizes for optimal performance
- **Filtering options**: Citation count, year range, and quality filters
- **API rate limiting**: Built-in delays and backoff strategies
- **Validation**: Comprehensive data validation before import

### 🖥️ Multiple Interfaces
- **Streamlit Web UI**: Interactive interface with real-time progress
- **Command Line Interface**: Scriptable imports for automation
- **Python API**: Programmatic access for custom workflows

## 🚀 Quick Start

### Using the Streamlit Interface

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Navigate to Data Import**:
   - Go to "Data Management" → "Data Import" in the sidebar

3. **Configure your import**:
   - Choose import method (Search Query, Paper IDs, or Sample Config)
   - Set maximum papers and batch size
   - Configure advanced options (citations, authors, filtering)

4. **Start importing**:
   - Click "▶️ Start Import"
   - Monitor progress in real-time
   - View detailed statistics and any errors

### Using the Command Line

1. **Search-based import**:
   ```bash
   python -m src.cli.import_data search "machine learning" --max-papers 100
   ```

2. **Import specific paper IDs**:
   ```bash
   python -m src.cli.import_data ids paper1 paper2 paper3 --batch-size 10
   ```

3. **Import from file**:
   ```bash
   python -m src.cli.import_data ids --ids-file paper_ids.txt
   ```

4. **Advanced filtering**:
   ```bash
   python -m src.cli.import_data search "neural networks" \
     --min-citations 10 \
     --year-range 2020 2024 \
     --batch-size 50
   ```

### Using the Python API

```python
from src.data.import_pipeline import (
    ImportConfiguration, 
    DataImportPipeline,
    quick_import_by_search
)

# Quick import
progress = quick_import_by_search(
    search_query="machine learning",
    max_papers=100,
    progress_callback=lambda p: print(f"Progress: {p.overall_progress_percent:.1f}%")
)

# Advanced configuration
config = ImportConfiguration(
    search_query="deep learning",
    max_papers=500,
    batch_size=50,
    include_citations=True,
    include_authors=True,
    min_citation_count=5,
    year_range=(2020, 2024)
)

pipeline = DataImportPipeline(config)
progress = pipeline.import_papers_by_search("deep learning", 500)
```

## 📋 Configuration Options

### Basic Configuration
- **max_papers**: Maximum number of papers to import (1-100,000)
- **batch_size**: Papers processed per batch (1-1,000, recommended: 50-100)
- **api_delay**: Delay between API requests in seconds (0.1-10.0)

### Content Options
- **include_citations**: Import citation relationships (recommended: True)
- **include_references**: Import reference relationships (recommended: True)
- **include_authors**: Import author information (recommended: True)
- **include_venues**: Import publication venue information (recommended: True)

### Filtering Options
- **min_citation_count**: Filter papers with fewer citations (default: 0)
- **year_range**: Publication year range, e.g., (2020, 2024)
- **min_year/max_year**: Individual year limits

### Advanced Options
- **save_progress**: Enable resumable imports (recommended: True)
- **progress_file**: Custom progress file path
- **max_workers**: Number of concurrent workers (1-8)
- **retry_attempts**: Number of retry attempts for failed operations

## 🔍 Import Methods

### Search Query Import
Import papers based on search terms:

```python
# Examples of effective search queries
"machine learning"
"deep learning neural networks"
"computer vision transformers"
"natural language processing BERT"
"reinforcement learning robotics"
```

### Paper ID Import
Import specific papers by Semantic Scholar ID:

```python
paper_ids = [
    "649def34f8be52c8b66281af98ae884c09aef38f9",
    "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
    "2b8a9c9c9d8f7e6d5c4b3a29f8e7d6c5b4a39f8e"
]
```

### Sample Configuration
Use pre-configured settings for testing:
- Search query: "machine learning"
- Max papers: 100
- Includes citations, authors, venues
- Minimum 5 citations per paper
- Year range: 2020-2024

## 📊 Progress Monitoring

### Status Indicators
- 🟡 **PENDING**: Import not yet started
- 🔵 **IN_PROGRESS**: Import currently running
- 🟢 **COMPLETED**: Import finished successfully
- 🔴 **FAILED**: Import encountered errors
- 🟠 **CANCELLED**: Import cancelled by user
- 🟤 **PAUSED**: Import temporarily paused

### Metrics Tracked
- **Papers**: Total/processed/created counts
- **Citations**: Citation relationships created
- **Authors**: Author records created
- **Venues**: Publication venue records created
- **Elapsed Time**: Total time since import start
- **Errors/Warnings**: Detailed error reporting

### Progress Callbacks
Register callback functions to receive real-time updates:

```python
def progress_callback(progress):
    print(f"Status: {progress.status.value}")
    print(f"Papers: {progress.processed_papers}/{progress.total_papers}")
    print(f"Progress: {progress.overall_progress_percent:.1f}%")

pipeline.add_progress_callback(progress_callback)
```

## 🛠️ Data Quality and Validation

### Input Validation
- **Paper data**: Validates required fields (paperId, title)
- **Citation data**: Validates source/target relationships
- **Author data**: Validates author IDs and names
- **Configuration**: Validates import parameters

### Quality Filters
- **Minimum citation count**: Filter low-impact papers
- **Year range filtering**: Focus on recent research
- **Abstract length checks**: Identify incomplete records
- **Duplicate detection**: Prevent duplicate imports

### Error Handling
- **Graceful failures**: Continue processing despite individual errors
- **Detailed logging**: Comprehensive error messages and stack traces
- **Recovery mechanisms**: Retry logic for transient failures
- **State preservation**: Save progress for resumable imports

## 🎯 Best Practices

### Performance Optimization
- **Start small**: Test with 10-100 papers before large imports
- **Appropriate batch size**: Use 50-100 for most imports
- **Monitor database**: Watch Neo4j performance during large imports
- **API rate limiting**: Use 1-2 second delays to avoid rate limits

### Data Strategy
- **Focused queries**: Use specific search terms for better quality
- **Incremental imports**: Import data in stages rather than all at once
- **Quality thresholds**: Set minimum citation counts for academic relevance
- **Year filtering**: Focus on recent publications for current research

### Troubleshooting
- **Database connection**: Ensure Neo4j is running and accessible
- **Memory usage**: Monitor system memory during large imports
- **API limits**: Watch for rate limiting messages
- **Disk space**: Ensure sufficient space for progress files and logs

## 📁 File Structure

New files added to the platform:

```
src/data/
├── import_pipeline.py          # Main import pipeline implementation
└── ...

src/utils/
├── validation.py               # Enhanced validation functions
└── ...

src/cli/
├── import_data.py              # Command-line interface
└── ...

src/streamlit_app/pages/
├── Data_Import.py              # Streamlit interface
└── ...

tests/
├── test_import_pipeline.py     # Comprehensive test suite
└── ...
```

## 🔧 Configuration Files

### Example Configuration JSON
```json
{
  "search_query": "machine learning",
  "max_papers": 500,
  "batch_size": 50,
  "include_citations": true,
  "include_authors": true,
  "include_venues": true,
  "min_citation_count": 5,
  "year_range": [2020, 2024],
  "api_delay": 1.0
}
```

### Environment Variables
Ensure your `.env` file is configured:
```env
NEO4J_URI=neo4j+s://your-database-url
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
SEMANTIC_SCHOLAR_API_KEY=your-api-key  # Optional but recommended
```

## 🚨 Troubleshooting

### Common Issues

1. **Import fails to start**
   - Check Neo4j database connection
   - Verify environment variables in `.env`
   - Ensure database has proper constraints

2. **Rate limiting errors**
   - Increase `api_delay` parameter
   - Add Semantic Scholar API key
   - Reduce batch size

3. **Memory issues**
   - Reduce batch size (try 10-25)
   - Monitor system memory usage
   - Close other applications

4. **Validation errors**
   - Check search query syntax
   - Verify paper IDs format
   - Review configuration parameters

### Debug Mode
Enable verbose logging:
```bash
python -m src.cli.import_data search "test" --verbose
```

### Log Files
Check these files for detailed information:
- `import.log`: CLI import logs
- `logs/app.log`: Application logs
- Import progress files: `import_progress_*.json`

## 📈 Performance Benchmarks

### Typical Performance
- **Small imports (10-100 papers)**: 1-5 minutes
- **Medium imports (100-1000 papers)**: 5-30 minutes  
- **Large imports (1000+ papers)**: 30+ minutes

### Factors Affecting Speed
- **API delay settings**: Lower delays = faster but risk rate limiting
- **Batch size**: Larger batches = fewer database transactions
- **Network speed**: Affects API response times
- **Database performance**: Neo4j configuration and hardware
- **Include options**: More data = longer processing time

## 🔮 Future Enhancements

### Planned Features
- **Incremental updates**: Refresh existing papers with new citation data
- **Advanced filtering**: More sophisticated quality metrics
- **Export formats**: JSON, CSV, and other output options
- **Scheduling**: Automated periodic imports
- **Multi-source support**: Additional academic databases

### API Extensions
- **Batch endpoints**: More efficient API usage
- **Webhook support**: Real-time import notifications
- **Import templates**: Pre-configured import scenarios
- **Data lineage**: Track import provenance and history

## 💡 Tips for Success

1. **Start with test imports** to understand data quality and timing
2. **Use specific search queries** for more relevant results  
3. **Monitor database performance** during large imports
4. **Set appropriate filters** to focus on high-quality papers
5. **Save configurations** for repeatable import processes
6. **Use resumable imports** for large datasets
7. **Regular backups** before major import operations

---

## 🎉 What's Next?

The data import pipeline provides the foundation for populating your academic citation database. Once you have imported papers and citations:

1. **Train ML models** using the populated data
2. **Explore network analysis** with the enhanced visualization tools
3. **Generate predictions** using the ML prediction interface
4. **Export results** for academic publications

The platform now provides a complete workflow from data import to research insights! 🚀