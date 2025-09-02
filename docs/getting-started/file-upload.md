# File Upload - Import Your Research Collections

Quickly import large collections of academic papers using drag-and-drop file upload. Upload .txt or .csv files containing paper IDs from your research collections, literature reviews, or bibliographic databases.

## 🚀 Quick Start

### Step 1: Prepare Your File
=== "Text File (.txt)"
    Create a simple text file with one paper ID per line:
    ```txt
    649def34f8be52c8b66281af98ae884c09aef38f9
    204e3073870fae3d05bcbc2f6a8e263d9b72e776
    2b8a9c9c9d8f7e6d5c4b3a29f8e7d6c5b4a39f8e
    ```

=== "CSV File (.csv)"
    Create a CSV file with paper IDs in the first column:
    ```csv
    paper_id,title,source
    649def34f8be52c8b66281af98ae884c09aef38f9,"Attention Is All You Need","literature_review"
    204e3073870fae3d05bcbc2f6a8e263d9b72e776,"BERT: Pre-training","user_collection"
    ```

### Step 2: Upload in Interface
1. **Navigate to Data Import** page in Streamlit
2. **Select "Paper IDs"** import method
3. **Click "📁 File Upload"** tab
4. **Drag and drop** your file or click "Choose a file"
5. **Preview the paper IDs** (first 10 shown)
6. **Configure import settings** 
7. **Click "▶️ Start Import"**

### Step 3: Monitor Progress
- Watch **real-time progress bars** for import status
- Monitor **performance metrics** (papers/second, success rate)
- Review **error logs** for any issues
- Check **completion statistics** when finished

## 📁 Supported File Formats

### Text Files (.txt)
- **One paper ID per line** - Simple format for paper ID lists
- **Comments supported** - Lines starting with `#` are ignored
- **Empty lines ignored** - Flexible formatting allowed
- **UTF-8 encoding** - International character support

**Example with comments:**
```txt
# Machine Learning Survey Papers - Updated 2024
649def34f8be52c8b66281af98ae884c09aef38f9
204e3073870fae3d05bcbc2f6a8e263d9b72e776

# Additional papers from recent conference
2b8a9c9c9d8f7e6d5c4b3a29f8e7d6c5b4a39f8e
```

### CSV Files (.csv)
- **First column contains paper IDs** - Additional columns ignored
- **Header row supported** - Column names can be included
- **Standard CSV format** - Comma-separated values
- **Metadata preservation** - Keep additional information in extra columns

**Example with metadata:**
```csv
paper_id,title,journal,year,notes
649def34f8be52c8b66281af98ae884c09aef38f9,"Attention Is All You Need","NIPS",2017,"Transformer architecture"
204e3073870fae3d05bcbc2f6a8e263d9b72e776,"BERT","NAACL",2019,"Bidirectional encoder"
```

## 📖 Step-by-Step Guide

### Creating Compatible Files

=== "From Zotero"
    1. **Select papers** in your Zotero library
    2. **Right-click** → Export Items
    3. **Choose format**: "CSV" or create custom format
    4. **Extract paper IDs** from URLs or DOIs
    5. **Save as .txt** with one ID per line

=== "From Mendeley"
    1. **Go to File** → Export
    2. **Choose "Plain Text List"** format
    3. **Extract Semantic Scholar IDs** from paper metadata
    4. **Create .txt file** with extracted IDs
    5. **Validate format** before upload

=== "From Google Scholar"
    1. **Perform your search** in Google Scholar
    2. **Copy paper URLs** from search results
    3. **Extract paper IDs** from Semantic Scholar links
    4. **Create CSV** with IDs and titles
    5. **Upload to platform** using file interface

=== "From Spreadsheet"
    1. **Create column** with paper IDs
    2. **Add metadata columns** (optional: titles, sources)
    3. **Save as CSV** format
    4. **Ensure paper IDs** are in first column
    5. **Test with small sample** first

### File Upload Process

#### Interface Navigation
1. **Launch Streamlit app**: `streamlit run app.py`
2. **Open sidebar menu** (hamburger icon if collapsed)
3. **Navigate to "Data Management"** → **"Data Import"**
4. **Select import method**: Choose **"Paper IDs"**

#### Upload Configuration
=== "Basic Settings"
    - **Max Papers**: Limit total papers imported (1-10,000)
    - **Batch Size**: Papers processed per batch (recommended: 25-50)
    - **API Delay**: Time between requests (recommended: 1-2 seconds)

=== "Content Options"
    - **Include Citations**: ✅ Import citation relationships (recommended)
    - **Include Authors**: ✅ Import author information
    - **Include Venues**: ✅ Import journal/conference data
    - **Include References**: ✅ Import reference relationships

=== "Quality Filters"
    - **Min Citations**: Filter papers with fewer citations
    - **Year Range**: Publication year filtering
    - **Quality Thresholds**: Remove incomplete records

#### Upload Steps
1. **Switch to "📁 File Upload" tab**
2. **Click "Choose a file"** or drag-and-drop
3. **Wait for file validation** (automatic)
4. **Review preview** of first 10 paper IDs
5. **Adjust settings** if needed
6. **Click "▶️ Start Import"**

#### Progress Monitoring
- **Status indicator**: 🟡 Pending → 🔵 In Progress → 🟢 Complete
- **Progress bars**: Overall progress and current batch
- **Statistics**: Papers processed, citations found, errors encountered
- **Performance metrics**: Import speed, success rate, time remaining

## 📊 Sample Files

The platform provides sample files for testing:

### Download Sample Files
1. **Go to Data Import** page
2. **Select "Paper IDs"** method  
3. **Click "📁 File Upload"** tab
4. **Expand "📁 Download Sample Files"** section
5. **Download either format**:
   - `sample_paper_ids.txt` - Text format with 10 ML papers
   - `sample_paper_ids.csv` - CSV format with metadata

### Sample File Contents

**sample_paper_ids.txt**:
```txt
# Sample Machine Learning Papers for Testing
649def34f8be52c8b66281af98ae884c09aef38f9
204e3073870fae3d05bcbc2f6a8e263d9b72e776
2b8a9c9c9d8f7e6d5c4b3a29f8e7d6c5b4a39f8e
# ... 7 more papers
```

**sample_paper_ids.csv**:
```csv
paper_id,title,source
649def34f8be52c8b66281af98ae884c09aef38f9,"Attention Is All You Need","transformer_survey"
204e3073870fae3d05bcbc2f6a8e263d9b72e776,"BERT: Pre-training","nlp_collection"
# ... 8 more papers with titles and sources
```

## 💡 Common Use Cases

### Academic Research Workflows

=== "📚 Literature Reviews"
    - **Export from reference manager** (Zotero, Mendeley, EndNote)
    - **Create paper ID lists** from systematic reviews
    - **Import citation networks** for meta-analysis
    - **Track research evolution** over time

=== "🔬 Research Projects"
    - **Import dataset paper collections** for reproducibility
    - **Upload conference proceedings** for field analysis
    - **Process collaboration networks** between research groups
    - **Analyze venue-specific research** trends

=== "📊 Bibliometric Studies"
    - **Import large paper collections** for statistical analysis
    - **Process citation databases** for network metrics
    - **Upload institutional publications** for impact assessment
    - **Analyze temporal research** patterns

### File Preparation Strategies

=== "Quality Control"
    - **Remove duplicate IDs** before upload
    - **Validate paper ID format** (32-40 character strings)
    - **Check for invalid characters** (only alphanumeric allowed)
    - **Test with small samples** first

=== "Performance Optimization"
    - **Split large files** (>1000 papers) for better performance
    - **Use descriptive filenames** for organization
    - **Add comments** in .txt files explaining data source
    - **Keep original and processed** versions

=== "Error Prevention"
    - **Ensure UTF-8 encoding** for international characters
    - **Validate paper IDs** exist in Semantic Scholar
    - **Check network connection** stability
    - **Have adequate disk space** for progress files

## 🔧 Technical Specifications

### File Limits & Requirements

| Specification | Limit | Recommendation |
|---------------|-------|----------------|
| **File Size** | 200MB max | <10MB for best performance |
| **Paper Count** | 10,000 per import | Start with 100-500 |
| **Paper ID Length** | 32-40 characters | Standard Semantic Scholar format |
| **Encoding** | UTF-8 | Ensure international compatibility |
| **Line Endings** | Any format | Unix, Windows, Mac supported |

### Validation Rules
- **Paper ID Format**: Alphanumeric strings, 32-40 characters
- **No Duplicates**: Within the same file (duplicates across imports are handled)
- **Valid Characters**: Letters (a-z, A-Z) and numbers (0-9) only
- **Empty Handling**: Empty lines and comment lines (#) are ignored

### Error Handling
- **Invalid Format**: Clear error messages with line numbers
- **File Read Errors**: UTF-8 encoding problems automatically detected
- **Empty Files**: Validation prevents empty or invalid imports
- **Malformed CSV**: Pandas parsing with automatic error recovery

## 🚨 Troubleshooting

### Upload Issues

=== "File Upload Failures"
    **"No valid paper IDs found"**
    - ✅ Check file format (one ID per line for .txt)
    - ✅ Verify paper ID length (32-40 characters)
    - ✅ Ensure alphanumeric characters only
    - ✅ Remove any extra whitespace or special characters

    **"Error reading file"**
    - ✅ Save file with UTF-8 encoding
    - ✅ Check for corrupted or binary content
    - ✅ Try opening file in text editor first
    - ✅ Re-save from original application

    **"File too large"**
    - ✅ Split into multiple smaller files
    - ✅ Remove unnecessary metadata columns
    - ✅ Compress using standard text compression
    - ✅ Upload in smaller batches

=== "Import Errors"
    **"Paper not found" errors**
    - ⚠️ Some paper IDs may not exist in Semantic Scholar
    - ✅ Verify IDs are from Semantic Scholar, not other databases
    - ✅ Check for typos in paper ID strings
    - ✅ Test with known valid IDs first

    **"API rate limiting" errors**
    - ✅ Increase API delay to 2-3 seconds
    - ✅ Reduce batch size to 10-25 papers
    - ✅ Add Semantic Scholar API key to `.env`
    - ✅ Wait a few minutes and retry

    **"Database connection" errors**
    - ✅ Verify Neo4j database is running
    - ✅ Check `.env` file configuration
    - ✅ Test database connection separately
    - ✅ Ensure network connectivity

=== "Performance Issues"
    **Slow upload processing**
    - ✅ Check file size and reduce if needed
    - ✅ Verify stable internet connection
    - ✅ Close other applications using memory
    - ✅ Consider uploading during off-peak hours

    **Memory errors during import**
    - ✅ Reduce batch size to 10-20 papers
    - ✅ Close other applications
    - ✅ Restart Streamlit application
    - ✅ Consider upgrading system memory

## 📈 Performance Tips

### Optimal Configuration

For most use cases, these settings provide the best balance of speed and reliability:

```python
# Recommended settings for file uploads
max_papers = 500           # Start small, increase gradually
batch_size = 25           # Good balance of speed and memory usage
api_delay = 1.5           # Avoid rate limiting while maintaining speed
include_citations = True  # Essential for network analysis
include_authors = True    # Valuable for collaboration analysis
min_citation_count = 5    # Focus on impactful papers
```

### Large File Strategies
1. **Test first**: Upload 10-50 papers to validate process
2. **Split files**: Break >1000 papers into 500-paper chunks
3. **Schedule uploads**: Run large imports during off-hours
4. **Monitor resources**: Watch memory and CPU usage
5. **Backup progress**: Save progress files for resumable imports

### Quality Assurance
1. **Validate sources**: Ensure paper IDs come from reliable sources
2. **Check samples**: Preview uploaded data before full import
3. **Monitor error rates**: Watch for patterns in failed imports
4. **Document provenance**: Keep notes about data sources and dates

## 🎉 Success Stories

File upload has enabled researchers to:

- **Import 2,000+ papers** from systematic literature reviews in minutes
- **Process conference proceedings** with complete citation networks  
- **Upload bibliographic exports** from institutional repositories
- **Batch import research datasets** for reproducibility studies
- **Create curated paper collections** for specific research domains

## 🔗 Integration with Other Features

After successful file upload:

1. **[Train ML models](../user-guide/ml-predictions.md)** with your imported data
2. **[Analyze networks](../user-guide/network-analysis.md)** using community detection
3. **[Explore interactively](../user-guide/interactive-features.md)** with visualization tools
4. **[Generate reports](../user-guide/results-interpretation.md)** for publications

---

## 🔗 **Related Guides**

**Next Steps**:
- **[Data Import Pipeline](../user-guide/data-import.md)** - Advanced import features and configuration
- **[Interactive Features](../user-guide/interactive-features.md)** - Using file upload in the web interface
- **[Quick Start](quick-start.md)** - Complete workflow after uploading data

**Getting Started**:
- **[Demo Mode](demo-mode.md)** - Try file upload with sample data first (recommended!)
- **[Installation](installation.md)** - Platform setup requirements
- **[Configuration](configuration.md)** - Database and environment setup

**Ready to import?** Start with the [sample files](#sample-files) to test the process, then upload your own research collections! 

**Need help?** Check the [troubleshooting section](#troubleshooting) or visit the comprehensive [Data Import guide](../user-guide/data-import.md) for advanced options.