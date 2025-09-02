# File Upload Guide for Data Import

The Academic Citation Platform now supports **file upload** for importing paper IDs, making it easy to import large lists of papers from your research collections, literature reviews, or bibliographic databases.

## 🚀 **New File Upload Features**

### **📁 File Upload Interface**
- **Two Input Methods**: Manual text input or file upload via tabs
- **Multiple Formats**: Support for .txt and .csv files
- **Real-time Preview**: See the first 10 paper IDs after upload
- **Validation**: Automatic validation of paper ID format
- **Error Handling**: Clear error messages for invalid files

### **📋 Supported File Formats**

#### **Text Files (.txt)**
- One paper ID per line
- Comments supported (lines starting with #)
- Empty lines ignored
- UTF-8 encoding

**Example format:**
```
649def34f8be52c8b66281af98ae884c09aef38f9
204e3073870fae3d05bcbc2f6a8e263d9b72e776
2b8a9c9c9d8f7e6d5c4b3a29f8e7d6c5b4a39f8e
```

#### **CSV Files (.csv)**
- Paper IDs in the first column
- Additional columns ignored (can contain metadata)
- Header row supported
- Standard CSV format

**Example format:**
```csv
paper_id,title,source
649def34f8be52c8b66281af98ae884c09aef38f9,"Attention Is All You Need","user_collection"
204e3073870fae3d05bcbc2f6a8e263d9b72e776,"BERT: Pre-training","literature_review"
```

## 📖 **How to Use File Upload**

### **Step 1: Prepare Your File**
1. **Create a text file** with paper IDs (one per line)
2. **OR create a CSV file** with paper IDs in the first column
3. **Save the file** with .txt or .csv extension

### **Step 2: Upload in Streamlit Interface**
1. Navigate to **Data Import** page
2. Select **"Paper IDs"** import method
3. Click on **"📁 File Upload"** tab
4. Click **"Choose a file"** button
5. Select your .txt or .csv file
6. **Preview** the loaded paper IDs
7. Configure import settings
8. Click **"▶️ Start Import"**

### **Step 3: Monitor Progress**
- Watch real-time progress bars
- Monitor performance metrics
- Review any errors or warnings
- Check completion statistics

## 🎯 **Sample Files Available**

The platform provides sample files you can download and test:

### **📄 sample_paper_ids.txt**
- 10 machine learning paper IDs
- Text format with comments
- Ready to upload and test

### **📊 sample_paper_ids.csv**
- 10 paper IDs with titles and sources
- CSV format with metadata
- Demonstrates column structure

**To get sample files:**
1. Go to Data Import page
2. Select "Paper IDs" method
3. Click "📁 File Upload" tab
4. Expand "📁 Download Sample Files"
5. Download either .txt or .csv sample

## 💡 **Use Cases**

### **📚 Academic Research**
- Import papers from your **Zotero/Mendeley** library
- Upload **literature review** paper lists
- Import **citation lists** from academic papers
- Process **bibliography** exports

### **🔬 Research Projects**
- Import **dataset paper collections**
- Upload **conference proceeding** lists
- Process **author collaboration** networks
- Import **venue-specific** paper collections

### **📊 Data Analysis**
- Import papers for **network analysis**
- Upload **citation graph** node lists
- Process **bibliometric** study datasets
- Import **temporal analysis** paper sets

## 🛠️ **Creating Compatible Files**

### **From Bibliographic Software**
1. **Zotero**: Export → Format: "CSV" or "Plain Text"
2. **Mendeley**: File → Export → "Plain Text List"
3. **EndNote**: Export → "Tab Delimited" format

### **From Academic Databases**
1. **Google Scholar**: Copy paper URLs and extract IDs
2. **Semantic Scholar**: Export search results as CSV
3. **DBLP**: Export paper lists in text format

### **From Spreadsheets**
1. Create column with paper IDs
2. Save as CSV format
3. Ensure paper IDs are in first column

## ⚙️ **Technical Details**

### **File Size Limits**
- **Maximum file size**: 200MB (Streamlit default)
- **Recommended**: Under 10MB for best performance
- **Paper ID limit**: 10,000 papers per import

### **Validation Rules**
- Paper IDs must be **32-40 characters** (typical Semantic Scholar format)
- **Alphanumeric** characters only
- **No duplicates** within the same file
- **Empty lines** and **comments** (#) ignored

### **Error Handling**
- **Invalid format**: Clear error messages with line numbers
- **File read errors**: UTF-8 encoding issues handled
- **Empty files**: Validation prevents empty imports
- **Malformed CSV**: Pandas parsing with error recovery

## 🔧 **Advanced Usage**

### **Command Line Alternative**
You can also use the CLI for file-based imports:

```bash
# Import from text file
python -m src.cli.import_data ids --ids-file paper_ids.txt

# Import with custom settings
python -m src.cli.import_data ids --ids-file paper_ids.txt \
    --batch-size 25 \
    --api-delay 1.5 \
    --no-citations
```

### **Python API**
For programmatic access:

```python
from src.data.import_pipeline import quick_import_by_ids

# Load paper IDs from file
with open('paper_ids.txt', 'r') as f:
    paper_ids = [line.strip() for line in f if line.strip()]

# Import with progress tracking
progress = quick_import_by_ids(
    paper_ids,
    progress_callback=lambda p: print(f"Progress: {p.overall_progress_percent:.1f}%")
)
```

## 📈 **Performance Tips**

### **File Preparation**
- **Remove duplicates** before uploading
- **Validate paper IDs** in external tools first
- **Split large files** (>1000 papers) for better performance
- **Use descriptive filenames** for organization

### **Import Configuration**
- **Start small**: Test with 10-50 papers first
- **Batch size**: Use 25-50 for file imports
- **API delay**: Use 1-2 seconds to avoid rate limiting
- **Monitor progress**: Watch for errors and warnings

### **System Resources**
- **Memory usage**: Monitor during large imports
- **Database performance**: Ensure Neo4j has adequate resources
- **Network stability**: Stable connection for API calls

## 🚨 **Troubleshooting**

### **File Upload Issues**
- **"No valid paper IDs found"**: Check file format and content
- **"Error reading file"**: Ensure UTF-8 encoding
- **"File too large"**: Split into smaller files
- **"Upload failed"**: Try refreshing the page

### **Paper ID Issues**
- **Invalid format**: Ensure 32-40 character alphanumeric strings
- **Not found errors**: Some paper IDs may not exist in Semantic Scholar
- **Access denied**: Some papers may have restricted access

### **Performance Issues**
- **Slow uploads**: Check file size and internet connection
- **Memory errors**: Reduce batch size and file size
- **API timeouts**: Increase API delay setting

## 📋 **Best Practices**

### **File Organization**
- **Naming convention**: Use descriptive names (e.g., `ml_survey_2024.txt`)
- **Version control**: Keep original and processed versions
- **Documentation**: Add comments in .txt files explaining source
- **Backup**: Keep copies of important paper ID collections

### **Quality Control**
- **Validate sources**: Ensure paper IDs are from reliable sources
- **Check duplicates**: Remove duplicate entries before import
- **Preview results**: Use sample files to test process first
- **Monitor imports**: Watch progress and error rates

### **Data Management**
- **Incremental imports**: Import in stages rather than all at once
- **Error tracking**: Save error logs for problematic paper IDs
- **Progress monitoring**: Use progress callbacks for large imports
- **Result verification**: Check imported data in database

## 🎉 **Get Started**

1. **Download sample files** from the Data Import page
2. **Test the upload process** with small samples
3. **Prepare your own files** using the format guidelines
4. **Start importing** your research paper collections!

The file upload feature makes it easy to import large collections of papers from your research workflow into the Academic Citation Platform. 🚀