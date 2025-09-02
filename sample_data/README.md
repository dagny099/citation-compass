# Sample Data for Testing Data Import

This directory contains sample files to help you test the file upload functionality of the Academic Citation Platform.

## 📁 Available Files

### **sample_paper_ids.txt**
- **Format**: Plain text file
- **Content**: 10 machine learning paper IDs (one per line)
- **Features**: Includes comments and formatting examples
- **Use case**: Test basic text file upload

### **sample_paper_ids.csv**
- **Format**: CSV file
- **Content**: 10 paper IDs with titles and metadata
- **Features**: Demonstrates CSV structure with multiple columns
- **Use case**: Test CSV file upload and metadata handling

## 🚀 How to Use

### **Method 1: Download from Streamlit Interface**
1. Navigate to **Data Import** page
2. Select **"Paper IDs"** import method  
3. Click **"📁 File Upload"** tab
4. Expand **"📁 Download Sample Files"**
5. Click download buttons for sample files

### **Method 2: Use Files Directly**
1. Copy files from this directory
2. Upload them via the Data Import interface
3. Test the import process with real data

## 📋 File Formats

### **Text File Format (sample_paper_ids.txt)**
```
# Sample Paper IDs for Testing Data Import
# These are real Semantic Scholar paper IDs for machine learning papers
# You can use this file to test the file upload functionality

649def34f8be52c8b66281af98ae884c09aef38f9
204e3073870fae3d05bcbc2f6a8e263d9b72e776
2b8a9c9c9d8f7e6d5c4b3a29f8e7d6c5b4a39f8e
```

### **CSV File Format (sample_paper_ids.csv)**
```csv
paper_id,title,source
649def34f8be52c8b66281af98ae884c09aef38f9,"Attention Is All You Need","user_collection"
204e3073870fae3d05bcbc2f6a8e263d9b72e776,"BERT: Pre-training of Deep Bidirectional Transformers","literature_review"
```

## 💡 Testing Scenarios

### **Small Dataset Test**
- Use these sample files (10 papers)
- Quick validation of upload functionality
- Test progress tracking with small dataset

### **File Format Testing**
- Test both .txt and .csv formats
- Verify proper parsing of each format
- Check metadata handling in CSV

### **Error Testing**
- Modify files to test error handling
- Try invalid paper IDs
- Test empty files or malformed CSV

## 🔧 Customization

### **Create Your Own Test Files**

**For .txt files:**
```bash
# Create custom text file
echo "your_paper_id_1" > my_test_papers.txt
echo "your_paper_id_2" >> my_test_papers.txt
```

**For .csv files:**
```bash
# Create custom CSV file
echo "paper_id,title,notes" > my_test_papers.csv
echo "your_paper_id_1,Paper Title 1,Important paper" >> my_test_papers.csv
```

## 📊 Expected Results

When you upload these sample files:

- **Papers Found**: 10 paper IDs detected
- **Format Validation**: All IDs should pass validation
- **Preview**: First 10 IDs shown in interface
- **Import Ready**: Files ready for import process

## ⚠️ Notes

- **Paper ID Format**: These are example format IDs for testing
- **Real Import**: For real imports, use actual Semantic Scholar paper IDs
- **API Limits**: Sample files are small to avoid rate limiting during testing
- **Test Environment**: Perfect for development and testing workflows

## 🎯 Next Steps

1. **Download** sample files from the interface
2. **Test upload** functionality with samples
3. **Create your own** files using these as templates
4. **Import real data** for your research projects

These sample files provide a safe way to test and learn the file upload functionality before importing your actual research paper collections! 🚀