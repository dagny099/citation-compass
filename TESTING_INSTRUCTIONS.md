# Academic Citation Platform - Testing Instructions

## Overview

This document provides comprehensive step-by-step instructions for testing the Academic Citation Platform. The platform combines interactive web interfaces, ML prediction capabilities, and robust data collection from three reference codebases.

## Prerequisites

### System Requirements
- Python 3.8+ (recommended: 3.10+)
- 4GB+ RAM (for ML model loading)
- 500MB+ disk space (for models and cache)
- Internet connection (for Semantic Scholar API)

### Environment Setup
```bash
# 1. Navigate to project directory
cd /Users/bhs/PROJECTS/academic-citation-platform

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify PyTorch installation (required for ML service)
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Section 1: Core Infrastructure Testing

### 1.1 Data Model Validation
```bash
# Run data model tests
python -m pytest tests/test_models_simple.py -v

# Expected output: All tests should pass
# Tests cover: Paper, Author, Citation models with Pydantic v2 validation
```

### 1.2 Fixture Infrastructure Testing
```bash
# Test all fixtures (44+ fixtures)
python -m pytest tests/test_fixtures.py -v

# Expected output: All fixture tests pass
# Validates: Data fixtures, model fixtures, configuration fixtures
```

### 1.3 Configuration System Testing
```bash
# Test configuration loading
python -c "
from src.data.api_config import get_config
config = get_config()
print(f'Platform config loaded: {config.platform.name}')
print(f'API rate limit: {config.semantic_scholar.requests_per_minute}')
"

# Expected output: Configuration values should be displayed
```

## Section 2: API Client Testing

### 2.1 Unified API Client Tests
```bash
# Run API client tests (uses mocks)
python -m pytest tests/test_unified_api_client.py -v

# Expected output: All API client functionality tests pass
# Tests cover: Rate limiting, caching, pagination, error handling
```

### 2.2 Live API Testing (Optional - requires internet)
```bash
# Test live API connection (optional, uses real API)
python -c "
from src.data.unified_api_client import UnifiedSemanticScholarClient
client = UnifiedSemanticScholarClient()
result = client.search_papers('machine learning', limit=1)
print(f'API test successful: {len(result.get(\"data\", []))} papers found')
"

# Expected output: Should return 1 paper result
# Note: This uses real API calls, so run sparingly
```

## Section 3: ML Service Testing

### 3.1 ML Service Initialization
```bash
# Test ML service can load (creates mock model if needed)
python -m pytest tests/test_ml_service.py::TestTransEModel::test_model_initialization -v

# Expected output: ML model initialization test passes
```

### 3.2 Prediction Generation Testing
```bash
# Test prediction generation
python -m pytest tests/test_ml_service.py::TestTransEModel -v

# Expected output: All ML model tests pass
# Tests cover: Model loading, predictions, caching, embeddings
```

### 3.3 Prediction Cache Testing
```bash
# Test prediction caching
python -m pytest tests/test_ml_service.py::TestPredictionCache -v

# Expected output: Cache tests pass
# Validates: Cache hit/miss, expiration, memory management
```

## Section 4: Database Integration Testing

### 4.1 Database Layer Testing
```bash
# Test database functionality (uses mocks)
python -m pytest tests/test_unified_database.py -v

# Expected output: All database tests pass
# Tests cover: Schema validation, query library, connection management
```

### 4.2 Schema Validation Testing
```bash
# Test schema validator
python -m pytest tests/test_unified_database.py::TestSchemaValidator -v

# Expected output: Schema validation tests pass
```

## Section 5: Integration Testing

### 5.1 Service Integration Tests
```bash
# Run comprehensive integration tests
python -m pytest tests/test_integration.py -v --tb=short

# Expected output: All integration tests pass
# Tests cover: Service coordination, end-to-end workflows, error handling
```

### 5.2 End-to-End Prediction Workflow
```bash
# Test complete prediction workflow
python -m pytest tests/test_integration.py::TestServiceIntegration::test_end_to_end_prediction_workflow -v

# Expected output: Complete workflow test passes
# Validates: ML service + API client coordination
```

## Section 6: Validation and Security Testing

### 6.1 Data Validation Tests
```bash
# Run validation test suite
python -m pytest tests/test_validation.py -v

# Expected output: All validation tests pass
# Tests cover: Input validation, security, business logic, performance
```

### 6.2 Security Validation
```bash
# Test security aspects
python -m pytest tests/test_validation.py::TestSecurityValidation -v

# Expected output: Security validation tests pass
# Validates: Input sanitization, path traversal protection, API key handling
```

## Section 7: Streamlit Application Testing

### 7.1 Application Structure Validation
```bash
# Test Streamlit app structure
python -m pytest tests/test_integration.py::TestStreamlitIntegration -v

# Expected output: Streamlit structure tests pass
# Validates: Page files exist, imports work, configuration present
```

### 7.2 Manual Streamlit Testing
```bash
# Launch Streamlit application
streamlit run app.py

# Manual testing steps:
# 1. Verify home page loads with navigation sidebar
# 2. Navigate to "ML Predictions" page
# 3. Test paper ID input (use: "649def34f8be52c8b66281af98ae884c09aef38f9")
# 4. Generate predictions and verify results display
# 5. Navigate to "Embedding Explorer" page
# 6. Test embedding visualization
# 7. Navigate to "Enhanced Visualizations" page
# 8. Test network visualization with prediction overlays
# 9. Navigate to "Notebook Pipeline" page  
# 10. Test interactive analysis workflow

# Expected behavior:
# - All pages load without errors
# - Navigation sidebar works correctly
# - ML predictions generate and display properly
# - Visualizations render interactive charts
# - Export functionality works
```

## Section 8: Performance Testing

### 8.1 Performance Validation
```bash
# Test performance bounds
python -m pytest tests/test_validation.py::TestPerformanceValidation -v

# Expected output: Performance tests pass
# Validates: Response times, memory usage, concurrent access
```

### 8.2 Load Testing (Optional)
```bash
# Test with multiple concurrent predictions
python -c "
import concurrent.futures
from src.services.ml_service import get_ml_service
import time

ml_service = get_ml_service()
paper_ids = ['paper_1', 'paper_2', 'paper_3']

start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(ml_service.predict_citations, pid) for pid in paper_ids]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

print(f'Concurrent predictions completed in {time.time() - start_time:.2f}s')
print(f'Results: {len(results)} prediction sets generated')
"

# Expected output: Concurrent operations complete successfully
```

## Section 9: Complete System Testing

### 9.1 Full Test Suite
```bash
# Run complete test suite
python -m pytest tests/ -v --tb=short

# Expected output: All tests pass (may take 2-3 minutes)
# This runs all 100+ tests across all modules
```

### 9.2 Test Coverage Analysis
```bash
# Install coverage tool
pip install pytest-cov

# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Expected output: Coverage report showing >80% coverage
# HTML report available in htmlcov/index.html
```

## Section 10: Production Readiness Testing

### 10.1 Configuration Testing
```bash
# Test different environment configurations
python -c "
import os
os.environ['ENVIRONMENT'] = 'production'
from src.data.api_config import get_config
config = get_config()
print(f'Production config loaded: {config.platform.environment}')
"

# Expected output: Production configuration loads correctly
```

### 10.2 Model Health Check
```bash
# Test ML service health check
python -c "
from src.services.ml_service import get_ml_service
ml_service = get_ml_service()
health = ml_service.health_check()
print(f'ML Service Health: {health}')
"

# Expected output: Health check returns 'healthy' status
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: PyTorch Installation Problems
```bash
# Solution: Install PyTorch manually
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Issue 2: Streamlit Import Errors
```bash
# Solution: Install missing dependencies
pip install streamlit plotly networkx pandas numpy
```

#### Issue 3: Model Loading Failures
```bash
# Solution: Check model files exist and are readable
ls -la models/
python -c "import torch; print('PyTorch working correctly')"
```

#### Issue 4: API Rate Limiting
```bash
# Solution: Reduce test frequency or use mocks
export SEMANTIC_SCHOLAR_API_KEY="your_key_here"  # If available
```

#### Issue 5: Memory Issues During Testing
```bash
# Solution: Run tests in smaller batches
python -m pytest tests/test_ml_service.py -v  # Test ML service only
python -m pytest tests/test_integration.py -v  # Test integration only
```

## Test Results Interpretation

### Expected Test Outcomes

1. **All unit tests pass (>90 tests)**: Validates individual component functionality
2. **Integration tests pass (>20 tests)**: Validates component interaction
3. **Validation tests pass (>15 tests)**: Validates data integrity and security
4. **Streamlit app launches successfully**: Validates web interface
5. **ML predictions generate correctly**: Validates core ML functionality
6. **Visualizations render properly**: Validates interactive components
7. **Navigation works correctly**: Validates multi-page structure

### Performance Benchmarks

- **Single prediction**: < 100ms
- **Batch predictions (10 papers)**: < 1s
- **Model loading**: < 5s
- **Streamlit page load**: < 2s
- **Memory usage**: < 500MB during operation

## Next Steps After Testing

1. **If all tests pass**: System is ready for production deployment
2. **If tests fail**: Review error messages, check troubleshooting guide
3. **Performance issues**: Consider model optimization or caching improvements
4. **Feature requests**: Use the modular architecture to add new capabilities

## Test Automation

For continuous testing, consider setting up:

```bash
# Create test automation script
cat > run_tests.sh << 'EOF'
#!/bin/bash
set -e

echo "Running Academic Citation Platform Test Suite..."
echo "=============================================="

echo "1. Running unit tests..."
python -m pytest tests/test_models_simple.py tests/test_fixtures.py -v

echo "2. Running service tests..."
python -m pytest tests/test_ml_service.py tests/test_unified_api_client.py -v

echo "3. Running integration tests..."
python -m pytest tests/test_integration.py -v

echo "4. Running validation tests..."
python -m pytest tests/test_validation.py -v

echo "All tests completed successfully!"
EOF

chmod +x run_tests.sh
./run_tests.sh
```

This completes the comprehensive testing instructions for the Academic Citation Platform.