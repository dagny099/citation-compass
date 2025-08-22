"""
Data validation utilities for Academic Citation Platform.
"""

import re
from typing import List, Dict, Any, Optional
import pandas as pd


def validate_paper_id(paper_id: str) -> bool:
    """Validate paper ID format."""
    if not paper_id or not isinstance(paper_id, str):
        return False
    return len(paper_id.strip()) > 0 and len(paper_id) <= 100


def validate_author_id(author_id: str) -> bool:
    """Validate author ID format."""
    if not author_id or not isinstance(author_id, str):
        return False
    return len(author_id.strip()) > 0 and len(author_id) <= 100


def validate_doi(doi: str) -> bool:
    """Validate DOI format."""
    if not doi:
        return True  # DOI is optional
    
    # Basic DOI format validation
    doi_pattern = r'^10\.\d+/.+'
    return bool(re.match(doi_pattern, doi))


def validate_year(year: Optional[int]) -> bool:
    """Validate publication year."""
    if year is None:
        return True  # Year is optional
    return 1900 <= year <= 2030


def validate_dataframe_schema(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """
    Validate DataFrame schema against required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "is_valid": True,
        "missing_columns": [],
        "extra_columns": [],
        "row_count": len(df),
        "column_count": len(df.columns)
    }
    
    # Check for missing columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        validation_results["missing_columns"] = list(missing_cols)
        validation_results["is_valid"] = False
    
    # Check for extra columns (informational only)
    extra_cols = set(df.columns) - set(required_columns)
    if extra_cols:
        validation_results["extra_columns"] = list(extra_cols)
    
    return validation_results