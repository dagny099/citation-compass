"""
Data validation utilities for academic citation data.
Ensures data quality and consistency throughout the pipeline.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Set, Tuple, Optional
import ast

logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation and quality checking utilities."""
    
    @staticmethod
    def validate_paper_data(data: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Validate paper data and return clean data with error log.
        
        Args:
            data: List of paper dictionaries
            
        Returns:
            Tuple of (validated_data, error_messages)
        """
        validated_data = []
        errors = []
        
        for i, item in enumerate(data):
            # Handle string representations of data
            if isinstance(item, str):
                try:
                    item = ast.literal_eval(item)
                except (ValueError, SyntaxError):
                    errors.append(f"Row {i}: Cannot parse string data")
                    continue
            
            if not isinstance(item, dict):
                errors.append(f"Row {i}: Data is not a dictionary")
                continue
            
            # Check required fields
            if not item.get('paperId'):
                errors.append(f"Row {i}: Missing paperId")
                continue
                
            if not item.get('title'):
                errors.append(f"Row {i}: Missing title")
                continue
            
            # Validate numeric fields
            for field in ['referenceCount', 'citationCount', 'year']:
                if field in item and item[field] is not None:
                    try:
                        item[field] = int(float(item[field]))
                    except (ValueError, TypeError):
                        errors.append(f"Row {i}: Invalid {field} value: {item[field]}")
                        item[field] = 0
            
            # Validate authors field
            if 'authors' in item and item['authors']:
                if isinstance(item['authors'], str):
                    try:
                        item['authors'] = ast.literal_eval(item['authors'])
                    except (ValueError, SyntaxError):
                        errors.append(f"Row {i}: Cannot parse authors field")
                        item['authors'] = []
            
            # Validate fieldsOfStudy
            if 'fieldsOfStudy' in item and item['fieldsOfStudy']:
                if isinstance(item['fieldsOfStudy'], str):
                    try:
                        item['fieldsOfStudy'] = ast.literal_eval(item['fieldsOfStudy'])
                    except (ValueError, SyntaxError):
                        errors.append(f"Row {i}: Cannot parse fieldsOfStudy")
                        item['fieldsOfStudy'] = []
            
            validated_data.append(item)
        
        return validated_data, errors
    
    @staticmethod
    def check_data_completeness(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, float]:
        """
        Check completeness of data across required columns.
        
        Args:
            df: DataFrame to check
            required_columns: List of required column names
            
        Returns:
            Dictionary of column names to completeness percentages
        """
        completeness = {}
        
        for col in required_columns:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                completeness[col] = (non_null_count / len(df)) * 100
            else:
                completeness[col] = 0.0
        
        return completeness
    
    @staticmethod
    def detect_duplicates(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
        """
        Detect and return duplicate records based on ID column.
        
        Args:
            df: DataFrame to check
            id_column: Column name containing unique identifiers
            
        Returns:
            DataFrame containing duplicate rows
        """
        return df[df.duplicated(subset=[id_column], keep=False)]
    
    @staticmethod
    def validate_relationships(relationships_df: pd.DataFrame, 
                             entities_df: pd.DataFrame, 
                             entity_id_col: str) -> List[str]:
        """
        Validate that all relationship entities exist in the entities table.
        
        Args:
            relationships_df: DataFrame with relationship data
            entities_df: DataFrame with entity data
            entity_id_col: Column name for entity IDs
            
        Returns:
            List of validation error messages
        """
        errors = []
        entity_ids = set(entities_df[entity_id_col].dropna())
        
        # Check if all relationship entities exist
        for col in relationships_df.columns:
            if col.endswith('Id') or col == entity_id_col:
                missing_entities = set(relationships_df[col].dropna()) - entity_ids
                if missing_entities:
                    errors.append(f"Missing entities in {col}: {len(missing_entities)} records")
        
        return errors
    
    @staticmethod
    def clean_text_fields(text: str) -> str:
        """
        Clean and normalize text fields.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters that might cause issues
        text = text.replace('\x00', '')  # Null bytes
        text = text.replace('\r', ' ')   # Carriage returns
        text = text.replace('\n', ' ')   # Newlines
        
        return text.strip()
    
    @staticmethod
    def validate_citation_counts(df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate citation count statistics and detect outliers.
        
        Args:
            df: DataFrame with citation data
            
        Returns:
            Dictionary with validation statistics
        """
        stats = {}
        
        if 'citationCount' in df.columns:
            citation_counts = df['citationCount'].dropna()
            
            stats['total_papers'] = len(df)
            stats['papers_with_citations'] = len(citation_counts[citation_counts > 0])
            stats['max_citations'] = citation_counts.max()
            stats['mean_citations'] = citation_counts.mean()
            stats['median_citations'] = citation_counts.median()
            
            # Detect potential outliers (papers with extremely high citations)
            q99 = citation_counts.quantile(0.99)
            outliers = citation_counts[citation_counts > q99 * 3]
            stats['potential_outliers'] = len(outliers)
            
        return stats


def validate_and_report(data: List[Dict], data_type: str = "paper") -> Tuple[List[Dict], Dict]:
    """
    Comprehensive validation with detailed reporting.
    
    Args:
        data: List of data dictionaries to validate
        data_type: Type of data being validated
        
    Returns:
        Tuple of (validated_data, validation_report)
    """
    validator = DataValidator()
    
    # Validate data
    validated_data, errors = validator.validate_paper_data(data)
    
    # Create DataFrame for additional checks
    if validated_data:
        df = pd.DataFrame(validated_data)
        
        # Check completeness
        required_cols = ['paperId', 'title'] if data_type == "paper" else ['authorId', 'name']
        completeness = validator.check_data_completeness(df, required_cols)
        
        # Check for duplicates
        id_col = 'paperId' if data_type == "paper" else 'authorId'
        duplicates = validator.detect_duplicates(df, id_col)
        
        # Citation statistics (for papers)
        citation_stats = {}
        if data_type == "paper":
            citation_stats = validator.validate_citation_counts(df)
    else:
        completeness = {}
        duplicates = pd.DataFrame()
        citation_stats = {}
    
    # Compile report
    report = {
        'total_records': len(data),
        'valid_records': len(validated_data),
        'error_count': len(errors),
        'errors': errors[:10],  # First 10 errors
        'completeness': completeness,
        'duplicate_count': len(duplicates),
        'citation_stats': citation_stats
    }
    
    # Log summary
    logger.info(f"Validation complete for {data_type} data:")
    logger.info(f"  Total records: {report['total_records']}")
    logger.info(f"  Valid records: {report['valid_records']}")
    logger.info(f"  Errors: {report['error_count']}")
    logger.info(f"  Duplicates: {report['duplicate_count']}")
    
    return validated_data, report