"""
Data validation utilities for Academic Citation Platform.

This module provides comprehensive validation functions for:
- Paper data validation
- Citation data validation  
- Author data validation
- General data quality checks
- Schema validation for database operations
"""

import re
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


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


def validate_paper_data(paper_data: Dict[str, Any]) -> bool:
    """
    Validate paper data from API for database import.
    
    Args:
        paper_data: Paper data dictionary from Semantic Scholar API
        
    Returns:
        True if paper data is valid, False otherwise
    """
    try:
        # Required fields
        if not paper_data.get('paperId'):
            logger.warning("Paper missing required paperId")
            return False
        
        if not validate_paper_id(paper_data['paperId']):
            logger.warning(f"Invalid paper ID: {paper_data['paperId']}")
            return False
        
        # Title is strongly recommended
        title = paper_data.get('title', '').strip()
        if not title:
            logger.warning(f"Paper {paper_data['paperId']} missing title")
            return False
        
        if len(title) > 1000:  # Reasonable title length limit
            logger.warning(f"Paper {paper_data['paperId']} has unusually long title")
            return False
        
        # Validate year if present
        year = paper_data.get('year')
        if year is not None and not validate_year(year):
            logger.warning(f"Paper {paper_data['paperId']} has invalid year: {year}")
            return False
        
        # Validate citation count if present
        citation_count = paper_data.get('citationCount')
        if citation_count is not None:
            if not isinstance(citation_count, int) or citation_count < 0:
                logger.warning(f"Paper {paper_data['paperId']} has invalid citation count: {citation_count}")
                return False
        
        # Validate DOI if present
        doi = paper_data.get('externalIds', {}).get('DOI')
        if doi and not validate_doi(doi):
            logger.warning(f"Paper {paper_data['paperId']} has invalid DOI: {doi}")
            return False
        
        # Validate abstract length
        abstract = paper_data.get('abstract', '')
        if abstract and len(abstract) > 10000:  # Reasonable abstract length limit
            logger.warning(f"Paper {paper_data['paperId']} has unusually long abstract")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating paper data: {e}")
        return False


def validate_citation_data(citation_data: Dict[str, Any]) -> bool:
    """
    Validate citation relationship data.
    
    Args:
        citation_data: Citation data dictionary with source_id and target_id
        
    Returns:
        True if citation data is valid, False otherwise
    """
    try:
        # Required fields
        source_id = citation_data.get('source_id')
        target_id = citation_data.get('target_id')
        
        if not source_id or not target_id:
            logger.warning("Citation missing required source_id or target_id")
            return False
        
        if not validate_paper_id(source_id) or not validate_paper_id(target_id):
            logger.warning(f"Invalid citation IDs: {source_id} -> {target_id}")
            return False
        
        # Self-citations are valid but worth noting
        if source_id == target_id:
            logger.info(f"Self-citation detected: {source_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating citation data: {e}")
        return False


def validate_author_data(author_data: Dict[str, Any]) -> bool:
    """
    Validate author data from API.
    
    Args:
        author_data: Author data dictionary from Semantic Scholar API
        
    Returns:
        True if author data is valid, False otherwise
    """
    try:
        # Author ID is required
        author_id = author_data.get('authorId')
        if not author_id:
            logger.warning("Author missing required authorId")
            return False
        
        if not validate_author_id(author_id):
            logger.warning(f"Invalid author ID: {author_id}")
            return False
        
        # Name is strongly recommended
        name = author_data.get('name', '').strip()
        if not name:
            logger.warning(f"Author {author_id} missing name")
            return False
        
        if len(name) > 200:  # Reasonable name length limit
            logger.warning(f"Author {author_id} has unusually long name")
            return False
        
        # Validate metrics if present
        metrics = ['paperCount', 'citationCount', 'hIndex']
        for metric in metrics:
            value = author_data.get(metric)
            if value is not None:
                if not isinstance(value, int) or value < 0:
                    logger.warning(f"Author {author_id} has invalid {metric}: {value}")
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating author data: {e}")
        return False


def validate_venue_data(venue_data: Dict[str, Any]) -> bool:
    """
    Validate venue data.
    
    Args:
        venue_data: Venue data dictionary
        
    Returns:
        True if venue data is valid, False otherwise
    """
    try:
        # Venue name is required
        name = venue_data.get('name', '').strip()
        if not name:
            logger.warning("Venue missing required name")
            return False
        
        if len(name) > 500:  # Reasonable venue name length limit
            logger.warning(f"Venue has unusually long name: {name[:50]}...")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating venue data: {e}")
        return False


def validate_import_batch(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate a batch of papers for import.
    
    Args:
        papers: List of paper data dictionaries
        
    Returns:
        Dictionary with validation results and statistics
    """
    results = {
        'total_papers': len(papers),
        'valid_papers': 0,
        'invalid_papers': 0,
        'validation_errors': [],
        'duplicate_ids': [],
        'quality_warnings': []
    }
    
    seen_ids = set()
    
    for i, paper_data in enumerate(papers):
        paper_id = paper_data.get('paperId', f'unknown_{i}')
        
        # Check for duplicates
        if paper_id in seen_ids:
            results['duplicate_ids'].append(paper_id)
            continue
        seen_ids.add(paper_id)
        
        # Validate paper
        if validate_paper_data(paper_data):
            results['valid_papers'] += 1
            
            # Quality checks
            if not paper_data.get('abstract'):
                results['quality_warnings'].append(f"Paper {paper_id} missing abstract")
            
            if not paper_data.get('year'):
                results['quality_warnings'].append(f"Paper {paper_id} missing year")
            
            if not paper_data.get('authors'):
                results['quality_warnings'].append(f"Paper {paper_id} missing authors")
                
        else:
            results['invalid_papers'] += 1
            results['validation_errors'].append(f"Invalid paper data: {paper_id}")
    
    results['validation_success_rate'] = (results['valid_papers'] / results['total_papers']) * 100 if results['total_papers'] > 0 else 0
    
    return results


def validate_database_constraints(db_manager) -> Dict[str, Any]:
    """
    Validate database constraints and indexes.
    
    Args:
        db_manager: UnifiedDatabaseManager instance
        
    Returns:
        Dictionary with constraint validation results
    """
    try:
        # Get current constraints
        constraints_df = db_manager.query("SHOW CONSTRAINTS")
        indexes_df = db_manager.query("SHOW INDEXES")
        
        required_constraints = [
            'paper_id_unique',
            'author_id_unique',
            'venue_name_unique'
        ]
        
        existing_constraints = set(constraints_df['name'].tolist() if not constraints_df.empty else [])
        
        results = {
            'total_constraints': len(existing_constraints),
            'total_indexes': len(indexes_df) if not indexes_df.empty else 0,
            'missing_constraints': [],
            'validation_passed': True
        }
        
        for constraint in required_constraints:
            if constraint not in existing_constraints:
                results['missing_constraints'].append(constraint)
                results['validation_passed'] = False
        
        return results
        
    except Exception as e:
        logger.error(f"Error validating database constraints: {e}")
        return {
            'error': str(e),
            'validation_passed': False
        }


def validate_import_configuration(config) -> Dict[str, Any]:
    """
    Validate import configuration parameters.
    
    Args:
        config: ImportConfiguration instance
        
    Returns:
        Dictionary with configuration validation results
    """
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Validate limits
    if config.max_papers <= 0:
        results['errors'].append("max_papers must be greater than 0")
        results['is_valid'] = False
    
    if config.max_papers > 100000:
        results['warnings'].append("max_papers is very large, consider smaller batches")
    
    if config.batch_size <= 0:
        results['errors'].append("batch_size must be greater than 0")
        results['is_valid'] = False
    
    if config.batch_size > 1000:
        results['warnings'].append("Large batch_size may cause memory issues")
    
    # Validate search parameters
    if not config.search_query and not config.paper_ids:
        results['errors'].append("Either search_query or paper_ids must be provided")
        results['is_valid'] = False
    
    # Validate year range
    if config.year_range:
        start_year, end_year = config.year_range
        if start_year > end_year:
            results['errors'].append("year_range start must be <= end")
            results['is_valid'] = False
        
        if not validate_year(start_year) or not validate_year(end_year):
            results['errors'].append("Invalid years in year_range")
            results['is_valid'] = False
    
    # Validate API delay
    if config.api_delay < 0:
        results['errors'].append("api_delay cannot be negative")
        results['is_valid'] = False
    
    if config.api_delay > 10:
        results['warnings'].append("Large api_delay will slow down imports significantly")
    
    return results