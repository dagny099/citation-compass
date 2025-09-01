"""
Data Import Pipeline for Academic Citation Platform.

This module provides comprehensive data import functionality that:
- Fetches papers and citations from Semantic Scholar API
- Builds citation relationship networks
- Supports batch processing with progress tracking
- Handles resumable imports with state management
- Validates data integrity throughout the process
- Provides comprehensive logging and error handling

The pipeline is designed for robustness in production environments with
large-scale data import requirements.
"""

import logging
import json
import time
import pickle
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import hashlib
import threading
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from .unified_api_client import UnifiedSemanticScholarClient
from .unified_database import UnifiedDatabaseManager, get_database
from ..models.paper import Paper
from ..models.author import Author
from ..models.venue import Venue
from ..models.citation import Citation
from ..utils.logging import get_logger
from ..utils.validation import validate_paper_data, validate_citation_data


class ImportStatus(Enum):
    """Status of import operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ImportProgress:
    """Tracks progress of import operations."""
    
    total_papers: int = 0
    processed_papers: int = 0
    total_citations: int = 0
    processed_citations: int = 0
    total_authors: int = 0
    processed_authors: int = 0
    
    papers_created: int = 0
    papers_updated: int = 0
    citations_created: int = 0
    authors_created: int = 0
    venues_created: int = 0
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: ImportStatus = ImportStatus.PENDING
    
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def elapsed_time(self) -> Optional[timedelta]:
        """Calculate elapsed time since start."""
        if self.start_time is None:
            return None
        end_time = self.end_time or datetime.now()
        return end_time - self.start_time
    
    @property
    def papers_progress_percent(self) -> float:
        """Calculate papers processing progress percentage."""
        if self.total_papers == 0:
            return 0.0
        return (self.processed_papers / self.total_papers) * 100
    
    @property
    def citations_progress_percent(self) -> float:
        """Calculate citations processing progress percentage."""
        if self.total_citations == 0:
            return 0.0
        return (self.processed_citations / self.total_citations) * 100
    
    @property
    def overall_progress_percent(self) -> float:
        """Calculate overall progress percentage."""
        total_items = self.total_papers + self.total_citations
        processed_items = self.processed_papers + self.processed_citations
        
        if total_items == 0:
            return 0.0
        return (processed_items / total_items) * 100


@dataclass
class ImportConfiguration:
    """Configuration for import operations."""
    
    # Source data configuration
    search_query: Optional[str] = None
    paper_ids: Optional[List[str]] = None
    author_ids: Optional[List[str]] = None
    venue_names: Optional[List[str]] = None
    fields_of_study: Optional[List[str]] = None
    year_range: Optional[Tuple[int, int]] = None
    
    # Import scope configuration
    max_papers: int = 10000
    max_depth: int = 2  # Citation network traversal depth
    include_citations: bool = True
    include_references: bool = True
    include_authors: bool = True
    include_venues: bool = True
    include_fields: bool = True
    
    # Processing configuration
    batch_size: int = 100
    max_workers: int = 4
    api_delay: float = 1.0
    retry_attempts: int = 3
    
    # Data filtering configuration
    min_citation_count: int = 0
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    
    # Output and persistence configuration
    save_progress: bool = True
    progress_file: Optional[Path] = None
    output_format: str = "database"  # "database", "json", "csv"
    output_path: Optional[Path] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.progress_file is None and self.save_progress:
            self.progress_file = Path(f"import_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Ensure reasonable limits
        if self.max_papers > 100000:
            raise ValueError("max_papers cannot exceed 100,000 for safety")
        
        if self.batch_size > 1000:
            raise ValueError("batch_size cannot exceed 1,000 for memory management")


class ImportStateManager:
    """
    Manages import state for resumable operations.
    
    Tracks processed items, handles checkpoints, and enables resumption
    of interrupted import operations.
    """
    
    def __init__(self, state_file: Optional[Path] = None):
        """
        Initialize state manager.
        
        Args:
            state_file: Optional path to state file. Auto-generated if None.
        """
        self.state_file = state_file or Path("import_state.json")
        self.logger = get_logger(__name__)
        
        # State tracking
        self.processed_papers: Set[str] = set()
        self.processed_citations: Set[Tuple[str, str]] = set()
        self.processed_authors: Set[str] = set()
        self.failed_items: Dict[str, str] = {}  # item_id -> error_message
        
        # Load existing state if available
        self.load_state()
        
        # Thread safety
        self._lock = threading.Lock()
    
    def load_state(self) -> None:
        """Load state from file if it exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                
                self.processed_papers = set(state_data.get('processed_papers', []))
                self.processed_authors = set(state_data.get('processed_authors', []))
                self.failed_items = state_data.get('failed_items', {})
                
                # Convert citation tuples from list format
                citations_list = state_data.get('processed_citations', [])
                self.processed_citations = {tuple(c) for c in citations_list}
                
                self.logger.info(f"Loaded import state: {len(self.processed_papers)} papers, "
                               f"{len(self.processed_citations)} citations, "
                               f"{len(self.processed_authors)} authors processed")
                
            except Exception as e:
                self.logger.warning(f"Failed to load import state: {e}")
                # Continue with empty state
    
    def save_state(self) -> None:
        """Save current state to file."""
        try:
            with self._lock:
                state_data = {
                    'processed_papers': list(self.processed_papers),
                    'processed_authors': list(self.processed_authors),
                    'processed_citations': [list(c) for c in self.processed_citations],
                    'failed_items': self.failed_items,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Write to temporary file first, then rename for atomic operation
                temp_file = self.state_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(state_data, f, indent=2)
                
                temp_file.rename(self.state_file)
                
        except Exception as e:
            self.logger.error(f"Failed to save import state: {e}")
    
    def mark_paper_processed(self, paper_id: str) -> None:
        """Mark a paper as processed."""
        with self._lock:
            self.processed_papers.add(paper_id)
    
    def mark_citation_processed(self, source_id: str, target_id: str) -> None:
        """Mark a citation as processed."""
        with self._lock:
            self.processed_citations.add((source_id, target_id))
    
    def mark_author_processed(self, author_id: str) -> None:
        """Mark an author as processed."""
        with self._lock:
            self.processed_authors.add(author_id)
    
    def mark_item_failed(self, item_id: str, error_message: str) -> None:
        """Mark an item as failed with error message."""
        with self._lock:
            self.failed_items[item_id] = error_message
    
    def is_paper_processed(self, paper_id: str) -> bool:
        """Check if a paper has already been processed."""
        return paper_id in self.processed_papers
    
    def is_citation_processed(self, source_id: str, target_id: str) -> bool:
        """Check if a citation has already been processed."""
        return (source_id, target_id) in self.processed_citations
    
    def is_author_processed(self, author_id: str) -> bool:
        """Check if an author has already been processed."""
        return author_id in self.processed_authors
    
    def get_progress_stats(self) -> Dict[str, int]:
        """Get current progress statistics."""
        return {
            'processed_papers': len(self.processed_papers),
            'processed_citations': len(self.processed_citations),
            'processed_authors': len(self.processed_authors),
            'failed_items': len(self.failed_items)
        }
    
    def reset_state(self) -> None:
        """Reset all state tracking."""
        with self._lock:
            self.processed_papers.clear()
            self.processed_citations.clear()
            self.processed_authors.clear()
            self.failed_items.clear()
            
            # Remove state file
            if self.state_file.exists():
                self.state_file.unlink()


class DataImportPipeline:
    """
    Comprehensive data import pipeline for academic citation data.
    
    Features:
    - Multi-source data fetching (search queries, paper IDs, author IDs)
    - Citation network traversal with configurable depth
    - Batch processing with progress tracking
    - Resumable imports with state management
    - Data validation and quality checks
    - Comprehensive error handling and logging
    - Performance optimization with concurrent processing
    """
    
    def __init__(self, 
                 config: Optional[ImportConfiguration] = None,
                 api_client: Optional[UnifiedSemanticScholarClient] = None,
                 database: Optional[UnifiedDatabaseManager] = None):
        """
        Initialize the data import pipeline.
        
        Args:
            config: Import configuration. Uses defaults if None.
            api_client: Semantic Scholar API client. Creates new if None.
            database: Database manager. Uses singleton if None.
        """
        self.config = config or ImportConfiguration()
        self.api_client = api_client or UnifiedSemanticScholarClient()
        self.database = database or get_database()
        
        self.logger = get_logger(__name__)
        self.progress = ImportProgress()
        self.state_manager = ImportStateManager(self.config.progress_file)
        
        # Progress tracking
        self._progress_callbacks: List[callable] = []
        self._cancelled = False
        self._paused = False
    
    def add_progress_callback(self, callback: callable) -> None:
        """
        Add callback function to receive progress updates.
        
        Args:
            callback: Function that accepts ImportProgress object
        """
        self._progress_callbacks.append(callback)
    
    def _notify_progress(self) -> None:
        """Notify all progress callbacks with current progress."""
        for callback in self._progress_callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")
    
    def cancel_import(self) -> None:
        """Cancel the ongoing import operation."""
        self._cancelled = True
        self.progress.status = ImportStatus.CANCELLED
        self.logger.info("Import operation cancelled by user")
    
    def pause_import(self) -> None:
        """Pause the ongoing import operation."""
        self._paused = True
        self.progress.status = ImportStatus.PAUSED
        self.logger.info("Import operation paused")
    
    def resume_import(self) -> None:
        """Resume the paused import operation."""
        self._paused = False
        self.progress.status = ImportStatus.IN_PROGRESS
        self.logger.info("Import operation resumed")
    
    def import_papers_by_search(self, 
                               search_query: str,
                               max_results: Optional[int] = None) -> ImportProgress:
        """
        Import papers based on search query.
        
        Args:
            search_query: Search query string
            max_results: Maximum number of papers to import. Uses config if None.
            
        Returns:
            ImportProgress object with operation results
        """
        max_results = max_results or self.config.max_papers
        
        self.logger.info(f"Starting paper import by search: '{search_query}' (max: {max_results})")
        
        self.progress = ImportProgress()
        self.progress.start_time = datetime.now()
        self.progress.status = ImportStatus.IN_PROGRESS
        
        try:
            # Search for papers
            self.logger.info(f"Searching for papers with query: {search_query}")
            
            papers_data = []
            processed_count = 0
            
            # Use the API client's search functionality with pagination
            for paper_batch in self.api_client.search_papers_paginated(
                query=search_query,
                limit=max_results,
                fields=['paperId', 'title', 'abstract', 'citationCount', 'year', 
                       'publicationDate', 'authors', 'venue', 's2FieldsOfStudy', 'citations', 'references']
            ):
                if self._cancelled:
                    break
                
                while self._paused:
                    time.sleep(1)
                
                # Filter papers based on configuration
                filtered_papers = self._filter_papers(paper_batch)
                
                if filtered_papers:
                    # Process batch
                    batch_results = self._process_paper_batch(filtered_papers)
                    papers_data.extend(batch_results)
                    
                    processed_count += len(filtered_papers)
                    self.progress.processed_papers = processed_count
                    self._notify_progress()
                
                # Save state periodically
                if processed_count % (self.config.batch_size * 5) == 0:
                    self.state_manager.save_state()
            
            # Set total counts
            self.progress.total_papers = processed_count
            
            # Process citations if requested
            if self.config.include_citations and papers_data:
                self._process_citation_networks(papers_data)
            
            # Mark as completed
            self.progress.status = ImportStatus.COMPLETED if not self._cancelled else ImportStatus.CANCELLED
            self.progress.end_time = datetime.now()
            
            self.logger.info(f"Import completed: {self.progress.processed_papers} papers, "
                           f"{self.progress.processed_citations} citations processed")
            
        except Exception as e:
            self.logger.error(f"Import failed: {e}")
            self.progress.status = ImportStatus.FAILED
            self.progress.errors.append(str(e))
            self.progress.end_time = datetime.now()
        
        finally:
            # Save final state
            if self.config.save_progress:
                self.state_manager.save_state()
            
            self._notify_progress()
        
        return self.progress
    
    def import_papers_by_ids(self, paper_ids: List[str]) -> ImportProgress:
        """
        Import specific papers by their IDs.
        
        Args:
            paper_ids: List of paper IDs to import
            
        Returns:
            ImportProgress object with operation results
        """
        self.logger.info(f"Starting paper import by IDs: {len(paper_ids)} papers")
        
        self.progress = ImportProgress()
        self.progress.start_time = datetime.now()
        self.progress.status = ImportStatus.IN_PROGRESS
        self.progress.total_papers = len(paper_ids)
        
        try:
            # Filter out already processed papers
            remaining_ids = [
                pid for pid in paper_ids 
                if not self.state_manager.is_paper_processed(pid)
            ]
            
            self.logger.info(f"Importing {len(remaining_ids)} new papers "
                           f"(skipping {len(paper_ids) - len(remaining_ids)} already processed)")
            
            # Process in batches
            papers_data = []
            
            for i in range(0, len(remaining_ids), self.config.batch_size):
                if self._cancelled:
                    break
                
                while self._paused:
                    time.sleep(1)
                
                batch_ids = remaining_ids[i:i + self.config.batch_size]
                
                # Fetch paper details
                batch_papers = []
                for paper_id in batch_ids:
                    try:
                        paper_data = self.api_client.get_paper_details(
                            paper_id=paper_id,
                            fields=['paperId', 'title', 'abstract', 'citationCount', 'year', 
                                   'publicationDate', 'authors', 'venue', 's2FieldsOfStudy', 
                                   'citations', 'references']
                        )
                        
                        if paper_data:
                            batch_papers.append(paper_data)
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to fetch paper {paper_id}: {e}")
                        self.state_manager.mark_item_failed(paper_id, str(e))
                
                # Process batch
                if batch_papers:
                    filtered_papers = self._filter_papers(batch_papers)
                    batch_results = self._process_paper_batch(filtered_papers)
                    papers_data.extend(batch_results)
                
                self.progress.processed_papers += len(batch_papers)
                self._notify_progress()
                
                # Rate limiting
                time.sleep(self.config.api_delay)
            
            # Process citations if requested
            if self.config.include_citations and papers_data:
                self._process_citation_networks(papers_data)
            
            # Mark as completed
            self.progress.status = ImportStatus.COMPLETED if not self._cancelled else ImportStatus.CANCELLED
            self.progress.end_time = datetime.now()
            
            self.logger.info(f"Import by IDs completed: {self.progress.processed_papers} papers processed")
            
        except Exception as e:
            self.logger.error(f"Import by IDs failed: {e}")
            self.progress.status = ImportStatus.FAILED
            self.progress.errors.append(str(e))
            self.progress.end_time = datetime.now()
        
        finally:
            # Save final state
            if self.config.save_progress:
                self.state_manager.save_state()
            
            self._notify_progress()
        
        return self.progress
    
    def _filter_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter papers based on configuration criteria.
        
        Args:
            papers: List of paper data dictionaries
            
        Returns:
            Filtered list of papers
        """
        filtered = []
        
        for paper in papers:
            # Check citation count threshold
            citation_count = paper.get('citationCount', 0)
            if citation_count < self.config.min_citation_count:
                continue
            
            # Check year range
            year = paper.get('year')
            if year:
                if self.config.min_year and year < self.config.min_year:
                    continue
                if self.config.max_year and year > self.config.max_year:
                    continue
            
            # Additional filtering can be added here
            
            filtered.append(paper)
        
        return filtered
    
    def _process_paper_batch(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of papers for database insertion.
        
        Args:
            papers: List of paper data dictionaries
            
        Returns:
            List of processed paper data
        """
        processed_papers = []
        
        for paper_data in papers:
            paper_id = paper_data.get('paperId')
            
            # Skip if already processed
            if self.state_manager.is_paper_processed(paper_id):
                continue
            
            try:
                # Validate paper data
                if not validate_paper_data(paper_data):
                    self.progress.warnings.append(f"Invalid paper data for {paper_id}")
                    continue
                
                # Convert to paper model
                paper = self._convert_to_paper_model(paper_data)
                
                # Store in database
                self._store_paper_in_database(paper, paper_data)
                
                # Process related entities
                if self.config.include_authors:
                    self._process_paper_authors(paper_data)
                
                if self.config.include_venues:
                    self._process_paper_venue(paper_data)
                
                # Mark as processed
                self.state_manager.mark_paper_processed(paper_id)
                processed_papers.append(paper_data)
                
                self.progress.papers_created += 1
                
            except Exception as e:
                self.logger.error(f"Failed to process paper {paper_id}: {e}")
                self.state_manager.mark_item_failed(paper_id, str(e))
                self.progress.errors.append(f"Paper {paper_id}: {str(e)}")
        
        return processed_papers
    
    def _convert_to_paper_model(self, paper_data: Dict[str, Any]) -> Paper:
        """
        Convert API paper data to Paper model.
        
        Args:
            paper_data: Raw paper data from API
            
        Returns:
            Paper model instance
        """
        # Extract publication date
        pub_date = None
        if paper_data.get('publicationDate'):
            try:
                pub_date = datetime.fromisoformat(paper_data['publicationDate'].replace('Z', '+00:00'))
            except (ValueError, TypeError):
                pass
        
        return Paper(
            paper_id=paper_data['paperId'],
            title=paper_data.get('title', ''),
            abstract=paper_data.get('abstract', ''),
            citation_count=paper_data.get('citationCount', 0),
            reference_count=paper_data.get('referenceCount', 0),
            publication_date=pub_date,
            year=paper_data.get('year'),
            doi=paper_data.get('externalIds', {}).get('DOI'),
            s2_fields_of_study=paper_data.get('s2FieldsOfStudy', []),
            influential_citation_count=paper_data.get('influentialCitationCount', 0),
            is_open_access=paper_data.get('isOpenAccess', False),
            open_access_pdf=paper_data.get('openAccessPdf'),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def _store_paper_in_database(self, paper: Paper, raw_data: Dict[str, Any]) -> None:
        """
        Store paper in database with all related data.
        
        Args:
            paper: Paper model instance
            raw_data: Raw API data for additional processing
        """
        # Convert to database format
        paper_dict = {
            'paperId': paper.paper_id,
            'title': paper.title,
            'abstract': paper.abstract,
            'citationCount': paper.citation_count,
            'year': paper.year,
            'publicationDate': paper.publication_date.isoformat() if paper.publication_date else None
        }
        
        # Use batch create for efficiency
        self.database.batch_create_papers([paper_dict])
        
        self.logger.debug(f"Stored paper in database: {paper.paper_id}")
    
    def _process_paper_authors(self, paper_data: Dict[str, Any]) -> None:
        """
        Process and store paper authors.
        
        Args:
            paper_data: Paper data containing author information
        """
        authors = paper_data.get('authors', [])
        paper_id = paper_data['paperId']
        
        for author_data in authors:
            author_id = author_data.get('authorId')
            if not author_id or self.state_manager.is_author_processed(author_id):
                continue
            
            try:
                # Create author record
                author_dict = {
                    'authorId': author_id,
                    'name': author_data.get('name', ''),
                    'paperCount': author_data.get('paperCount', 0),
                    'citationCount': author_data.get('citationCount', 0),
                    'hIndex': author_data.get('hIndex', 0)
                }
                
                # Store author (this would need a batch_create_authors method)
                # For now, we'll use individual transactions
                self.database.execute(
                    """
                    MERGE (a:Author {authorId: $authorId})
                    SET a.name = $name,
                        a.paperCount = $paperCount,
                        a.citationCount = $citationCount,
                        a.hIndex = $hIndex,
                        a.updated_at = datetime()
                    """,
                    author_dict
                )
                
                # Create authorship relationship
                self.database.execute(
                    """
                    MATCH (p:Paper {paperId: $paperId})
                    MATCH (a:Author {authorId: $authorId})
                    MERGE (a)-[:AUTHORED]->(p)
                    """,
                    {'paperId': paper_id, 'authorId': author_id}
                )
                
                self.state_manager.mark_author_processed(author_id)
                self.progress.authors_created += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to process author {author_id}: {e}")
    
    def _process_paper_venue(self, paper_data: Dict[str, Any]) -> None:
        """
        Process and store paper venue information.
        
        Args:
            paper_data: Paper data containing venue information
        """
        venue_data = paper_data.get('venue')
        if not venue_data:
            return
        
        paper_id = paper_data['paperId']
        venue_name = venue_data if isinstance(venue_data, str) else venue_data.get('name', '')
        
        if not venue_name:
            return
        
        try:
            # Create venue record
            self.database.execute(
                """
                MERGE (v:PubVenue {name: $name})
                SET v.updated_at = datetime()
                """,
                {'name': venue_name}
            )
            
            # Create publication relationship
            self.database.execute(
                """
                MATCH (p:Paper {paperId: $paperId})
                MATCH (v:PubVenue {name: $venueName})
                MERGE (p)-[:PUBLISHED_IN]->(v)
                """,
                {'paperId': paper_id, 'venueName': venue_name}
            )
            
            self.progress.venues_created += 1
            
        except Exception as e:
            self.logger.warning(f"Failed to process venue for paper {paper_id}: {e}")
    
    def _process_citation_networks(self, papers_data: List[Dict[str, Any]]) -> None:
        """
        Process citation networks for the imported papers.
        
        Args:
            papers_data: List of paper data with citation information
        """
        self.logger.info(f"Processing citation networks for {len(papers_data)} papers")
        
        # Extract all citation relationships
        citations_to_process = []
        
        for paper_data in papers_data:
            paper_id = paper_data['paperId']
            
            # Process outgoing citations (references)
            if self.config.include_references:
                references = paper_data.get('references', [])
                for ref in references:
                    ref_id = ref.get('paperId')
                    if ref_id and not self.state_manager.is_citation_processed(paper_id, ref_id):
                        citations_to_process.append((paper_id, ref_id))
            
            # Process incoming citations
            if self.config.include_citations:
                citations = paper_data.get('citations', [])
                for cite in citations:
                    cite_id = cite.get('paperId')
                    if cite_id and not self.state_manager.is_citation_processed(cite_id, paper_id):
                        citations_to_process.append((cite_id, paper_id))
        
        self.progress.total_citations = len(citations_to_process)
        self.logger.info(f"Found {len(citations_to_process)} citation relationships to process")
        
        # Process citations in batches
        batch_size = self.config.batch_size
        
        for i in range(0, len(citations_to_process), batch_size):
            if self._cancelled:
                break
            
            while self._paused:
                time.sleep(1)
            
            batch = citations_to_process[i:i + batch_size]
            
            # Convert to database format
            citation_batch = []
            for source_id, target_id in batch:
                citation_batch.append({
                    'source_id': source_id,
                    'target_id': target_id
                })
            
            try:
                # Batch create citations
                created_count = self.database.batch_create_citations(citation_batch)
                
                # Mark as processed
                for source_id, target_id in batch:
                    self.state_manager.mark_citation_processed(source_id, target_id)
                
                self.progress.processed_citations += len(batch)
                self.progress.citations_created += created_count
                self._notify_progress()
                
            except Exception as e:
                self.logger.error(f"Failed to process citation batch: {e}")
                self.progress.errors.append(f"Citation batch: {str(e)}")
        
        self.logger.info(f"Citation processing completed: {self.progress.citations_created} relationships created")
    
    def get_import_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of import operation.
        
        Returns:
            Dictionary with detailed import statistics
        """
        return {
            'progress': {
                'total_papers': self.progress.total_papers,
                'processed_papers': self.progress.processed_papers,
                'total_citations': self.progress.total_citations,
                'processed_citations': self.progress.processed_citations,
                'papers_created': self.progress.papers_created,
                'citations_created': self.progress.citations_created,
                'authors_created': self.progress.authors_created,
                'venues_created': self.progress.venues_created,
            },
            'status': {
                'current_status': self.progress.status.value,
                'start_time': self.progress.start_time.isoformat() if self.progress.start_time else None,
                'end_time': self.progress.end_time.isoformat() if self.progress.end_time else None,
                'elapsed_time': str(self.progress.elapsed_time) if self.progress.elapsed_time else None,
                'papers_progress_percent': self.progress.papers_progress_percent,
                'citations_progress_percent': self.progress.citations_progress_percent,
                'overall_progress_percent': self.progress.overall_progress_percent,
            },
            'errors_and_warnings': {
                'errors': self.progress.errors,
                'warnings': self.progress.warnings,
                'error_count': len(self.progress.errors),
                'warning_count': len(self.progress.warnings),
            },
            'state_manager_stats': self.state_manager.get_progress_stats(),
            'configuration': {
                'max_papers': self.config.max_papers,
                'batch_size': self.config.batch_size,
                'include_citations': self.config.include_citations,
                'include_authors': self.config.include_authors,
                'include_venues': self.config.include_venues,
            }
        }


def create_sample_import_config() -> ImportConfiguration:
    """
    Create a sample import configuration for demonstration.
    
    Returns:
        ImportConfiguration with reasonable defaults for testing
    """
    return ImportConfiguration(
        search_query="machine learning",
        max_papers=100,
        batch_size=10,
        include_citations=True,
        include_authors=True,
        include_venues=True,
        min_citation_count=5,
        year_range=(2020, 2024),
        save_progress=True
    )


# Utility functions for common import patterns

def quick_import_by_search(search_query: str, 
                          max_papers: int = 100,
                          progress_callback: Optional[callable] = None) -> ImportProgress:
    """
    Quick import utility for search-based paper import.
    
    Args:
        search_query: Search query string
        max_papers: Maximum number of papers to import
        progress_callback: Optional callback for progress updates
        
    Returns:
        ImportProgress with operation results
    """
    config = ImportConfiguration(
        search_query=search_query,
        max_papers=max_papers,
        batch_size=min(50, max_papers),
        include_citations=True,
        include_authors=True,
        include_venues=True
    )
    
    pipeline = DataImportPipeline(config)
    
    if progress_callback:
        pipeline.add_progress_callback(progress_callback)
    
    return pipeline.import_papers_by_search(search_query, max_papers)


def quick_import_by_ids(paper_ids: List[str],
                       progress_callback: Optional[callable] = None) -> ImportProgress:
    """
    Quick import utility for ID-based paper import.
    
    Args:
        paper_ids: List of paper IDs to import
        progress_callback: Optional callback for progress updates
        
    Returns:
        ImportProgress with operation results
    """
    config = ImportConfiguration(
        max_papers=len(paper_ids),
        batch_size=min(50, len(paper_ids)),
        include_citations=True,
        include_authors=True,
        include_venues=True
    )
    
    pipeline = DataImportPipeline(config)
    
    if progress_callback:
        pipeline.add_progress_callback(progress_callback)
    
    return pipeline.import_papers_by_ids(paper_ids)