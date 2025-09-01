"""
Demo Data Loader for Academic Citation Platform.

This module provides functionality to load demo datasets for users who want to
explore the platform without setting up Neo4j or importing real data. It supports:

- Offline mode with in-memory data structures
- Database mode with Neo4j integration  
- Progress tracking for data loading operations
- Automatic fallback to offline mode when database unavailable
- Caching for improved performance
"""

import json
import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
import threading
import time

from .demo_dataset import DemoDatasetInfo, get_available_datasets
from .unified_database import get_database, UnifiedDatabaseManager
from ..models.paper import Paper
from ..models.author import Author
from ..models.citation import Citation
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LoadProgress:
    """Track progress of demo data loading."""
    
    total_items: int = 0
    loaded_items: int = 0
    current_operation: str = "Initializing"
    start_time: Optional[datetime] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.loaded_items / self.total_items) * 100.0
    
    @property
    def elapsed_time(self) -> Optional[float]:
        """Calculate elapsed time in seconds."""
        if self.start_time is None:
            return None
        return (datetime.now() - self.start_time).total_seconds()


class OfflineDataStore:
    """
    In-memory data store for offline demo mode.
    
    Provides a Neo4j-like interface for querying demo data without requiring
    an actual database connection. Perfect for users who want to explore
    the platform immediately without setup.
    """
    
    def __init__(self):
        """Initialize empty data store."""
        self.papers: Dict[str, Dict] = {}
        self.authors: Dict[str, Dict] = {}
        self.venues: Dict[str, Dict] = {}
        self.citations: List[Dict] = []
        
        # Index for fast lookups
        self._paper_citations: Dict[str, List[str]] = {}  # paper_id -> list of citing papers
        self._paper_references: Dict[str, List[str]] = {}  # paper_id -> list of referenced papers
        self._author_papers: Dict[str, List[str]] = {}     # author_id -> list of papers
        
        self.loaded_dataset: Optional[str] = None
        self.load_time: Optional[datetime] = None
        
        logger.info("Initialized offline data store")
    
    def load_dataset(self, dataset_name: str, data_dir: Optional[Path] = None) -> bool:
        """
        Load a demo dataset into the offline store.
        
        Args:
            dataset_name: Name of dataset to load
            data_dir: Directory containing datasets
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            data_dir = data_dir or Path("data/demo_datasets")
            dataset_path = data_dir / dataset_name
            
            if not dataset_path.exists():
                logger.error(f"Dataset path not found: {dataset_path}")
                return False
            
            logger.info(f"Loading demo dataset: {dataset_name}")
            load_start = datetime.now()
            
            # Load papers
            papers_file = dataset_path / "papers.json"
            if papers_file.exists():
                with open(papers_file) as f:
                    papers_data = json.load(f)
                    self.papers = {p["paper_id"]: p for p in papers_data}
                logger.info(f"Loaded {len(self.papers)} papers")
            
            # Load authors
            authors_file = dataset_path / "authors.json"
            if authors_file.exists():
                with open(authors_file) as f:
                    authors_data = json.load(f)
                    self.authors = {a["author_id"]: a for a in authors_data}
                logger.info(f"Loaded {len(self.authors)} authors")
            
            # Load venues
            venues_file = dataset_path / "venues.json"
            if venues_file.exists():
                with open(venues_file) as f:
                    venues_data = json.load(f)
                    self.venues = {v["venue_id"]: v for v in venues_data}
                logger.info(f"Loaded {len(self.venues)} venues")
            
            # Load citations and build indexes
            citations_file = dataset_path / "citations.json"
            if citations_file.exists():
                with open(citations_file) as f:
                    self.citations = json.load(f)
                
                # Build citation indexes
                self._build_citation_indexes()
                logger.info(f"Loaded {len(self.citations)} citations")
            
            self.loaded_dataset = dataset_name
            self.load_time = datetime.now()
            
            load_duration = (self.load_time - load_start).total_seconds()
            logger.info(f"Dataset loaded successfully in {load_duration:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            return False
    
    def _build_citation_indexes(self):
        """Build indexes for fast citation lookups."""
        self._paper_citations = {}
        self._paper_references = {}
        self._author_papers = {}
        
        # Build citation indexes
        for citation in self.citations:
            source_id = citation["source_id"]
            target_id = citation["target_id"]
            
            # source cites target, so target is cited by source
            if target_id not in self._paper_citations:
                self._paper_citations[target_id] = []
            self._paper_citations[target_id].append(source_id)
            
            # source references target
            if source_id not in self._paper_references:
                self._paper_references[source_id] = []
            self._paper_references[source_id].append(target_id)
        
        # Build author-paper index
        for paper_id, paper in self.papers.items():
            for author_name in paper.get("authors", []):
                # Find author ID by name (simplified lookup)
                author_id = None
                for aid, author in self.authors.items():
                    if author["name"] == author_name:
                        author_id = aid
                        break
                
                if author_id:
                    if author_id not in self._author_papers:
                        self._author_papers[author_id] = []
                    self._author_papers[author_id].append(paper_id)
    
    def get_paper(self, paper_id: str) -> Optional[Dict]:
        """Get paper by ID."""
        return self.papers.get(paper_id)
    
    def get_papers(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get list of papers with pagination."""
        paper_list = list(self.papers.values())
        return paper_list[offset:offset + limit]
    
    def search_papers(self, query: str, limit: int = 50) -> List[Dict]:
        """Search papers by title or abstract."""
        query_lower = query.lower()
        results = []
        
        for paper in self.papers.values():
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            
            if query_lower in title or query_lower in abstract:
                results.append(paper)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_paper_citations(self, paper_id: str) -> List[Dict]:
        """Get papers that cite the given paper."""
        citing_paper_ids = self._paper_citations.get(paper_id, [])
        return [self.papers[pid] for pid in citing_paper_ids if pid in self.papers]
    
    def get_paper_references(self, paper_id: str) -> List[Dict]:
        """Get papers referenced by the given paper."""
        referenced_paper_ids = self._paper_references.get(paper_id, [])
        return [self.papers[pid] for pid in referenced_paper_ids if pid in self.papers]
    
    def get_author(self, author_id: str) -> Optional[Dict]:
        """Get author by ID."""
        return self.authors.get(author_id)
    
    def get_author_papers(self, author_id: str) -> List[Dict]:
        """Get papers by a given author."""
        paper_ids = self._author_papers.get(author_id, [])
        return [self.papers[pid] for pid in paper_ids if pid in self.papers]
    
    def search_authors(self, query: str, limit: int = 50) -> List[Dict]:
        """Search authors by name."""
        query_lower = query.lower()
        results = []
        
        for author in self.authors.values():
            name = author.get("name", "").lower()
            if query_lower in name:
                results.append(author)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_network_statistics(self) -> Dict[str, int]:
        """Get network statistics."""
        return {
            "papers": len(self.papers),
            "authors": len(self.authors),
            "venues": len(self.venues),
            "citations": len(self.citations),
            "fields": len(set(p.get("field", "Unknown") for p in self.papers.values())),
            "papers_with_embeddings": 0  # No embeddings in offline mode
        }
    
    def get_citation_network(self) -> Tuple[List[Dict], List[Dict]]:
        """Get complete citation network as nodes and edges."""
        # Nodes are papers
        nodes = [
            {
                "id": paper_id,
                "title": paper["title"],
                "authors": paper.get("authors", []),
                "year": paper.get("year"),
                "citation_count": paper.get("citation_count", 0),
                "field": paper.get("field", "Unknown")
            }
            for paper_id, paper in self.papers.items()
        ]
        
        # Edges are citations
        edges = [
            {
                "source": citation["source_id"],
                "target": citation["target_id"],
                "type": "cites"
            }
            for citation in self.citations
        ]
        
        return nodes, edges


class DemoDataLoader:
    """
    Main demo data loader with automatic fallback between database and offline modes.
    
    This class intelligently chooses between database storage and offline mode
    based on Neo4j availability, providing a seamless experience for users.
    """
    
    def __init__(self, prefer_database: bool = True):
        """
        Initialize demo data loader.
        
        Args:
            prefer_database: Whether to prefer database mode over offline mode
        """
        self.prefer_database = prefer_database
        self.logger = get_logger(__name__)
        
        # Try to initialize database connection
        self.database_available = False
        self.database: Optional[UnifiedDatabaseManager] = None
        
        if prefer_database:
            try:
                self.database = get_database()
                self.database_available = self.database.test_connection()
                self.logger.info(f"Database connection: {'available' if self.database_available else 'unavailable'}")
            except Exception as e:
                self.logger.warning(f"Database initialization failed: {e}")
                self.database_available = False
        
        # Initialize offline store as fallback
        self.offline_store = OfflineDataStore()
        self.current_mode = "offline"  # Will be updated after loading
        
        # Progress tracking
        self.progress_callbacks: List[callable] = []
        self.current_progress = LoadProgress()
        
        self.logger.info(f"Demo data loader initialized (database_available: {self.database_available})")
    
    def add_progress_callback(self, callback: callable) -> None:
        """Add callback for progress updates."""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self, operation: str, loaded: int, total: int) -> None:
        """Notify progress callbacks."""
        self.current_progress.current_operation = operation
        self.current_progress.loaded_items = loaded
        self.current_progress.total_items = total
        
        for callback in self.progress_callbacks:
            try:
                callback(self.current_progress)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")
    
    def load_demo_dataset(self, 
                         dataset_name: Optional[str] = None,
                         force_offline: bool = False) -> bool:
        """
        Load demo dataset with automatic mode selection.
        
        Args:
            dataset_name: Name of dataset to load (defaults to first available)
            force_offline: Force offline mode even if database is available
            
        Returns:
            True if loaded successfully
        """
        self.current_progress = LoadProgress(start_time=datetime.now())
        
        # Determine dataset to load
        if dataset_name is None:
            available_datasets = get_available_datasets()
            if not available_datasets:
                self.logger.error("No demo datasets available")
                return False
            dataset_name = available_datasets[0].name
        
        self.logger.info(f"Loading demo dataset: {dataset_name}")
        
        # Choose loading mode
        use_database = self.database_available and not force_offline
        
        if use_database:
            success = self._load_to_database(dataset_name)
            if success:
                self.current_mode = "database"
                self.logger.info("Demo dataset loaded to database")
            else:
                self.logger.warning("Database loading failed, falling back to offline mode")
                success = self._load_to_offline_store(dataset_name)
                if success:
                    self.current_mode = "offline"
        else:
            success = self._load_to_offline_store(dataset_name)
            if success:
                self.current_mode = "offline"
                self.logger.info("Demo dataset loaded in offline mode")
        
        if success:
            self.logger.info(f"Demo dataset '{dataset_name}' loaded successfully in {self.current_mode} mode")
        else:
            self.logger.error(f"Failed to load demo dataset: {dataset_name}")
        
        return success
    
    def _load_to_database(self, dataset_name: str) -> bool:
        """Load dataset to Neo4j database."""
        try:
            data_dir = Path("data/demo_datasets")
            dataset_path = data_dir / dataset_name
            
            if not dataset_path.exists():
                self.logger.error(f"Dataset path not found: {dataset_path}")
                return False
            
            # Load data files
            papers_file = dataset_path / "papers.json"
            citations_file = dataset_path / "citations.json"
            
            total_operations = 4
            current_op = 0
            
            # Load papers
            if papers_file.exists():
                current_op += 1
                self._notify_progress("Loading papers to database", current_op, total_operations)
                
                with open(papers_file) as f:
                    papers_data = json.load(f)
                
                # Convert to database format
                db_papers = []
                for paper in papers_data:
                    db_paper = {
                        "paperId": paper["paper_id"],
                        "title": paper["title"],
                        "abstract": paper.get("abstract", ""),
                        "citationCount": paper.get("citation_count", 0),
                        "year": paper.get("year"),
                        "publicationDate": None  # Not available in demo data
                    }
                    db_papers.append(db_paper)
                
                # Batch create papers
                created_count = self.database.batch_create_papers(db_papers)
                self.logger.info(f"Created {created_count} papers in database")
            
            # Load citations
            if citations_file.exists():
                current_op += 1
                self._notify_progress("Loading citations to database", current_op, total_operations)
                
                with open(citations_file) as f:
                    citations_data = json.load(f)
                
                # Convert to database format
                db_citations = []
                for citation in citations_data:
                    db_citation = {
                        "source_id": citation["source_id"],
                        "target_id": citation["target_id"]
                    }
                    db_citations.append(db_citation)
                
                # Batch create citations
                created_count = self.database.batch_create_citations(db_citations)
                self.logger.info(f"Created {created_count} citations in database")
            
            current_op += 1
            self._notify_progress("Database loading complete", current_op, total_operations)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Database loading failed: {e}")
            return False
    
    def _load_to_offline_store(self, dataset_name: str) -> bool:
        """Load dataset to offline store."""
        return self.offline_store.load_dataset(dataset_name)
    
    def get_data_interface(self):
        """Get appropriate data interface based on current mode."""
        if self.current_mode == "database" and self.database:
            return DatabaseInterface(self.database)
        else:
            return OfflineInterface(self.offline_store)
    
    def get_current_mode(self) -> str:
        """Get current data loading mode."""
        return self.current_mode
    
    def is_database_available(self) -> bool:
        """Check if database is available."""
        return self.database_available
    
    def get_load_status(self) -> Dict[str, Any]:
        """Get current load status information."""
        return {
            "mode": self.current_mode,
            "database_available": self.database_available,
            "loaded_dataset": getattr(self.offline_store, 'loaded_dataset', None),
            "load_time": getattr(self.offline_store, 'load_time', None),
            "progress": {
                "current_operation": self.current_progress.current_operation,
                "progress_percent": self.current_progress.progress_percent,
                "elapsed_time": self.current_progress.elapsed_time
            }
        }


class DatabaseInterface:
    """Interface adapter for database mode."""
    
    def __init__(self, database: UnifiedDatabaseManager):
        self.database = database
    
    def get_paper(self, paper_id: str) -> Optional[Dict]:
        return self.database.get_paper_details(paper_id)
    
    def search_papers(self, query: str, limit: int = 50) -> List[Dict]:
        df = self.database.find_papers_by_keyword(query)
        return df.head(limit).to_dict('records')
    
    def get_network_statistics(self) -> Dict[str, int]:
        return self.database.get_network_statistics()


class OfflineInterface:
    """Interface adapter for offline mode."""
    
    def __init__(self, offline_store: OfflineDataStore):
        self.store = offline_store
    
    def get_paper(self, paper_id: str) -> Optional[Dict]:
        return self.store.get_paper(paper_id)
    
    def search_papers(self, query: str, limit: int = 50) -> List[Dict]:
        return self.store.search_papers(query, limit)
    
    def get_network_statistics(self) -> Dict[str, int]:
        return self.store.get_network_statistics()


# Global demo loader instance
_demo_loader: Optional[DemoDataLoader] = None


def get_demo_loader(prefer_database: bool = True) -> DemoDataLoader:
    """
    Get global demo data loader instance.
    
    Args:
        prefer_database: Whether to prefer database mode
        
    Returns:
        DemoDataLoader instance
    """
    global _demo_loader
    
    if _demo_loader is None:
        _demo_loader = DemoDataLoader(prefer_database=prefer_database)
    
    return _demo_loader


def quick_load_demo(dataset_name: Optional[str] = None,
                   force_offline: bool = False,
                   progress_callback: Optional[callable] = None) -> bool:
    """
    Quick function to load demo data with minimal setup.
    
    Args:
        dataset_name: Dataset to load (auto-selects if None)
        force_offline: Force offline mode
        progress_callback: Optional progress callback
        
    Returns:
        True if loaded successfully
    """
    loader = get_demo_loader()
    
    if progress_callback:
        loader.add_progress_callback(progress_callback)
    
    return loader.load_demo_dataset(dataset_name, force_offline)