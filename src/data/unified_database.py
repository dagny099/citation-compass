"""
Unified database layer that combines capabilities from all three reference codebases.

This module provides:
- Enhanced Neo4j connection management (from knowledge-cartography)
- ML embedding storage support (for citation-map-dashboard models)
- Comprehensive query interface (from academic-citation-prediction)
- Caching and performance optimization
- Schema migration and validation

The UnifiedDatabaseManager provides a single interface for all database operations
across the integrated platform.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from neo4j import GraphDatabase, Session
from neo4j.exceptions import ServiceUnavailable, TransientError

from .api_config import get_config, Neo4jConfig


class DatabaseError(Exception):
    """Custom exception for database operation failures."""
    pass


class SchemaValidator:
    """
    Validates and manages database schema evolution across different versions.
    
    Ensures compatibility between the different schemas used in the reference codebases
    while supporting new features like ML embedding storage.
    """
    
    REQUIRED_CONSTRAINTS = [
        "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paperId IS UNIQUE",
        "CREATE CONSTRAINT author_id_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.authorId IS UNIQUE",
        "CREATE CONSTRAINT venue_name_unique IF NOT EXISTS FOR (v:PubVenue) REQUIRE v.name IS UNIQUE",
        "CREATE CONSTRAINT field_name_unique IF NOT EXISTS FOR (f:Field) REQUIRE f.name IS UNIQUE",
        "CREATE CONSTRAINT year_value_unique IF NOT EXISTS FOR (y:PubYear) REQUIRE y.year IS UNIQUE",
    ]
    
    REQUIRED_INDEXES = [
        "CREATE INDEX paper_title_index IF NOT EXISTS FOR (p:Paper) ON (p.title)",
        "CREATE INDEX paper_year_index IF NOT EXISTS FOR (p:Paper) ON (p.year)",
        "CREATE INDEX paper_citation_count_index IF NOT EXISTS FOR (p:Paper) ON (p.citationCount)",
        "CREATE INDEX author_name_index IF NOT EXISTS FOR (a:Author) ON (a.name)",
    ]
    
    # Schema versioning for migrations
    SCHEMA_VERSION = "2.0"  # Updated for ML integration
    
    def __init__(self, session: Session):
        """
        Initialize schema validator.
        
        Args:
            session: Neo4j database session
        """
        self.session = session
        self.logger = logging.getLogger(__name__)
    
    def validate_and_create_schema(self) -> Dict[str, Any]:
        """
        Validate existing schema and create missing constraints/indexes.
        
        Returns:
            Dictionary with validation results and actions taken
        """
        self.logger.info("Validating and creating database schema...")
        
        results = {
            'constraints_created': 0,
            'indexes_created': 0,
            'errors': [],
            'schema_version': self.SCHEMA_VERSION
        }
        
        # Create constraints
        for constraint in self.REQUIRED_CONSTRAINTS:
            try:
                self.session.run(constraint)
                results['constraints_created'] += 1
                self.logger.debug(f"Created/verified constraint: {constraint[:50]}...")
            except Exception as e:
                error_msg = f"Failed to create constraint: {str(e)}"
                results['errors'].append(error_msg)
                self.logger.warning(error_msg)
        
        # Create indexes
        for index in self.REQUIRED_INDEXES:
            try:
                self.session.run(index)
                results['indexes_created'] += 1
                self.logger.debug(f"Created/verified index: {index[:50]}...")
            except Exception as e:
                error_msg = f"Failed to create index: {str(e)}"
                results['errors'].append(error_msg)
                self.logger.warning(error_msg)
        
        # Set schema version metadata
        try:
            self.session.run("""
                MERGE (v:SchemaVersion {version: $version})
                SET v.updated_at = datetime()
            """, version=self.SCHEMA_VERSION)
        except Exception as e:
            self.logger.warning(f"Failed to set schema version: {e}")
        
        self.logger.info(f"Schema validation complete: {results['constraints_created']} constraints, "
                        f"{results['indexes_created']} indexes")
        
        return results
    
    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get current schema information including version and statistics.
        
        Returns:
            Dictionary with schema metadata
        """
        try:
            # Get schema version
            version_result = self.session.run("MATCH (v:SchemaVersion) RETURN v.version as version, v.updated_at as updated")
            version_data = version_result.single()
            
            # Get constraint information
            constraints_result = self.session.run("SHOW CONSTRAINTS")
            constraints = [record['name'] for record in constraints_result]
            
            # Get index information  
            indexes_result = self.session.run("SHOW INDEXES")
            indexes = [record['name'] for record in indexes_result]
            
            # Get node and relationship counts
            stats_result = self.session.run("""
                MATCH (n) 
                OPTIONAL MATCH ()-[r]->()
                RETURN 
                    count(DISTINCT n) as total_nodes,
                    count(r) as total_relationships,
                    count(DISTINCT labels(n)) as unique_labels
            """)
            stats = stats_result.single()
            
            return {
                'version': version_data['version'] if version_data else 'unknown',
                'updated_at': version_data['updated_at'] if version_data else None,
                'constraints': constraints,
                'indexes': indexes,
                'statistics': {
                    'total_nodes': stats['total_nodes'],
                    'total_relationships': stats['total_relationships'],
                    'unique_labels': stats['unique_labels']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get schema info: {e}")
            return {'error': str(e)}


class QueryLibrary:
    """
    Centralized library of all Cypher queries used across the platform.
    
    Consolidates queries from all three reference codebases and adds new ones
    for ML integration and enhanced functionality.
    """
    
    # Node count queries (from knowledge-cartography)
    GET_PAPERS_COUNT = "MATCH (p:Paper) RETURN count(p) as count"
    GET_AUTHORS_COUNT = "MATCH (a:Author) RETURN count(a) as count"
    GET_VENUES_COUNT = "MATCH (v:PubVenue) RETURN count(v) as count"
    GET_FIELDS_COUNT = "MATCH (f:Field) RETURN count(f) as count"
    
    # Citation queries (from citation-map-dashboard)
    GET_CITATION_EDGES = """
        MATCH (source:Paper)-[:CITES]->(target:Paper)
        RETURN source.paperId as source_id, target.paperId as target_id
    """
    
    # Paper details queries (from academic-citation-prediction)
    GET_PAPER_DETAILS = """
        MATCH (p:Paper {paperId: $paperId})
        OPTIONAL MATCH (p)<-[:AUTHORED]-(a:Author)
        OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:PubVenue)
        OPTIONAL MATCH (p)-[:IS_ABOUT]->(f:Field)
        OPTIONAL MATCH (p)-[:PUB_YEAR]->(y:PubYear)
        RETURN 
            p.paperId as paperId,
            p.title as title,
            p.abstract as abstract,
            p.citationCount as citationCount,
            p.year as year,
            p.publicationDate as publicationDate,
            collect(DISTINCT a.name) as authors,
            collect(DISTINCT v.name) as venues,
            collect(DISTINCT f.name) as fields,
            y.year as pubYear
    """
    
    # Network analysis queries
    GET_PAPER_CITATIONS = """
        MATCH (citing:Paper)-[:CITES]->(cited:Paper {paperId: $paperId})
        RETURN citing.paperId as paperId, citing.title as title, citing.citationCount as citationCount
        ORDER BY citing.citationCount DESC
    """
    
    GET_PAPER_REFERENCES = """
        MATCH (paper:Paper {paperId: $paperId})-[:CITES]->(referenced:Paper)
        RETURN referenced.paperId as paperId, referenced.title as title, referenced.citationCount as citationCount
        ORDER BY referenced.citationCount DESC
    """
    
    # Search queries
    FIND_PAPERS_BY_KEYWORD = """
        MATCH (p:Paper)
        WHERE toLower(p.title) CONTAINS toLower($keyword)
        RETURN p.paperId as paperId, p.title as title, p.citationCount as citationCount, p.year as year
        ORDER BY p.citationCount DESC
        LIMIT 50
    """
    
    # ML integration queries - NEW
    STORE_PAPER_EMBEDDING = """
        MATCH (p:Paper {paperId: $paperId})
        SET p.embedding = $embedding,
            p.embedding_model = $model_name,
            p.embedding_updated = datetime()
        RETURN p.paperId as paperId
    """
    
    GET_PAPERS_WITH_EMBEDDINGS = """
        MATCH (p:Paper)
        WHERE p.embedding IS NOT NULL
        RETURN p.paperId as paperId, p.embedding as embedding
    """
    
    # Batch operations
    BATCH_CREATE_PAPERS = """
        UNWIND $papers as paper
        MERGE (p:Paper {paperId: paper.paperId})
        SET p.title = paper.title,
            p.abstract = paper.abstract,
            p.citationCount = paper.citationCount,
            p.year = paper.year,
            p.publicationDate = paper.publicationDate,
            p.updated_at = datetime()
        RETURN count(p) as created_count
    """
    
    BATCH_CREATE_CITATIONS = """
        UNWIND $citations as citation
        MATCH (source:Paper {paperId: citation.source_id})
        MATCH (target:Paper {paperId: citation.target_id})
        MERGE (source)-[:CITES]->(target)
        RETURN count(*) as created_count
    """
    
    # Analytics queries (enhanced from all codebases)
    GET_TOP_CITED_PAPERS = """
        MATCH (p:Paper)
        WHERE p.citationCount > 0
        RETURN p.paperId as paperId, p.title as title, p.citationCount as citationCount
        ORDER BY p.citationCount DESC
        LIMIT $limit
    """
    
    GET_AUTHOR_COLLABORATION_NETWORK = """
        MATCH (a1:Author)-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author)
        WHERE a1.authorId < a2.authorId
        RETURN a1.name as author1, a2.name as author2, count(p) as collaborations
        ORDER BY collaborations DESC
        LIMIT $limit
    """


class UnifiedDatabaseManager:
    """
    Unified database manager combining capabilities from all three reference codebases.
    
    Features:
    - Enhanced connection management with retry logic
    - ML embedding storage and retrieval
    - Comprehensive query interface
    - Caching for performance optimization
    - Schema validation and migration
    - Batch operations for efficiency
    """
    
    def __init__(self, config: Optional[Neo4jConfig] = None):
        """
        Initialize the unified database manager.
        
        Args:
            config: Optional database configuration. Uses global config if None.
        """
        self.config = config or get_config().get_db_config()
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        if not self.config.validate():
            missing = self.config.get_missing_params()
            raise DatabaseError(f"Database configuration incomplete. Missing: {missing}")
        
        # Connection management
        self._driver = None
        self._connected = False
        self._connection_attempts = 0
        
        # Query cache for performance
        self._query_cache = {}
        self._cache_ttl = get_config().get_cache_config().db_query_ttl
        
        # Initialize connection
        self._initialize_connection()
    
    def _initialize_connection(self) -> None:
        """Initialize database connection with retry logic."""
        max_attempts = self.config.max_retry_attempts
        
        while self._connection_attempts < max_attempts:
            try:
                self.logger.info(f"Connecting to Neo4j at {self.config.uri}")
                
                self._driver = GraphDatabase.driver(
                    self.config.uri,
                    auth=(self.config.username, self.config.password),
                    connection_timeout=self.config.connection_timeout
                )
                
                # Test connection
                with self._driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    test_value = result.single()["test"]
                    
                    if test_value != 1:
                        raise DatabaseError(f"Connection test failed: expected 1, got {test_value}")
                
                self._connected = True
                self.logger.info("Successfully connected to Neo4j database")
                
                # Initialize schema
                self._initialize_schema()
                break
                
            except Exception as e:
                self._connection_attempts += 1
                self.logger.warning(f"Connection attempt {self._connection_attempts} failed: {e}")
                
                if self._connection_attempts >= max_attempts:
                    raise DatabaseError(f"Failed to connect after {max_attempts} attempts: {e}")
                
                time.sleep(2 ** self._connection_attempts)  # Exponential backoff
    
    def _initialize_schema(self) -> None:
        """Initialize and validate database schema."""
        try:
            with self._driver.session() as session:
                validator = SchemaValidator(session)
                schema_results = validator.validate_and_create_schema()
                
                if schema_results['errors']:
                    self.logger.warning(f"Schema validation had {len(schema_results['errors'])} errors")
                    for error in schema_results['errors']:
                        self.logger.warning(f"  - {error}")
                
                self.logger.info(f"Schema initialized: version {schema_results['schema_version']}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize schema: {e}")
            # Don't fail connection for schema issues
    
    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None, 
              use_cache: bool = True) -> pd.DataFrame:
        """
        Execute Cypher query and return results as DataFrame.
        
        Args:
            cypher: Cypher query string
            params: Query parameters
            use_cache: Whether to use query caching
            
        Returns:
            DataFrame containing query results
            
        Raises:
            DatabaseError: If query execution fails
        """
        if not self._connected:
            raise DatabaseError("Database not connected")
        
        # Check cache first
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(cypher, params)
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                return cached_result
        
        try:
            start_time = time.time()
            
            with self._driver.session() as session:
                self.logger.debug(f"Executing query: {cypher[:100]}...")
                result = session.run(cypher, params or {})
                
                # Convert to DataFrame
                data = [record.data() for record in result]
                df = pd.DataFrame(data)
                
                execution_time = time.time() - start_time
                self.logger.debug(f"Query completed in {execution_time:.3f}s, returned {len(df)} rows")
                
                # Cache result
                if use_cache and cache_key:
                    self._cache_result(cache_key, df)
                
                return df
                
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            self.logger.error(f"Query: {cypher}")
            self.logger.error(f"Parameters: {params}")
            raise DatabaseError(f"Query execution failed: {str(e)}")
    
    def execute(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute Cypher query without expecting return data (for writes).
        
        Args:
            cypher: Cypher query string
            params: Query parameters
            
        Returns:
            Dictionary containing execution statistics
            
        Raises:
            DatabaseError: If execution fails
        """
        if not self._connected:
            raise DatabaseError("Database not connected")
        
        try:
            start_time = time.time()
            
            with self._driver.session() as session:
                self.logger.debug(f"Executing command: {cypher[:100]}...")
                result = session.run(cypher, params or {})
                
                # Get execution statistics
                summary = result.consume()
                stats = {
                    'nodes_created': summary.counters.nodes_created,
                    'nodes_deleted': summary.counters.nodes_deleted,
                    'relationships_created': summary.counters.relationships_created,
                    'relationships_deleted': summary.counters.relationships_deleted,
                    'properties_set': summary.counters.properties_set,
                    'execution_time': time.time() - start_time
                }
                
                self.logger.debug(f"Execution completed: {stats}")
                return stats
                
        except Exception as e:
            self.logger.error(f"Command execution failed: {str(e)}")
            self.logger.error(f"Command: {cypher}")
            self.logger.error(f"Parameters: {params}")
            raise DatabaseError(f"Command execution failed: {str(e)}")
    
    def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive details for a single paper.
        
        Args:
            paper_id: Paper ID to retrieve
            
        Returns:
            Paper details dictionary or None if not found
        """
        df = self.query(QueryLibrary.GET_PAPER_DETAILS, {"paperId": paper_id})
        
        if df.empty:
            return None
        
        return df.iloc[0].to_dict()
    
    def get_paper_citations(self, paper_id: str, limit: int = 100) -> pd.DataFrame:
        """
        Get papers citing the specified paper.
        
        Args:
            paper_id: Paper ID to get citations for
            limit: Maximum number of citations to return
            
        Returns:
            DataFrame of citing papers
        """
        query = QueryLibrary.GET_PAPER_CITATIONS + f" LIMIT {limit}"
        return self.query(query, {"paperId": paper_id})
    
    def find_papers_by_keyword(self, keyword: str) -> pd.DataFrame:
        """
        Search papers by keyword in title.
        
        Args:
            keyword: Search keyword
            
        Returns:
            DataFrame of matching papers
        """
        return self.query(QueryLibrary.FIND_PAPERS_BY_KEYWORD, {"keyword": keyword})
    
    def store_paper_embeddings(self, embeddings: Dict[str, np.ndarray], 
                              model_name: str = "TransE") -> int:
        """
        Store ML embeddings for papers.
        
        Args:
            embeddings: Dictionary mapping paper IDs to embedding vectors
            model_name: Name of the model that generated embeddings
            
        Returns:
            Number of papers updated with embeddings
        """
        updated_count = 0
        
        for paper_id, embedding in embeddings.items():
            try:
                # Convert numpy array to list for Neo4j storage
                embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                
                result = self.execute(
                    QueryLibrary.STORE_PAPER_EMBEDDING,
                    {
                        "paperId": paper_id,
                        "embedding": embedding_list,
                        "model_name": model_name
                    }
                )
                
                if result.get('properties_set', 0) > 0:
                    updated_count += 1
                    
            except Exception as e:
                self.logger.warning(f"Failed to store embedding for paper {paper_id}: {e}")
        
        self.logger.info(f"Stored embeddings for {updated_count}/{len(embeddings)} papers")
        return updated_count
    
    def get_papers_with_embeddings(self) -> pd.DataFrame:
        """
        Get all papers that have ML embeddings stored.
        
        Returns:
            DataFrame with paper IDs and embeddings
        """
        return self.query(QueryLibrary.GET_PAPERS_WITH_EMBEDDINGS)
    
    def batch_create_papers(self, papers: List[Dict[str, Any]]) -> int:
        """
        Create or update multiple papers in a single transaction.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Number of papers created/updated
        """
        result = self.execute(QueryLibrary.BATCH_CREATE_PAPERS, {"papers": papers})
        return result.get('nodes_created', 0) + result.get('properties_set', 0) // len(papers)
    
    def batch_create_citations(self, citations: List[Dict[str, str]]) -> int:
        """
        Create multiple citation relationships in a single transaction.
        
        Args:
            citations: List of citation dictionaries with 'source_id' and 'target_id'
            
        Returns:
            Number of citation relationships created
        """
        result = self.execute(QueryLibrary.BATCH_CREATE_CITATIONS, {"citations": citations})
        return result.get('relationships_created', 0)
    
    def get_network_statistics(self) -> Dict[str, int]:
        """
        Get comprehensive network statistics.
        
        Returns:
            Dictionary with various network metrics
        """
        stats = {}
        
        # Basic counts
        stats['papers'] = int(self.query(QueryLibrary.GET_PAPERS_COUNT).iloc[0, 0])
        stats['authors'] = int(self.query(QueryLibrary.GET_AUTHORS_COUNT).iloc[0, 0])
        stats['venues'] = int(self.query(QueryLibrary.GET_VENUES_COUNT).iloc[0, 0])
        stats['fields'] = int(self.query(QueryLibrary.GET_FIELDS_COUNT).iloc[0, 0])
        
        # Citation statistics
        citation_df = self.query(QueryLibrary.GET_CITATION_EDGES)
        stats['citations'] = len(citation_df)
        
        # Embedding statistics
        embedding_df = self.query(QueryLibrary.GET_PAPERS_WITH_EMBEDDINGS)
        stats['papers_with_embeddings'] = len(embedding_df)
        
        return stats
    
    def _generate_cache_key(self, cypher: str, params: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for query and parameters."""
        import hashlib
        key_data = f"{cypher}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached query result if not expired."""
        if cache_key in self._query_cache:
            result, timestamp = self._query_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return result.copy()  # Return copy to avoid modification
            else:
                del self._query_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: pd.DataFrame) -> None:
        """Cache query result with timestamp."""
        self._query_cache[cache_key] = (result.copy(), time.time())
        
        # Simple cache size management
        if len(self._query_cache) > 100:  # Keep cache size reasonable
            # Remove oldest entries
            sorted_items = sorted(
                self._query_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            for key, _ in sorted_items[:20]:  # Remove oldest 20%
                del self._query_cache[key]
    
    def test_connection(self) -> bool:
        """
        Test database connectivity.
        
        Returns:
            True if connection is working, False otherwise
        """
        try:
            self.query("RETURN 1 as test", use_cache=False)
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get current database schema information.
        
        Returns:
            Dictionary with schema metadata
        """
        try:
            with self._driver.session() as session:
                validator = SchemaValidator(session)
                return validator.get_schema_info()
        except Exception as e:
            self.logger.error(f"Failed to get schema info: {e}")
            return {'error': str(e)}
    
    def close(self) -> None:
        """Close database connection and cleanup resources."""
        if self._driver:
            try:
                self._driver.close()
                self._connected = False
                self.logger.info("Database connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing database connection: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()


# Singleton instance for Streamlit caching
_db_instance: Optional[UnifiedDatabaseManager] = None

def get_database(config: Optional[Neo4jConfig] = None) -> UnifiedDatabaseManager:
    """
    Get a singleton database instance for the application.
    
    Args:
        config: Optional database configuration override
        
    Returns:
        UnifiedDatabaseManager instance
    """
    global _db_instance
    
    if _db_instance is None or config is not None:
        _db_instance = UnifiedDatabaseManager(config)
    
    return _db_instance


def reset_database() -> None:
    """Reset database singleton. Useful for testing."""
    global _db_instance
    if _db_instance:
        _db_instance.close()
    _db_instance = None