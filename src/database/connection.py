"""
Centralized Neo4j database access layer for Academic Citation Platform.

This module provides a unified interface for all Neo4j database operations,
including connection management, query execution, and data retrieval functions.
Adapted from knowledge-cartography with enhancements for multi-codebase integration.

Classes:
    Neo4jConnection: Main database connection wrapper
    Neo4jError: Custom exception for database operations
    
Functions:
    get_db: Get singleton database connection (cached)
    get_overview_counts: Retrieve network overview statistics
    find_papers_by_keyword: Search papers by title keyword
    get_network_data: Get network data for specific paper
"""

from __future__ import annotations
import os
import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from neo4j import GraphDatabase

# Set up logging
logger = logging.getLogger(__name__)


class Neo4jError(Exception):
    """Custom exception for Neo4j database operations."""
    pass


class Neo4jConnection:
    """
    Enhanced Neo4j database connection wrapper with error handling and logging.
    
    This class provides a unified interface for executing Cypher queries and
    returning results as pandas DataFrames. It includes connection validation,
    error handling, and proper resource management.
    
    Attributes:
        _driver: Neo4j GraphDatabase driver instance
        _connected: Boolean flag indicating connection status
        
    Methods:
        query: Execute Cypher query and return DataFrame
        execute: Execute Cypher query without return (for write operations)  
        test_connection: Verify database connectivity
        get_database_info: Get database version and statistics
        close: Close database connection
    """
    
    def __init__(self, validate_connection: bool = True):
        """
        Initialize Neo4j connection using environment variables.
        
        Args:
            validate_connection: Whether to test connection on initialization
            
        Raises:
            Neo4jError: If connection parameters are missing or connection fails
        """
        try:
            # Get connection parameters from environment
            self.uri = os.getenv("NEO4J_URI")
            self.user = os.getenv("NEO4J_USER") 
            self.password = os.getenv("NEO4J_PWD") or os.getenv("NEO4J_PASSWORD")
            
            # Support multiple environment variable naming conventions
            if not self.uri:
                self.uri = os.getenv("NEO4J_URL")
            if not self.user:
                self.user = os.getenv("NEO4J_USERNAME")
            
            # Validate required parameters
            if not all([self.uri, self.user, self.password]):
                missing = [key for key, val in {
                    "NEO4J_URI/NEO4J_URL": self.uri,
                    "NEO4J_USER/NEO4J_USERNAME": self.user, 
                    "NEO4J_PWD/NEO4J_PASSWORD": self.password
                }.items() if not val]
                raise Neo4jError(f"Missing required environment variables: {missing}")
            
            # Create driver connection
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self._connected = False
            
            # Test connection if requested
            if validate_connection:
                self.test_connection()
                
            logger.info(f"Neo4j connection initialized to {self.uri}")
            
        except Exception as e:
            raise Neo4jError(f"Failed to initialize Neo4j connection: {str(e)}") from e

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a Cypher query and return results as a pandas DataFrame.
        
        This method is optimized for read queries that return data. For write
        operations without return values, use the execute method instead.
        
        Args:
            cypher: Cypher query string to execute
            params: Optional dictionary of query parameters
            
        Returns:
            pandas DataFrame containing query results
            
        Raises:
            Neo4jError: If query execution fails or connection is lost
            
        Examples:
            >>> db = Neo4jConnection()
            >>> df = db.query("MATCH (p:Paper) RETURN count(p) as total")
            >>> print(f"Total papers: {df.iloc[0]['total']}")
        """
        try:
            with self._driver.session() as session:
                logger.debug(f"Executing query: {cypher[:100]}...")
                result = session.run(cypher, params or {})
                data = [record.data() for record in result]
                df = pd.DataFrame(data)
                logger.debug(f"Query returned {len(df)} rows")
                return df
                
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            logger.error(f"Query: {cypher}")
            logger.error(f"Parameters: {params}")
            raise Neo4jError(f"Query execution failed: {str(e)}") from e

    def execute(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a Cypher query without expecting return data (for writes/updates).
        
        This method is optimized for write operations like CREATE, MERGE, DELETE.
        It returns execution statistics instead of data.
        
        Args:
            cypher: Cypher query string to execute
            params: Optional dictionary of query parameters
            
        Returns:
            Dictionary containing execution statistics
            
        Raises:
            Neo4jError: If query execution fails
            
        Examples:
            >>> stats = db.execute("CREATE (p:Paper {title: $title})", {"title": "New Paper"})
            >>> print(f"Nodes created: {stats['nodes_created']}")
        """
        try:
            with self._driver.session() as session:
                logger.debug(f"Executing command: {cypher[:100]}...")
                result = session.run(cypher, params or {})
                
                # Consume the result and get statistics
                summary = result.consume()
                stats = {
                    'nodes_created': summary.counters.nodes_created,
                    'nodes_deleted': summary.counters.nodes_deleted,
                    'relationships_created': summary.counters.relationships_created,
                    'relationships_deleted': summary.counters.relationships_deleted,
                    'properties_set': summary.counters.properties_set,
                    'labels_added': summary.counters.labels_added,
                    'labels_removed': summary.counters.labels_removed
                }
                
                logger.debug(f"Execution completed: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"Command execution failed: {str(e)}")
            logger.error(f"Command: {cypher}")
            logger.error(f"Parameters: {params}")
            raise Neo4jError(f"Command execution failed: {str(e)}") from e

    def test_connection(self) -> bool:
        """
        Test the database connection and verify it's working.
        
        Returns:
            True if connection is successful, raises exception otherwise
            
        Raises:
            Neo4jError: If connection test fails
        """
        try:
            logger.info("Testing Neo4j connection...")
            with self._driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    self._connected = True
                    logger.info("Neo4j connection test successful")
                    return True
                else:
                    raise Neo4jError(f"Connection test returned unexpected value: {test_value}")
                    
        except Exception as e:
            self._connected = False
            logger.error(f"Neo4j connection test failed: {str(e)}")
            raise Neo4jError(f"Connection test failed: {str(e)}") from e

    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the connected Neo4j database.
        
        Returns:
            Dictionary containing database version and basic statistics
            
        Raises:
            Neo4jError: If unable to retrieve database information
        """
        try:
            info = {}
            
            # Get database version
            try:
                version_result = self.query("CALL dbms.components()")
                if not version_result.empty:
                    info['version'] = version_result.iloc[0]['versions'][0]
                    info['edition'] = version_result.iloc[0]['edition']
            except Exception:
                # Fallback for older Neo4j versions or restricted permissions
                info['version'] = 'Unknown'
                info['edition'] = 'Unknown'
            
            # Get basic statistics
            stats_queries = {
                'total_nodes': "MATCH (n) RETURN count(n) as count",
                'total_relationships': "MATCH ()-[r]->() RETURN count(r) as count"
            }
            
            # Try to get node labels and relationship types (may fail in some environments)
            try:
                labels_result = self.query("CALL db.labels()")
                info['node_labels'] = labels_result.iloc[:, 0].tolist() if not labels_result.empty else []
            except Exception:
                info['node_labels'] = []
                
            try:
                rel_types_result = self.query("CALL db.relationshipTypes()")
                info['relationship_types'] = rel_types_result.iloc[:, 0].tolist() if not rel_types_result.empty else []
            except Exception:
                info['relationship_types'] = []
            
            for key, query in stats_queries.items():
                try:
                    result = self.query(query)
                    info[key] = result.iloc[0]['count'] if not result.empty else 0
                except Exception as e:
                    logger.warning(f"Could not retrieve {key}: {str(e)}")
                    info[key] = 0
            
            logger.info(f"Retrieved database info: {info.get('version', 'unknown version')}")
            return info
            
        except Exception as e:
            raise Neo4jError(f"Failed to get database info: {str(e)}") from e

    def get_network_statistics(self) -> Dict[str, int]:
        """
        Get basic network statistics for academic citation data.
        
        Returns:
            Dictionary containing counts of papers, authors, citations, etc.
        """
        try:
            stats = {}
            
            # Basic entity counts
            entity_queries = {
                'papers': "MATCH (p:Paper) RETURN count(p) as count",
                'authors': "MATCH (a:Author) RETURN count(a) as count",
                'venues': "MATCH (v:PubVenue) RETURN count(v) as count",
                'fields': "MATCH (f:Field) RETURN count(f) as count",
                'years': "MATCH (y:PubYear) RETURN count(y) as count"
            }
            
            # Relationship counts
            relationship_queries = {
                'citations': "MATCH ()-[:CITES]->() RETURN count(*) as count",
                'authorships': "MATCH ()-[:AUTHORED]->() RETURN count(*) as count",
                'publications': "MATCH ()-[:PUBLISHED_IN]->() RETURN count(*) as count",
                'field_associations': "MATCH ()-[:IS_ABOUT]->() RETURN count(*) as count"
            }
            
            # Execute all queries
            for key, query in {**entity_queries, **relationship_queries}.items():
                try:
                    result = self.query(query)
                    stats[key] = int(result.iloc[0]['count']) if not result.empty else 0
                except Exception as e:
                    logger.warning(f"Could not retrieve {key} count: {str(e)}")
                    stats[key] = 0
            
            return stats
            
        except Exception as e:
            raise Neo4jError(f"Failed to get network statistics: {str(e)}") from e

    def close(self) -> None:
        """Close the database connection and cleanup resources."""
        try:
            if hasattr(self, '_driver') and self._driver:
                self._driver.close()
                self._connected = False
                logger.info("Neo4j connection closed")
        except Exception as e:
            logger.warning(f"Error closing Neo4j connection: {str(e)}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.close()


# Utility functions for common operations
def create_connection(validate: bool = True) -> Neo4jConnection:
    """
    Create a new Neo4j connection.
    
    Args:
        validate: Whether to test the connection on creation
        
    Returns:
        Neo4jConnection instance
    """
    return Neo4jConnection(validate_connection=validate)


def find_papers_by_keyword(db: Neo4jConnection, keyword: str) -> pd.DataFrame:
    """
    Search for papers containing a keyword in their title.
    
    Args:
        db: Neo4j connection instance
        keyword: Keyword to search for
        
    Returns:
        DataFrame with matching papers
    """
    query = """
    MATCH (p:Paper)
    WHERE toLower(p.title) CONTAINS toLower($keyword)
    RETURN p.paperId as paperId, p.title as title, 
           p.year as year, p.citationCount as citationCount
    ORDER BY p.citationCount DESC
    LIMIT 100
    """
    return db.query(query, {"keyword": keyword})


def get_paper_details(db: Neo4jConnection, paper_id: str) -> pd.DataFrame:
    """
    Get detailed information about a specific paper.
    
    Args:
        db: Neo4j connection instance
        paper_id: Paper ID to look up
        
    Returns:
        DataFrame with paper details
    """
    query = """
    MATCH (p:Paper {paperId: $paperId})
    OPTIONAL MATCH (p)<-[:AUTHORED]-(a:Author)
    OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:PubVenue)
    OPTIONAL MATCH (p)-[:IS_ABOUT]->(f:Field)
    OPTIONAL MATCH (p)-[:PUB_YEAR]->(y:PubYear)
    RETURN p.paperId as paperId, p.title as title, p.abstract as abstract,
           p.year as year, p.citationCount as citationCount,
           collect(DISTINCT a.authorName) as authors,
           collect(DISTINCT v.venue) as venues,
           collect(DISTINCT f.field) as fields,
           y.year as pubYear
    """
    return db.query(query, {"paperId": paper_id})


def get_citation_network(db: Neo4jConnection, paper_id: str, depth: int = 1) -> pd.DataFrame:
    """
    Get citation network data for a specific paper.
    
    Args:
        db: Neo4j connection instance
        paper_id: Central paper ID
        depth: How many citation hops to include
        
    Returns:
        DataFrame with network triples (source, relationship, target)
    """
    if depth == 1:
        query = """
        MATCH (p:Paper {paperId: $paperId})
        OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
        OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
        WITH p, collect(cited) as cited_papers, collect(citing) as citing_papers
        UNWIND cited_papers as cited
        RETURN p.paperId as source_id, p.title as source_label, 'Paper' as source_type,
               'CITES' as relationship_type,
               cited.paperId as target_id, cited.title as target_label, 'Paper' as target_type
        UNION
        UNWIND citing_papers as citing
        RETURN citing.paperId as source_id, citing.title as source_label, 'Paper' as source_type,
               'CITES' as relationship_type,
               p.paperId as target_id, p.title as target_label, 'Paper' as target_type
        """
    else:
        # For deeper networks, we might want to limit the results
        query = f"""
        MATCH path = (p:Paper {{paperId: $paperId}})-[:CITES*1..{depth}]-(connected:Paper)
        WITH p, connected, length(path) as distance
        WHERE distance <= {depth}
        RETURN p.paperId as source_id, p.title as source_label, 'Paper' as source_type,
               'CITES' as relationship_type,
               connected.paperId as target_id, connected.title as target_label, 'Paper' as target_type,
               distance
        ORDER BY distance
        LIMIT 1000
        """
    
    return db.query(query, {"paperId": paper_id})