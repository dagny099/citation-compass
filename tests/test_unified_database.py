"""
Test suite for the unified database layer.

Tests the enhanced database capabilities including ML embedding storage,
schema validation, and unified query interface.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time

from src.data.unified_database import (
    UnifiedDatabaseManager,
    SchemaValidator,
    QueryLibrary,
    DatabaseError,
    get_database,
    reset_database
)
from src.data.api_config import Neo4jConfig


class TestSchemaValidator:
    """Test schema validation and management."""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock Neo4j session."""
        session = Mock()
        session.run.return_value = Mock()
        return session
    
    def test_schema_validator_initialization(self, mock_session):
        """Test schema validator initializes correctly."""
        validator = SchemaValidator(mock_session)
        assert validator.session == mock_session
        assert validator.SCHEMA_VERSION == "2.0"
    
    def test_validate_and_create_schema_success(self, mock_session):
        """Test successful schema validation and creation."""
        validator = SchemaValidator(mock_session)
        
        # Mock successful constraint/index creation
        mock_session.run.return_value = Mock()
        
        results = validator.validate_and_create_schema()
        
        assert results['schema_version'] == "2.0"
        assert results['constraints_created'] == len(validator.REQUIRED_CONSTRAINTS)
        assert results['indexes_created'] == len(validator.REQUIRED_INDEXES)
        assert len(results['errors']) == 0
    
    def test_validate_and_create_schema_with_errors(self, mock_session):
        """Test schema validation with some failures."""
        validator = SchemaValidator(mock_session)
        
        # Mock some failures
        def side_effect(query):
            if "CREATE CONSTRAINT" in query:
                raise Exception("Constraint already exists")
            return Mock()
        
        mock_session.run.side_effect = side_effect
        
        results = validator.validate_and_create_schema()
        
        assert len(results['errors']) > 0
        assert "Constraint already exists" in str(results['errors'])
    
    def test_get_schema_info(self, mock_session):
        """Test schema information retrieval."""
        validator = SchemaValidator(mock_session)
        
        # Mock responses for different queries
        def run_side_effect(query, params=None):
            if "MATCH (v:SchemaVersion)" in query:
                result = Mock()
                result.single.return_value = {'version': '2.0', 'updated_at': '2023-01-01'}
                return result
            elif "SHOW CONSTRAINTS" in query:
                return [{'name': 'constraint1'}, {'name': 'constraint2'}]
            elif "SHOW INDEXES" in query:
                return [{'name': 'index1'}, {'name': 'index2'}]
            else:  # Stats query
                result = Mock()
                result.single.return_value = {
                    'total_nodes': 1000,
                    'total_relationships': 2000,
                    'unique_labels': 5
                }
                return result
        
        mock_session.run.side_effect = run_side_effect
        
        info = validator.get_schema_info()
        
        assert info['version'] == '2.0'
        assert len(info['constraints']) == 2
        assert len(info['indexes']) == 2
        assert info['statistics']['total_nodes'] == 1000


class TestQueryLibrary:
    """Test the centralized query library."""
    
    def test_query_constants_exist(self):
        """Test that all expected queries are defined."""
        expected_queries = [
            'GET_PAPERS_COUNT',
            'GET_AUTHORS_COUNT', 
            'GET_CITATION_EDGES',
            'GET_PAPER_DETAILS',
            'FIND_PAPERS_BY_KEYWORD',
            'STORE_PAPER_EMBEDDING',
            'GET_PAPERS_WITH_EMBEDDINGS',
            'BATCH_CREATE_PAPERS',
            'BATCH_CREATE_CITATIONS'
        ]
        
        for query_name in expected_queries:
            assert hasattr(QueryLibrary, query_name)
            assert isinstance(getattr(QueryLibrary, query_name), str)
            assert len(getattr(QueryLibrary, query_name)) > 0
    
    def test_parameterized_queries(self):
        """Test that parameterized queries contain parameter placeholders."""
        parameterized_queries = [
            ('GET_PAPER_DETAILS', '$paperId'),
            ('FIND_PAPERS_BY_KEYWORD', '$keyword'),
            ('STORE_PAPER_EMBEDDING', '$paperId'),
            ('BATCH_CREATE_PAPERS', '$papers'),
            ('BATCH_CREATE_CITATIONS', '$citations')
        ]
        
        for query_name, expected_param in parameterized_queries:
            query = getattr(QueryLibrary, query_name)
            assert expected_param in query


class TestUnifiedDatabaseManager:
    """Test the main database manager."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock database configuration."""
        config = Neo4jConfig()
        config.uri = "bolt://localhost:7687"
        config.username = "neo4j"
        config.password = "password"
        return config
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = Mock()
        session = Mock()
        result = Mock()
        
        # Mock successful connection test
        result.single.return_value = {"test": 1}
        session.run.return_value = result
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        
        return driver, session, result
    
    @patch('src.data.unified_database.GraphDatabase.driver')
    def test_database_manager_initialization(self, mock_driver_class, mock_config, mock_driver):
        """Test database manager initializes correctly."""
        driver, session, result = mock_driver
        mock_driver_class.return_value = driver
        
        db = UnifiedDatabaseManager(config=mock_config)
        
        assert db.config == mock_config
        assert db._connected is True
        mock_driver_class.assert_called_once()
    
    @patch('src.data.unified_database.GraphDatabase.driver')
    def test_database_connection_failure(self, mock_driver_class, mock_config):
        """Test database manager handles connection failures."""
        mock_driver_class.side_effect = Exception("Connection failed")
        
        with pytest.raises(DatabaseError, match="Failed to connect"):
            UnifiedDatabaseManager(config=mock_config)
    
    @patch('src.data.unified_database.GraphDatabase.driver')
    def test_query_execution(self, mock_driver_class, mock_config, mock_driver):
        """Test successful query execution."""
        driver, session, result = mock_driver
        mock_driver_class.return_value = driver
        
        # Mock query result
        result.single.return_value = {"test": 1}  # For connection test
        
        # Create separate result for actual query
        query_result = Mock()
        test_data = [{"count": 42}]
        query_result.data.return_value = test_data
        session.run.side_effect = [result, query_result]  # Connection test + actual query
        
        db = UnifiedDatabaseManager(config=mock_config)
        
        # Mock the data method for DataFrame creation
        query_result.__iter__ = Mock(return_value=iter([Mock(data=lambda: {"count": 42})]))
        
        df = db.query("MATCH (p:Paper) RETURN count(p) as count")
        
        assert isinstance(df, pd.DataFrame)
        session.run.assert_called()
    
    @patch('src.data.unified_database.GraphDatabase.driver')
    def test_query_caching(self, mock_driver_class, mock_config, mock_driver):
        """Test query result caching."""
        driver, session, result = mock_driver
        mock_driver_class.return_value = driver
        
        # Mock results
        result.single.return_value = {"test": 1}  # Connection test
        query_result = Mock()
        query_result.__iter__ = Mock(return_value=iter([Mock(data=lambda: {"count": 42})]))
        session.run.side_effect = [result, query_result]  # Connection test + query
        
        db = UnifiedDatabaseManager(config=mock_config)
        
        # First query
        df1 = db.query("SELECT 1", use_cache=True)
        
        # Second query (should use cache)
        df2 = db.query("SELECT 1", use_cache=True)
        
        # Should only call run twice (connection test + first query)
        assert session.run.call_count == 2
        assert df1.equals(df2)
    
    @patch('src.data.unified_database.GraphDatabase.driver')
    def test_execute_command(self, mock_driver_class, mock_config, mock_driver):
        """Test command execution without return data."""
        driver, session, result = mock_driver
        mock_driver_class.return_value = driver
        
        # Mock connection test
        result.single.return_value = {"test": 1}
        
        # Mock execute result with summary
        execute_result = Mock()
        summary = Mock()
        summary.counters.nodes_created = 5
        summary.counters.relationships_created = 10
        summary.counters.properties_set = 15
        execute_result.consume.return_value = summary
        
        session.run.side_effect = [result, execute_result]
        
        db = UnifiedDatabaseManager(config=mock_config)
        
        stats = db.execute("CREATE (n:Test) RETURN n")
        
        assert stats['nodes_created'] == 5
        assert stats['relationships_created'] == 10
        assert stats['properties_set'] == 15
        assert 'execution_time' in stats
    
    @patch('src.data.unified_database.GraphDatabase.driver')
    def test_get_paper_details(self, mock_driver_class, mock_config, mock_driver):
        """Test getting paper details."""
        driver, session, result = mock_driver
        mock_driver_class.return_value = driver
        
        # Mock connection test
        result.single.return_value = {"test": 1}
        
        # Mock paper details query
        paper_result = Mock()
        paper_data = {
            "paperId": "123",
            "title": "Test Paper",
            "citationCount": 42,
            "authors": ["Author 1", "Author 2"]
        }
        paper_result.__iter__ = Mock(return_value=iter([Mock(data=lambda: paper_data)]))
        
        session.run.side_effect = [result, paper_result]
        
        db = UnifiedDatabaseManager(config=mock_config)
        
        paper = db.get_paper_details("123")
        
        assert paper is not None
        assert paper["paperId"] == "123"
        assert paper["title"] == "Test Paper"
    
    @patch('src.data.unified_database.GraphDatabase.driver')
    def test_store_paper_embeddings(self, mock_driver_class, mock_config, mock_driver):
        """Test storing ML embeddings."""
        driver, session, result = mock_driver
        mock_driver_class.return_value = driver
        
        # Mock connection test
        result.single.return_value = {"test": 1}
        
        # Mock embedding storage
        embedding_result = Mock()
        summary = Mock()
        summary.counters.properties_set = 3  # embedding, model_name, updated_at
        embedding_result.consume.return_value = summary
        
        session.run.side_effect = [result, embedding_result, embedding_result]
        
        db = UnifiedDatabaseManager(config=mock_config)
        
        # Test with numpy arrays
        embeddings = {
            "paper1": np.array([0.1, 0.2, 0.3]),
            "paper2": np.array([0.4, 0.5, 0.6])
        }
        
        updated_count = db.store_paper_embeddings(embeddings, "TransE")
        
        assert updated_count == 2  # Both papers updated
    
    @patch('src.data.unified_database.GraphDatabase.driver')
    def test_batch_operations(self, mock_driver_class, mock_config, mock_driver):
        """Test batch create operations."""
        driver, session, result = mock_driver
        mock_driver_class.return_value = driver
        
        # Mock connection test
        result.single.return_value = {"test": 1}
        
        # Mock batch operations
        batch_result = Mock()
        summary = Mock()
        summary.counters.nodes_created = 5
        summary.counters.relationships_created = 10
        batch_result.consume.return_value = summary
        
        session.run.side_effect = [result, batch_result, batch_result]
        
        db = UnifiedDatabaseManager(config=mock_config)
        
        # Test batch create papers
        papers = [
            {"paperId": "1", "title": "Paper 1"},
            {"paperId": "2", "title": "Paper 2"}
        ]
        created_papers = db.batch_create_papers(papers)
        assert created_papers >= 0  # Should work without error
        
        # Test batch create citations
        citations = [
            {"source_id": "1", "target_id": "2"},
            {"source_id": "2", "target_id": "3"}
        ]
        created_citations = db.batch_create_citations(citations)
        assert created_citations == 10  # From mock
    
    @patch('src.data.unified_database.GraphDatabase.driver')
    def test_network_statistics(self, mock_driver_class, mock_config, mock_driver):
        """Test network statistics retrieval."""
        driver, session, result = mock_driver
        mock_driver_class.return_value = driver
        
        # Mock connection test
        result.single.return_value = {"test": 1}
        
        # Mock different count queries
        def run_side_effect(query, params=None):
            if "count(p)" in query:
                mock_result = Mock()
                mock_result.__iter__ = Mock(return_value=iter([Mock(data=lambda: {"count": 100})]))
                return mock_result
            elif "count(a)" in query:
                mock_result = Mock()
                mock_result.__iter__ = Mock(return_value=iter([Mock(data=lambda: {"count": 200})]))
                return mock_result
            elif "source_id" in query:  # Citation edges
                mock_result = Mock()
                mock_result.__iter__ = Mock(return_value=iter([
                    Mock(data=lambda: {"source_id": "1", "target_id": "2"}),
                    Mock(data=lambda: {"source_id": "2", "target_id": "3"})
                ]))
                return mock_result
            else:
                return result
        
        session.run.side_effect = run_side_effect
        
        db = UnifiedDatabaseManager(config=mock_config)
        
        stats = db.get_network_statistics()
        
        assert 'papers' in stats
        assert 'authors' in stats
        assert 'citations' in stats
        assert isinstance(stats['papers'], int)
    
    @patch('src.data.unified_database.GraphDatabase.driver')
    def test_connection_test(self, mock_driver_class, mock_config, mock_driver):
        """Test connection testing method."""
        driver, session, result = mock_driver
        mock_driver_class.return_value = driver
        
        result.single.return_value = {"test": 1}
        session.run.return_value = result
        
        db = UnifiedDatabaseManager(config=mock_config)
        
        # Should return True for successful connection
        assert db.test_connection() is True
    
    @patch('src.data.unified_database.GraphDatabase.driver')
    def test_context_manager(self, mock_driver_class, mock_config, mock_driver):
        """Test context manager functionality."""
        driver, session, result = mock_driver
        mock_driver_class.return_value = driver
        result.single.return_value = {"test": 1}
        
        with UnifiedDatabaseManager(config=mock_config) as db:
            assert db._connected is True
        
        # Should call close
        driver.close.assert_called_once()


class TestDatabaseSingleton:
    """Test singleton database instance management."""
    
    def teardown_method(self):
        """Reset singleton after each test."""
        reset_database()
    
    @patch('src.data.unified_database.UnifiedDatabaseManager')
    def test_get_database_singleton(self, mock_db_class):
        """Test singleton behavior of get_database."""
        mock_instance = Mock()
        mock_db_class.return_value = mock_instance
        
        # First call creates instance
        db1 = get_database()
        assert db1 == mock_instance
        mock_db_class.assert_called_once()
        
        # Second call returns same instance
        db2 = get_database()
        assert db2 == mock_instance
        assert db1 is db2
        # Should not create new instance
        mock_db_class.assert_called_once()
    
    @patch('src.data.unified_database.UnifiedDatabaseManager')
    def test_get_database_with_config_override(self, mock_db_class):
        """Test that config override creates new instance."""
        mock_instance1 = Mock()
        mock_instance2 = Mock()
        mock_db_class.side_effect = [mock_instance1, mock_instance2]
        
        # First call
        db1 = get_database()
        assert db1 == mock_instance1
        
        # Call with config override should create new instance
        config = Neo4jConfig()
        db2 = get_database(config)
        assert db2 == mock_instance2
        assert db1 is not db2
        
        assert mock_db_class.call_count == 2
    
    @patch('src.data.unified_database.UnifiedDatabaseManager')
    def test_reset_database(self, mock_db_class):
        """Test database reset functionality."""
        mock_instance = Mock()
        mock_db_class.return_value = mock_instance
        
        # Create instance
        db1 = get_database()
        assert db1 == mock_instance
        
        # Reset
        reset_database()
        mock_instance.close.assert_called_once()
        
        # Next call should create new instance
        mock_instance2 = Mock()
        mock_db_class.return_value = mock_instance2
        
        db2 = get_database()
        assert db2 == mock_instance2
        assert db1 is not db2


if __name__ == "__main__":
    pytest.main([__file__])