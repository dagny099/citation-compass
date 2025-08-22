"""
Unit tests for database connection layer.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.database.connection import Neo4jConnection, Neo4jError, create_connection


class TestNeo4jConnection:
    """Test cases for Neo4jConnection class."""
    
    def test_init_missing_credentials(self):
        """Test initialization with missing credentials."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Neo4jError, match="Missing required environment variables"):
                Neo4jConnection(validate_connection=False)
    
    def test_init_with_credentials(self):
        """Test successful initialization with credentials."""
        env_vars = {
            "NEO4J_URI": "neo4j://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password"
        }
        
        with patch.dict(os.environ, env_vars):
            with patch("src.database.connection.GraphDatabase.driver") as mock_driver:
                conn = Neo4jConnection(validate_connection=False)
                assert conn.uri == env_vars["NEO4J_URI"]
                assert conn.user == env_vars["NEO4J_USER"]
                assert conn.password == env_vars["NEO4J_PASSWORD"]
                mock_driver.assert_called_once()
    
    def test_alternative_env_vars(self):
        """Test initialization with alternative environment variable names."""
        env_vars = {
            "NEO4J_URL": "neo4j://localhost:7687",
            "NEO4J_USERNAME": "neo4j", 
            "NEO4J_PWD": "password"
        }
        
        with patch.dict(os.environ, env_vars):
            with patch("src.database.connection.GraphDatabase.driver"):
                conn = Neo4jConnection(validate_connection=False)
                assert conn.uri == env_vars["NEO4J_URL"]
                assert conn.user == env_vars["NEO4J_USERNAME"]
                assert conn.password == env_vars["NEO4J_PWD"]
    
    def test_query_success(self):
        """Test successful query execution."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        # Mock query result
        mock_record = MagicMock()
        mock_record.data.return_value = {"count": 5}
        mock_result.__iter__.return_value = [mock_record]
        
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch.dict(os.environ, {"NEO4J_URI": "neo4j://test", "NEO4J_USER": "test", "NEO4J_PASSWORD": "test"}):
            with patch("src.database.connection.GraphDatabase.driver", return_value=mock_driver):
                conn = Neo4jConnection(validate_connection=False)
                result = conn.query("MATCH (n) RETURN count(n) as count")
                
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 1
                assert result.iloc[0]["count"] == 5
    
    def test_query_failure(self):
        """Test query execution failure."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.run.side_effect = Exception("Database error")
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch.dict(os.environ, {"NEO4J_URI": "neo4j://test", "NEO4J_USER": "test", "NEO4J_PASSWORD": "test"}):
            with patch("src.database.connection.GraphDatabase.driver", return_value=mock_driver):
                conn = Neo4jConnection(validate_connection=False)
                
                with pytest.raises(Neo4jError, match="Query execution failed"):
                    conn.query("INVALID QUERY")
    
    def test_get_network_statistics(self):
        """Test network statistics retrieval."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        
        # Mock different query results
        def mock_run(query, params=None):
            mock_result = MagicMock()
            if "Paper" in query:
                mock_record = MagicMock()
                mock_record.data.return_value = {"count": 100}
                mock_result.__iter__.return_value = [mock_record]
            elif "Author" in query:
                mock_record = MagicMock()
                mock_record.data.return_value = {"count": 50}
                mock_result.__iter__.return_value = [mock_record]
            else:
                mock_record = MagicMock()
                mock_record.data.return_value = {"count": 0}
                mock_result.__iter__.return_value = [mock_record]
            return mock_result
        
        mock_session.run.side_effect = mock_run
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch.dict(os.environ, {"NEO4J_URI": "neo4j://test", "NEO4J_USER": "test", "NEO4J_PASSWORD": "test"}):
            with patch("src.database.connection.GraphDatabase.driver", return_value=mock_driver):
                conn = Neo4jConnection(validate_connection=False)
                stats = conn.get_network_statistics()
                
                assert "papers" in stats
                assert "authors" in stats
                assert stats["papers"] == 100
                assert stats["authors"] == 50


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_connection(self):
        """Test connection creation function."""
        with patch.dict(os.environ, {"NEO4J_URI": "neo4j://test", "NEO4J_USER": "test", "NEO4J_PASSWORD": "test"}):
            with patch("src.database.connection.GraphDatabase.driver"):
                conn = create_connection(validate=False)
                assert isinstance(conn, Neo4jConnection)
    
    def test_find_papers_by_keyword(self):
        """Test paper search by keyword."""
        mock_conn = MagicMock()
        mock_df = pd.DataFrame([
            {"paperId": "123", "title": "Test Paper", "year": 2020, "citationCount": 10}
        ])
        mock_conn.query.return_value = mock_df
        
        from src.database.connection import find_papers_by_keyword
        result = find_papers_by_keyword(mock_conn, "test")
        
        assert isinstance(result, pd.DataFrame)
        mock_conn.query.assert_called_once()
        
        # Check query was called with correct parameters
        call_args = mock_conn.query.call_args
        assert "keyword" in call_args[1]
        assert call_args[1]["keyword"] == "test"