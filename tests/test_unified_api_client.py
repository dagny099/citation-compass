"""
Test suite for the unified API client.

Tests the integration of capabilities from all three reference codebases
and ensures the unified client works correctly.
"""

import pytest
import requests
import time
from unittest.mock import Mock, patch, MagicMock
import json

from src.data.unified_api_client import (
    UnifiedSemanticScholarClient, 
    RateLimiter, 
    APICache, 
    APIMetrics,
    get_unified_client
)
from src.data.api_config import SemanticScholarConfig


class TestRateLimiter:
    """Test the rate limiting functionality."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initializes with correct defaults."""
        limiter = RateLimiter()
        assert limiter.base_pause == 1.0
        assert limiter.max_pause == 60.0
        assert limiter.current_pause == 1.0
        assert limiter.consecutive_failures == 0
    
    def test_rate_limit_handling(self):
        """Test rate limit response handling."""
        limiter = RateLimiter(base_pause=0.1)  # Faster for testing
        
        # Mock 429 response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {'Retry-After': '2'}
        
        with patch('time.sleep') as mock_sleep:
            should_retry = limiter.handle_rate_limit_response(mock_response)
            
        assert should_retry is True
        assert limiter.consecutive_failures == 1
        mock_sleep.assert_called_once_with(2)
    
    def test_successful_request_resets_failures(self):
        """Test that successful requests reset failure count."""
        limiter = RateLimiter()
        limiter.consecutive_failures = 3
        limiter.current_pause = 8.0
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        
        should_retry = limiter.handle_rate_limit_response(mock_response)
        
        assert should_retry is False
        assert limiter.consecutive_failures == 0
        assert limiter.current_pause == 1.0


class TestAPICache:
    """Test the API caching functionality."""
    
    def test_cache_initialization(self):
        """Test cache initializes correctly."""
        cache = APICache(max_size=10, default_ttl=300)
        assert cache.max_size == 10
        assert cache.default_ttl == 300
        assert len(cache.cache) == 0
    
    def test_cache_set_and_get(self):
        """Test basic cache operations."""
        cache = APICache(default_ttl=300)
        
        # Set a value
        cache.set("http://test.com", {"param": "value"}, {"result": "data"})
        
        # Get the value
        result = cache.get("http://test.com", {"param": "value"})
        assert result == {"result": "data"}
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = APICache()
        result = cache.get("http://nonexistent.com", {})
        assert result is None
    
    def test_cache_expiration(self):
        """Test that expired items are removed."""
        cache = APICache(default_ttl=0)  # Immediate expiration
        
        cache.set("http://test.com", {}, {"data": "test"})
        time.sleep(0.1)  # Wait for expiration
        
        result = cache.get("http://test.com", {})
        assert result is None
    
    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = APICache(max_size=2)
        
        # Fill cache
        cache.set("url1", {}, "data1")
        cache.set("url2", {}, "data2")
        
        # Add third item (should evict first)
        cache.set("url3", {}, "data3")
        
        assert cache.get("url1", {}) is None  # Evicted
        assert cache.get("url2", {}) == "data2"  # Still there
        assert cache.get("url3", {}) == "data3"  # Newly added


class TestAPIMetrics:
    """Test the API metrics tracking."""
    
    def test_metrics_initialization(self):
        """Test metrics initialize to zero."""
        metrics = APIMetrics()
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.success_rate == 0.0
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = APIMetrics()
        metrics.total_requests = 10
        metrics.successful_requests = 8
        
        assert metrics.success_rate == 80.0
    
    def test_average_response_time(self):
        """Test average response time calculation."""
        metrics = APIMetrics()
        metrics.successful_requests = 4
        metrics.total_response_time = 8.0
        
        assert metrics.average_response_time == 2.0
    
    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        metrics = APIMetrics()
        metrics.cache_hits = 7
        metrics.cache_misses = 3
        
        assert metrics.cache_hit_rate == 70.0


class TestUnifiedSemanticScholarClient:
    """Test the main unified API client."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = SemanticScholarConfig()
        config.rate_limit_pause = 0.1  # Faster for testing
        config.request_timeout = 5
        return config
    
    @pytest.fixture
    def client(self, mock_config):
        """Create a client instance for testing."""
        return UnifiedSemanticScholarClient(config=mock_config)
    
    def test_client_initialization(self, client):
        """Test client initializes correctly."""
        assert client.config is not None
        assert client.rate_limiter is not None
        assert client.cache is not None
        assert client.metrics is not None
        assert client.session is not None
    
    @patch('requests.Session.get')
    def test_successful_get_request(self, mock_get, client):
        """Test successful GET request handling."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        response = client._make_request("http://test.com", {"param": "value"})
        
        assert response.status_code == 200
        assert client.metrics.successful_requests == 1
        assert client.metrics.total_requests == 1
    
    @patch('requests.Session.get')
    def test_request_with_caching(self, mock_get, client):
        """Test that successful requests are cached."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # First request
        response1 = client._make_request("http://test.com", {"param": "value"})
        
        # Second request should use cache
        response2 = client._make_request("http://test.com", {"param": "value"})
        
        # Should only call the actual HTTP method once
        assert mock_get.call_count == 1
        assert client.metrics.cache_hits == 1
        assert client.metrics.cache_misses == 1
    
    @patch('requests.Session.get')
    def test_rate_limit_retry(self, mock_get, client):
        """Test automatic retry on rate limit."""
        # First call returns 429, second succeeds
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {'Retry-After': '1'}
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {"data": "test"}
        success_response.raise_for_status.return_value = None
        
        mock_get.side_effect = [rate_limit_response, success_response]
        
        with patch('time.sleep'):  # Speed up test
            response = client._make_request("http://test.com")
        
        assert response.status_code == 200
        assert mock_get.call_count == 2
        assert client.metrics.rate_limited_requests == 1
    
    @patch('src.data.unified_api_client.UnifiedSemanticScholarClient._make_request')
    def test_search_papers(self, mock_request, client):
        """Test paper search functionality."""
        mock_request.return_value.json.return_value = {
            "data": [{"paperId": "123", "title": "Test Paper"}]
        }
        
        result = client.search_papers("machine learning", bulk=True)
        
        mock_request.assert_called_once()
        assert "data" in result
    
    @patch('src.data.unified_api_client.UnifiedSemanticScholarClient._make_request')
    def test_get_paper_details(self, mock_request, client):
        """Test getting paper details."""
        mock_request.return_value.json.return_value = {
            "paperId": "123",
            "title": "Test Paper",
            "citationCount": 42
        }
        
        result = client.get_paper_details("123")
        
        assert result["paperId"] == "123"
        assert result["title"] == "Test Paper"
    
    @patch('src.data.unified_api_client.UnifiedSemanticScholarClient.paginate_api_requests')
    def test_get_paper_citations(self, mock_paginate, client):
        """Test getting paper citations."""
        mock_paginate.return_value = [
            {"citingPaper": {"paperId": "456", "title": "Citing Paper 1"}},
            {"citingPaper": {"paperId": "789", "title": "Citing Paper 2"}}
        ]
        
        citations = client.get_paper_citations("123")
        
        assert len(citations) == 2
        assert citations[0]["paperId"] == "456"
        assert citations[1]["paperId"] == "789"
    
    @patch('src.data.unified_api_client.UnifiedSemanticScholarClient._make_request')
    def test_batch_paper_details(self, mock_request, client):
        """Test batch paper details retrieval."""
        mock_request.return_value.json.return_value = [
            {"paperId": "123", "title": "Paper 1"},
            {"paperId": "456", "title": "Paper 2"},
            None  # Should be filtered out
        ]
        
        papers = client.batch_paper_details(["123", "456", "789"])
        
        assert len(papers) == 2  # None filtered out
        assert papers[0]["paperId"] == "123"
        assert papers[1]["paperId"] == "456"
    
    def test_get_ml_training_data_structure(self, client):
        """Test ML training data format."""
        with patch.object(client, 'get_paper_citations') as mock_citations:
            mock_citations.return_value = [
                {"paperId": "citing1"},
                {"paperId": "citing2"}
            ]
            
            df = client.get_ml_training_data(["paper1", "paper2"])
            
            # Check DataFrame structure
            expected_columns = ['source_id', 'target_id', 'relationship_type']
            assert all(col in df.columns for col in expected_columns)
            
            # Check relationship type
            if len(df) > 0:
                assert all(df['relationship_type'] == 'CITES')
    
    def test_get_visualization_data_structure(self, client):
        """Test visualization data format."""
        with patch.object(client, 'batch_paper_details') as mock_batch, \
             patch.object(client, 'get_paper_citations') as mock_citations:
            
            mock_batch.return_value = [
                {
                    "paperId": "123",
                    "title": "Test Paper",
                    "citationCount": 10,
                    "year": 2023,
                    "authors": [{"name": "Author 1"}, {"name": "Author 2"}]
                }
            ]
            
            mock_citations.return_value = [
                {"paperId": "456", "title": "Citing Paper"}
            ]
            
            result = client.get_visualization_data(["123", "456"])  # Include citing paper in set
            
            # Check structure
            assert "nodes" in result
            assert "edges" in result
            
            # Check nodes DataFrame
            nodes_df = result["nodes"]
            expected_node_columns = ['id', 'label', 'type', 'citation_count', 'year', 'authors']
            assert all(col in nodes_df.columns for col in expected_node_columns)
            
            # Check edges DataFrame  
            edges_df = result["edges"]
            if len(edges_df) > 0:  # Only check if edges exist
                expected_edge_columns = ['source', 'target', 'relationship', 'source_label', 'target_label']
                assert all(col in edges_df.columns for col in expected_edge_columns)


def test_get_unified_client():
    """Test the convenience function for getting a client instance."""
    client = get_unified_client()
    assert isinstance(client, UnifiedSemanticScholarClient)


class TestBackwardCompatibility:
    """Test backward compatibility functions."""
    
    @patch('src.data.unified_api_client.get_unified_client')
    def test_retrieve_citations_by_id(self, mock_get_client):
        """Test legacy citation retrieval function."""
        from src.data.unified_api_client import retrieve_citations_by_id
        
        mock_client = Mock()
        mock_client.get_paper_citations.return_value = [
            {"paperId": "123"},
            {"paperId": "456"}
        ]
        mock_get_client.return_value = mock_client
        
        result = retrieve_citations_by_id("paper1")
        
        assert result == ["123", "456"]
        mock_client.get_paper_citations.assert_called_once_with("paper1", fields="paperId")
    
    @patch('src.data.unified_api_client.get_unified_client')
    def test_batch_paper_details_legacy(self, mock_get_client):
        """Test legacy batch paper details function."""
        from src.data.unified_api_client import batch_paper_details
        
        mock_client = Mock()
        mock_client.batch_paper_details.return_value = [{"paperId": "123"}]
        mock_get_client.return_value = mock_client
        
        result = batch_paper_details(["123", "456"])
        
        assert result == [{"paperId": "123"}]
        mock_client.batch_paper_details.assert_called_once_with(["123", "456"])


if __name__ == "__main__":
    pytest.main([__file__])