"""
Unified API client that consolidates capabilities from all three reference codebases.

This module combines:
- Robust pagination and rate limiting from academic-citation-prediction
- Interactive exploration needs from knowledge-cartography  
- ML training data requirements from citation-map-dashboard

The UnifiedSemanticScholarClient provides a single interface for all Semantic Scholar
API interactions across the integrated platform.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Generator, Union, Tuple, Any
from urllib.parse import urlencode
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .api_config import get_config, SemanticScholarConfig


@dataclass
class APIMetrics:
    """Track API usage metrics for monitoring and optimization."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    total_response_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time in seconds."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests == 0:
            return 0.0
        return (self.cache_hits / total_cache_requests) * 100


class RateLimiter:
    """
    Intelligent rate limiter that adapts to API responses.
    
    Features adaptive rate limiting based on response headers and 429 status codes,
    with exponential backoff for robust error recovery.
    """
    
    def __init__(self, base_pause: float = 1.0, max_pause: float = 60.0):
        """
        Initialize rate limiter.
        
        Args:
            base_pause: Base pause time between requests in seconds
            max_pause: Maximum pause time for exponential backoff
        """
        self.base_pause = base_pause
        self.max_pause = max_pause
        self.current_pause = base_pause
        self.last_request_time = 0.0
        self.consecutive_failures = 0
        
        self.logger = logging.getLogger(__name__)
    
    def wait_if_needed(self) -> None:
        """Wait appropriate time since last request to respect rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.current_pause:
            sleep_time = self.current_pause - elapsed
            self.logger.debug(f"Rate limiting: sleeping {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def handle_rate_limit_response(self, response: requests.Response) -> bool:
        """
        Handle rate limit response and adjust timing.
        
        Args:
            response: HTTP response object
            
        Returns:
            True if rate limited (should retry), False otherwise
        """
        if response.status_code == 429:
            self.consecutive_failures += 1
            
            # Check for Retry-After header
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                try:
                    wait_time = int(retry_after)
                except ValueError:
                    wait_time = self.current_pause * (2 ** self.consecutive_failures)
            else:
                wait_time = self.current_pause * (2 ** self.consecutive_failures)
            
            wait_time = min(wait_time, self.max_pause)
            
            self.logger.warning(f"Rate limited. Waiting {wait_time} seconds before retry.")
            time.sleep(wait_time)
            
            self.current_pause = wait_time
            return True
        
        # Reset on successful request
        if response.status_code == 200:
            self.consecutive_failures = 0
            self.current_pause = self.base_pause
        
        return False


class APICache:
    """
    Simple in-memory cache for API responses to reduce redundant requests.
    
    Implements LRU eviction and TTL expiration for efficient memory usage.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Initialize API cache.
        
        Args:
            max_size: Maximum number of cached responses
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.access_order: List[str] = []
        
        self.logger = logging.getLogger(__name__)
    
    def _generate_key(self, url: str, params: Dict = None) -> str:
        """Generate cache key from URL and parameters."""
        key_data = f"{url}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, url: str, params: Dict = None) -> Optional[Any]:
        """
        Retrieve cached response if available and not expired.
        
        Args:
            url: Request URL
            params: Request parameters
            
        Returns:
            Cached response data or None if not found/expired
        """
        key = self._generate_key(url, params)
        
        if key in self.cache:
            data, timestamp = self.cache[key]
            
            # Check if expired
            if datetime.now() - timestamp > timedelta(seconds=self.default_ttl):
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                return None
            
            # Update access order for LRU
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.logger.debug(f"Cache hit for key: {key[:16]}...")
            return data
        
        return None
    
    def set(self, url: str, params: Dict, data: Any) -> None:
        """
        Store response in cache.
        
        Args:
            url: Request URL
            params: Request parameters  
            data: Response data to cache
        """
        key = self._generate_key(url, params)
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            if self.access_order:
                oldest_key = self.access_order.pop(0)
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
        
        self.cache[key] = (data, datetime.now())
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        self.logger.debug(f"Cached response for key: {key[:16]}...")


class UnifiedSemanticScholarClient:
    """
    Unified API client combining capabilities from all three reference codebases.
    
    This class provides:
    - Robust pagination handling (from academic-citation-prediction)
    - Interactive exploration support (from knowledge-cartography)
    - ML training data preparation (from citation-map-dashboard)
    - Advanced caching and rate limiting
    - Comprehensive error handling and recovery
    """
    
    def __init__(self, config: Optional[SemanticScholarConfig] = None):
        """
        Initialize the unified API client.
        
        Args:
            config: Optional configuration override. Uses global config if None.
        """
        self.config = config or get_config().get_api_config()
        
        # Initialize components
        self.rate_limiter = RateLimiter(
            base_pause=self.config.rate_limit_pause,
            max_pause=60.0
        )
        self.cache = APICache(
            max_size=get_config().get_cache_config().max_cache_size,
            default_ttl=get_config().get_cache_config().api_response_ttl
        )
        self.metrics = APIMetrics()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Configure HTTP session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        if self.config.api_key:
            self.session.headers.update({"x-api-key": self.config.api_key})
        
        self.logger.info("Unified Semantic Scholar API client initialized")
    
    def _make_request(self, url: str, params: Dict = None, method: str = 'GET', 
                     json_data: Dict = None, use_cache: bool = True) -> requests.Response:
        """
        Make HTTP request with comprehensive error handling, caching, and rate limiting.
        
        Args:
            url: Full URL for the request
            params: Query parameters
            method: HTTP method ('GET' or 'POST')
            json_data: JSON data for POST requests
            use_cache: Whether to use caching for this request
            
        Returns:
            Response object
            
        Raises:
            requests.exceptions.RequestException: For unrecoverable request failures
        """
        self.metrics.total_requests += 1
        
        # Check cache first for GET requests
        if method == 'GET' and use_cache:
            cached_response = self.cache.get(url, params)
            if cached_response is not None:
                self.metrics.cache_hits += 1
                # Create mock response object
                mock_response = requests.Response()
                mock_response._content = json.dumps(cached_response).encode()
                mock_response.status_code = 200
                return mock_response
            else:
                self.metrics.cache_misses += 1
        
        # Rate limiting
        self.rate_limiter.wait_if_needed()
        
        start_time = time.time()
        
        try:
            # Make the actual request
            if method == 'GET':
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=self.config.request_timeout
                )
            elif method == 'POST':
                response = self.session.post(
                    url, 
                    params=params, 
                    json=json_data,
                    timeout=self.config.request_timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Record response time
            response_time = time.time() - start_time
            self.metrics.total_response_time += response_time
            
            # Handle rate limiting
            if self.rate_limiter.handle_rate_limit_response(response):
                self.metrics.rate_limited_requests += 1
                # Retry the request
                return self._make_request(url, params, method, json_data, use_cache)
            
            # Check for other errors
            response.raise_for_status()
            
            # Cache successful GET responses
            if method == 'GET' and use_cache and response.status_code == 200:
                try:
                    response_data = response.json()
                    self.cache.set(url, params, response_data)
                except json.JSONDecodeError:
                    self.logger.warning("Failed to cache response: invalid JSON")
            
            self.metrics.successful_requests += 1
            return response
            
        except requests.exceptions.RequestException as e:
            self.metrics.failed_requests += 1
            self.logger.error(f"Request failed: {e}")
            raise
    
    def search_papers(self, query: str, bulk: bool = True, fields: str = None, 
                     limit: int = None) -> Dict:
        """
        Search for papers by query string with enhanced options.
        
        Args:
            query: Search query (paper title, keywords, etc.)
            bulk: Use bulk search endpoint for more results
            fields: Comma-separated fields to retrieve
            limit: Maximum number of results to return
            
        Returns:
            Dictionary containing search results
        """
        endpoint = (self.config.endpoints['paper_search_bulk'] if bulk 
                   else self.config.endpoints['paper_search'])
        url = f"{self.config.base_url}{endpoint}"
        
        params = {
            'query': query,
            'fields': fields or self.config.paper_fields
        }
        
        if limit:
            params['limit'] = min(limit, self.config.max_pagination_limit)
        
        self.logger.info(f"Searching papers: '{query}' (bulk={bulk})")
        response = self._make_request(url, params)
        return response.json()
    
    def get_paper_details(self, paper_id: str, fields: str = None) -> Optional[Dict]:
        """
        Get detailed information for a single paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            fields: Comma-separated fields to retrieve
            
        Returns:
            Paper details dictionary or None if not found
        """
        endpoint = self.config.endpoints['paper_details'].format(paper_id=paper_id)
        url = f"{self.config.base_url}{endpoint}"
        
        params = {'fields': fields or self.config.paper_fields}
        
        try:
            response = self._make_request(url, params)
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.logger.warning(f"Paper not found: {paper_id}")
                return None
            raise
    
    def get_paper_citations(self, paper_id: str, fields: str = None, 
                           limit: int = None) -> List[Dict]:
        """
        Get all papers citing a given paper with pagination support.
        
        Args:
            paper_id: Semantic Scholar paper ID
            fields: Fields to retrieve for citing papers
            limit: Maximum number of citations to retrieve
            
        Returns:
            List of citing paper dictionaries
        """
        endpoint_url = f"{self.config.base_url}{self.config.endpoints['paper_citations'].format(paper_id=paper_id)}"
        params = {"fields": fields or self.config.citation_fields}
        
        self.logger.info(f"Retrieving citations for paper: {paper_id}")
        
        try:
            citations = []
            retrieved = 0
            max_retrieve = limit or self.config.max_pagination_limit
            
            for citation_data in self.paginate_api_requests(endpoint_url, params):
                if 'citingPaper' in citation_data and citation_data['citingPaper']:
                    citations.append(citation_data['citingPaper'])
                    retrieved += 1
                    
                    if retrieved >= max_retrieve:
                        break
            
            self.logger.info(f"Retrieved {len(citations)} citations for {paper_id}")
            return citations
            
        except Exception as e:
            self.logger.error(f"Error retrieving citations for {paper_id}: {e}")
            return []
    
    def get_paper_references(self, paper_id: str, fields: str = None,
                           limit: int = None) -> List[Dict]:
        """
        Get all papers referenced by a given paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            fields: Fields to retrieve for referenced papers
            limit: Maximum number of references to retrieve
            
        Returns:
            List of referenced paper dictionaries
        """
        endpoint_url = f"{self.config.base_url}{self.config.endpoints['paper_references'].format(paper_id=paper_id)}"
        params = {"fields": fields or self.config.citation_fields}
        
        try:
            references = []
            retrieved = 0
            max_retrieve = limit or self.config.max_pagination_limit
            
            for reference_data in self.paginate_api_requests(endpoint_url, params):
                if 'citedPaper' in reference_data and reference_data['citedPaper']:
                    references.append(reference_data['citedPaper'])
                    retrieved += 1
                    
                    if retrieved >= max_retrieve:
                        break
            
            return references
            
        except Exception as e:
            self.logger.error(f"Error retrieving references for {paper_id}: {e}")
            return []
    
    def batch_paper_details(self, paper_ids: List[str], fields: str = None) -> List[Dict]:
        """
        Get detailed information for multiple papers efficiently.
        
        Args:
            paper_ids: List of paper IDs (automatically batched if > max_batch_size)
            fields: Comma-separated fields to retrieve
            
        Returns:
            List of paper detail dictionaries (None entries filtered out)
        """
        if not paper_ids:
            return []
        
        all_papers = []
        batch_size = self.config.max_batch_size
        
        # Process in batches
        for i in range(0, len(paper_ids), batch_size):
            batch_ids = paper_ids[i:i + batch_size]
            
            url = f"{self.config.base_url}{self.config.endpoints['paper_batch']}"
            params = {'fields': fields or self.config.paper_fields}
            json_data = {"ids": batch_ids}
            
            self.logger.info(f"Batch request for {len(batch_ids)} papers")
            
            try:
                response = self._make_request(url, params, method='POST', 
                                            json_data=json_data, use_cache=False)
                batch_papers = response.json()
                
                # Filter out None responses
                valid_papers = [paper for paper in batch_papers if paper is not None]
                all_papers.extend(valid_papers)
                
            except Exception as e:
                self.logger.error(f"Batch request failed for papers {i}-{i+len(batch_ids)}: {e}")
                continue
        
        self.logger.info(f"Retrieved {len(all_papers)} valid papers from {len(paper_ids)} requested")
        return all_papers
    
    def paginate_api_requests(self, endpoint_url: str, params: Dict = None,
                             offset_key: str = 'offset', limit_key: str = 'limit',
                             next_key: str = 'next', data_key: str = 'data',
                             page_size: int = None) -> Generator[Dict, None, None]:
        """
        Generic pagination handler for any Semantic Scholar API endpoint.
        
        This is the core pagination engine used by all other methods. It handles
        the complexity of paginated responses and provides a simple iterator interface.
        
        Args:
            endpoint_url: Full URL for the API endpoint
            params: Dictionary of query parameters
            offset_key: Name of the offset parameter in requests
            limit_key: Name of the limit parameter in requests  
            next_key: Key in response containing next offset
            data_key: Key in response containing data array
            page_size: Items per page (uses config default if None)
            
        Yields:
            Individual items from paginated responses
        """
        params = params or {}
        offset = 0
        total_retrieved = 0
        page_size = page_size or self.config.default_page_size
        
        while True:
            # Prepare request parameters
            current_params = {
                **params,
                offset_key: offset,
                limit_key: page_size
            }
            
            # Make request
            self.logger.debug(f"Paginating: offset={offset}, limit={page_size}")
            
            try:
                response = self._make_request(endpoint_url, current_params)
                
                if response.status_code == 404:
                    self.logger.debug("Pagination complete: no more results")
                    break
                    
                data = response.json()
                
                # Extract items from response
                items = data.get(data_key, [])
                if not items:
                    self.logger.debug("Pagination complete: empty response")
                    break
                
                # Yield each item
                for item in items:
                    total_retrieved += 1
                    yield item
                
                # Check for continuation
                if (next_key not in data or 
                    total_retrieved >= self.config.max_pagination_limit):
                    self.logger.debug(f"Pagination complete: {total_retrieved} total items")
                    break
                
                # Update offset for next page
                offset = data[next_key]
                
            except Exception as e:
                self.logger.error(f"Pagination error at offset {offset}: {e}")
                break
    
    def expand_citation_network(self, seed_papers: List[Dict], max_depth: int = 2,
                               citation_threshold: int = 0, 
                               max_papers_per_depth: int = 1000) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Recursively expand citation network from seed papers.
        
        This implements the unique network expansion capability from academic-citation-prediction,
        enhanced with better progress tracking and limits for web interface use.
        
        Args:
            seed_papers: Initial papers to expand from
            max_depth: Maximum expansion depth
            citation_threshold: Minimum citations required for expansion
            max_papers_per_depth: Limit papers processed per depth level
            
        Returns:
            Tuple of (expanded_papers_list, expansion_stats)
        """
        expanded_papers = list(seed_papers)
        processed_ids = {paper['paperId'] for paper in seed_papers}
        
        stats = {
            'initial_papers': len(seed_papers),
            'depth_stats': [],
            'total_api_calls': 0,
            'total_papers_found': len(seed_papers)
        }
        
        self.logger.info(f"Starting citation network expansion: {len(seed_papers)} seed papers")
        
        for depth in range(max_depth):
            depth_start_time = time.time()
            self.logger.info(f"Expanding citation network - depth {depth + 1}/{max_depth}")
            
            new_paper_ids = set()
            papers_processed_this_depth = 0
            
            # Get papers to process at this depth (limit for performance)
            current_papers = expanded_papers[-len(expanded_papers):]
            if len(current_papers) > max_papers_per_depth:
                # Sort by citation count and take top papers
                current_papers = sorted(
                    current_papers, 
                    key=lambda p: p.get('citationCount', 0), 
                    reverse=True
                )[:max_papers_per_depth]
            
            for paper in current_papers:
                if papers_processed_this_depth >= max_papers_per_depth:
                    break
                    
                if paper.get('citationCount', 0) > citation_threshold:
                    citing_papers = self.get_paper_citations(
                        paper['paperId'], 
                        fields="paperId,title,citationCount",
                        limit=100  # Limit citations per paper for performance
                    )
                    stats['total_api_calls'] += 1
                    
                    new_ids = {p['paperId'] for p in citing_papers 
                              if p['paperId'] not in processed_ids}
                    new_paper_ids.update(new_ids)
                    
                    papers_processed_this_depth += 1
            
            if not new_paper_ids:
                self.logger.info(f"No new papers found at depth {depth + 1}")
                break
            
            # Get details for new papers in batches
            new_paper_ids_list = list(new_paper_ids)
            new_papers = self.batch_paper_details(new_paper_ids_list)
            stats['total_api_calls'] += len(new_paper_ids_list) // self.config.max_batch_size + 1
            
            # Add to expanded set
            expanded_papers.extend(new_papers)
            processed_ids.update(new_paper_ids)
            
            # Record depth statistics
            depth_time = time.time() - depth_start_time
            depth_stat = {
                'depth': depth + 1,
                'papers_processed': papers_processed_this_depth,
                'new_papers_found': len(new_papers),
                'time_seconds': depth_time,
                'cumulative_papers': len(expanded_papers)
            }
            stats['depth_stats'].append(depth_stat)
            
            self.logger.info(f"Depth {depth + 1} complete: {len(new_papers)} new papers "
                           f"({depth_time:.1f}s)")
        
        stats['total_papers_found'] = len(expanded_papers)
        stats['total_expansion_time'] = sum(d['time_seconds'] for d in stats['depth_stats'])
        
        self.logger.info(f"Citation network expansion complete: "
                        f"{stats['total_papers_found']} total papers, "
                        f"{stats['total_api_calls']} API calls")
        
        return expanded_papers, stats
    
    def get_ml_training_data(self, paper_ids: List[str]) -> pd.DataFrame:
        """
        Prepare data specifically formatted for ML model training.
        
        This method creates the data format expected by the TransE model from
        citation-map-dashboard, with proper edge lists and entity mappings.
        
        Args:
            paper_ids: List of paper IDs to include in training data
            
        Returns:
            DataFrame with columns: source_id, target_id, relationship_type
        """
        self.logger.info(f"Preparing ML training data for {len(paper_ids)} papers")
        
        edges = []
        processed = 0
        
        for paper_id in paper_ids:
            try:
                # Get citations (incoming edges)
                citations = self.get_paper_citations(
                    paper_id, 
                    fields="paperId",
                    limit=50  # Limit for training data
                )
                
                for citing_paper in citations:
                    edges.append({
                        'source_id': citing_paper['paperId'],
                        'target_id': paper_id,
                        'relationship_type': 'CITES'
                    })
                
                processed += 1
                if processed % 100 == 0:
                    self.logger.info(f"Processed {processed}/{len(paper_ids)} papers for ML data")
                    
            except Exception as e:
                self.logger.warning(f"Failed to get citations for {paper_id}: {e}")
                continue
        
        df = pd.DataFrame(edges)
        self.logger.info(f"Created ML training dataset: {len(df)} citation edges")
        
        return df
    
    def get_visualization_data(self, paper_ids: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Prepare data optimized for visualization in knowledge-cartography style.
        
        Args:
            paper_ids: List of paper IDs to visualize
            
        Returns:
            Dictionary containing 'nodes' and 'edges' DataFrames
        """
        self.logger.info(f"Preparing visualization data for {len(paper_ids)} papers")
        
        # Get paper details
        papers = self.batch_paper_details(paper_ids)
        
        # Create nodes DataFrame
        nodes_data = []
        for paper in papers:
            nodes_data.append({
                'id': paper['paperId'],
                'label': paper.get('title', 'Unknown Title')[:50] + '...',
                'type': 'Paper',
                'citation_count': paper.get('citationCount', 0),
                'year': paper.get('year'),
                'authors': ', '.join([a.get('name', '') for a in paper.get('authors', [])][:3])
            })
        
        # Create edges DataFrame
        edges_data = []
        for paper in papers:
            # Get a subset of citations for visualization
            citations = self.get_paper_citations(
                paper['paperId'], 
                fields="paperId,title",
                limit=10  # Limit for visualization performance
            )
            
            for citing_paper in citations:
                if citing_paper['paperId'] in paper_ids:  # Only include edges within our set
                    edges_data.append({
                        'source': citing_paper['paperId'],
                        'target': paper['paperId'],
                        'relationship': 'CITES',
                        'source_label': citing_paper.get('title', 'Unknown')[:30] + '...',
                        'target_label': paper.get('title', 'Unknown')[:30] + '...'
                    })
        
        result = {
            'nodes': pd.DataFrame(nodes_data),
            'edges': pd.DataFrame(edges_data)
        }
        
        self.logger.info(f"Created visualization data: {len(result['nodes'])} nodes, "
                        f"{len(result['edges'])} edges")
        
        return result
    
    def get_metrics(self) -> APIMetrics:
        """Get current API usage metrics."""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset API usage metrics."""
        self.metrics = APIMetrics()
        self.logger.info("API metrics reset")


# Convenience functions for backward compatibility with reference codebases
def get_unified_client() -> UnifiedSemanticScholarClient:
    """Get a configured instance of the unified API client."""
    return UnifiedSemanticScholarClient()


# Legacy function wrappers for backward compatibility
def paginate_api_requests(base_url: str, params: Dict = None, **kwargs) -> Generator[Dict, None, None]:
    """Backward compatibility wrapper for pagination."""
    client = get_unified_client()
    return client.paginate_api_requests(base_url, params, **kwargs)


def retrieve_citations_by_id(paper_id: str) -> List[str]:
    """Backward compatibility wrapper for citations."""
    client = get_unified_client()
    citations = client.get_paper_citations(paper_id, fields="paperId")
    return [c['paperId'] for c in citations]


def batch_paper_details(paper_ids: List[str]) -> List[Dict]:
    """Backward compatibility wrapper for batch details."""
    client = get_unified_client()
    return client.batch_paper_details(paper_ids)