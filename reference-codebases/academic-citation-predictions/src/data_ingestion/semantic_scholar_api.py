"""
Semantic Scholar API client for academic citation data ingestion.
Consolidates all API interaction functions with proper error handling and rate limiting.
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Generator, Union
from urllib.parse import urlencode
import json

from config.settings import (
    SEMANTIC_SCHOLAR_BASE_URL, DEFAULT_RATE_LIMIT_PAUSE, MAX_BATCH_SIZE,
    MAX_PAGINATION_LIMIT, DEFAULT_REQUEST_TIMEOUT, ENDPOINTS,
    PAPER_FIELDS, AUTHOR_FIELDS, CITATION_FIELDS
)


class SemanticScholarAPI:
    """
    Client for interacting with the Semantic Scholar API.
    Handles pagination, rate limiting, and error recovery.
    """
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_pause: float = DEFAULT_RATE_LIMIT_PAUSE):
        """
        Initialize the API client.
        
        Args:
            api_key: Optional API key for higher rate limits
            rate_limit_pause: Pause between requests in seconds
        """
        self.base_url = SEMANTIC_SCHOLAR_BASE_URL
        self.headers = {"x-api-key": api_key} if api_key else {}
        self.rate_limit_pause = rate_limit_pause
        self.timeout = DEFAULT_REQUEST_TIMEOUT
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _make_request(self, url: str, params: Dict = None, method: str = 'GET', 
                     json_data: Dict = None) -> requests.Response:
        """
        Make a request with error handling and rate limiting.
        
        Args:
            url: Full URL for the request
            params: Query parameters
            method: HTTP method ('GET' or 'POST')
            json_data: JSON data for POST requests
            
        Returns:
            Response object
            
        Raises:
            requests.exceptions.RequestException: For request failures
        """
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=self.headers, timeout=self.timeout)
            elif method == 'POST':
                response = requests.post(url, params=params, json=json_data, 
                                       headers=self.headers, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            if response.status_code == 429:  # Too Many Requests
                self.logger.warning("Rate limit hit. Waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(url, params, method, json_data)
            
            response.raise_for_status()
            time.sleep(self.rate_limit_pause)
            return response
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise
    
    def search_papers(self, query: str, bulk: bool = True, fields: str = None) -> Dict:
        """
        Search for papers by query string.
        
        Args:
            query: Search query (paper title, keywords, etc.)
            bulk: Use bulk search endpoint for more results
            fields: Comma-separated fields to retrieve
            
        Returns:
            Dictionary containing search results
        """
        endpoint = ENDPOINTS['paper_search_bulk'] if bulk else ENDPOINTS['paper_search']
        url = f"{self.base_url}{endpoint}"
        
        params = {
            'query': query,
            'fields': fields or PAPER_FIELDS
        }
        
        response = self._make_request(url, params)
        return response.json()
    
    def search_authors(self, query: str, bulk: bool = False, fields: str = None) -> Dict:
        """
        Search for authors by query string.
        
        Args:
            query: Search query (author name, etc.)
            bulk: Use bulk search endpoint
            fields: Comma-separated fields to retrieve
            
        Returns:
            Dictionary containing search results
        """
        endpoint = ENDPOINTS['author_search_bulk'] if bulk else ENDPOINTS['author_search']
        url = f"{self.base_url}{endpoint}"
        
        params = {
            'query': query,
            'fields': fields or AUTHOR_FIELDS
        }
        
        response = self._make_request(url, params)
        return response.json()
    
    def paginate_api_requests(self, endpoint_url: str, params: Dict = None,
                             offset_key: str = 'offset', limit_key: str = 'limit',
                             next_key: str = 'next', data_key: str = 'data',
                             limit: int = 100) -> Generator[Dict, None, None]:
        """
        Generic function to handle pagination for API requests.
        
        Args:
            endpoint_url: Full URL for the API endpoint
            params: Dictionary of query parameters
            offset_key: Name of the offset parameter in the request
            limit_key: Name of the limit parameter in the request
            next_key: Key in the response that contains the next offset
            data_key: Key in the response that contains the data array
            limit: Number of items to request per page
            
        Yields:
            Individual items from the paginated response
        """
        params = params or {}
        offset = 0
        total_retrieved = 0
        
        while True:
            # Update parameters with pagination values
            current_params = {
                **params,
                offset_key: offset,
                limit_key: limit
            }
            
            # Create full URL with parameters
            url = f"{endpoint_url}?{urlencode(current_params)}"
            self.logger.info(f"Fetching results {offset} to {offset + limit} from {url}")
            
            response = self._make_request(url)
            
            if response.status_code == 404:
                self.logger.info("No more results to retrieve")
                break
                
            data = response.json()
            
            # Check if we got any results
            items = data.get(data_key, [])
            if not items:
                self.logger.info("No more results to retrieve")
                break
            
            # Process each item in the current batch
            for item in items:
                total_retrieved += 1
                yield item
            
            # Check if we've retrieved all results or hit max limit
            if next_key not in data or total_retrieved >= MAX_PAGINATION_LIMIT:
                self.logger.info(f"Retrieved all results. Total: {total_retrieved}")
                break
            
            # Update offset for next batch
            offset = data[next_key]
    
    def get_paper_citations(self, paper_id: str, fields: str = None) -> List[str]:
        """
        Get all papers citing a given paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            fields: Fields to retrieve for citing papers
            
        Returns:
            List of citing paper IDs
        """
        endpoint_url = f"{self.base_url}{ENDPOINTS['paper_citations'].format(paper_id=paper_id)}"
        params = {"fields": fields or CITATION_FIELDS}
        
        try:
            cited_paper_ids = list(self.paginate_api_requests(endpoint_url, params))
            citing_paper_ids = [d['citingPaper']['paperId'] for d in cited_paper_ids 
                              if 'citingPaper' in d and 'paperId' in d['citingPaper']]
            return citing_paper_ids
            
        except Exception as e:
            self.logger.error(f"Error retrieving citations for {paper_id}: {e}")
            return []
    
    def get_paper_references(self, paper_id: str, fields: str = None) -> List[str]:
        """
        Get all papers referenced by a given paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            fields: Fields to retrieve for referenced papers
            
        Returns:
            List of referenced paper IDs
        """
        endpoint_url = f"{self.base_url}{ENDPOINTS['paper_references'].format(paper_id=paper_id)}"
        params = {"fields": fields or CITATION_FIELDS}
        
        try:
            referenced_paper_ids = list(self.paginate_api_requests(endpoint_url, params))
            ref_paper_ids = [d['citedPaper']['paperId'] for d in referenced_paper_ids 
                           if 'citedPaper' in d and 'paperId' in d['citedPaper']]
            return ref_paper_ids
            
        except Exception as e:
            self.logger.error(f"Error retrieving references for {paper_id}: {e}")
            return []
    
    def batch_paper_details(self, paper_ids: List[str], fields: str = None) -> List[Dict]:
        """
        Get detailed information for multiple papers at once.
        
        Args:
            paper_ids: List of paper IDs (max 500)
            fields: Comma-separated fields to retrieve
            
        Returns:
            List of paper detail dictionaries
        """
        if len(paper_ids) > MAX_BATCH_SIZE:
            self.logger.warning(f"Batch size {len(paper_ids)} exceeds max {MAX_BATCH_SIZE}. "
                              f"Consider splitting into smaller batches.")
        
        url = f"{self.base_url}{ENDPOINTS['paper_batch']}"
        params = {'fields': fields or PAPER_FIELDS}
        json_data = {"ids": paper_ids}
        
        response = self._make_request(url, params, method='POST', json_data=json_data)
        details = response.json()
        
        # Filter out None responses (papers not found)
        details = [paper for paper in details if paper is not None]
        
        return details
    
    def batch_author_details(self, author_ids: List[str], fields: str = None) -> List[Dict]:
        """
        Get detailed information for multiple authors at once.
        
        Args:
            author_ids: List of author IDs (max 500)
            fields: Comma-separated fields to retrieve
            
        Returns:
            List of author detail dictionaries
        """
        if len(author_ids) > MAX_BATCH_SIZE:
            self.logger.warning(f"Batch size {len(author_ids)} exceeds max {MAX_BATCH_SIZE}. "
                              f"Consider splitting into smaller batches.")
        
        url = f"{self.base_url}{ENDPOINTS['author_batch']}"
        params = {'fields': fields or AUTHOR_FIELDS}
        json_data = {"ids": author_ids}
        
        response = self._make_request(url, params, method='POST', json_data=json_data)
        details = response.json()
        
        # Filter out None responses (authors not found)
        details = [author for author in details if author is not None]
        
        return details
    
    def expand_citation_network(self, seed_papers: List[Dict], max_depth: int = 2,
                               citation_threshold: int = 0) -> List[Dict]:
        """
        Recursively expand citation network from seed papers.
        
        This is a unique feature that automatically discovers and expands 
        citation networks based on citation relationships.
        
        Args:
            seed_papers: Initial papers to expand from
            max_depth: Maximum expansion depth
            citation_threshold: Minimum citations required for expansion
            
        Returns:
            Expanded list of papers with citation relationships
        """
        expanded_papers = list(seed_papers)
        processed_ids = {paper['paperId'] for paper in seed_papers}
        
        for depth in range(max_depth):
            self.logger.info(f"Expanding citation network - depth {depth + 1}")
            new_paper_ids = []
            
            for paper in expanded_papers:
                if paper.get('citationCount', 0) > citation_threshold:
                    citing_ids = self.get_paper_citations(paper['paperId'])
                    new_ids = [pid for pid in citing_ids if pid not in processed_ids]
                    new_paper_ids.extend(new_ids)
                    processed_ids.update(new_ids)
            
            if not new_paper_ids:
                break
                
            # Get details for new papers in batches
            batch_size = MAX_BATCH_SIZE
            for i in range(0, len(new_paper_ids), batch_size):
                batch_ids = new_paper_ids[i:i + batch_size]
                new_papers = self.batch_paper_details(batch_ids)
                expanded_papers.extend(new_papers)
                
                # Pause between batches
                time.sleep(self.rate_limit_pause * 2)
        
        return expanded_papers


# Convenience functions for backward compatibility
def paginate_api_requests(base_url: str, params: Dict = None, **kwargs) -> Generator[Dict, None, None]:
    """Backward compatibility wrapper for pagination."""
    api = SemanticScholarAPI()
    return api.paginate_api_requests(base_url, params, **kwargs)


def retrieve_citations_by_id(paper_id: str) -> List[str]:
    """Backward compatibility wrapper for citations."""
    api = SemanticScholarAPI()
    return api.get_paper_citations(paper_id)


def batch_paper_details(paper_ids: List[str]) -> List[Dict]:
    """Backward compatibility wrapper for batch details."""
    api = SemanticScholarAPI()
    return api.batch_paper_details(paper_ids)


def search_endpoint_nopaginate(endpoint: str = '/paper/search', query: str = 'the semantic web') -> Optional[Dict]:
    """Backward compatibility wrapper for search."""
    api = SemanticScholarAPI()
    if 'paper' in endpoint:
        return api.search_papers(query, bulk=False)
    elif 'author' in endpoint:
        return api.search_authors(query, bulk=False)
    else:
        return None