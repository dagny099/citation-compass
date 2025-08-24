"""
Helper functions for interacting with Scholar API and building a Knowledge Graph.
Project: Academic Citation Network
Author: Barbara Hidalgo-Sotelo
"""

import requests
import time
from typing import Dict, List, Optional, Generator, Union
import json
import logging
import pandas as pd
from urllib.parse import urlencode

# ENRICH KG with cited papers (paperID as the index)
def build_triples(details_list_dicts: list, subject_key: str, relation: str, predicate_key: str) -> pd.DataFrame:

    rows = [] 
    for index, item in enumerate(details_list_dicts):
        if subject_key in item.keys():
            subject = item[subject_key]
        else:
            subject = subject_key
        if predicate_key in item.keys():
            predicate = item[predicate_key]
        else:
            predicate = predicate_key
        
        rows.append( [subject, relation, predicate ] )

    return pd.DataFrame(rows, columns=['subject','predicate','object'])


def search_endpoint_nopaginate(endpoint: str = '/paper/search', query: str = 'the semantic web')  -> Optional[Dict]:
    """
    Access search endpoint of the Semantic Scholar API.
    
    Args:
        query: Query string to search for, e.g. paper title or author name
        
    Returns:
        Dictionary containing search results or None if request fails
    """
    base_url = "https://api.semanticscholar.org/graph/v1"
    if endpoint.search('/paper/search') == -1:
        endpoint = '/paper/search'
    else:
        endpoint
    params = {
        'query': query,
        'fields': 'paperId,title,authors,year,venue'
    }
    
    try:
        response = requests.get(f"{base_url}{endpoint}", params=params)  
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error searching {endpoint} for {query}: {e}")
        return None
    

def paginate_api_requests(
    base_url: str,
    params: Dict = None,
    offset_key: str = 'offset',
    limit_key: str = 'limit',
    next_key: str = 'next',
    data_key: str = 'data',
    limit: int = 100,
    rate_limit_pause: float = 1.0
) -> Generator[Dict, None, None]:
    """
    Generic function to handle pagination for API requests.
    
    Args:
        base_url: Base URL for the API endpoint
        params: Dictionary of query parameters
        offset_key: Name of the offset parameter in the request
        limit_key: Name of the limit parameter in the request
        next_key: Key in the response that contains the next offset
        data_key: Key in the response that contains the data array
        limit: Number of items to request per page
        rate_limit_pause: Time to wait between requests in seconds
        
    Yields:
        Individual items from the paginated response
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize parameters
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
        url = f"{base_url}?{urlencode(current_params)}"
        logger.info(f"Fetching results {offset} to {offset + limit} from {url}")
        
        try:
            # Make API request
            response = requests.get(url)
            if response.status_code == 404:
                logger.info("No more results to retrieve")
                break
            response.raise_for_status()
            data = response.json()
            
            # Check if we got any results
            items = data.get(data_key, [])
            if not items:
                logger.info("No more results to retrieve")
                break
            
            # Process each item in the current batch
            for item in items:
                total_retrieved += 1
                yield item
            
            # Check if we've retrieved all results
            if next_key not in data:
                logger.info(f"Retrieved all results. Total: {total_retrieved}")
                break
            elif total_retrieved >= 9999:
                print(f"Retrieved MAXIMUM RESULTS: {total_retrieved}")
                logger.info(f"Retrieved MAXIMUM RESULTS: {total_retrieved}")
                break

            
            # Update offset for next batch
            offset = data[next_key]
            
            # Respect rate limits
            time.sleep(rate_limit_pause)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching results: {e}")
            if response.status_code == 429:  # Too Many Requests
                logger.info("Rate limit hit. Waiting 60 seconds...")
                time.sleep(60)
                continue
            else:
                raise


def retrieve_citations_by_id(paper_id: str) -> Union[Dict, pd.DataFrame]:
    try:
        # Use the paginate function in helper_func.py b/c >100 results are expected
        cited_paper_ids = list( paginate_api_requests(
                                    base_url=f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations",
                                    params={"fields": "paperId"}
                                    )
                              )            
        # Get paperIDs of citing papers to do batch lookup of desired details -- Including original paper_id -- NEXT STEP: Create nodes
        citedPaperIds = []
        [citedPaperIds.append(d['citingPaper']['paperId']) for d in cited_paper_ids]
        return citedPaperIds
    except requests.exceptions.RequestException as e:
        print(f"Error: {e} when querying the citations endpoint")
        return {}, pd.DataFrame()  # Return empty dictionary and DataFrame in case of error


def batch_paper_details(paperIds: list) -> list:
    
    # DEFINE METADATA:
    paper_params="paperId,title,authors,year,publicationDate,venue,abstract,referenceCount,citationCount,fieldsOfStudy"
    
    # QUERY API
    r = requests.post(
        'https://api.semanticscholar.org/graph/v1/paper/batch',
        params={'fields': paper_params},
        json={"ids": paperIds}
    )
    
    details = r.json()    
    
    # If some paperIds did not have info , returned as NoneType, so eliminate
    details = [{} if x is None else x for x in details]
    
    return details
