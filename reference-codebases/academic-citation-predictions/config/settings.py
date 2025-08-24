"""
Configuration settings for Academic Citation Predictions project.
"""

import os
from typing import Dict, List

# API Configuration
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
DEFAULT_RATE_LIMIT_PAUSE = 1.0
MAX_BATCH_SIZE = 500
MAX_PAGINATION_LIMIT = 9999
DEFAULT_REQUEST_TIMEOUT = 30

# API Endpoints
ENDPOINTS = {
    'paper_search': '/paper/search',
    'paper_search_bulk': '/paper/search/bulk', 
    'author_search': '/author/search',
    'author_search_bulk': '/author/search/bulk',
    'paper_batch': '/paper/batch',
    'author_batch': '/author/batch',
    'paper_citations': '/paper/{paper_id}/citations',
    'paper_references': '/paper/{paper_id}/references'
}

# Field specifications for different data types
PAPER_FIELDS = "paperId,title,authors,year,publicationDate,venue,abstract,referenceCount,citationCount,fieldsOfStudy"
AUTHOR_FIELDS = "authorId,name,affiliations,paperCount,citationCount,hIndex,url"
CITATION_FIELDS = "paperId"

# Neo4j Configuration
NEO4J_CONFIG = {
    'uri': os.getenv("NEO4J_URI"),
    'user': os.getenv("NEO4J_USER"), 
    'password': os.getenv("NEO4J_PWD")
}

# File paths
DATA_PATHS = {
    'storage': 'storage/',
    'transformed': 'transformed/',
    'raw_data': 'storage/MEGA_paper_details.json',
    'paper_info': 'storage/MEGA_paper_INFO.csv'
}

# Graph schema configuration
GRAPH_SCHEMA = {
    'node_types': ['Paper', 'Author', 'PubVenue', 'PubYear', 'Field'],
    'relationships': {
        'CITED_BY': ('Paper', 'Paper'),
        'CO_AUTHORED': ('Author', 'Paper'),
        'PUBLISHED_IN': ('Paper', 'PubVenue'),
        'PUB_YEAR': ('Paper', 'PubYear'),
        'IS_ABOUT': ('Paper', 'Field'),
        'AFFILIATED_WITH': ('Author', 'Institution'),
        'LOCATED_IN': ('Institution', 'Place')
    }
}

# ML Model Configuration
MODEL_CONFIG = {
    'embedding_models': ['TransE', 'RotatE', 'DistMult'],
    'default_model': 'TransE',
    'embedding_dim': 100,
    'training_epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.01,
    'test_split': 0.2
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'top_authors_limit': 10,
    'top_venues_limit': 10,
    'rising_stars_min_years': 3,
    'citation_histogram_bins': 50
}