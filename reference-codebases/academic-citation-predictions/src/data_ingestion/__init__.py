"""Data ingestion package for academic citation predictions."""

from .semantic_scholar_api import SemanticScholarAPI
from .graph_builder import GraphBuilder

__all__ = ['SemanticScholarAPI', 'GraphBuilder']