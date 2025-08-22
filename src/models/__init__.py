"""
Core data models for Academic Citation Platform.

This module provides unified data models for academic entities,
combining patterns from all three reference codebases.
"""

from .paper import Paper, PaperCreate, PaperUpdate
from .author import Author, AuthorCreate, AuthorUpdate  
from .venue import Venue, VenueCreate, VenueUpdate
from .field import Field, FieldCreate, FieldUpdate
from .citation import Citation, CitationCreate

__all__ = [
    "Paper", "PaperCreate", "PaperUpdate",
    "Author", "AuthorCreate", "AuthorUpdate", 
    "Venue", "VenueCreate", "VenueUpdate",
    "Field", "FieldCreate", "FieldUpdate",
    "Citation", "CitationCreate"
]