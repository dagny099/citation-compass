"""
Venue data models for Academic Citation Platform.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class VenueBase(BaseModel):
    """Base venue model with common fields."""
    name: str = Field(..., min_length=1, max_length=500)
    venue_type: Optional[str] = Field(None, description="Journal, Conference, etc.")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Venue name cannot be empty')
        return v.strip()


class Venue(VenueBase):
    """Complete venue model."""
    venue_id: Optional[str] = None
    url: Optional[str] = None
    issn: Optional[str] = None
    
    # Statistics
    paper_count: Optional[int] = Field(None, ge=0)
    total_citations: Optional[int] = Field(None, ge=0)
    avg_citations_per_paper: Optional[float] = Field(None, ge=0.0)
    
    class Config:
        populate_by_name = True
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j storage."""
        return {
            "name": self.name,
            "type": self.venue_type,
            "url": self.url,
            "issn": self.issn
        }


class VenueCreate(VenueBase):
    """Model for creating new venues."""
    pass


class VenueUpdate(BaseModel):
    """Model for updating existing venues."""
    name: Optional[str] = Field(None, min_length=1, max_length=500)
    venue_type: Optional[str] = None
    url: Optional[str] = None
    issn: Optional[str] = None