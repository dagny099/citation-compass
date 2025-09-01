"""
Citation data models for Academic Citation Platform.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class CitationBase(BaseModel):
    """Base citation model."""
    source_paper_id: str = Field(..., description="Paper that cites")
    target_paper_id: str = Field(..., description="Paper being cited")
    
    # Citation context and metadata
    context: Optional[str] = Field(None, description="Citation context text")
    intent_category: Optional[str] = Field(None, description="Citation intent")
    is_influential: Optional[bool] = Field(None, description="Whether citation is influential")


class Citation(CitationBase):
    """Complete citation model."""
    citation_id: Optional[str] = None
    
    # Additional metadata
    section: Optional[str] = Field(None, description="Paper section where citation appears")
    citation_offset: Optional[int] = Field(None, description="Character offset in paper")
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        populate_by_name = True
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j relationship properties."""
        data = {}
        
        if self.context:
            data["citationContext"] = self.context
        if self.intent_category:
            data["intentCategory"] = self.intent_category
        if self.is_influential is not None:
            data["isInfluential"] = self.is_influential
        if self.section:
            data["section"] = self.section
        if self.citation_offset is not None:
            data["citationOffset"] = self.citation_offset
            
        return data


class CitationCreate(CitationBase):
    """Model for creating new citations."""
    pass