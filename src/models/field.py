"""
Field data models for Academic Citation Platform.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class FieldBase(BaseModel):
    """Base field model with common fields."""
    name: str = Field(..., min_length=1, max_length=200)
    category: Optional[str] = Field(None, description="Broader category")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Field name cannot be empty')
        return v.strip()


class Field(FieldBase):
    """Complete field model."""
    s2_field_id: Optional[str] = None
    
    # Statistics
    paper_count: Optional[int] = Field(None, ge=0)
    author_count: Optional[int] = Field(None, ge=0)
    
    class Config:
        allow_population_by_field_name = True
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j storage."""
        return {
            "name": self.name,
            "category": self.category,
            "s2FieldId": self.s2_field_id
        }


class FieldCreate(FieldBase):
    """Model for creating new fields."""
    pass


class FieldUpdate(BaseModel):
    """Model for updating existing fields."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    category: Optional[str] = None