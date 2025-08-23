"""
Field data models for Academic Citation Platform.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field as PydanticField, field_validator, ConfigDict


class FieldBase(BaseModel):
    """Base field model with common fields."""
    name: str = PydanticField(..., min_length=1, max_length=200)
    category: Optional[str] = PydanticField(None, description="Broader category")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Field name cannot be empty')
        return v.strip()


class ResearchField(FieldBase):
    """Complete field model."""
    model_config = ConfigDict(validate_assignment=True)
    
    s2_field_id: Optional[str] = None
    
    # Statistics
    paper_count: Optional[int] = PydanticField(None, ge=0)
    author_count: Optional[int] = PydanticField(None, ge=0)
    
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
    name: Optional[str] = PydanticField(None, min_length=1, max_length=200)
    category: Optional[str] = None


# For backward compatibility
Field = ResearchField