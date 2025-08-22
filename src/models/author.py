"""
Author data models for Academic Citation Platform.

Unified author model supporting multiple naming conventions and affiliations.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator


class AuthorBase(BaseModel):
    """Base author model with common fields."""
    name: str = Field(..., min_length=1, max_length=200, description="Author name")
    author_id: Optional[str] = Field(None, description="Unique author identifier")
    
    # Alternative name fields for compatibility
    author_name: Optional[str] = Field(None, description="Alternative name field")
    
    # Academic metrics
    paper_count: Optional[int] = Field(None, ge=0)
    citation_count: Optional[int] = Field(None, ge=0)
    h_index: Optional[int] = Field(None, ge=0)
    
    # Profile information
    url: Optional[str] = Field(None, description="Author profile URL")
    affiliations: Optional[List[str]] = Field(default_factory=list)
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Author name cannot be empty')
        return v.strip()
    
    @property
    def display_name(self) -> str:
        """Get the best available name for display."""
        return self.name or self.author_name or "Unknown Author"


class Author(AuthorBase):
    """Complete author model with all fields."""
    
    # Research areas and topics
    research_fields: Optional[List[str]] = Field(default_factory=list)
    
    # Collaboration data
    collaborators: Optional[List[str]] = Field(default_factory=list)
    frequent_venues: Optional[List[str]] = Field(default_factory=list)
    
    # Temporal data
    first_publication_year: Optional[int] = Field(None, ge=1900, le=2030)
    last_publication_year: Optional[int] = Field(None, ge=1900, le=2030)
    active_years: Optional[List[int]] = Field(default_factory=list)
    
    # ML features
    embedding: Optional[List[float]] = Field(None, description="Author embedding vector")
    
    class Config:
        """Pydantic configuration."""
        allow_population_by_field_name = True
        validate_assignment = True
    
    @property
    def is_prolific(self, threshold: int = 50) -> bool:
        """Check if author is prolific (high paper count)."""
        return bool(self.paper_count and self.paper_count >= threshold)
    
    @property
    def is_highly_cited(self, threshold: int = 1000) -> bool:
        """Check if author is highly cited."""
        return bool(self.citation_count and self.citation_count >= threshold)
    
    @property
    def career_span(self) -> Optional[int]:
        """Calculate career span in years."""
        if self.first_publication_year and self.last_publication_year:
            return self.last_publication_year - self.first_publication_year + 1
        return None
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j storage."""
        data = {
            "name": self.display_name,
        }
        
        # Add identifier if available
        if self.author_id:
            data["authorId"] = self.author_id
        
        # Use both name fields for compatibility
        if self.author_name:
            data["authorName"] = self.author_name
        
        # Add optional fields if they exist
        if self.paper_count is not None:
            data["paperCount"] = self.paper_count
        if self.citation_count is not None:
            data["citationCount"] = self.citation_count
        if self.h_index is not None:
            data["hIndex"] = self.h_index
        if self.url:
            data["url"] = self.url
        if self.affiliations:
            data["affiliations"] = self.affiliations
        if self.research_fields:
            data["researchFields"] = self.research_fields
            
        return data
    
    @classmethod
    def from_neo4j_record(cls, record: Dict[str, Any]) -> Author:
        """Create Author instance from Neo4j record."""
        return cls(
            author_id=record.get("authorId"),
            name=record.get("name", "") or record.get("authorName", ""),
            author_name=record.get("authorName"),
            paper_count=record.get("paperCount"),
            citation_count=record.get("citationCount"),
            h_index=record.get("hIndex"),
            url=record.get("url"),
            affiliations=record.get("affiliations", []),
            research_fields=record.get("researchFields", []),
            collaborators=record.get("collaborators", []),
            frequent_venues=record.get("frequentVenues", [])
        )
    
    @classmethod
    def from_semantic_scholar_response(cls, data: Dict[str, Any]) -> Author:
        """Create Author instance from Semantic Scholar API response."""
        # Extract affiliations
        affiliations = []
        if "affiliations" in data and data["affiliations"]:
            affiliations = [
                aff for aff in data["affiliations"] 
                if isinstance(aff, str) and aff.strip()
            ]
        
        return cls(
            author_id=data.get("authorId"),
            name=data.get("name", ""),
            paper_count=data.get("paperCount"),
            citation_count=data.get("citationCount"),
            h_index=data.get("hIndex"),
            url=data.get("url"),
            affiliations=affiliations
        )


class AuthorCreate(AuthorBase):
    """Model for creating new authors."""
    
    def to_author(self) -> Author:
        """Convert to full Author model."""
        return Author(**self.dict())


class AuthorUpdate(BaseModel):
    """Model for updating existing authors."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    author_name: Optional[str] = Field(None, min_length=1, max_length=200)
    paper_count: Optional[int] = Field(None, ge=0)
    citation_count: Optional[int] = Field(None, ge=0)
    h_index: Optional[int] = Field(None, ge=0)
    url: Optional[str] = None
    affiliations: Optional[List[str]] = None
    research_fields: Optional[List[str]] = None
    
    def apply_to_author(self, author: Author) -> Author:
        """Apply updates to an existing author."""
        update_data = self.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(author, field, value)
        return author


# ====================================================================
# COLLABORATION MODELS
# ====================================================================

class Collaboration(BaseModel):
    """Model for author collaboration relationships."""
    author1_id: str
    author2_id: str
    paper_count: int = Field(ge=1, description="Number of co-authored papers")
    papers: Optional[List[str]] = Field(default_factory=list, description="List of co-authored paper IDs")
    
    first_collaboration_year: Optional[int] = None
    last_collaboration_year: Optional[int] = None
    
    @property
    def collaboration_span(self) -> Optional[int]:
        """Calculate collaboration span in years."""
        if self.first_collaboration_year and self.last_collaboration_year:
            return self.last_collaboration_year - self.first_collaboration_year + 1
        return None


class AuthorNetwork(BaseModel):
    """Model for author collaboration network."""
    author_id: str
    collaborators: List[Collaboration]
    total_collaborators: int
    total_collaborations: int
    
    @property
    def frequent_collaborators(self, min_papers: int = 3) -> List[Collaboration]:
        """Get collaborators with minimum number of joint papers."""
        return [
            collab for collab in self.collaborators 
            if collab.paper_count >= min_papers
        ]


# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def normalize_author_name(name: str) -> str:
    """
    Normalize author name for consistent storage and matching.
    
    Args:
        name: Raw author name
        
    Returns:
        Normalized author name
    """
    if not name:
        return ""
    
    # Basic normalization
    normalized = name.strip()
    
    # Remove extra whitespace
    normalized = " ".join(normalized.split())
    
    # Basic title case (might need more sophisticated handling)
    if normalized.isupper() or normalized.islower():
        normalized = normalized.title()
    
    return normalized


def create_author_from_dict(data: Dict[str, Any]) -> Author:
    """
    Create Author instance from dictionary with flexible field mapping.
    
    Args:
        data: Dictionary with author data
        
    Returns:
        Author instance
    """
    # Map common field variations
    field_mapping = {
        "id": "author_id",
        "authorId": "author_id",
        "author_id": "author_id",
        "name": "name",
        "authorName": "author_name",
        "paperCount": "paper_count",
        "citationCount": "citation_count",
        "hIndex": "h_index",
        "url": "url"
    }
    
    mapped_data = {}
    for original_field, standard_field in field_mapping.items():
        if original_field in data:
            mapped_data[standard_field] = data[original_field]
    
    # Handle affiliations
    if "affiliations" in data and data["affiliations"]:
        if isinstance(data["affiliations"], list):
            mapped_data["affiliations"] = [
                str(aff) for aff in data["affiliations"] if aff
            ]
        else:
            mapped_data["affiliations"] = [str(data["affiliations"])]
    
    # Ensure we have a name
    if "name" not in mapped_data:
        mapped_data["name"] = mapped_data.get("author_name", "Unknown Author")
    
    return Author(**mapped_data)


def find_potential_duplicates(authors: List[Author], similarity_threshold: float = 0.8) -> List[tuple]:
    """
    Find potential duplicate authors based on name similarity.
    
    Args:
        authors: List of authors to check
        similarity_threshold: Minimum similarity score for potential duplicates
        
    Returns:
        List of tuples containing potential duplicate pairs
    """
    from difflib import SequenceMatcher
    
    duplicates = []
    
    for i, author1 in enumerate(authors):
        for j, author2 in enumerate(authors[i+1:], i+1):
            # Compare names
            name1 = normalize_author_name(author1.display_name).lower()
            name2 = normalize_author_name(author2.display_name).lower()
            
            similarity = SequenceMatcher(None, name1, name2).ratio()
            
            if similarity >= similarity_threshold:
                duplicates.append((author1, author2, similarity))
    
    return duplicates


# ====================================================================
# DATACLASS VERSION (for backward compatibility)
# ====================================================================

@dataclass
class AuthorDataClass:
    """Dataclass version of Author for simpler use cases."""
    author_id: Optional[str]
    name: str
    paper_count: Optional[int] = None
    citation_count: Optional[int] = None
    h_index: Optional[int] = None
    affiliations: List[str] = field(default_factory=list)
    
    def to_pydantic(self) -> Author:
        """Convert to Pydantic Author model."""
        return Author(
            author_id=self.author_id,
            name=self.name,
            paper_count=self.paper_count,
            citation_count=self.citation_count,
            h_index=self.h_index,
            affiliations=self.affiliations
        )