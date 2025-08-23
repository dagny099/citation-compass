"""
Paper data models for Academic Citation Platform.

Unified paper model combining schemas from all three reference codebases:
- academic-citation-predictions: Comprehensive metadata
- citation-map-dashboard: ML-optimized features
- knowledge-cartography: Production data validation
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from pydantic import BaseModel, Field as PydanticField, field_validator, ConfigDict


class PaperBase(BaseModel):
    """Base paper model with common fields."""
    title: str = PydanticField(..., min_length=1, max_length=1000)
    abstract: Optional[str] = PydanticField(None, max_length=10000)
    year: Optional[int] = PydanticField(None, ge=1900, le=2030)
    citation_count: Optional[int] = PydanticField(None, ge=0)
    reference_count: Optional[int] = PydanticField(None, ge=0)
    publication_date: Optional[str] = None
    doi: Optional[str] = None
    is_open_access: Optional[bool] = None
    
    @field_validator('year')
    @classmethod
    def validate_year(cls, v):
        if v is not None and (v < 1900 or v > 2030):
            raise ValueError('Year must be between 1900 and 2030')
        return v
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Title cannot be empty')
        return v.strip()


class Paper(PaperBase):
    """Complete paper model with all fields."""
    model_config = ConfigDict(validate_assignment=True)
    
    paper_id: str = PydanticField(..., description="Unique paper identifier")
    
    # Additional metadata from Semantic Scholar
    influential_citation_count: Optional[int] = PydanticField(None, ge=0)
    s2_fields_of_study: Optional[List[str]] = PydanticField(default_factory=list)
    open_access_pdf: Optional[str] = None
    
    # Computed fields
    authors: Optional[List[str]] = PydanticField(default_factory=list)
    venues: Optional[List[str]] = PydanticField(default_factory=list)
    fields: Optional[List[str]] = PydanticField(default_factory=list)
    
    # ML features
    embedding: Optional[List[float]] = PydanticField(None, description="Paper embedding vector")
        
    @property
    def has_abstract(self) -> bool:
        """Check if paper has an abstract."""
        return bool(self.abstract and len(self.abstract.strip()) > 0)
    
    @property
    def is_highly_cited(self, threshold: int = 100) -> bool:
        """Check if paper is highly cited."""
        return bool(self.citation_count and self.citation_count >= threshold)
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j storage."""
        data = {
            "paperId": self.paper_id,
            "title": self.title,
            "citationCount": self.citation_count or 0,
            "referenceCount": self.reference_count or 0,
        }
        
        # Add optional fields if they exist
        if self.abstract:
            data["abstract"] = self.abstract
        if self.year:
            data["year"] = self.year
        if self.publication_date:
            data["publicationDate"] = self.publication_date
        if self.doi:
            data["doi"] = self.doi
        if self.is_open_access is not None:
            data["isOpenAccess"] = self.is_open_access
        if self.influential_citation_count is not None:
            data["influentialCitationCount"] = self.influential_citation_count
        if self.s2_fields_of_study:
            data["s2FieldsOfStudy"] = self.s2_fields_of_study
        if self.open_access_pdf:
            data["openAccessPdf"] = self.open_access_pdf
            
        return data
    
    @classmethod
    def from_neo4j_record(cls, record: Dict[str, Any]) -> Paper:
        """Create Paper instance from Neo4j record."""
        return cls(
            paper_id=record.get("paperId", ""),
            title=record.get("title", ""),
            abstract=record.get("abstract"),
            year=record.get("year"),
            citation_count=record.get("citationCount"),
            reference_count=record.get("referenceCount"),
            publication_date=record.get("publicationDate"),
            doi=record.get("doi"),
            is_open_access=record.get("isOpenAccess"),
            influential_citation_count=record.get("influentialCitationCount"),
            s2_fields_of_study=record.get("s2FieldsOfStudy", []),
            open_access_pdf=record.get("openAccessPdf"),
            authors=record.get("authors", []),
            venues=record.get("venues", []),
            fields=record.get("fields", [])
        )
    
    @classmethod
    def from_semantic_scholar_response(cls, data: Dict[str, Any]) -> Paper:
        """Create Paper instance from Semantic Scholar API response."""
        # Extract authors
        authors = []
        if "authors" in data and data["authors"]:
            authors = [author.get("name", "") for author in data["authors"] if author.get("name")]
        
        # Extract venue
        venues = []
        if "venue" in data and data["venue"]:
            venues = [data["venue"]]
        
        # Extract fields of study
        fields = []
        if "fieldsOfStudy" in data and data["fieldsOfStudy"]:
            fields = [field for field in data["fieldsOfStudy"] if field]
        
        # Extract S2 fields
        s2_fields = []
        if "s2FieldsOfStudy" in data and data["s2FieldsOfStudy"]:
            s2_fields = [field.get("category", "") for field in data["s2FieldsOfStudy"] if field.get("category")]
        
        return cls(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract"),
            year=data.get("year"),
            citation_count=data.get("citationCount"),
            reference_count=data.get("referenceCount"),
            publication_date=data.get("publicationDate"),
            doi=data.get("doi"),
            is_open_access=data.get("isOpenAccess"),
            influential_citation_count=data.get("influentialCitationCount"),
            s2_fields_of_study=s2_fields,
            open_access_pdf=data.get("openAccessPdf", {}).get("url") if data.get("openAccessPdf") else None,
            authors=authors,
            venues=venues,
            fields=fields
        )


class PaperCreate(PaperBase):
    """Model for creating new papers."""
    paper_id: str = PydanticField(..., description="Unique paper identifier")
    
    def to_paper(self) -> Paper:
        """Convert to full Paper model."""
        return Paper(**self.model_dump())


class PaperUpdate(BaseModel):
    """Model for updating existing papers."""
    title: Optional[str] = PydanticField(None, min_length=1, max_length=1000)
    abstract: Optional[str] = PydanticField(None, max_length=10000)
    year: Optional[int] = PydanticField(None, ge=1900, le=2030)
    citation_count: Optional[int] = PydanticField(None, ge=0)
    reference_count: Optional[int] = PydanticField(None, ge=0)
    publication_date: Optional[str] = None
    doi: Optional[str] = None
    is_open_access: Optional[bool] = None
    influential_citation_count: Optional[int] = PydanticField(None, ge=0)
    
    def apply_to_paper(self, paper: Paper) -> Paper:
        """Apply updates to an existing paper."""
        update_data = self.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(paper, field, value)
        return paper


# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def validate_paper_id(paper_id: str) -> bool:
    """
    Validate paper ID format.
    
    Args:
        paper_id: Paper ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not paper_id or not isinstance(paper_id, str):
        return False
    
    # Basic validation - adjust based on your ID format requirements
    return len(paper_id.strip()) > 0 and len(paper_id) <= 100


def create_paper_from_dict(data: Dict[str, Any]) -> Paper:
    """
    Create Paper instance from dictionary with flexible field mapping.
    
    Args:
        data: Dictionary with paper data
        
    Returns:
        Paper instance
    """
    # Map common field variations
    field_mapping = {
        "id": "paper_id",
        "paperId": "paper_id", 
        "paper_id": "paper_id",
        "title": "title",
        "abstract": "abstract",
        "year": "year",
        "publicationDate": "publication_date",
        "citationCount": "citation_count",
        "referenceCount": "reference_count",
        "doi": "doi",
        "isOpenAccess": "is_open_access",
        "influentialCitationCount": "influential_citation_count"
    }
    
    mapped_data = {}
    for original_field, standard_field in field_mapping.items():
        if original_field in data:
            mapped_data[standard_field] = data[original_field]
    
    # Handle authors list
    if "authors" in data:
        if isinstance(data["authors"], list):
            mapped_data["authors"] = [
                author.get("name", str(author)) if isinstance(author, dict) else str(author)
                for author in data["authors"]
            ]
    
    # Handle venue
    if "venue" in data and data["venue"]:
        mapped_data["venues"] = [data["venue"]]
    
    # Handle fields of study
    if "fieldsOfStudy" in data and data["fieldsOfStudy"]:
        mapped_data["fields"] = data["fieldsOfStudy"]
    
    return Paper(**mapped_data)


# ====================================================================
# DATACLASS VERSION (for backward compatibility)
# ====================================================================

@dataclass
class PaperDataClass:
    """Dataclass version of Paper for simpler use cases."""
    paper_id: str
    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    citation_count: Optional[int] = None
    reference_count: Optional[int] = None
    publication_date: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    venues: List[str] = field(default_factory=list)
    fields: List[str] = field(default_factory=list)
    
    def to_pydantic(self) -> Paper:
        """Convert to Pydantic Paper model."""
        return Paper(
            paper_id=self.paper_id,
            title=self.title,
            abstract=self.abstract,
            year=self.year,
            citation_count=self.citation_count,
            reference_count=self.reference_count,
            publication_date=self.publication_date,
            authors=self.authors,
            venues=self.venues,
            fields=self.fields
        )