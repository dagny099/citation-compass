"""
API models for the Academic Citation Platform.

This module provides data models for API requests and responses, supporting:
- Semantic Scholar API integration
- Internal API endpoints
- Batch processing requests
- Error handling and status codes
- Pagination and filtering

These models ensure consistent data exchange across all API interactions.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Union, Generic, TypeVar
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
from pydantic.generics import GenericModel

# Type variable for generic responses
T = TypeVar('T')


class APIStatus(str, Enum):
    """API response status codes."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PROCESSING = "processing"


class ErrorType(str, Enum):
    """Types of API errors."""
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    EXTERNAL_API_ERROR = "external_api_error"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"


class SortDirection(str, Enum):
    """Sort direction options."""
    ASC = "asc"
    DESC = "desc"


class PaperSortField(str, Enum):
    """Fields available for sorting papers."""
    TITLE = "title"
    YEAR = "year"
    CITATION_COUNT = "citation_count"
    CREATED_AT = "created_at"
    RELEVANCE = "relevance"


class AuthorSortField(str, Enum):
    """Fields available for sorting authors."""
    NAME = "name"
    PAPER_COUNT = "paper_count"
    CITATION_COUNT = "citation_count"
    H_INDEX = "h_index"


class APIError(BaseModel):
    """
    Standardized error response model.
    
    Provides consistent error reporting across all API endpoints
    with detailed information for debugging and user feedback.
    """
    
    error_type: ErrorType = Field(..., description="Type of error")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[str] = Field(None, description="Additional error details")
    error_code: Optional[str] = Field(None, description="Internal error code")
    field: Optional[str] = Field(None, description="Field that caused validation error")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    
    @classmethod
    def validation_error(cls, message: str, field: Optional[str] = None, 
                        details: Optional[str] = None) -> APIError:
        """Create validation error."""
        return cls(
            error_type=ErrorType.VALIDATION_ERROR,
            message=message,
            field=field,
            details=details
        )
    
    @classmethod
    def not_found(cls, resource: str, identifier: str) -> APIError:
        """Create not found error."""
        return cls(
            error_type=ErrorType.NOT_FOUND,
            message=f"{resource} not found",
            details=f"No {resource.lower()} found with identifier: {identifier}"
        )
    
    @classmethod
    def rate_limited(cls, retry_after: Optional[int] = None) -> APIError:
        """Create rate limit error."""
        details = f"Retry after {retry_after} seconds" if retry_after else None
        return cls(
            error_type=ErrorType.RATE_LIMITED,
            message="Rate limit exceeded",
            details=details
        )
    
    @classmethod
    def server_error(cls, message: str = "Internal server error", 
                    details: Optional[str] = None) -> APIError:
        """Create server error."""
        return cls(
            error_type=ErrorType.SERVER_ERROR,
            message=message,
            details=details
        )


class PaginationParams(BaseModel):
    """
    Pagination parameters for API requests.
    
    Supports both offset-based and cursor-based pagination
    for different use cases and performance requirements.
    """
    
    # Offset-based pagination
    offset: int = Field(default=0, ge=0, description="Number of items to skip")
    limit: int = Field(default=50, ge=1, le=1000, description="Maximum items to return")
    
    # Cursor-based pagination (alternative)
    cursor: Optional[str] = Field(None, description="Cursor for pagination")
    
    # Sorting
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_direction: SortDirection = Field(default=SortDirection.DESC, description="Sort direction")
    
    @validator('limit')
    def validate_limit(cls, v):
        """Ensure reasonable limit values."""
        if v > 1000:
            raise ValueError("Limit cannot exceed 1000")
        return v


class PaginationMeta(BaseModel):
    """
    Pagination metadata for responses.
    
    Provides information about the current page, total results,
    and navigation links for paginated responses.
    """
    
    total_count: int = Field(..., ge=0, description="Total number of items")
    page_count: int = Field(..., ge=0, description="Total number of pages")
    current_page: int = Field(..., ge=1, description="Current page number")
    per_page: int = Field(..., ge=1, description="Items per page")
    
    # Navigation
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")
    next_cursor: Optional[str] = Field(None, description="Cursor for next page")
    previous_cursor: Optional[str] = Field(None, description="Cursor for previous page")
    
    @classmethod
    def from_params(cls, params: PaginationParams, total_count: int) -> PaginationMeta:
        """Create pagination metadata from request parameters."""
        per_page = params.limit
        current_page = (params.offset // per_page) + 1
        page_count = (total_count + per_page - 1) // per_page
        
        return cls(
            total_count=total_count,
            page_count=page_count,
            current_page=current_page,
            per_page=per_page,
            has_next=current_page < page_count,
            has_previous=current_page > 1
        )


class APIResponse(GenericModel, Generic[T]):
    """
    Generic API response wrapper.
    
    Provides consistent response structure across all endpoints
    with status, data, metadata, and error handling.
    """
    
    status: APIStatus = Field(..., description="Response status")
    data: Optional[T] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Optional message")
    errors: Optional[List[APIError]] = Field(None, description="List of errors")
    meta: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    
    @classmethod
    def success(cls, data: T, message: Optional[str] = None, 
               meta: Optional[Dict[str, Any]] = None) -> APIResponse[T]:
        """Create successful response."""
        return cls(
            status=APIStatus.SUCCESS,
            data=data,
            message=message,
            meta=meta
        )
    
    @classmethod
    def error(cls, errors: Union[APIError, List[APIError]], 
             message: Optional[str] = None) -> APIResponse[T]:
        """Create error response."""
        if isinstance(errors, APIError):
            errors = [errors]
        
        return cls(
            status=APIStatus.ERROR,
            errors=errors,
            message=message
        )
    
    @classmethod
    def partial(cls, data: T, errors: Union[APIError, List[APIError]], 
               message: Optional[str] = None) -> APIResponse[T]:
        """Create partial success response."""
        if isinstance(errors, APIError):
            errors = [errors]
        
        return cls(
            status=APIStatus.PARTIAL,
            data=data,
            errors=errors,
            message=message
        )


class PaginatedResponse(GenericModel, Generic[T]):
    """
    Paginated API response wrapper.
    
    Extends the standard API response with pagination metadata
    for endpoints that return multiple items.
    """
    
    status: APIStatus = Field(..., description="Response status")
    data: List[T] = Field(default_factory=list, description="List of response items")
    pagination: PaginationMeta = Field(..., description="Pagination metadata")
    message: Optional[str] = Field(None, description="Optional message")
    errors: Optional[List[APIError]] = Field(None, description="List of errors")
    meta: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    
    @classmethod
    def success(cls, data: List[T], pagination: PaginationMeta,
               message: Optional[str] = None, 
               meta: Optional[Dict[str, Any]] = None) -> PaginatedResponse[T]:
        """Create successful paginated response."""
        return cls(
            status=APIStatus.SUCCESS,
            data=data,
            pagination=pagination,
            message=message,
            meta=meta
        )


class SearchRequest(BaseModel):
    """
    Request model for search operations.
    
    Supports full-text search with filtering, sorting, and pagination
    across different entity types.
    """
    
    query: str = Field(..., min_length=1, description="Search query string")
    entity_types: Optional[List[str]] = Field(None, description="Types of entities to search")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional filters")
    pagination: PaginationParams = Field(default_factory=PaginationParams, description="Pagination parameters")
    
    # Search options
    exact_match: bool = Field(default=False, description="Whether to use exact matching")
    include_abstracts: bool = Field(default=True, description="Whether to search in abstracts")
    min_year: Optional[int] = Field(None, ge=1900, description="Minimum publication year")
    max_year: Optional[int] = Field(None, le=2030, description="Maximum publication year")
    min_citations: Optional[int] = Field(None, ge=0, description="Minimum citation count")
    
    @validator('query')
    def validate_query(cls, v):
        """Ensure query is not empty after stripping."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    @validator('max_year')
    def validate_year_range(cls, v, values):
        """Ensure year range is valid."""
        if 'min_year' in values and v is not None and values['min_year'] is not None:
            if v < values['min_year']:
                raise ValueError("max_year must be >= min_year")
        return v


class BatchRequest(BaseModel):
    """
    Request model for batch operations.
    
    Supports batch processing of multiple items with progress tracking
    and error handling for individual items.
    """
    
    items: List[Dict[str, Any]] = Field(..., min_items=1, max_items=1000, description="Items to process")
    operation: str = Field(..., description="Operation to perform")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Operation options")
    
    # Processing options
    batch_size: int = Field(default=100, ge=1, le=500, description="Items per batch")
    continue_on_error: bool = Field(default=True, description="Continue processing if individual items fail")
    return_errors: bool = Field(default=True, description="Include errors in response")
    
    @validator('items')
    def validate_items_not_empty(cls, v):
        """Ensure items list is not empty."""
        if not v:
            raise ValueError("Items list cannot be empty")
        return v


class BatchResponse(BaseModel):
    """
    Response model for batch operations.
    
    Provides detailed results for each processed item including
    successes, failures, and processing statistics.
    """
    
    total_items: int = Field(..., description="Total items processed")
    successful_items: int = Field(..., description="Number of successful items")
    failed_items: int = Field(..., description="Number of failed items")
    
    # Results
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Results for each item")
    errors: List[APIError] = Field(default_factory=list, description="Errors encountered")
    
    # Processing metadata
    processing_time_seconds: float = Field(..., description="Total processing time")
    started_at: datetime = Field(..., description="Processing start time")
    completed_at: datetime = Field(..., description="Processing completion time")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.successful_items / self.total_items) * 100


class CitationSearchRequest(BaseModel):
    """
    Specialized search request for citation networks.
    
    Provides citation-specific filtering and search options
    for exploring academic networks.
    """
    
    # Basic search
    query: Optional[str] = Field(None, description="Search query")
    paper_ids: Optional[List[str]] = Field(None, description="Specific paper IDs to include")
    
    # Citation filtering
    min_citations: int = Field(default=0, ge=0, description="Minimum citation count")
    max_citations: Optional[int] = Field(None, ge=0, description="Maximum citation count")
    citation_depth: int = Field(default=1, ge=1, le=3, description="Citation network depth")
    
    # Temporal filtering
    year_range: Optional[Tuple[int, int]] = Field(None, description="Publication year range")
    recent_years_only: Optional[int] = Field(None, ge=1, description="Only papers from last N years")
    
    # Author and venue filtering
    authors: Optional[List[str]] = Field(None, description="Author names to include")
    venues: Optional[List[str]] = Field(None, description="Venue names to include")
    fields: Optional[List[str]] = Field(None, description="Research fields to include")
    
    # Network options
    include_references: bool = Field(default=True, description="Include paper references")
    include_citations: bool = Field(default=True, description="Include citing papers")
    max_network_size: int = Field(default=1000, ge=1, le=5000, description="Maximum network size")
    
    @validator('max_citations')
    def validate_citation_range(cls, v, values):
        """Ensure citation range is valid."""
        if v is not None and 'min_citations' in values:
            if v < values['min_citations']:
                raise ValueError("max_citations must be >= min_citations")
        return v
    
    @validator('year_range')
    def validate_year_range(cls, v):
        """Ensure year range is valid."""
        if v is not None:
            start_year, end_year = v
            if start_year > end_year:
                raise ValueError("Start year must be <= end year")
            if start_year < 1900 or end_year > 2030:
                raise ValueError("Years must be between 1900 and 2030")
        return v


class PredictionRequest(BaseModel):
    """
    Request model for citation predictions.
    
    Supports different types of prediction requests including
    single predictions, batch predictions, and network-wide analysis.
    """
    
    # Source papers
    source_paper_ids: List[str] = Field(..., min_items=1, description="Papers that would cite")
    
    # Target options
    target_paper_ids: Optional[List[str]] = Field(None, description="Specific target papers")
    target_search_query: Optional[str] = Field(None, description="Search for target papers")
    
    # Prediction options
    model_name: str = Field(default="TransE", description="ML model to use")
    top_k: int = Field(default=10, ge=1, le=100, description="Top K predictions per source")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence")
    
    # Filtering
    exclude_existing_citations: bool = Field(default=True, description="Exclude existing citations")
    exclude_self_citations: bool = Field(default=True, description="Exclude self-citations")
    min_target_citations: int = Field(default=1, ge=0, description="Minimum citations for targets")
    
    # Output options
    include_explanations: bool = Field(default=False, description="Include prediction explanations")
    include_confidence_intervals: bool = Field(default=False, description="Include confidence intervals")
    
    @validator('source_paper_ids')
    def validate_source_papers(cls, v):
        """Ensure source papers list is valid."""
        if not v:
            raise ValueError("Must provide at least one source paper")
        return v


class NetworkAnalysisRequest(BaseModel):
    """
    Request model for network analysis operations.
    
    Supports comprehensive network analysis including centrality measures,
    community detection, and structural analysis.
    """
    
    # Network specification
    paper_ids: Optional[List[str]] = Field(None, description="Specific papers to include")
    search_query: Optional[str] = Field(None, description="Search query for papers")
    max_network_size: int = Field(default=1000, ge=10, le=10000, description="Maximum network size")
    
    # Analysis options
    compute_centrality: bool = Field(default=True, description="Compute centrality measures")
    compute_communities: bool = Field(default=True, description="Detect communities")
    compute_shortest_paths: bool = Field(default=False, description="Compute shortest paths")
    compute_clustering: bool = Field(default=True, description="Compute clustering coefficient")
    
    # Centrality measures
    centrality_measures: List[str] = Field(
        default=["degree", "betweenness", "closeness", "pagerank"],
        description="Centrality measures to compute"
    )
    
    # Community detection
    community_algorithm: str = Field(default="louvain", description="Community detection algorithm")
    
    # Output options
    include_node_attributes: bool = Field(default=True, description="Include detailed node attributes")
    include_edge_attributes: bool = Field(default=True, description="Include detailed edge attributes")
    format_for_visualization: bool = Field(default=True, description="Format output for visualization")
    
    @validator('centrality_measures')
    def validate_centrality_measures(cls, v):
        """Ensure centrality measures are valid."""
        valid_measures = {"degree", "betweenness", "closeness", "pagerank", "eigenvector"}
        for measure in v:
            if measure not in valid_measures:
                raise ValueError(f"Invalid centrality measure: {measure}")
        return v


class HealthCheckResponse(BaseModel):
    """
    Response model for health check endpoints.
    
    Provides system status and health information for monitoring
    and diagnostics.
    """
    
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    
    # Component status
    database_status: str = Field(..., description="Database connection status")
    api_status: str = Field(..., description="External API status")
    ml_models_status: str = Field(..., description="ML models status")
    
    # Performance metrics
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    
    # Resource usage
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    cpu_usage_percent: Optional[float] = Field(None, description="CPU usage percentage")
    
    # Version information
    version: str = Field(..., description="Application version")
    build_date: Optional[str] = Field(None, description="Build date")
    
    @classmethod
    def healthy(cls, response_time_ms: float, uptime_seconds: float) -> HealthCheckResponse:
        """Create healthy status response."""
        return cls(
            status="healthy",
            database_status="connected",
            api_status="operational",
            ml_models_status="loaded",
            response_time_ms=response_time_ms,
            uptime_seconds=uptime_seconds,
            version="1.0.0"
        )
    
    @classmethod
    def unhealthy(cls, issues: List[str]) -> HealthCheckResponse:
        """Create unhealthy status response."""
        return cls(
            status="unhealthy",
            database_status="error" if "database" in str(issues) else "unknown",
            api_status="error" if "api" in str(issues) else "unknown", 
            ml_models_status="error" if "models" in str(issues) else "unknown",
            response_time_ms=0.0,
            uptime_seconds=0.0,
            version="1.0.0"
        )