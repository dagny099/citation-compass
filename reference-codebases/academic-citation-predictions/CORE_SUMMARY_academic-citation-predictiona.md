# Academic Citation Predictions - Core Functionality Summary

## Project Overview
**Academic citation network analysis system** using Semantic Scholar API for data ingestion and Neo4j for graph storage/analysis. While designed for PyTorch-based link prediction models, the current implementation focuses on data collection, transformation, and graph analytics. Machine learning capabilities are planned but not yet implemented.

## Data Flow Architecture

```
Semantic Scholar API → Data Ingestion → ETL Processing → Graph Storage → Analysis
                                    ↓
                               Knowledge Graph
                            (Papers, Authors, Venues)
                                    ↓
                              Neo4j Analytics
                           (Citation patterns, trends)
```

**Note**: PyKEEN and PyTorch Geometric integrations are planned but not currently implemented.

## Core Scripts & Functions

### 1. `helper_func.py` - API & Utility Functions ✅ **EXISTS**
**Purpose**: Core utilities for Semantic Scholar API interaction and data processing

#### Key Functions:
- **`paginate_api_requests(base_url, params, ...)`** (lines 63-155)
  - Handles paginated API requests with rate limiting
  - Generic function for any Semantic Scholar endpoint
  - Returns generator of individual items
  - **Data Flow**: API → Paginated Results → Individual Records

- **`retrieve_citations_by_id(paper_id)`** (lines 157-172)  
  - Gets list of papers citing a given paper
  - Returns list of citing paper IDs
  - **Data Flow**: Paper ID → Citations Endpoint → List of Citing Paper IDs

- **`batch_paper_details(paperIds)`** (lines 174-192)
  - Bulk retrieval of paper metadata for multiple IDs
  - Uses POST batch endpoint for efficiency
  - **Data Flow**: List of Paper IDs → Batch API → Detailed Paper Records

- **`build_triples(details_list_dicts, subject_key, relation, predicate_key)`** (lines 16-32)
  - Converts structured data to RDF triples format
  - **Data Flow**: Raw Data Records → Triple Format (Subject, Predicate, Object)

- **`search_endpoint_nopaginate(endpoint, query)`** (lines 34-61)
  - Simple search without pagination
  - **Data Flow**: Search Query → API Search → Results JSON

### 2. `etl_papers_authors.py` - Data Transformation Pipeline ✅ **EXISTS**
**Purpose**: Transforms raw API data into normalized graph entities and relationships

#### Key Functions:
- **`transform_papers(data)`** (lines 70-86)
  - Extracts paper nodes with metadata
  - **Data Flow**: Raw API Data → Paper Entities (paperId, title, citations, etc.)

- **`transform_authors(data)`** (lines 34-66)
  - Extracts author nodes and co-authorship relationships  
  - **Data Flow**: Raw API Data → (Author Entities, Co-authorship Relations)

- **`transform_fields_of_study(data)`** (lines 6-31)
  - Extracts research fields and paper-field relationships
  - **Data Flow**: Raw API Data → (Field Entities, Paper-Field Relations)

- **`transform_venues(data)`** (lines 89-108)
  - Extracts publication venues and paper-venue relationships
  - **Data Flow**: Raw API Data → (Venue Entities, Paper-Venue Relations)

- **`transform_years(data)`** (lines 111-138)
  - Extracts publication years and paper-year relationships
  - **Data Flow**: Raw API Data → (Year Entities, Paper-Year Relations)

### 3. `neo4j_insights.py` - Graph Analysis & Querying ✅ **EXISTS**
**Purpose**: Analytics and insights from the knowledge graph stored in Neo4j

#### Key Functions:
- **`run_query(cypher_query, parameters)`** (lines 25-39)
  - Executes Cypher queries and returns pandas DataFrames
  - **Data Flow**: Cypher Query → Neo4j Database → DataFrame Results

- **`get_top_authors(limit)`** (lines 45-58)
  - Finds authors by total citation count
  - **Data Flow**: Neo4j Graph → Author Citation Aggregation → Ranked List

- **`get_top_venues(limit)`** (lines 61-74)
  - Finds venues by average citation count per paper
  - **Data Flow**: Neo4j Graph → Venue Citation Analysis → Ranked List

- **`get_coauthors(author_name, limit)`** (lines 77-93)
  - Finds frequent collaborators for a given author
  - **Data Flow**: Author Name → Co-authorship Network → Collaboration Rankings

- **`get_citation_histogram()`** (lines 96-106)
  - Retrieves citation distribution data
  - **Data Flow**: Neo4j Graph → Citation Counts → Distribution Data

- **`get_yearly_author_citations()`** (lines 109-121)
  - Gets temporal citation patterns by author
  - **Data Flow**: Neo4j Graph → Time-Series Citation Data → Author Trends

- **`get_rising_stars(min_years, top_n)`** (lines 124-148)
  - Identifies authors with increasing citation trends using linear regression
  - **Data Flow**: Yearly Citations → Linear Regression Analysis → Growth Rankings

### 4. `1-Data-Ingestion.ipynb` - Primary Data Collection Workflow ✅ **EXISTS**
**Purpose**: Interactive data collection from Semantic Scholar API

#### Core Workflow:
1. **Search Phase**: Query API for seed paper by title
2. **Expansion Phase**: Recursively collect citing papers 
3. **Enrichment Phase**: Gather author, venue, field metadata
4. **Storage Phase**: Save raw data as JSON/CSV
5. **Graph Construction**: Convert to RDF triples and PyTorch Geometric objects

#### Key Data Structures Created:
- Citation networks (paper → citing papers)
- Co-authorship networks (authors → shared papers)  
- Knowledge graph triples (subject, predicate, object)

### 5. `2- Enrich Data-.ipynb` - Data Processing & Transformation ✅ **EXISTS**
**Purpose**: Clean and structure raw data for graph database import

#### Core Workflow:
1. **Load Raw Data**: Read collected JSON data (9,212 papers) ✅
2. **Apply Transformations**: Use ETL functions to create normalized entities ✅
3. **Generate Relationships**: Create relationship tables for Neo4j import ✅
4. **Network Analysis**: Build co-authorship networks with NetworkX ✅
5. **Export Structured Data**: Save as CSV files for database import ✅

### 6. Supporting Files ✅ **ALL EXIST**

#### `neo4j_full_import_script.cypher` ✅
- Complete Neo4j database schema and import script
- Creates constraints for data integrity
- Imports all entity types and relationships

#### `database-schema.txt` ✅ 
- Documents Neo4j graph schema
- Node types: Paper, Author, PubVenue, PubYear, Field
- Relationship types: PUBLISHED_IN, CO_AUTHORED, IS_ABOUT, etc.

#### `src/data_ingestion/semantic_scholar_api.py` ✅ **NEW - ENHANCED VERSION**
- Consolidated API client with proper error handling
- Class-based architecture with configuration management
- Enhanced citation network expansion capabilities

## Unique Functionality & Design Patterns

### 1. **Recursive Citation Network Expansion** ✅ **IMPLEMENTED**
- `expand_nodes()` function in notebooks ✅
- `expand_citation_network()` method in `SemanticScholarAPI` class ✅
- Automatically discovers and expands citation networks
- **Unique Feature**: Self-expanding knowledge graph based on citation relationships

### 2. **Multi-Modal Data Integration** ✅ **IMPLEMENTED**
- Combines bibliometric data (citations, references) ✅
- Author collaboration networks ✅
- Temporal publication patterns ✅
- Research field classifications ✅

### 3. **Hybrid Storage Strategy** ✅ **PARTIALLY IMPLEMENTED**
- Raw data in JSON files ✅
- Processed data in CSV format ✅
- Graph data in Neo4j ✅
- PyTorch Geometric objects for ML ❌ **PLANNED BUT NOT IMPLEMENTED**

### 4. **Scalable API Interaction** ✅ **FULLY IMPLEMENTED**
- Generic pagination handling ✅
- Rate limiting and error recovery ✅
- Batch processing for efficiency ✅

### 5. **Advanced Graph Analytics** ✅ **IMPLEMENTED**
- Rising star detection using linear regression ✅
- Centrality analysis for author importance ✅ (in notebooks)
- Temporal trend analysis ✅

## Data Pipeline Summary

```
1. INGESTION: Semantic Scholar API → Raw JSON Data ✅
2. EXTRACTION: Raw Data → Pandas DataFrames ✅ 
3. TRANSFORMATION: DataFrames → Graph Entities + Relationships ✅
4. LOADING: Structured Data → Neo4j Graph Database ✅
5. ANALYSIS: Neo4j → Insights + Visualizations ✅
6. MODELING: Graph Data → PyTorch Geometric → Link Predictions ❌ PLANNED
```

## Key Dependencies
- **Data Sources**: Semantic Scholar API ✅
- **Storage**: Neo4j graph database, JSON/CSV files ✅
- **Processing**: pandas ✅, NetworkX ✅, PyTorch Geometric ❌ **PLANNED**
- **ML Framework**: PyKEEN ❌ **PLANNED** (listed in pyproject.toml but not used)
- **Analysis**: scipy for statistical analysis ✅
- **Graph Database**: RDFLib ✅ for triple storage

## Implementation Status Summary

### ✅ **FULLY IMPLEMENTED**
- Data ingestion pipeline with Semantic Scholar API
- ETL transformation functions for graph entities
- Neo4j graph database integration
- Advanced analytics (rising stars, centrality, trends)
- Hybrid storage strategy (JSON, CSV, Neo4j)
- Rate limiting and error handling
- Citation network expansion algorithms

### ❌ **PLANNED BUT NOT IMPLEMENTED**
- PyTorch Geometric graph objects
- PyKEEN knowledge graph embeddings  
- Link prediction models
- Machine learning pipeline

### 📂 **DATA ASSETS CREATED**
- 9,212 academic papers collected
- Citation network with 318+ citing relationships
- Author collaboration networks
- Transformed CSV files ready for Neo4j import
- RDF triples for semantic web applications

## Comparison Points for Other Codebases
When comparing with other projects, focus on:

1. **API Integration Strategy** ✅ - Robust pagination and rate limiting
2. **Graph Construction Approach** ✅ - Multi-entity modeling with relationships
3. **ML Model Architecture** ❌ - **NOT YET IMPLEMENTED**
4. **Scalability Patterns** ✅ - Batch processing and recursive expansion
5. **Analytics Capabilities** ✅ - Unique rising star detection and trend analysis
6. **Storage Architecture** ✅ - Multi-format storage strategy