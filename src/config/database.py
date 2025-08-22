"""
Database-specific configuration and query definitions.

This module contains Neo4j queries and database-specific settings,
adapted from knowledge-cartography's query organization patterns.
"""

# ====================================================================
# CORE CYPHER QUERIES
# ====================================================================

# Basic count queries for overview statistics
GET_PAPERS_COUNT = "MATCH (p:Paper) RETURN count(p) as count"
GET_AUTHORS_COUNT = "MATCH (a:Author) RETURN count(a) as count"
GET_VENUES_COUNT = "MATCH (v:PubVenue) RETURN count(v) as count"
GET_FIELDS_COUNT = "MATCH (f:Field) RETURN count(f) as count"
GET_PUBYEARS_COUNT = "MATCH (y:PubYear) RETURN count(y) as count"

# Relationship count queries
GET_CITATIONS_COUNT = "MATCH ()-[:CITES]->() RETURN count(*) as count"
GET_AUTHORED_PAPERS_COUNT = "MATCH ()-[:AUTHORED]->() RETURN count(*) as count"
GET_PUBLISHED_IN_VENUES_COUNT = "MATCH ()-[:PUBLISHED_IN]->() RETURN count(*) as count"
GET_IS_ABOUT_FIELDS_COUNT = "MATCH ()-[:IS_ABOUT]->() RETURN count(*) as count"

# ====================================================================
# NETWORK ANALYSIS QUERIES
# ====================================================================

# Citation network queries
GET_CITATION_EDGES = """
MATCH (source:Paper)-[:CITES]->(target:Paper)
RETURN source.paperId as source_id, target.paperId as target_id
"""

GET_CITATION_NETWORK_WITH_METADATA = """
MATCH (source:Paper)-[:CITES]->(target:Paper)
RETURN source.paperId as source_id, source.title as source_title,
       target.paperId as target_id, target.title as target_title,
       source.citationCount as source_citations,
       target.citationCount as target_citations
"""

# Paper exploration queries  
FIND_PAPERS_BY_KEYWORD = """
MATCH (p:Paper)
WHERE toLower(p.title) CONTAINS toLower($keyword)
RETURN p.paperId as paperId, p.title as title, 
       p.year as year, p.citationCount as citationCount
ORDER BY p.citationCount DESC
LIMIT 100
"""

GET_PAPER_DETAILS = """
MATCH (p:Paper {paperId: $paperId})
OPTIONAL MATCH (p)<-[:AUTHORED]-(a:Author)
OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:PubVenue)
OPTIONAL MATCH (p)-[:IS_ABOUT]->(f:Field)
OPTIONAL MATCH (p)-[:PUB_YEAR]->(y:PubYear)
RETURN p.paperId as paperId, p.title as title, p.abstract as abstract,
       p.year as year, p.citationCount as citationCount,
       collect(DISTINCT a.authorName) as authors,
       collect(DISTINCT v.name) as venues,
       collect(DISTINCT f.name) as fields,
       y.year as pubYear
"""

# ====================================================================
# STATISTICAL ANALYSIS QUERIES
# ====================================================================

# Publication year analysis
GET_PUB_YEAR_RANGE = """
MATCH (p:Paper)
WHERE p.year IS NOT NULL
RETURN min(p.year) as min_year, max(p.year) as max_year
"""

GET_PUB_YEAR_HIST = """
MATCH (p:Paper)-[:PUB_YEAR]->(y:PubYear)
RETURN y.year as year, count(p) as paper_count
ORDER BY y.year
"""

# Field analysis
GET_PAPER_COUNTS_PER_FIELD = """
MATCH (p:Paper)-[:IS_ABOUT]->(f:Field)
RETURN f.name as field_name, count(p) as paper_count
ORDER BY paper_count DESC
"""

GET_AUTHOR_COUNTS_PER_FIELD = """
MATCH (a:Author)-[:AUTHORED]->(p:Paper)-[:IS_ABOUT]->(f:Field)
RETURN f.name as field_name, count(DISTINCT a) as author_count
ORDER BY author_count DESC
"""

# Citation analysis
GET_CITATIONS_PER_PAPER_n = """
MATCH (p:Paper)
OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
RETURN p.paperId as paper_id, p.title as title, count(citing) as citations
ORDER BY citations DESC
"""

GET_CITATIONS_PER_PAPER_a = """
MATCH (p:Paper)
RETURN p.paperId as paper_id, p.title as title, 
       COALESCE(p.citationCount, 0) as citations
ORDER BY citations DESC
"""

# ====================================================================
# AUTHOR ANALYSIS QUERIES
# ====================================================================

GET_TOP_AUTHORS_BY_CITATIONS = """
MATCH (a:Author)-[:AUTHORED]->(p:Paper)
RETURN a.authorId as authorId, 
       COALESCE(a.authorName, a.name) as authorName,
       count(p) as paper_count,
       sum(p.citationCount) as total_citations
ORDER BY total_citations DESC
LIMIT $limit
"""

GET_AUTHOR_COLLABORATIONS = """
MATCH (a:Author {authorName: $authorName})-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(coauthor:Author)
WHERE a <> coauthor
RETURN coauthor.authorName as coauthor_name, count(p) as collaboration_count
ORDER BY collaboration_count DESC
LIMIT $limit
"""

GET_RISING_STARS_DATA = """
MATCH (a:Author)-[:AUTHORED]->(p:Paper)-[:PUB_YEAR]->(y:PubYear)
WHERE y.year >= $min_year
WITH a, y.year as year, sum(p.citationCount) as year_citations
RETURN a.authorId as authorId, 
       COALESCE(a.authorName, a.name) as authorName,
       collect({year: year, citations: year_citations}) as yearly_data
"""

# ====================================================================
# NETWORK STRUCTURE QUERIES
# ====================================================================

GET_NETWORK_SCHEMA = """
MATCH (source)-[r]->(target)
RETURN DISTINCT 
       labels(source)[0] as source_type,
       type(r) as relationship_type, 
       labels(target)[0] as target_type,
       count(*) as count
ORDER BY count DESC
"""

GET_NETWORK_PAPERID = """
MATCH (p:Paper {paperId: $paperId})
OPTIONAL MATCH (p)-[r1:CITES]->(cited:Paper)
OPTIONAL MATCH (citing:Paper)-[r2:CITES]->(p)
OPTIONAL MATCH (p)<-[r3:AUTHORED]-(a:Author)
OPTIONAL MATCH (p)-[r4:PUBLISHED_IN]->(v:PubVenue)
OPTIONAL MATCH (p)-[r5:IS_ABOUT]->(f:Field)

WITH p, 
     collect(DISTINCT {type: 'CITES', target: cited.paperId, target_title: cited.title}) as cites,
     collect(DISTINCT {type: 'CITED_BY', source: citing.paperId, source_title: citing.title}) as cited_by,
     collect(DISTINCT {type: 'AUTHORED_BY', author: COALESCE(a.authorName, a.name)}) as authors,
     collect(DISTINCT {type: 'PUBLISHED_IN', venue: v.name}) as venues,
     collect(DISTINCT {type: 'IS_ABOUT', field: f.name}) as fields

UNWIND (cites + cited_by + authors + venues + fields) as relationship
RETURN p.paperId as source_id, p.title as source_label, 'Paper' as source_type,
       relationship.type as relationship_type,
       COALESCE(relationship.target, relationship.source, relationship.author, relationship.venue, relationship.field) as target_id,
       COALESCE(relationship.target_title, relationship.source_title, relationship.author, relationship.venue, relationship.field) as target_label,
       CASE relationship.type
         WHEN 'CITES' THEN 'Paper'
         WHEN 'CITED_BY' THEN 'Paper'
         WHEN 'AUTHORED_BY' THEN 'Author'
         WHEN 'PUBLISHED_IN' THEN 'PubVenue'
         WHEN 'IS_ABOUT' THEN 'Field'
       END as target_type
"""

# ====================================================================
# ML/PREDICTION QUERIES
# ====================================================================

GET_ALL_PAPERS_FOR_ML = """
MATCH (p:Paper)
RETURN p.paperId as paper_id, p.title as title, 
       p.citationCount as citation_count, p.year as year
"""

GET_PAPER_FEATURES = """
MATCH (p:Paper {paperId: $paperId})
OPTIONAL MATCH (p)<-[:AUTHORED]-(a:Author)
OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:PubVenue)
OPTIONAL MATCH (p)-[:IS_ABOUT]->(f:Field)
OPTIONAL MATCH (p)-[:PUB_YEAR]->(y:PubYear)
RETURN p.paperId as paper_id,
       count(DISTINCT a) as author_count,
       collect(DISTINCT f.name) as fields,
       v.name as venue,
       y.year as year,
       p.citationCount as citation_count
"""

# ====================================================================
# DATA QUALITY QUERIES
# ====================================================================

GET_ORPHANED_PAPERS = """
MATCH (p:Paper)
WHERE NOT (p)<-[:AUTHORED]-()
RETURN count(p) as orphaned_count
"""

GET_PAPERS_WITHOUT_YEAR = """
MATCH (p:Paper)
WHERE NOT (p)-[:PUB_YEAR]->() AND p.year IS NULL
RETURN count(p) as no_year_count
"""

GET_AUTHORS_WITHOUT_PAPERS = """
MATCH (a:Author)
WHERE NOT (a)-[:AUTHORED]->()
RETURN count(a) as authorless_count
"""

# ====================================================================
# PERFORMANCE OPTIMIZATION QUERIES
# ====================================================================

# Queries for creating indexes and constraints (defined in schema.py)
# These are here for reference and documentation

PERFORMANCE_INDEXES = [
    "CREATE INDEX paper_citation_count IF NOT EXISTS FOR (p:Paper) ON (p.citationCount)",
    "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
    "CREATE FULLTEXT INDEX paper_title_fulltext IF NOT EXISTS FOR (p:Paper) ON EACH [p.title, p.abstract]"
]

# ====================================================================
# QUERY COLLECTIONS
# ====================================================================

# Group queries by functional area for easy access
OVERVIEW_QUERIES = {
    "papers_count": GET_PAPERS_COUNT,
    "authors_count": GET_AUTHORS_COUNT,
    "venues_count": GET_VENUES_COUNT,
    "fields_count": GET_FIELDS_COUNT,
    "years_count": GET_PUBYEARS_COUNT,
    "citations_count": GET_CITATIONS_COUNT,
    "authorships_count": GET_AUTHORED_PAPERS_COUNT
}

PAPER_QUERIES = {
    "find_by_keyword": FIND_PAPERS_BY_KEYWORD,
    "get_details": GET_PAPER_DETAILS,
    "get_network": GET_NETWORK_PAPERID,
    "citations_per_paper": GET_CITATIONS_PER_PAPER_n
}

AUTHOR_QUERIES = {
    "top_by_citations": GET_TOP_AUTHORS_BY_CITATIONS,
    "collaborations": GET_AUTHOR_COLLABORATIONS,
    "rising_stars_data": GET_RISING_STARS_DATA
}

NETWORK_QUERIES = {
    "citation_edges": GET_CITATION_EDGES,
    "schema": GET_NETWORK_SCHEMA,
    "paper_network": GET_NETWORK_PAPERID
}

ML_QUERIES = {
    "all_papers": GET_ALL_PAPERS_FOR_ML,
    "paper_features": GET_PAPER_FEATURES,
    "citation_network": GET_CITATION_NETWORK_WITH_METADATA
}

QUALITY_QUERIES = {
    "orphaned_papers": GET_ORPHANED_PAPERS,
    "papers_without_year": GET_PAPERS_WITHOUT_YEAR,
    "authors_without_papers": GET_AUTHORS_WITHOUT_PAPERS
}