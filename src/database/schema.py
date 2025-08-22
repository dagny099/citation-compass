"""
Unified Neo4j database schema for Academic Citation Platform.

This module defines the complete database schema combining the best elements
from all three reference codebases:
- academic-citation-predictions: Comprehensive entity modeling
- citation-map-dashboard: ML-optimized citation relationships
- knowledge-cartography: Production-ready constraints and indexes

Schema Components:
- Node types: Paper, Author, PubVenue, PubYear, Field, Institution
- Relationships: CITES, AUTHORED, PUBLISHED_IN, PUB_YEAR, IS_ABOUT, AFFILIATED_WITH
- Constraints: Unique identifiers and required properties
- Indexes: Performance optimization for common queries
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# ====================================================================
# SCHEMA DEFINITIONS
# ====================================================================

NODE_TYPES = {
    "Paper": {
        "required_properties": ["paperId", "title"],
        "optional_properties": [
            "abstract", "referenceCount", "citationCount", 
            "publicationDate", "year", "doi", "s2FieldsOfStudy",
            "influentialCitationCount", "isOpenAccess", "openAccessPdf"
        ],
        "constraints": ["paperId"],
        "indexes": ["title", "year", "citationCount"]
    },
    "Author": {
        "required_properties": ["authorId"],
        "optional_properties": [
            "authorName", "name", "url", "paperCount", 
            "citationCount", "hIndex", "affiliations"
        ],
        "constraints": ["authorId"],
        "indexes": ["authorName", "name"]
    },
    "PubVenue": {
        "required_properties": ["name"],
        "optional_properties": [
            "venue", "type", "url", "issn", "volume", 
            "issue", "firstPage", "lastPage"
        ],
        "constraints": ["name"],
        "indexes": ["name"]
    },
    "PubYear": {
        "required_properties": ["year"],
        "optional_properties": [],
        "constraints": ["year"],
        "indexes": ["year"]
    },
    "Field": {
        "required_properties": ["name"],
        "optional_properties": ["category", "s2FieldId"],
        "constraints": ["name"],
        "indexes": ["name"]
    },
    "Institution": {
        "required_properties": ["name"],
        "optional_properties": ["institutionId", "url", "country", "type"],
        "constraints": ["name"],
        "indexes": ["name", "country"]
    }
}

RELATIONSHIP_TYPES = {
    "CITES": {
        "from": "Paper",
        "to": "Paper",
        "properties": ["citationContext", "intentCategory", "isInfluential"]
    },
    "AUTHORED": {
        "from": "Author", 
        "to": "Paper",
        "properties": ["authorOrder", "isCorresponding"]
    },
    "CO_AUTHORED": {
        "from": "Author",
        "to": "Paper", 
        "properties": ["authorOrder", "isCorresponding"]
    },
    "PUBLISHED_IN": {
        "from": "Paper",
        "to": "PubVenue",
        "properties": ["volume", "issue", "pages"]
    },
    "PUB_YEAR": {
        "from": "Paper",
        "to": "PubYear",
        "properties": []
    },
    "IS_ABOUT": {
        "from": "Paper",
        "to": "Field",
        "properties": ["confidence", "source"]
    },
    "AFFILIATED_WITH": {
        "from": "Author",
        "to": "Institution", 
        "properties": ["startDate", "endDate", "position"]
    }
}

# ====================================================================
# CONSTRAINT QUERIES
# ====================================================================

CONSTRAINT_QUERIES = [
    # Node uniqueness constraints
    "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paperId IS UNIQUE",
    "CREATE CONSTRAINT author_id_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.authorId IS UNIQUE", 
    "CREATE CONSTRAINT venue_name_unique IF NOT EXISTS FOR (v:PubVenue) REQUIRE v.name IS UNIQUE",
    "CREATE CONSTRAINT year_value_unique IF NOT EXISTS FOR (y:PubYear) REQUIRE y.year IS UNIQUE",
    "CREATE CONSTRAINT field_name_unique IF NOT EXISTS FOR (f:Field) REQUIRE f.name IS UNIQUE",
    "CREATE CONSTRAINT institution_name_unique IF NOT EXISTS FOR (i:Institution) REQUIRE i.name IS UNIQUE",
    
    # Property existence constraints (Neo4j 4.0+)
    "CREATE CONSTRAINT paper_title_required IF NOT EXISTS FOR (p:Paper) REQUIRE p.title IS NOT NULL",
    "CREATE CONSTRAINT author_id_required IF NOT EXISTS FOR (a:Author) REQUIRE a.authorId IS NOT NULL"
]

# ====================================================================
# INDEX QUERIES  
# ====================================================================

INDEX_QUERIES = [
    # Full-text search indexes
    "CREATE FULLTEXT INDEX paper_title_fulltext IF NOT EXISTS FOR (p:Paper) ON EACH [p.title, p.abstract]",
    "CREATE FULLTEXT INDEX author_name_fulltext IF NOT EXISTS FOR (a:Author) ON EACH [a.authorName, a.name]",
    
    # Range indexes for numerical queries
    "CREATE INDEX paper_citation_count IF NOT EXISTS FOR (p:Paper) ON (p.citationCount)",
    "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
    "CREATE INDEX author_citation_count IF NOT EXISTS FOR (a:Author) ON (a.citationCount)",
    "CREATE INDEX author_h_index IF NOT EXISTS FOR (a:Author) ON (a.hIndex)",
    
    # Composite indexes for common query patterns
    "CREATE INDEX paper_year_citations IF NOT EXISTS FOR (p:Paper) ON (p.year, p.citationCount)",
    "CREATE INDEX venue_name_idx IF NOT EXISTS FOR (v:PubVenue) ON (v.name)"
]

# ====================================================================
# SCHEMA VALIDATION QUERIES
# ====================================================================

VALIDATION_QUERIES = {
    "check_constraints": "SHOW CONSTRAINTS",
    "check_indexes": "SHOW INDEXES",
    "node_counts": {
        "papers": "MATCH (p:Paper) RETURN count(p) as count",
        "authors": "MATCH (a:Author) RETURN count(a) as count", 
        "venues": "MATCH (v:PubVenue) RETURN count(v) as count",
        "years": "MATCH (y:PubYear) RETURN count(y) as count",
        "fields": "MATCH (f:Field) RETURN count(f) as count",
        "institutions": "MATCH (i:Institution) RETURN count(i) as count"
    },
    "relationship_counts": {
        "citations": "MATCH ()-[:CITES]->() RETURN count(*) as count",
        "authorships": "MATCH ()-[:AUTHORED]->() RETURN count(*) as count",
        "co_authorships": "MATCH ()-[:CO_AUTHORED]->() RETURN count(*) as count",
        "publications": "MATCH ()-[:PUBLISHED_IN]->() RETURN count(*) as count",
        "year_associations": "MATCH ()-[:PUB_YEAR]->() RETURN count(*) as count",
        "field_associations": "MATCH ()-[:IS_ABOUT]->() RETURN count(*) as count",
        "affiliations": "MATCH ()-[:AFFILIATED_WITH]->() RETURN count(*) as count"
    },
    "data_quality": {
        "papers_without_title": "MATCH (p:Paper) WHERE p.title IS NULL OR p.title = '' RETURN count(p) as count",
        "papers_without_year": "MATCH (p:Paper) WHERE p.year IS NULL RETURN count(p) as count", 
        "authors_without_name": "MATCH (a:Author) WHERE (a.authorName IS NULL OR a.authorName = '') AND (a.name IS NULL OR a.name = '') RETURN count(a) as count",
        "duplicate_papers": "MATCH (p:Paper) WITH p.paperId as id, count(*) as cnt WHERE cnt > 1 RETURN count(id) as count",
        "orphaned_papers": "MATCH (p:Paper) WHERE NOT (p)<-[:AUTHORED]-() RETURN count(p) as count"
    }
}

# ====================================================================
# SCHEMA MANAGEMENT CLASS
# ====================================================================

class SchemaManager:
    """
    Manages database schema creation, validation, and migration.
    """
    
    def __init__(self, db_connection):
        """
        Initialize schema manager with database connection.
        
        Args:
            db_connection: Neo4jConnection instance
        """
        self.db = db_connection
        
    def create_constraints(self) -> Dict[str, Any]:
        """
        Create all database constraints.
        
        Returns:
            Dictionary with creation results and any errors
        """
        results = {"created": [], "errors": []}
        
        for query in CONSTRAINT_QUERIES:
            try:
                self.db.execute(query)
                constraint_name = query.split("CREATE CONSTRAINT ")[1].split(" IF NOT EXISTS")[0]
                results["created"].append(constraint_name)
                logger.info(f"Created constraint: {constraint_name}")
            except Exception as e:
                error_msg = f"Failed to create constraint: {query} - Error: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
                
        return results
        
    def create_indexes(self) -> Dict[str, Any]:
        """
        Create all database indexes.
        
        Returns:
            Dictionary with creation results and any errors
        """
        results = {"created": [], "errors": []}
        
        for query in INDEX_QUERIES:
            try:
                self.db.execute(query)
                index_name = query.split("CREATE ")[1].split(" INDEX ")[1].split(" IF NOT EXISTS")[0]
                results["created"].append(index_name)
                logger.info(f"Created index: {index_name}")
            except Exception as e:
                error_msg = f"Failed to create index: {query} - Error: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
                
        return results
        
    def validate_schema(self) -> Dict[str, Any]:
        """
        Validate current database schema against expected structure.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "constraints": {},
            "indexes": {},
            "node_counts": {},
            "relationship_counts": {},
            "data_quality": {}
        }
        
        # Check constraints
        try:
            constraints_df = self.db.query(VALIDATION_QUERIES["check_constraints"])
            validation_results["constraints"] = {
                "total": len(constraints_df),
                "details": constraints_df.to_dict('records') if not constraints_df.empty else []
            }
        except Exception as e:
            validation_results["constraints"]["error"] = str(e)
            
        # Check indexes
        try:
            indexes_df = self.db.query(VALIDATION_QUERIES["check_indexes"])
            validation_results["indexes"] = {
                "total": len(indexes_df),
                "details": indexes_df.to_dict('records') if not indexes_df.empty else []
            }
        except Exception as e:
            validation_results["indexes"]["error"] = str(e)
            
        # Check node counts
        for node_type, query in VALIDATION_QUERIES["node_counts"].items():
            try:
                result = self.db.query(query)
                validation_results["node_counts"][node_type] = int(result.iloc[0]["count"]) if not result.empty else 0
            except Exception as e:
                validation_results["node_counts"][node_type] = f"Error: {str(e)}"
                
        # Check relationship counts
        for rel_type, query in VALIDATION_QUERIES["relationship_counts"].items():
            try:
                result = self.db.query(query)
                validation_results["relationship_counts"][rel_type] = int(result.iloc[0]["count"]) if not result.empty else 0
            except Exception as e:
                validation_results["relationship_counts"][rel_type] = f"Error: {str(e)}"
                
        # Check data quality
        for quality_check, query in VALIDATION_QUERIES["data_quality"].items():
            try:
                result = self.db.query(query)
                validation_results["data_quality"][quality_check] = int(result.iloc[0]["count"]) if not result.empty else 0
            except Exception as e:
                validation_results["data_quality"][quality_check] = f"Error: {str(e)}"
                
        return validation_results
        
    def setup_complete_schema(self) -> Dict[str, Any]:
        """
        Set up complete database schema (constraints + indexes).
        
        Returns:
            Dictionary with setup results
        """
        logger.info("Setting up complete database schema...")
        
        results = {
            "constraints": self.create_constraints(),
            "indexes": self.create_indexes()
        }
        
        # Validate the setup
        validation = self.validate_schema()
        results["validation"] = validation
        
        logger.info("Schema setup completed")
        return results
        
    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get comprehensive schema information.
        
        Returns:
            Dictionary with complete schema details
        """
        return {
            "node_types": NODE_TYPES,
            "relationship_types": RELATIONSHIP_TYPES,
            "validation": self.validate_schema()
        }

# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def get_create_node_query(node_type: str, properties: Dict[str, Any]) -> str:
    """
    Generate Cypher query for creating a node.
    
    Args:
        node_type: Type of node (Paper, Author, etc.)
        properties: Dictionary of properties to set
        
    Returns:
        Cypher CREATE/MERGE query string
    """
    if node_type not in NODE_TYPES:
        raise ValueError(f"Unknown node type: {node_type}")
        
    # Get the primary key for MERGE operation
    node_config = NODE_TYPES[node_type]
    primary_key = node_config["constraints"][0]
    
    if primary_key not in properties:
        raise ValueError(f"Missing required property '{primary_key}' for {node_type}")
        
    # Build MERGE query with primary key
    merge_props = f"{primary_key}: ${primary_key}"
    query = f"MERGE (n:{node_type} {{{merge_props}}})\n"
    
    # Add SET clause for other properties
    set_clauses = []
    for prop, value in properties.items():
        if prop != primary_key:
            set_clauses.append(f"n.{prop} = ${prop}")
            
    if set_clauses:
        query += "SET " + ", ".join(set_clauses)
        
    return query


def get_create_relationship_query(rel_type: str, from_node: Dict[str, Any], 
                                 to_node: Dict[str, Any], rel_props: Dict[str, Any] = None) -> str:
    """
    Generate Cypher query for creating a relationship.
    
    Args:
        rel_type: Type of relationship (CITES, AUTHORED, etc.)
        from_node: Source node info {"type": "Paper", "key": "paperId", "value": "123"}
        to_node: Target node info {"type": "Author", "key": "authorId", "value": "456"}
        rel_props: Optional relationship properties
        
    Returns:
        Cypher MATCH + MERGE query string
    """
    if rel_type not in RELATIONSHIP_TYPES:
        raise ValueError(f"Unknown relationship type: {rel_type}")
        
    query = f"""
    MATCH (from:{from_node['type']} {{{from_node['key']}: $from_value}})
    MATCH (to:{to_node['type']} {{{to_node['key']}: $to_value}})
    MERGE (from)-[r:{rel_type}]->(to)
    """
    
    if rel_props:
        set_clauses = [f"r.{prop} = ${prop}" for prop in rel_props.keys()]
        query += "SET " + ", ".join(set_clauses)
        
    return query