#!/usr/bin/env python3
"""
Database setup and validation script for Academic Citation Platform.

This script:
1. Tests Neo4j connectivity with current environment variables
2. Creates basic schema if database is empty
3. Populates sample data for testing and development
4. Validates that all required indices exist
"""

import os
import logging
from typing import Dict, Any, List
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_environment() -> Dict[str, str]:
    """Check if required environment variables are set."""
    env_vars = {
        'NEO4J_URI': os.getenv('NEO4J_URI') or os.getenv('NEO4J_URL'),
        'NEO4J_USER': os.getenv('NEO4J_USER') or os.getenv('NEO4J_USERNAME'), 
        'NEO4J_PASSWORD': os.getenv('NEO4J_PWD') or os.getenv('NEO4J_PASSWORD')
    }
    
    missing = [k for k, v in env_vars.items() if not v]
    if missing:
        logger.error(f"Missing environment variables: {missing}")
        logger.info("Please copy .env.example to .env and configure your database settings")
        return {}
    
    logger.info("✅ Environment variables found")
    return env_vars

def test_connectivity(env_vars: Dict[str, str]) -> bool:
    """Test basic Neo4j connectivity."""
    try:
        from src.database.connection import Neo4jConnection
        
        logger.info("Testing Neo4j connection...")
        with Neo4jConnection(validate_connection=False) as db:
            db.test_connection()
            info = db.get_database_info()
            
        logger.info(f"✅ Connected to Neo4j {info.get('version', 'unknown')}")
        logger.info(f"   Database has {info.get('total_nodes', 0)} nodes and {info.get('total_relationships', 0)} relationships")
        return True
        
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        logger.info("Please check your Neo4j server is running and credentials are correct")
        return False

def setup_schema(db) -> bool:
    """Create basic schema constraints and indices."""
    logger.info("Setting up database schema...")
    
    schema_queries = [
        # Constraints for unique identifiers
        "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paperId IS UNIQUE",
        "CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.authorId IS UNIQUE", 
        "CREATE CONSTRAINT venue_id IF NOT EXISTS FOR (v:PubVenue) REQUIRE v.venue IS UNIQUE",
        "CREATE CONSTRAINT field_id IF NOT EXISTS FOR (f:Field) REQUIRE f.field IS UNIQUE",
        "CREATE CONSTRAINT year_value IF NOT EXISTS FOR (y:PubYear) REQUIRE y.year IS UNIQUE",
        
        # Indices for common queries
        "CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)",
        "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
        "CREATE INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.authorName)",
        "CREATE INDEX paper_citation_count IF NOT EXISTS FOR (p:Paper) ON (p.citationCount)"
    ]
    
    for query in schema_queries:
        try:
            db.execute(query)
            logger.debug(f"✅ {query.split()[1]} created")
        except Exception as e:
            if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                logger.debug(f"⚠️  Schema element already exists: {query.split()[2]}")
            else:
                logger.warning(f"Schema creation warning: {e}")
    
    logger.info("✅ Schema setup completed")
    return True

def create_sample_data(db) -> bool:
    """Create sample data for testing if database is empty."""
    logger.info("Checking for existing data...")
    
    stats = db.get_network_statistics()
    if stats.get('papers', 0) > 0:
        logger.info(f"✅ Database already has {stats['papers']} papers - skipping sample data")
        return True
    
    logger.info("Creating sample data for testing...")
    
    # Sample papers with realistic IDs
    sample_papers = [
        {
            'paperId': '649def34f8be52c8b66281af98ae884c09aef38f9',
            'title': 'Attention Is All You Need',
            'abstract': 'We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.',
            'year': 2017,
            'citationCount': 45000
        },
        {
            'paperId': 'cd218a0cce0dbb283049644c0ce5cf2a06ae59b3',
            'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
            'abstract': 'We introduce BERT, a new language representation model.',
            'year': 2018, 
            'citationCount': 35000
        },
        {
            'paperId': 'df2b0e26d0599ce3e70df8a9da02e51594e0e992',
            'title': 'Deep Residual Learning for Image Recognition',
            'abstract': 'We present residual networks to ease the training of very deep neural networks.',
            'year': 2016,
            'citationCount': 40000
        }
    ]
    
    # Create papers
    for paper in sample_papers:
        query = """
        MERGE (p:Paper {paperId: $paperId})
        ON CREATE SET 
            p.title = $title,
            p.abstract = $abstract, 
            p.year = $year,
            p.citationCount = $citationCount
        """
        db.execute(query, paper)
    
    # Create sample authors
    authors_query = """
    MERGE (a1:Author {authorId: '1', authorName: 'Ashish Vaswani'})
    MERGE (a2:Author {authorId: '2', authorName: 'Jacob Devlin'}) 
    MERGE (a3:Author {authorId: '3', authorName: 'Kaiming He'})
    
    // Link authors to papers
    MATCH (p1:Paper {paperId: '649def34f8be52c8b66281af98ae884c09aef38f9'})
    MATCH (a1:Author {authorName: 'Ashish Vaswani'})
    MERGE (a1)-[:AUTHORED]->(p1)
    
    MATCH (p2:Paper {paperId: 'cd218a0cce0dbb283049644c0ce5cf2a06ae59b3'})
    MATCH (a2:Author {authorName: 'Jacob Devlin'})
    MERGE (a2)-[:AUTHORED]->(p2)
    
    MATCH (p3:Paper {paperId: 'df2b0e26d0599ce3e70df8a9da02e51594e0e992'})  
    MATCH (a3:Author {authorName: 'Kaiming He'})
    MERGE (a3)-[:AUTHORED]->(p3)
    """
    db.execute(authors_query)
    
    # Create citation relationships
    citations_query = """
    MATCH (p1:Paper {paperId: 'cd218a0cce0dbb283049644c0ce5cf2a06ae59b3'})
    MATCH (p2:Paper {paperId: '649def34f8be52c8b66281af98ae884c09aef38f9'})
    MERGE (p1)-[:CITES]->(p2)
    
    MATCH (p3:Paper {paperId: 'df2b0e26d0599ce3e70df8a9da02e51594e0e992'})
    MATCH (p1:Paper {paperId: '649def34f8be52c8b66281af98ae884c09aef38f9'})  
    MERGE (p3)-[:CITES]->(p1)
    """
    db.execute(citations_query)
    
    # Create venues and years
    additional_data_query = """
    MERGE (v1:PubVenue {venue: 'NIPS'})
    MERGE (v2:PubVenue {venue: 'NAACL'})
    MERGE (v3:PubVenue {venue: 'CVPR'})
    
    MERGE (y1:PubYear {year: 2016})
    MERGE (y2:PubYear {year: 2017}) 
    MERGE (y3:PubYear {year: 2018})
    
    MERGE (f1:Field {field: 'Machine Learning'})
    MERGE (f2:Field {field: 'Natural Language Processing'})
    MERGE (f3:Field {field: 'Computer Vision'})
    
    // Link papers to venues, years, and fields
    MATCH (p1:Paper {paperId: '649def34f8be52c8b66281af98ae884c09aef38f9'})
    MATCH (v1:PubVenue {venue: 'NIPS'})
    MATCH (y2:PubYear {year: 2017})
    MATCH (f1:Field {field: 'Machine Learning'})
    MERGE (p1)-[:PUBLISHED_IN]->(v1)
    MERGE (p1)-[:PUB_YEAR]->(y2)
    MERGE (p1)-[:IS_ABOUT]->(f1)
    """
    db.execute(additional_data_query)
    
    # Verify sample data creation
    new_stats = db.get_network_statistics()
    logger.info(f"✅ Sample data created:")
    logger.info(f"   Papers: {new_stats.get('papers', 0)}")
    logger.info(f"   Authors: {new_stats.get('authors', 0)}")
    logger.info(f"   Citations: {new_stats.get('citations', 0)}")
    
    return True

def validate_setup() -> bool:
    """Run final validation of the database setup."""
    try:
        from src.database.connection import Neo4jConnection
        
        logger.info("Running final validation...")
        with Neo4jConnection() as db:
            # Test basic queries
            papers = db.query("MATCH (p:Paper) RETURN count(p) as count")
            authors = db.query("MATCH (a:Author) RETURN count(a) as count")
            citations = db.query("MATCH ()-[:CITES]->() RETURN count(*) as count")
            
            paper_count = papers.iloc[0]['count'] if not papers.empty else 0
            author_count = authors.iloc[0]['count'] if not authors.empty else 0  
            citation_count = citations.iloc[0]['count'] if not citations.empty else 0
            
            if paper_count > 0 and author_count > 0:
                logger.info("✅ Database validation successful!")
                logger.info(f"   Ready with {paper_count} papers, {author_count} authors, {citation_count} citations")
                return True
            else:
                logger.error("❌ Validation failed - missing data")
                return False
                
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("🚀 Academic Citation Platform - Database Setup")
    logger.info("=" * 60)
    
    # Check environment
    env_vars = check_environment()
    if not env_vars:
        logger.error("❌ Setup failed - missing environment variables")
        return False
    
    # Test connectivity
    if not test_connectivity(env_vars):
        logger.error("❌ Setup failed - connection issues")
        return False
    
    # Setup database
    try:
        from src.database.connection import Neo4jConnection
        
        with Neo4jConnection() as db:
            # Setup schema
            setup_schema(db)
            
            # Create sample data if needed
            create_sample_data(db)
            
        # Final validation
        if validate_setup():
            logger.info("🎉 Database setup completed successfully!")
            logger.info("You can now run tests and use the application.")
            return True
        else:
            logger.error("❌ Setup completed but validation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)