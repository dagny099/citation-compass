
// ===============================================
// Section 1: Establish Constraints 🎯
// ===============================================
CREATE CONSTRAINT author_id_unique IF NOT EXISTS
FOR (a:Author) REQUIRE a.authorId IS UNIQUE;

CREATE CONSTRAINT paper_id_unique IF NOT EXISTS
FOR (p:Paper) REQUIRE p.paperId IS UNIQUE;

CREATE CONSTRAINT venue_name_unique IF NOT EXISTS
FOR (v:PubVenue) REQUIRE v.name IS UNIQUE;

CREATE CONSTRAINT year_value_unique IF NOT EXISTS
FOR (y:PubYear) REQUIRE y.year IS UNIQUE;

CREATE CONSTRAINT field_name_unique IF NOT EXISTS
FOR (f:Field) REQUIRE f.name IS UNIQUE;

// ===============================================
// Section 2: Import Author Nodes ✍️
// ===============================================
LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/dagny099/PyMesh/refs/heads/main/data/authors.csv' AS row
WITH row WHERE row.authorId IS NOT NULL AND row.author IS NOT NULL
MERGE (:Author {authorId: row.authorId, name: row.author});

// ===============================================
// Section 3: Import Paper Nodes 📄
// Conditionally include publicationDate only if it exists

LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/dagny099/PyMesh/refs/heads/main/data/papers.csv' AS row
WITH row 
WHERE row.paperId IS NOT NULL AND row.title IS NOT NULL
MERGE (p:Paper {paperId: row.paperId})
SET p.title = row.title,
    p.referenceCount = toInteger(row.referenceCount),
    p.citationCount = toInteger(row.citationCount)
WITH p, row
WHERE row.publicationDate IS NOT NULL
SET p.publicationDate = row.publicationDate;

// ===============================================
// Section 4: Import Venue Nodes & Relationships 🏛️
LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/dagny099/PyMesh/refs/heads/main/data/venues.csv' AS row
WITH row WHERE row.name IS NOT NULL
MERGE (:PubVenue {name: row.name});

LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/dagny099/PyMesh/refs/heads/main/data/paper_venues.csv' AS row
WITH row WHERE row.paperId IS NOT NULL AND row.venue IS NOT NULL
MATCH (p:Paper {paperId: row.paperId})
MERGE (v:PubVenue {name: row.venue})
MERGE (p)-[:PUBLISHED_IN]->(v);

// ===============================================
// Section 5: Import Year Nodes & Relationships 📆
LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/dagny099/PyMesh/refs/heads/main/data/years.csv' AS row
WITH row WHERE row.year IS NOT NULL
MERGE (:PubYear {year: row.year});

LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/dagny099/PyMesh/refs/heads/main/data/paper_years.csv' AS row
WITH row WHERE row.paperId IS NOT NULL AND row.year IS NOT NULL
MATCH (p:Paper {paperId: row.paperId})
MERGE (y:PubYear {year: row.year})
MERGE (p)-[:PUB_YEAR]->(y);

// ===============================================
// Section 6: Import Fields of Study & Relationships 🧠
LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/dagny099/PyMesh/refs/heads/main/data/fields_of_study.csv' AS row
WITH row WHERE row.field IS NOT NULL
MERGE (:Field {name: row.field});

LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/dagny099/PyMesh/refs/heads/main/data/paper_fields.csv' AS row
WITH row WHERE row.paperId IS NOT NULL AND row.field IS NOT NULL
MATCH (p:Paper {paperId: row.paperId})
MERGE (f:Field {name: row.field})
MERGE (p)-[:IS_ABOUT]->(f);

// ===============================================
// Section 7: Co-Authorship Relationships 🤝
LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/dagny099/PyMesh/refs/heads/main/data/co_authorships.csv' AS row
WITH row WHERE row.paperId IS NOT NULL AND row.authorId IS NOT NULL
MATCH (p:Paper {paperId: row.paperId})
MATCH (a:Author {authorId: row.authorId})
MERGE (a)-[:CO_AUTHORED]->(p);

// ===============================================
// Section 8: Cleanup (Optional) 🧹
/*
To delete all nodes and relationships:
MATCH (n) DETACH DELETE n;

To delete all constraints:
SHOW CONSTRAINTS YIELD name
CALL {
  WITH name
  CALL db.constraints.drop(name)
  RETURN count(*) AS dropped
}
RETURN sum(dropped) AS totalDropped;

DROP CONSTRAINT authorId_Author_uniq IF EXISTS;
DROP CONSTRAINT fieldName_Field_uniq IF EXISTS;
DROP CONSTRAINT name_PubVenue_uniq IF EXISTS;
DROP CONSTRAINT paperId_Paper_uniq IF EXISTS;
DROP CONSTRAINT year_PubYear_uniq IF EXISTS;
*/
