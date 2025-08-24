:param {
  // Define the file path root and the individual file names required for loading.
  // https://neo4j.com/docs/operations-manual/current/configuration/file-locations/
  file_path_root: 'file:///', // Change this to the folder your script can access the files at.
  file_0: 'authors.csv',
  file_1: 'venues.csv',
  file_2: 'papers.csv',
  file_3: 'years.csv',
  file_4: 'fields_of_study.csv',
  file_5: 'co_authorships.csv',
  file_6: 'paper_venues.csv',
  file_7: 'paper_years.csv',
  file_8: 'paper_fields.csv'
};

// CONSTRAINT creation
// -------------------
//
// Create node uniqueness constraints, ensuring no duplicates for the given node label and ID property exist in the database. This also ensures no duplicates are introduced in future.
//
// NOTE: The following constraint creation syntax is generated based on the current connected database version 5.26-aura.
CREATE CONSTRAINT `paperId_Paper_uniq` IF NOT EXISTS
FOR (n: `Paper`)
REQUIRE (n.`paperId`) IS UNIQUE;
CREATE CONSTRAINT `authorId_Author_uniq` IF NOT EXISTS
FOR (n: `Author`)
REQUIRE (n.`authorId`) IS UNIQUE;
CREATE CONSTRAINT `name_PubVenue_uniq` IF NOT EXISTS
FOR (n: `PubVenue`)
REQUIRE (n.`name`) IS UNIQUE;
CREATE CONSTRAINT `year_PubYear_uniq` IF NOT EXISTS
FOR (n: `PubYear`)
REQUIRE (n.`year`) IS UNIQUE;
CREATE CONSTRAINT `name_Field_uniq` IF NOT EXISTS
FOR (n: `Field`)
REQUIRE (n.`name`) IS UNIQUE;

:param {
  idsToSkip: []
};

// NODE load
// ---------
//
// Load nodes in batches, one node label at a time. Nodes will be created using a MERGE statement to ensure a node with the same label and ID property remains unique. Pre-existing nodes found by a MERGE statement will have their other properties set to the latest values encountered in a load file.
//
// NOTE: Any nodes with IDs in the 'idsToSkip' list parameter will not be loaded.
LOAD CSV WITH HEADERS FROM ($file_path_root + $file_2) AS row
WITH row
WHERE NOT row.`paperId` IN $idsToSkip AND NOT row.`paperId` IS NULL
CALL {
  WITH row
  MERGE (n: `Paper` { `paperId`: row.`paperId` })
  SET n.`paperId` = row.`paperId`
  SET n.`title` = row.`title`
  SET n.`referenceCount` = toInteger(trim(row.`referenceCount`))
  SET n.`citationCount` = toInteger(trim(row.`citationCount`))
  // Your script contains the datetime datatype. Our app attempts to convert dates to ISO 8601 date format before passing them to the Cypher function.
  // This conversion cannot be done in a Cypher script load. Please ensure that your CSV file columns are in ISO 8601 date format to ensure equivalent loads.
  SET n.`publicationDate` = datetime(row.`publicationDate`)
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM ($file_path_root + $file_0) AS row
WITH row
WHERE NOT row.`authorId` IN $idsToSkip AND NOT row.`authorId` IS NULL
CALL {
  WITH row
  MERGE (n: `Author` { `authorId`: row.`authorId` })
  SET n.`authorId` = row.`authorId`
  SET n.`name` = row.`name`
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM ($file_path_root + $file_1) AS row
WITH row
WHERE NOT row.`name` IN $idsToSkip AND NOT row.`name` IS NULL
CALL {
  WITH row
  MERGE (n: `PubVenue` { `name`: row.`name` })
  SET n.`name` = row.`name`
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM ($file_path_root + $file_3) AS row
WITH row
WHERE NOT row.`year` IN $idsToSkip AND NOT datetime(row.`year`) IS NULL
CALL {
  WITH row
  MERGE (n: `PubYear` { `year`: datetime(row.`year`) })
  // Your script contains the datetime datatype. Our app attempts to convert dates to ISO 8601 date format before passing them to the Cypher function.
  // This conversion cannot be done in a Cypher script load. Please ensure that your CSV file columns are in ISO 8601 date format to ensure equivalent loads.
  SET n.`year` = datetime(row.`year`)
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM ($file_path_root + $file_4) AS row
WITH row
WHERE NOT row.`name` IN $idsToSkip AND NOT row.`name` IS NULL
CALL {
  WITH row
  MERGE (n: `Field` { `name`: row.`name` })
  SET n.`name` = row.`name`
} IN TRANSACTIONS OF 10000 ROWS;


// RELATIONSHIP load
// -----------------
//
// Load relationships in batches, one relationship type at a time. Relationships are created using a MERGE statement, meaning only one relationship of a given type will ever be created between a pair of nodes.
LOAD CSV WITH HEADERS FROM ($file_path_root + $file_5) AS row
WITH row 
CALL {
  WITH row
  MATCH (source: `Author` { `authorId`: row.`authorId` })
  MATCH (target: `Paper` { `paperId`: row.`paperId` })
  MERGE (source)-[r: `CO_AUTHORED`]->(target)
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM ($file_path_root + $file_6) AS row
WITH row 
CALL {
  WITH row
  MATCH (source: `Paper` { `paperId`: row.`paperId` })
  MATCH (target: `PubVenue` { `name`: row.`venue` })
  MERGE (source)-[r: `PUBLISHED_IN`]->(target)
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM ($file_path_root + $file_7) AS row
WITH row 
CALL {
  WITH row
  MATCH (source: `Paper` { `paperId`: row.`paperId` })
  MATCH (target: `PubYear` { `year`: datetime(row.`year`) })
  MERGE (source)-[r: `PUBLISHED_YEAR`]->(target)
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM ($file_path_root + $file_8) AS row
WITH row 
CALL {
  WITH row
  MATCH (source: `Paper` { `paperId`: row.`paperId` })
  MATCH (target: `Field` { `name`: row.`field` })
  MERGE (source)-[r: `IS_ABOUT`]->(target)
} IN TRANSACTIONS OF 10000 ROWS;
