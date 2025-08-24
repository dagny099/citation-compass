import os
import pandas as pd
from neo4j import GraphDatabase
from scipy.stats import linregress

# -----------------------------------------------------------------------------
# Configuration: Read connection details from environment variables
# -----------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PWD = os.getenv("NEO4J_PWD")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PWD]):
    raise EnvironmentError("Please set NEO4J_URI, NEO4J_USER, and NEO4J_PWD environment variables.")

# -----------------------------------------------------------------------------
# Establish Neo4j Driver
# -----------------------------------------------------------------------------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))


# -----------------------------------------------------------------------------
# Helper function to run a Cypher query and return a pandas DataFrame
# -----------------------------------------------------------------------------
def run_query(cypher_query: str, parameters: dict = None) -> pd.DataFrame:
    """
    Executes a Cypher query against the Neo4j database and returns the result as a pandas DataFrame.

    Args:
        cypher_query (str): The Cypher query string.
        parameters (dict, optional): Dictionary of parameters for parameterized queries.

    Returns:
        pd.DataFrame: A DataFrame containing the query results.
    """
    with driver.session() as session:
        result = session.run(cypher_query, parameters or {})
        df = pd.DataFrame([record.data() for record in result])
    return df


# -----------------------------------------------------------------------------
# Insight Functions
# -----------------------------------------------------------------------------
def get_top_authors(limit: int = 10) -> pd.DataFrame:
    """
    Retrieves the top authors by total citation count of their co-authored papers.
    Returns a DataFrame with columns: author, totalCitations, numPapers
    """
    cypher = f"""
    MATCH (a:Author)-[:CO_AUTHORED]->(p:Paper)
    WHERE p.citationCount IS NOT NULL
    WITH a.name AS author, SUM(p.citationCount) AS totalCitations, COUNT(p) AS numPapers
    ORDER BY totalCitations DESC
    LIMIT {limit}
    RETURN author, totalCitations, numPapers;
    """
    return run_query(cypher)


def get_top_venues(limit: int = 10) -> pd.DataFrame:
    """
    Retrieves the top publication venues by average citation count per paper.
    Returns a DataFrame with columns: venue, paperCount, avgCitations
    """
    cypher = f"""
    MATCH (p:Paper)-[:PUBLISHED_IN]->(v:PubVenue)
    WHERE p.citationCount IS NOT NULL
    WITH v.name AS venue, COUNT(p) AS paperCount, AVG(p.citationCount) AS avgCitations
    ORDER BY avgCitations DESC
    LIMIT {limit}
    RETURN venue, paperCount, avgCitations;
    """
    return run_query(cypher)


def get_coauthors(author_name: str, limit: int = 10) -> pd.DataFrame:
    """
    Finds the most common co-authors for a given author.
    Args:
        author_name (str): Exact name of the author.
        limit (int): Number of co-authors to return.
    Returns a DataFrame with columns: coAuthor, sharedPapers.
    """
    cypher = f"""
    MATCH (a1:Author {{name: $author_name}})-[:CO_AUTHORED]->(p:Paper)<-[:CO_AUTHORED]-(a2:Author)
    WHERE a1 <> a2
    WITH a2.name AS coAuthor, COUNT(p) AS sharedPapers
    ORDER BY sharedPapers DESC
    LIMIT {limit}
    RETURN coAuthor, sharedPapers;
    """
    return run_query(cypher, parameters={"author_name": author_name})


def get_citation_histogram() -> pd.DataFrame:
    """
    Retrieves citation counts for all papers to construct a histogram.
    Returns a DataFrame with column: citations
    """
    cypher = """
    MATCH (p:Paper)
    WHERE p.citationCount IS NOT NULL
    RETURN p.citationCount AS citations;
    """
    return run_query(cypher)


def get_yearly_author_citations() -> pd.DataFrame:
    """
    Retrieves total citation counts per author per publication year.
    Returns a DataFrame with columns: author, year, yearlyCitations
    """
    cypher = """
    MATCH (a:Author)-[:CO_AUTHORED]->(p:Paper)
    WHERE p.publicationDate IS NOT NULL AND p.citationCount IS NOT NULL
    WITH a.name AS author, date(p.publicationDate).year AS year, SUM(p.citationCount) AS yearlyCitations
    RETURN author, year, yearlyCitations
    ORDER BY author, year;
    """
    return run_query(cypher)


def get_rising_stars(min_years: int = 3, top_n: int = 10) -> pd.DataFrame:
    """
    Identifies authors whose citation counts are increasing over time ("rising stars").
    Args:
      min_years (int): Minimum distinct years of data to consider.
      top_n (int): Number of top rising authors to return.
    Returns a DataFrame with columns: author, growth_rate
    """
    # Fetch yearly citations per author
    yearly_df = get_yearly_author_citations()

    # Compute growth rates via linear regression
    rising_list = []
    for author, group in yearly_df.groupby("author"):
        if group["year"].nunique() >= min_years:
            slope, _, _, _, _ = linregress(group["year"], group["yearlyCitations"])
            if slope > 0:
                rising_list.append((author, slope))

    rising_stars_df = (
        pd.DataFrame(rising_list, columns=["author", "growth_rate"])
          .sort_values("growth_rate", ascending=False)
          .head(top_n)
    )
    return rising_stars_df


# -----------------------------------------------------------------------------
# Main: Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Top 10 authors by total citation count
    print("► Top 10 Authors by Total Citations")
    df_top_authors = get_top_authors(10)
    print(df_top_authors.to_string(index=False), "\n")

    # 2) Top 10 venues by average citations per paper
    print("► Top 10 Venues by Average Citations")
    df_top_venues = get_top_venues(10)
    print(df_top_venues.to_string(index=False), "\n")

    # 3) Example: Find top co-authors for a specific author
    example_author = "Barbara Hidalgo-Sotelo"
    print(f"► Top Co-authors for “{example_author}”")
    df_coauthors = get_coauthors(example_author, limit=5)
    print(df_coauthors.to_string(index=False), "\n")

    # 4) Citation histogram data (to be used for plotting later)
    print("► Citation Counts (first 10 rows)")
    df_hist = get_citation_histogram()
    print(df_hist.head(10).to_string(index=False), "\n")

    # 5) Rising stars: authors whose citations are trending upward
    print("► Rising Star Authors (growth rate of citations over time)")
    df_rising = get_rising_stars(min_years=3, top_n=10)
    print(df_rising.to_string(index=False), "\n")

    # Close the Neo4j driver when done
    driver.close()
