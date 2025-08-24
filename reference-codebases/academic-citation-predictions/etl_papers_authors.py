import ast
import pandas as pd
import numpy as np


def transform_fields_of_study(data):
    """Extract Fields of Study and their relationships to Papers."""
    fields = []
    paper_fields = []
    for item in data:
        #Weirdly some elements are strings...
        if type(item)!=dict:
            item = item[0]
            if item=='error':
                continue
            
        paper_id = item.get("paperId", '')
    
        if type(item['fieldsOfStudy'])==str:
            tmp = ast.literal_eval(item.get("fieldsOfStudy", []))
        else:
            tmp = item.get("fieldsOfStudy", [])    
        
        try:
            for field in tmp:
                fields.append({"name": field})
                paper_fields.append({"paperId": paper_id, "field": field})
        except (TypeError, KeyError):
            pass
    
    return pd.DataFrame(fields).drop_duplicates(), pd.DataFrame(paper_fields)


def transform_authors(data):
    """Extract Author nodes and their relationships to Papers."""
    authors = []
    co_authorships = []
    for item in data:
        #Weirdly some elements are strings...
        if type(item)!=dict:
            item = item[0]
            if item=='error':
                continue
            
        paper_id = item.get("paperId", '')
    
        if type(item['authors'])==str:
            tmp = ast.literal_eval(item.get("authors", []))
        else:
            tmp = item.get("authors", [])    
        
        try:
            # Attempt to iterate over authors
            for author in tmp:
                authors.append({
                    "authorId": author["authorId"],
                    "name": author["name"]
                })
                co_authorships.append({
                    "paperId": paper_id,
                    "authorId": author["authorId"]
                })
        except (TypeError, KeyError):
            pass
        
    return pd.DataFrame(authors).drop_duplicates(), pd.DataFrame(co_authorships)
        

# Transformation functions
def transform_papers(data):
    """Extract Paper nodes."""
    papers = []
    for item in data:
        if type(item)==dict:
            papers.append({
                    "paperId": item.get('paperId',''),
                    "title": item.get("title", ''),
                    "referenceCount": item.get("referenceCount", 0),
                    "citationCount": item.get("citationCount", 0),
                    "publicationDate": item.get("publicationDate", 0),
                    "venue": item.get("venue", ''),
                    "year": item.get("year", ''),
                    "fieldsOfStudy": item.get("fieldsOfStudy", [])
                })

    return pd.DataFrame(papers)


def transform_venues(data):
    """Extract unique nodes and their relationships to Papers."""
    venues = []
    paper_venues = []
    for item in data:
        #Weirdly some elements are strings...
        if type(item)!=dict:
            item = item[0]
            if item=='error':
                continue
        
        paper_id = item.get("paperId", '')
        venue = item.get("venue", '')
        if venue:  # Skip if venue is empty
            venues.append({"name": venue})
            paper_venues.append({
                "paperId": item["paperId"],
                "venue": venue
            })
    return pd.DataFrame(venues).drop_duplicates(), pd.DataFrame(paper_venues)


def transform_years(data):
    """Extract unique PubYear nodes and their relationships to Papers."""
    years = []
    paper_years = []
    for item in data:
        #Weirdly some elements are strings...
        if type(item)!=dict:
            item = item[0]
            if item=='error':
                continue

        year = item.get("year")
        if year:  # Skip if year is empty or null
            years.append({"year": year})
            paper_years.append({
                "paperId": item["paperId"],
                "year": year
            })

    # Create DataFrames
    years_df = pd.DataFrame(years).drop_duplicates()
    paper_years_df = pd.DataFrame(paper_years)

    # Ensure the 'year' column is stored as an integer
    years_df["year"] = pd.to_numeric(years_df["year"], errors="coerce").astype("Int64")
    paper_years_df["year"] = pd.to_numeric(paper_years_df["year"], errors="coerce").astype("Int64")

    return years_df, paper_years_df
