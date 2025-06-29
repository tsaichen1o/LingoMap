# column_clusterer.py

import json
import logging
import pandas as pd
from google.generativeai.generative_models import GenerativeModel
import streamlit as st
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Prior knowledge: A dictionary of column names and their descriptions.
# This provides the semantic context the LLM will use for clustering.
COLUMN_DESCRIPTIONS = {
    "ACQDATE": "Acquisition Date of the branch",
    "ADDRESS": "Primary street address of the branch",
    "ADDRESS2": "Secondary street address information (e.g., suite number)",
    "BKCLASS": "Institution Class or Type (e.g., National Bank, State Member Bank)",
    "CBSA": "Core Based Statistical Area Name",
    "CBSA_DIV": "Metropolitan Division Name within a CBSA",
    "CBSA_DIV_FLG": "Flag indicating if the branch is in a Metropolitan Division",
    "CBSA_DIV_NO": "Metropolitan Division Number",
    "CBSA_METRO": "Metropolitan Statistical Area Name",
    "CBSA_METRO_FLG": "Flag indicating if the branch is in a Metropolitan Area",
    "CBSA_METRO_NAME": "Metropolitan Statistical Area Name",
    "CBSA_MICRO_FLG": "Flag indicating if the branch is in a Micropolitan Area",
    "CBSA_NO": "Core Based Statistical Area Number",
    "CERT": "The unique FDIC Certificate number for the parent institution",
    "CITY": "Branch's city name",
    "COUNTY": "Branch's county name",
    "CSA": "Combined Statistical Area Name",
    "CSA_FLG": "Flag indicating if the branch is in a Combined Statistical Area",
    "CSA_NO": "Combined Statistical Area Number",
    "ESTYMD": "Date the branch was established",
    "FI_UNINUM": "FDIC unique number for the parent financial institution",
    "ID": "Generic record identifier",
    "LATITUDE": "Latitude coordinate of the branch",
    "LONGITUDE": "Longitude coordinate of the branch",
    "MAINOFF": "Flag indicating if this is the main office",
    "MDI_STATUS_CODE": "Minority Depository Institution Status Code",
    "MDI_STATUS_DESC": "Description of the Minority Depository Institution Status",
    "NAME": "Official name of the parent financial institution",
    "OFFNAME": "Official name of the branch office",
    "OFFNUM": "Branch office number, assigned by the institution",
    "RUNDATE": "The date the data report was generated",
    "SERVTYPE": "Service Type Code (e.g., Full Service, Limited Service)",
    "SERVTYPE_DESC": "Description of the branch's service type",
    "STALP": "Branch's state abbreviation (e.g., CA, NY)",
    "STCNTY": "FIPS code for the state and county",
    "STNAME": "Branch's full state name",
    "UNINUM": "FDIC unique identification number for a branch office",
    "ZIP": "Branch's postal ZIP code",
    "X": "X Coordinate, often redundant with Longitude",
    "Y": "Y Coordinate, often redundant with Latitude",
    "OBJECTID": "Internal unique ID for a row or feature"
}


def cluster_columns_with_llm(all_columns: List[str], n_clusters: int) -> List[Dict[str, Any]]:
    """
    Uses a Large Language Model to cluster columns based on their names and descriptions.

    Args:
        all_columns (List[str]): The list of column names from the DataFrame.
        n_clusters (int): The desired number of clusters.

    Returns:
        A list of dictionaries, where each dictionary represents a named cluster
        and contains the list of columns belonging to it.
    """
    logging.info(f"Starting LLM-based clustering for {len(all_columns)} columns into {n_clusters} clusters.")

    model = GenerativeModel('gemini-2.0-flash-exp')

    # Filter descriptions to only include columns present in the uploaded data
    available_descriptions = {col: desc for col, desc in COLUMN_DESCRIPTIONS.items() if col in all_columns}
    
    # Format the list of columns and their descriptions for the prompt
    formatted_column_list = "\n".join(
        f'- `{col}`: {desc}' for col, desc in available_descriptions.items()
    )

    # Design the prompt for the LLM
    prompt = f"""
# ROLE
You are an expert data architect and ontologist specializing in financial and geographic data.

# TASK
Your task is to group the following list of data columns into {n_clusters} semantically coherent clusters.
Analyze the column names and their descriptions to understand their meaning and relationship.
For example, columns related to addresses (`ADDRESS`, `CITY`, `STALP`, `ZIP`) should be in one group.
Columns related to the parent company (`NAME`, `CERT`) should be in another.

# INPUT: COLUMN LIST & DESCRIPTIONS
{formatted_column_list}

# INSTRUCTIONS
1.  Create exactly {n_clusters} clusters.
2.  Assign each column from the input list to one of the clusters.
3.  Give each cluster a short, descriptive name (e.g., "Institution Identity", "Branch Location", "Geospatial Coordinates").
4.  Return the result as a single, valid JSON object. The format must be a list of dictionaries,
    where each dictionary has two keys: "name" (the cluster name) and "columns" (a list of column names).
5.  Do not include any columns that were not in the input list.

# OUTPUT FORMAT (EXAMPLE)
[
  {{
    "name": "Institution Identity",
    "columns": ["CERT", "NAME", "BKCLASS"]
  }},
  {{
    "name": "Branch Location",
    "columns": ["ADDRESS", "CITY", "STALP", "ZIP", "COUNTY"]
  }}
]
"""

    try:
        logging.info("Sending clustering prompt to Gemini...")
        response = model.generate_content(prompt)
        
        # Clean and parse the JSON response
        cleaned_json_str = response.text.strip().lstrip('```json').rstrip('```').strip()
        clusters = json.loads(cleaned_json_str)
        
        # Basic validation of the returned structure
        if isinstance(clusters, list) and all(isinstance(c, dict) and 'name' in c and 'columns' in c for c in clusters):
            logging.info("Successfully received and parsed valid clusters from LLM.")
            return clusters
        else:
            logging.error("LLM returned a malformed JSON structure for clusters.")
            return []
            
    except Exception as e:
        logging.error(f"An error occurred during LLM clustering: {e}")
        st.error(f"Failed to get clusters from the LLM. Error: {e}")
        return []

# This function now acts as a wrapper to call the new LLM-based method.
# This ensures that the rest of your application (`core_engine.py`) doesn't need to change its call signature.
def cluster_columns(df: pd.DataFrame, n_clusters: int = 8) -> List[Dict[str, Any]]:
    """
    Main entry point for column clustering. This now uses the LLM-based approach.
    """
    return cluster_columns_with_llm(all_columns=df.columns.tolist(), n_clusters=n_clusters)