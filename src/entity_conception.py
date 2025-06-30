# entity_conception.py

import json
from google.generativeai.generative_models import GenerativeModel
import streamlit as st
from typing import Dict, List, Any
import pandas as pd

from rag_search import get_rag_searcher

# --- [NEW] Prior Knowledge: A curated list of common, high-level ontology classes ---
# This guides the LLM to select the most appropriate class for each entity.
RELEVANT_ENTITY_CLASSES = [
    # Core Financial & Business Entities from FIBO
    "fibo-fbc-fct-fse:FinancialInstitution", # For the parent company
    "cmns-org:Organization", # A general business organization
    "fibo-be-le-fbo:Branch", # For a branch office

    # Location and Address Entities from FIBO & Schema.org
    "fibo-fnd-plc-adr:PhysicalAddress", # A physical address
    "schema:PostalAddress", # A more general postal address

    # Geospatial Entities from GeoSPARQL & Commons
    "geo:Feature", # A generic geographic feature
    "geo:hasGeometry", # A generic geometry
    "sf:Point", # A specific point geometry
    "cmns-loc:Location", # A location as defined in OMG Commons

    # Statistical Area Entities from FIBO
    "fibo-ind-ei-ei:CombinedStatisticalArea",
    "fibo-fnd-utl-alx:StatisticalArea"
    "fibo-ind-ei-ei:MetropolitanStatisticalArea",
    "fibo-ind-ei-ei:MicropolitanStatisticalArea"
    "fibo-be-le-fbo:Division",

    # General Purpose Entities
    "dcat:Dataset", # For representing the dataset itself
    "skos:Concept", # For classification concepts like status or type
    "cmns-cls:Classifier" # A general classifier
]

def get_column_samples(df: pd.DataFrame, columns: List[str], n_samples: int = 3) -> Dict[str, List[Any]]:
    """
    Extracts a few sample values for specified columns from the DataFrame.
    """
    samples = {}
    for col in columns:
        if col in df:
            # Drop NA/None values before getting unique samples
            valid_samples = df[col].dropna().unique().tolist()
            samples[col] = valid_samples[:n_samples]
    return samples

def conceptualize_entity(
    cluster_name: str,
    columns: List[str],
    column_samples: Dict[str, List[Any]],
    rag_results: List[str] # Dynamic RAG search results
) -> Dict[str, Any]:
    """
    Builds the mega-prompt and calls the LLM to conceptualize an entity.

    Args:
        cluster_name (str): The name of the column cluster.
        columns (List[str]): The list of columns in the cluster.
        column_samples (Dict[str, List[Any]]): Sample data for each column.
        rag_results (List[str]): A list of relevant ontology terms from a RAG system.

    Returns:
        Dict[str, Any]: A dictionary representing the AI-generated entity definition.
    """

    model = GenerativeModel('gemini-2.0-flash-exp')

    # Dynamically build the field descriptions and examples section
    field_descriptions = "\n".join(
        f'- `{col}`: e.g., "{", ".join(map(str, samples))}"'
        for col, samples in column_samples.items()
    )
    
    # Format both knowledge sources for the prompt
    formatted_rag_results = "\n".join(f"- `{result}`" for result in rag_results)
    if not formatted_rag_results:
        formatted_rag_results = "No specific terms found in the real-time search."
        
    formatted_static_list = "\n".join(f"- `{cls}`" for cls in RELEVANT_ENTITY_CLASSES)

    prompt = f"""
# ROLE
You are a top-tier ontology modeling expert.

# TASK
Your goal is to define the most appropriate core business entity based on the provided information. You have TWO sources of knowledge to help you: real-time search results and a curated list of preferred classes.

# 1. COLUMN CLUSTER INFORMATION
- Cluster Name: "{cluster_name}"
- Included Columns: {columns}
- Column Examples:
{field_descriptions}

# 2. RELEVANT KNOWLEDGE BASE TERMS (from real-time RAG Search)
These terms were dynamically retrieved from the vector database as being potentially relevant:
{formatted_rag_results}

# 3. HIGHLY-RECOMMENDED ENTITY CLASSES (Prior Knowledge)
This is a curated list of high-quality, preferred classes. Your final choice should ideally come from this list unless the RAG search provides a clearly superior alternative.
{formatted_static_list}

# 4. YOUR INFERENCE AND DECISION
Synthesize all the above information. Give priority to the "Highly-Recommended" list, but consider the "RAG Search" results for more specific contexts. Return a single JSON object with the following keys:
- `entityId`: A Camel-cased ID ending with "Entity".
- `entityLabel`: A concise English label.
- `entityComment`: An English comment explaining the entity's purpose.
- `mapsToClass`: The single best ontology class URI, chosen based on your expert analysis of both knowledge sources.
"""

    try:
        response = model.generate_content(prompt)
        cleaned_json_str = response.text.strip().lstrip('```json').rstrip('```').strip()
        result = json.loads(cleaned_json_str)
        result['clusterName'] = cluster_name
        return result
    except Exception as e:
        st.error(f"Error calling Gemini API for cluster '{cluster_name}': {e}")
        return {
            "error": str(e),
            "clusterName": cluster_name
        }

def process_clusters_to_entities(
    df: pd.DataFrame,
    clusters: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Iterates through all column clusters and runs the entity conception process for each.
    """
    all_entities = []
    rag_searcher = get_rag_searcher()
    
    if not rag_searcher.collection:
        st.error("Cannot proceed with entity conception as the RAG database is unavailable.")
        return []
    
    progress_bar = st.progress(0, text="Conceptualizing entities...")
    
    for i, cluster in enumerate(clusters):
        name = cluster.get('name', f"Unnamed Cluster {i+1}")
        cols = cluster.get('columns', [])
        
        # 1. Create a query string and perform the RAG search
        query_text = f"Entity related to: {name}, with columns such as {', '.join(cols)}"
        rag_results = rag_searcher.search(query_text, n_results=5)
        
        # 2. Get column samples
        samples = get_column_samples(df, cols)
        
        # 3. Call the LLM with BOTH the dynamic RAG results and the static list
        entity_definition = conceptualize_entity(name, cols, samples, rag_results)
        all_entities.append(entity_definition)
        
        progress_bar.progress((i + 1) / len(clusters), text=f"Conceptualized: {name}")
        
    return all_entities
