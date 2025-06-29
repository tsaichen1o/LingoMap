# entity_conception.py

import json
from google.generativeai.generative_models import GenerativeModel
import streamlit as st
from typing import Dict, List, Any
import pandas as pd

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
    rag_results: List[str]
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
    field_descriptions = []
    for col, samples in column_samples.items():
        sample_str = ", ".join(map(str, samples))
        field_descriptions.append(f'  - {col}: e.g., "{sample_str}"')
    field_description_str = "\n".join(field_descriptions)

    # Build the RAG results section
    rag_str = "\n".join([f"- `{item}`" for item in rag_results])

    # Design the "Mega-Prompt"
    prompt = f"""
# ROLE
You are a top-tier ontology modeling expert, proficient in FIBO, Schema.org, and GeoSPARQL.

# TASK
Based on the provided column cluster information, define the most appropriate core business entity. Return your recommendation exclusively in JSON format.

# 1. COLUMN CLUSTER INFORMATION
- Cluster Name: "{cluster_name}"
- Included Columns: {columns}
- Column Examples:
{field_description_str}

# 2. RELEVANT KNOWLEDGE BASE TERMS (from RAG retrieval)
{rag_str}

# 3. YOUR INFERENCE AND DECISION
Synthesize all the above information and return a single JSON object with the following keys. Do not include markdown formatting.
- `entityId` (Camel-cased ID, e.g., physicalAddressEntity)
- `entityLabel` (English label, e.g., "Physical Address")
- `entityComment` (English comment explaining the entity's purpose)
- `mapsToClass` (The URI of the most appropriate ontology class)
"""

    try:
        response = model.generate_content(prompt)
        # Clean and parse the JSON string returned by Gemini
        cleaned_json_str = response.text.strip().lstrip('```json').rstrip('```').strip()
        result = json.loads(cleaned_json_str)
        result['clusterName'] = cluster_name # Add cluster name for reference
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
    
    # In a complete system, this would be a dynamic RAG call.
    # For now, we use a mock dictionary to simulate RAG results for specific clusters.
    mock_rag_data = {
        "Address": ["fibo-fnd-plc-adr:PhysicalAddress", "schema:PostalAddress", "geo:Feature"],
        "Institution": ["fibo-be-le-fbo:FormalBusinessOrganization", "schema:Organization", "fibo-fbc-fct-fse:FinancialInstitution"],
        "Branch": ["fibo-be-le-fbo:Branch", "schema:FinancialService", "schema:Store"],
        "Geospatial": ["sf:Point", "geo:Feature", "cmns-loc:Location"],
        "Identifier": ["cmns-id:Identifier", "schema:identifier"],
        "Date": ["cmns-dt:Date", "schema:Date"],
        "Status": ["cmns-cls:Classifier", "skos:Concept"]
    }
    
    progress_bar = st.progress(0)
    for i, cluster in enumerate(clusters):
        name = cluster.get('name', f"Unnamed Cluster {i+1}")
        cols = cluster.get('columns', [])
        
        # 1. Get sample data for the columns in the cluster
        samples = get_column_samples(df, cols)
        
        # 2. Simulate RAG retrieval for this cluster
        rag_results = mock_rag_data.get(name, ["rdfs:Class", "schema:Thing"]) # Default if not found

        # 3. Call the LLM to conceptualize the entity
        entity_definition = conceptualize_entity(name, cols, samples, rag_results)
        all_entities.append(entity_definition)
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(clusters))
        
    return all_entities
