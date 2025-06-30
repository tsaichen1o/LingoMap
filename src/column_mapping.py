import json
import logging
from google.generativeai.generative_models import GenerativeModel
import streamlit as st
from typing import Dict, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Prior Knowledge: A list of common properties for data fields.
# This helps guide the LLM's suggestions.
RELEVANT_DATA_PROPERTIES = [
    # Schema.org properties
    "schema:identifier", "schema:name", "schema:description", "schema:streetAddress",
    "schema:addressLocality", "schema:addressRegion", "schema:postalCode", "schema:latitude", "schema:longitude",
    # FIBO properties
    "fibo-fnd-plc-adr:hasCounty", "fibo-fnd-plc-adr:hasSubdivision", "fibo-fbc-pas-caa:hasOpenDate",
    "fibo-fnd-dt-fd:hasAcquisitionDate",
    # Commons properties (from LCC)
    "lcc-lr:hasName", "lcc-lr:hasTextValue",
    # DC Terms
    "dcterms:issued", "dcterms:type", "dcterms:identifier"
]

def get_property_mapping_with_llm(
    column_name: str,
    column_profile: Dict[str, Any],
    parent_entity: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Uses an LLM to suggest a specific property mapping for a single column,
    within the context of its parent entity.

    Args:
        column_name: The name of the column to map.
        column_profile: A dictionary containing the data profile of the column.
        parent_entity: The entity dictionary (from Stage 1) that this column belongs to.

    Returns:
        A dictionary containing the suggested mapping details.
    """
    logging.info(f"Requesting property mapping for column '{column_name}' within entity '{parent_entity.get('entityId', 'N/A')}'.")

    model = GenerativeModel('gemini-2.0-flash-exp')

    # Format the input data for the prompt
    formatted_profile = json.dumps(column_profile, indent=2)
    
    # Simulate RAG by providing a relevant subset of properties
    rag_results = "\n".join(f"- `{prop}`" for prop in RELEVANT_DATA_PROPERTIES)

    # Design the "Mega-Prompt"
    prompt = f"""
# ROLE
You are a top-tier semantic data modeling expert, specializing in FIBO and Schema.org.

# TASK
Your task is to recommend the best semantic property for the given CSV column. You must consider the column's data profile AND the context of the entity it belongs to.

# 1. PARENT ENTITY CONTEXT (from Stage 1)
- Entity ID: `{parent_entity.get('entityId', 'N/A')}`
- Entity Label: `{parent_entity.get('entityLabel', 'N/A')}`
- Entity Maps To: `{parent_entity.get('mapsToClass', 'N/A')}`
- Entity Comment: `{parent_entity.get('entityComment', 'N/A')}`

# 2. AUTOMATED DATA PROFILE REPORT (from data_profiler)
- Column Name: `{column_name}`
- Column Profile:
```json
{formatted_profile}

# RELEVANT KNOWLEDGE BASE PROPERTIES (from RAG retrieval)
{rag_results}

# INSTRUCTIONS
1. Analyze all the provided information.
2. Select the single best property from the knowledge base that accurately describes the column within its entity context.
3. Return the result as a single, valid JSON object.
4. The JSON object must contain these keys:
  * partOfEntity: The ID of the parent entity ({parent_entity.get('entityId', 'N/A')}).
  * mapsToProperty: The full URI of the recommended property.
  * confidenceScore: A score from 0.0 to 1.0 indicating your confidence.
  * justification: A concise, one-sentence explanation for your choice.
  * mappingType: The type of mapping, choose one of ["ColumnMapping", "IdentifierMapping", "ClassificationMapping"].
  
# OUTPUT FORMAT (EXAMPLE)
```json
{{
"partOfEntity": "...",
"mapsToProperty": "...",
"confidenceScore": 0.95,
"justification": "...",
"mappingType": "ColumnMapping"
}}
```
"""

    try:
        logging.info(f"Sending mapping prompt for '{column_name}' to Gemini...")
        response = model.generate_content(prompt)

        cleaned_json_str = response.text.strip().lstrip('```json').rstrip('```').strip()
        mapping_suggestion = json.loads(cleaned_json_str)
        
        logging.info(f"Successfully received mapping for '{column_name}'.")
        # time.sleep(6)  # Sleep to avoid hitting rate limits
        return mapping_suggestion
        
    except Exception as e:
        logging.error(f"An error occurred during column mapping for '{column_name}': {e}")
        st.error(f"Failed to get mapping for column '{column_name}'. Error: {e}")
        return {"error": str(e)}
