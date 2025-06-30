import json
import logging
from google.generativeai.generative_models import GenerativeModel
import streamlit as st
from typing import Dict, Any, List
import time

from rag_search import get_rag_searcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Prior Knowledge: A list of common properties for data fields.
# This helps guide the LLM's suggestions.
RELEVANT_DATA_PROPERTIES = [
    # Schema.org properties
    "schema:identifier", "schema:name", "schema:description", "schema:streetAddress",
    "schema:addressLocality", "schema:addressRegion", "schema:postalCode", "schema:latitude", "schema:longitude",
    "schema:location", "schema:parentOrganization", "schema:subOrganization", "schema:address"
    # FIBO properties
    "fibo-fnd-plc-adr:hasCounty", "fibo-fnd-plc-adr:hasSubdivision", "fibo-fbc-pas-caa:hasOpenDate",
    "fibo-fnd-dt-fd:hasAcquisitionDate", "fibo-be-oac-cctl:Affiliate", "fibo-be-le-fbo:Branch",
    # Commons properties (from LCC)
    "lcc-lr:hasName", "cmns-txt:hasTextValue", "cmns-loc:PhysicalLocation",
    # DC Terms
    "dcterms:issued", "dcterms:type", "dcterms:identifier",
    # Geo
    "geo:hasGeometry", "geor:sfWithin",
]

def get_property_mapping_with_llm(
    column_name: str,
    column_profile: Dict[str, Any],
    parent_entity: Dict[str, Any],
    rag_results: List[str]
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
    rag_searcher = get_rag_searcher()

    # --- 1. 動態建立一個豐富的 RAG 查詢字串 ---
    # 這個查詢結合了所有可用的上下文，以獲得最精準的搜尋結果
    profile_summary = f"data type is {column_profile.get('data_type_inferred', 'unknown')}, with example values like {list(column_profile.get('top_5_values', {}).keys())[:3]}"
    query_text = (
        f"A property for a column named '{column_name}' "
        f"which is part of a '{parent_entity.get('entityLabel', 'generic')}' entity. "
        f"The column's {profile_summary}."
    )
    
    # --- 2. 執行 RAG 搜尋 ---
    logging.info(f"Executing RAG search with query: '{query_text}'")
    rag_results = rag_searcher.search(query_text, n_results=10) # 獲取更多候選屬性

    # 格式化 RAG 結果以注入提示
    formatted_rag_results = "\n".join(f"- `{result}`" for result in rag_results)
    if not formatted_rag_results:
        formatted_rag_results = "No relevant properties were found in the knowledge base."

    # 格式化欄位剖析報告
    formatted_profile = json.dumps(column_profile, indent=2)
    formatted_static_list = "\n".join(f"- `{prop}`" for prop in RELEVANT_DATA_PROPERTIES)

    # --- 3. 設計整合了 RAG 結果的「超級提示」 ---
    prompt = f"""
# ROLE
You are a top-tier semantic data modeling expert.

# TASK
Your task is to recommend the best semantic property for the given CSV column. You must synthesize information from three sources: the column's data profile, its parent entity's context, and a list of potentially relevant properties retrieved from a knowledge base.

# 1. PARENT ENTITY CONTEXT
- Entity ID: `{parent_entity.get('entityId', 'N/A')}`
- Entity Label: `{parent_entity.get('entityLabel', 'N/A')}`
- Entity Maps To: `{parent_entity.get('mapsToClass', 'N/A')}`

# 2. AUTOMATED DATA PROFILE REPORT
- Column Name: `{column_name}`
- Column Profile:
```json
{formatted_profile}

# 3. RELEVANT PROPERTIES (from real-time RAG Search)
These properties were dynamically retrieved from the vector database as being the most semantically similar to the column's data. They provide specific context.
{formatted_rag_results}

# 4. HIGHLY-RECOMMENDED PROPERTIES (Prior Knowledge)
This is a curated list of high-quality, commonly used properties. Your final choice should ideally be a property that appears in BOTH the RAG results and this list, or if not, one from this list.
{formatted_static_list}

# 5. YOUR INFERENCE AND DECISION
1. Analyze all the provided information.
2. Select the single best and most precise property from the RAG search results that accurately describes the column.
3. Also determine the most appropriate data type (e.g., xsd:string, xsd:date, xsd:decimal).
4. The JSON object must contain these keys:
  * partOfEntity: The ID of the parent entity.
  * mapsToProperty: The full URI of the recommended property.
  * hasDataType: The recommended XSD data type (optional, can be null).
  * confidenceScore: A score from 0.0 to 1.0.
  * justification: A concise, one-sentence explanation for your choice.
  * mappingType: One of ["ColumnMapping", "IdentifierMapping", "ClassificationMapping"].

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
