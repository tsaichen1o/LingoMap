"""
LingoMap Core Mapping Engine (v2.1)
- Supports entity assignment based on a user-defined data model.
- Includes re-evaluation logic and an enhanced standalone test mode.
"""
import os
import json
import pandas as pd
from google.generativeai.client import configure
from google.generativeai.generative_models import GenerativeModel
from typing import Dict, Any, List
from dotenv import load_dotenv
import streamlit as st

from data_profiler import profile_column
from vocabulary_processor import VocabularyProcessor
from column_clusterer import cluster_columns
from entity_conception import process_clusters_to_entities
from relationship_definition import define_relationships_with_llm

load_dotenv()

class CoreMappingEngine:
    """
    The core engine that orchestrates profiling, retrieval, and LLM reasoning.
    """
    def __init__(self):
        """
        Initializes the engine with a vocabulary retriever and configures the LLM.
        """
        print("🚀 Initializing Core Mapping Engine...")
        
        self.retriever = VocabularyProcessor(model_name='all-mpnet-base-v2')
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY not found in environment variables.")
        
        print("   - Configuring Gemini API...")
        configure(api_key=api_key)
        
        self.llm = GenerativeModel('gemini-2.0-flash-exp')
        print("✅ Core Mapping Engine initialized successfully with Gemini 2.0 Flash.")
        
    def generate_semantic_entities(self, df: pd.DataFrame, n_clusters: int) -> List[Dict[str, Any]]:
        """
        Orchestrates the full process from raw data to conceptual entities.
        
        Step 1: Cluster columns based on their content.
        Step 2: Run entity conception for each cluster using an LLM.

        Args:
            df (pd.DataFrame): The input DataFrame.
            clustering_params (dict): Parameters for the clustering algorithm.

        Returns:
            List[Dict[str, Any]]: A list of AI-generated entity definitions.
        """
        st.write("Step 1: Clustering columns based on data similarity...")
        # Ensure your cluster_columns function returns a dictionary like:
        # {'Bank Address': ['ADDRESS', 'CITY', ...], 'Institution Info': ['NAME', ...]}
        # If it returns a list of dicts, we adapt to it.
        clusters = cluster_columns(df=df, n_clusters=n_clusters) 
        st.write(f"Clustering complete. Found {len(clusters)} potential entity groups.")
        
        st.write("\nStep 2: Conceptualizing entities from clusters via LLM...")
        # Call the new function to process these clusters
        # Convert clusters_dict back to a list of dicts as expected by process_clusters_to_entities
        entities = process_clusters_to_entities(df, clusters)
        st.write("Entity conception complete.")
        
        return entities
    
    def generate_entity_relationships(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Orchestrates the process of defining relationships between conceptual entities.
        
        Args:
            entities: The list of entity dictionaries generated in Stage 1.
        
        Returns:
            A list of relationship dictionaries inferred by the LLM.
        """
        st.write("\nStep 3: Defining relationships between entities via LLM...")
        if not entities:
            st.warning("Cannot define relationships because no entities were provided.")
            return []
        
        relationships = define_relationships_with_llm(entities)
        st.write("Relationship definition complete.")
        
        return relationships

    def _build_rag_query(self, profile: Dict[str, Any]) -> str:
        """Builds a richer query string for the RAG retriever."""
        column_name = profile.get('column_name', '')
        semantic_type = profile.get('inferred_semantic_type', '')
        samples = profile.get("sample_values", [])
        keywords = column_name.replace('_', ' ').lower()
        domain_context = "For a dataset about FDIC insured financial institutions,"
        
        query_parts = [
            f"{domain_context} find the best semantic property for a data column with these characteristics:",
            f"- Column Name: {column_name}", f"- Keywords: {keywords}",
            f"- Inferred Data Type: {semantic_type}", f"- Sample Values: {', '.join(map(str, samples))}"
        ]
        return "\n".join(query_parts)

    def _build_mega_prompt(self, profile: Dict[str, Any], rag_results: List[Dict], model_entities: list, cluster_name: str) -> str:
        """
        Constructs the final, context-rich prompt for the LLM for initial suggestion.
        """
        profile_str = json.dumps(profile, indent=2, ensure_ascii=False)
        rag_str = ""
        if rag_results:
            for i, res in enumerate(rag_results):
                rag_str += f"- Term {i+1}: {res.get('label', 'N/A')}\n  URI: {res.get('uri')}\n"
        else:
            rag_str = "No relevant terms found in the knowledge base."

        # We not only give ID, but also give label and description to help AI make better choices
        entities_description = "\n".join(
            [f"  - `{e.get('ID')}`: {e.get('Label')} ({e.get('Comment')})" for e in model_entities]
        )
        # Separate the entity ID list for output specification constraints
        entity_ids = [f"'{e.get('ID')}'" for e in model_entities]
        entity_id_list_str = f"[{', '.join(entity_ids)}]"

        prompt_template = f"""
# MISSION (Highest instruction)
You are a meticulous data architect. Your primary mission is to assign a CSV column to ONE of the PRE-DEFINED entities and suggest a corresponding property. Adherence to the provided entity list is mandatory.

# TASK
Analyze the column profile, its semantic cluster, and the retrieved vocabulary terms. Then, generate a single, valid JSON object that recommends the best mapping.

# STEP 1: UNDERSTAND THE DATA MODEL (Understand the data model)
First, study the ONLY available entities in the target data model. You CANNOT invent new entities.
The available entities are:
{entities_description}

# STEP 2: ANALYZE THE INPUT
Next, analyze the following information about the CSV column.
- Column Name: `{profile.get('column_name')}`
- Semantic Cluster: This column belongs to the '{cluster_name}' group.
- Data Profile: {profile.get('inferred_semantic_type')} with sample values like [{profile.get('sample_values', ['N/A'])[0]}].
- This JSON object contains detailed statistics about the column's data.
```json
{profile_str}
```
- Relevant Concepts from Knowledge Base (RAG):
{rag_str if rag_str else "  (None found)"}

# STEP 3: DECIDE AND JUSTIFY
Based on your analysis from STEP 1 and 2, perform the following two actions:
1.  **CHOOSE an ENTITY**: From the list of available entities, select the single most logical entity for this column.
2.  **SUGGEST a PROPERTY**: Find the best property from the knowledge base that fits the column and the chosen entity.

# STEP 4: GENERATE JSON OUTPUT
Finally, format your decision into a single, valid JSON object. The JSON object MUST adhere to the following strict schema.

## OUTPUT SCHEMA
{{
  "mapping_type": "string",
  "source_column": "string",
  "part_of": "string",
  "maps_to_property": "string",
  "confidence_score": "number",
  "justification": "string"
}}

## SCHEMA RULES
- The value for "part_of" MUST be an EXACT string match from this list: {entity_id_list_str}.

# Your final JSON output:
```json
"""
        return prompt_template

    def suggest_mapping(self, column_name: str, series: pd.Series, model_entities: list, cluster_name: str = "Unknown", temperature: float = 0.2) -> Dict[str, Any]:
        """
        The main method to generate a mapping suggestion for a given column.
        """
        print("\n" + "="*50)
        print(f"🧠 Generating suggestion for column: '{column_name}'")
        print("="*50)

        print("   - Step 1: Profiling column data...")
        profile = profile_column(column_name, series)

        print("   - Step 2: Retrieving relevant terms (RAG)...")
        rag_query = self._build_rag_query(profile)
        rag_results = self.retriever.search(rag_query, n_results=5)

        print("   - Step 3: Constructing the Mega-Prompt with Data Model context...")
        final_prompt = self._build_mega_prompt(profile, rag_results, model_entities, cluster_name)

        print(f"   - Step 4: Calling Gemini API (Temperature: {temperature})...")
        try:
            generation_config: Dict[str, Any] = {
                "temperature": temperature,
                "response_mime_type": "application/json"
            }
            response = self.llm.generate_content(final_prompt, generation_config=generation_config)  # type: ignore
            suggestion = json.loads(response.text)
            print("   - Suggestion received.")
            return suggestion
        except Exception as e:
            print(f"❌ Gemini API call failed: {e}")
            return {"error": str(e)}

    # --- Re-evaluation logic from your script ---
    def _build_reevaluation_prompt(self, profile: Dict[str, Any], user_uri: str, model_entities: list) -> str:
        """Constructs a prompt for the LLM to re-evaluate a user-provided URI."""
        profile_str = json.dumps(profile, indent=2, ensure_ascii=False)
        entities_str = "\n".join([f"- ID: {e.get('ID')}, Label: {e.get('Label')}" for e in model_entities])
        entity_ids = [f"'{e.get('ID')}'" for e in model_entities]

        prompt = f"""
# ROLE
You are a world-class expert in data architecture. A user has provided a mapping suggestion. Your task is to critically evaluate it.

# CONTEXT
## 1. Data Model Entities
{entities_str}

## 2. Column Profile
```json
{profile_str}
```

## 3. User's Suggested Mapping URI
The user suggests mapping this column to: {user_uri}

# YOUR TASK
Critically evaluate the user's suggestion. Your JSON output MUST include:
- "part_of": The ID of the entity from the list that the user's suggested property would belong to. MUST be one of [{', '.join(entity_ids)}].
- "maps_to_property": This MUST be the user's provided URI.
- "confidence_score": Your confidence in this mapping (1-100, where 100 is highest confidence).
- "justification": Your analysis of the user's choice.
- "mapping_type": Classify the user's suggestion ("ColumnMapping", "IdentifierMapping", etc.).

Provide only the JSON object.
"""
        return prompt

    def reevaluate_mapping(self, profile: Dict[str, Any], user_provided_uri: str, model_entities: list) -> Dict[str, Any]:
        """
        Takes a user's URI and re-evaluates it with the LLM.
        """
        print("\n" + "="*50)
        print(f"🤔 Re-evaluating user suggestion for column: '{profile.get('column_name')}'")
        print(f"   - User's URI: {user_provided_uri}")
        print("="*50)

        print("   - Step 1: Constructing the Re-evaluation Prompt...")
        final_prompt = self._build_reevaluation_prompt(profile, user_provided_uri, model_entities)

        print("   - Step 2: Calling Gemini API for re-evaluation...")
        try:
            # Re-evaluation can be more creative
            generation_config: Dict[str, Any] = {
                "temperature": 0.5,
                "response_mime_type": "application/json"
            }
            response = self.llm.generate_content(final_prompt, generation_config=generation_config)  # type: ignore
            suggestion = json.loads(response.text)
            print("   - Re-evaluation received.")
            return suggestion
        except Exception as e:
            print(f"❌ Gemini API call failed during re-evaluation: {e}")
            return {"error": str(e)}

def main():
    """Main function to test the core engine functionality"""
    
    csv_file = "FDIC_Insured_Banks.csv"
    if not os.path.exists(csv_file):
        print(f"❌ Error: Data file '{csv_file}' not found.")
        return

    try:
        print(f"📊 Loading data from {csv_file}...")
        df = pd.read_csv(csv_file, low_memory=False)
        print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns.")

        print("\n🔄 Running semantic clustering to get group context...")
        named_clusters = cluster_columns(df=df, n_clusters=8)
        
        column_to_cluster_name = {
            col: cluster['name'] 
            for cluster in named_clusters 
            for col in cluster['columns']
        }
        print("   ✅ Column cluster context is ready.")
        
        # --- Simulate the entity model defined in app.py ---
        mock_model_entities = [
            {"ID": "BankInstitutionEntity", "Label": "Bank Institution", "Comment": "A legal entity that provides financial services."},
            {"ID": "BankBranchEntity", "Label": "Bank Branch", "Comment": "A branch office of a bank institution."},
            {"ID": "PhysicalAddressEntity", "Label": "Physical Address", "Comment": "Combines all address-related fields."},
            {"ID": "GeometryEntity", "Label": "Geometry", "Comment": "Stores coordinate information."},
        ]
        
        engine = CoreMappingEngine()
        
        test_column_name = 'STALP'
        if test_column_name in df.columns:
            test_series = df[test_column_name]
            
            test_cluster_name = column_to_cluster_name.get(test_column_name, "Cluster not found")
            print(f"\n🧪 Testing column '{test_column_name}' which belongs to cluster: '{test_cluster_name}'")
            
            final_suggestion = engine.suggest_mapping(
                test_column_name, 
                test_series,
                mock_model_entities,
                test_cluster_name
            )
            
            print("\n" + "="*50)
            print("✨ Final Mapping Suggestion ✨")
            print("="*50)
            print(json.dumps(final_suggestion, indent=4, ensure_ascii=False))
        else:
            print(f"❌ Column '{test_column_name}' not found in the data file.")
            
    except Exception as e:
        print(f"❌ Error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
