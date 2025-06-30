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
import time

from data_profiler import profile_column
from vocabulary_processor import VocabularyProcessor
from column_clusterer import cluster_columns
from entity_conception import process_clusters_to_entities
from relationship_definition import define_relationships_with_llm
from column_mapping import get_property_mapping_with_llm
from rule_generator import generate_conditional_rule_with_llm
from rag_search import get_rag_searcher

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
        self.rag_searcher = get_rag_searcher()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY not found in environment variables.")
        
        print("   - Configuring Gemini API...")
        configure(api_key=api_key)
        
        self.llm = GenerativeModel('gemini-2.0-flash-exp')
        print("✅ Core Mapping Engine initialized successfully with Gemini 2.0 Flash.")
        
    def generate_semantic_entities(self, df: pd.DataFrame, n_clusters: int) -> tuple[List[Dict], List[Dict]]:
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
        if not clusters:
            return [], []
        st.write(f"Clustering complete. Found {len(clusters)} potential entity groups.")
        
        st.write("\nStep 2: Conceptualizing entities from clusters via LLM...")
        # Call the new function to process these clusters
        # Convert clusters_dict back to a list of dicts as expected by process_clusters_to_entities
        entities = process_clusters_to_entities(df, clusters)
        st.write("Entity conception complete.")
        
        return clusters, entities
    
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

    def suggest_column_mapping(
        self,
        column_name: str,
        df: pd.DataFrame,
        entities: List[Dict[str, Any]],
        clusters: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Orchestrates Stage 3: Generates a detailed mapping for a single column,
        first checking if it's a conditional rule.
        """
        st.write(f"Step 4: Generating detailed mapping for column `{column_name}`...")
        
        column_profile = profile_column(column_name, df[column_name])
        
        rule_suggestion = generate_conditional_rule_with_llm(
            column_name=column_name,
            column_profile=column_profile,
            entities=entities
        )
        
        if rule_suggestion.get("isRule"):
            return rule_suggestion
        
        # 1. Find the parent entity for the given column
        parent_entity = None
        cluster_name = "Unknown"
        for cluster in clusters:
            if column_name in cluster.get('columns', []):
                cluster_name = cluster.get('name', 'Unnamed')
                # Find the corresponding entity generated in Stage 1
                for entity in entities:
                    if entity.get('clusterName') == cluster_name:
                        parent_entity = entity
                        break
                break
        
        if not parent_entity:
            st.error(f"Could not find a parent entity for column `{column_name}`. Aborting.")
            return {"error": "Parent entity not found."}
        
        # --- 4. [NEW] DYNAMIC RAG SEARCH ---
        st.write(f"Searching knowledge base for terms related to `{column_name}`...")
        
        # 建立一個豐富的查詢字串
        # 這裡的 COLUMN_DESCRIPTIONS 應該從 column_clusterer 模組中引入或定義
        from column_clusterer import COLUMN_DESCRIPTIONS 
        col_desc = COLUMN_DESCRIPTIONS.get(column_name, "No description")
        query_text = (
            f"Data property for a column named '{column_name}' "
            f"described as '{col_desc}'. "
            f"It is part of the '{parent_entity.get('entityLabel', '')}' entity. "
            f"Sample values include: {column_profile.get('top_5_values', {}).keys()}"
        )
        
        # 執行搜尋
        rag_results = self.rag_searcher.search(query_text, n_results=10)
        # --------------------------------
        
        # 2. Generate a data profile for the column
        st.write(f"Profiling data for `{column_name}`...")
        column_profile = profile_column(column_name, df[column_name])
        
        # --- 2. NEW: FIRST, CHECK IF IT'S A CONDITIONAL RULE ---
        st.write(f"Checking if `{column_name}` is a candidate for a conditional rule...")
        suggestion = get_property_mapping_with_llm(
            column_name=column_name,
            column_profile=column_profile,
            parent_entity=parent_entity,
            rag_results=rag_results
        )
        time.sleep(6) # Add another delay
        # If the LLM says it's a rule, return that rule immediately.
        if rule_suggestion.get("isRule"):
            st.success(f"`{column_name}` was identified as a conditional rule!")
            return rule_suggestion
        # ---------------------------------------------------------
        
        # 3. If it's not a rule, proceed with the normal property mapping logic.
        st.write(f"`{column_name}` is regular data. Requesting property mapping...")
        
        # 5. 呼叫 LLM 進行最終推薦，並傳入 RAG 結果
        st.write("Requesting final mapping from LLM with RAG context...")
        suggestion = get_property_mapping_with_llm(
            column_name=column_name,
            column_profile=column_profile,
            parent_entity=parent_entity,
            rag_results=rag_results,
        )
        
        return suggestion

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

def run_standalone_test():
    """
    A standalone function to test the full engine workflow from the command line.
    """
    print("="*60)
    print("🔬 RUNNING CORE ENGINE IN STANDALONE TEST MODE 🔬")
    print("="*60)

    try:
        # --- Setup ---
        test_csv_path = 'FDIC_Insured_Banks.csv'
        if not os.path.exists(test_csv_path):
            print(f"❌ Test data file not found at '{test_csv_path}'. Please ensure it's in the root directory.")
            return
            
        df = pd.read_csv(test_csv_path)
        engine = CoreMappingEngine()
        n_clusters = 8

        # --- Stage 1: Clustering & Entity Conception ---
        print("\n--- STAGE 1: Generating Semantic Entities ---")
        # THIS IS THE FIX: The 'clusters' variable is now correctly captured here.
        clusters, entities = engine.generate_semantic_entities(df, n_clusters)
        if not entities:
            print("❌ Stage 1 failed. No entities were generated.")
            return
        print(f"✅ Stage 1 complete. Generated {len(entities)} entities.")
        # print(json.dumps(entities, indent=2)) # Uncomment for detailed view

        # --- Stage 2: Relationship Definition ---
        print("\n--- STAGE 2: Defining Entity Relationships ---")
        relationships = engine.generate_entity_relationships(entities)
        if not relationships:
            print("⚠️ Stage 2 generated no relationships (this might be expected).")
        else:
            print(f"✅ Stage 2 complete. Generated {len(relationships)} relationships.")
            # print(json.dumps(relationships, indent=2)) # Uncomment for detailed view

        # --- Stage 3: Detailed Column Mapping (Test two columns) ---
        print("\n--- STAGE 3: Testing Detailed Column Mapping ---")
        test_columns = ['STALP', 'MAINOFF']
        
        for col_to_test in test_columns:
            if col_to_test not in df.columns:
                print(f"⚠️ Skipping test for column '{col_to_test}' as it's not in the DataFrame.")
                continue

            print(f"\n🧪 Testing column: '{col_to_test}'...")
            
            # And 'clusters' is now correctly passed here.
            final_suggestion = engine.suggest_column_mapping(
                column_name=col_to_test,
                df=df,
                entities=entities,
                clusters=clusters
            )
            
            print("\n" + "="*50)
            print(f"✨ Final Suggestion for '{col_to_test}' ✨")
            print("="*50)
            print(json.dumps(final_suggestion, indent=4))

    except Exception as e:
        print(f"\n❌ AN ERROR OCCURRED DURING THE STANDALONE TEST: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_standalone_test()
