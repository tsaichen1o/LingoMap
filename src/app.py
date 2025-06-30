"""
LingoMap Semantic Mapping Tool
A smart tool that combines data analysis, RAG retrieval, and large language models to help you add standardized semantics to your data sets.
"""

import streamlit as st
import pandas as pd
from typing import Optional
import os
from datetime import datetime
import sys

sys.path.append(os.path.dirname(__file__))

try:
    from core_engine import CoreMappingEngine
    from data_profiler import profile_column
    from column_clusterer import cluster_columns
except ImportError as e:
    st.error(f"Error: Unable to import necessary modules. Please check if your file structure is correct. Error message: {e}")
    st.stop()

st.set_page_config(
    page_title="LingoMap Semantic Mapping Tool",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_theme():
    """Apply a custom CSS theme to create a more professional visual interface."""
    custom_css = """
    <style>
        /* --- Global Variable Definitions --- */
        :root {
            --primary-color: #1a9691; /* A subtle blue-green as the primary color */
            --background-color: #0c1821; /* Very dark blue-gray background */
            --secondary-background-color: #1b2a41; /* Slightly lighter secondary background, used for containers */
            --text-color: #e0e0e0; /* Soft gray-white text */
            --light-text-color: #a0a0a0; /* Darker secondary text */
            --border-color: #324a5f; /* Border color */
            --success-color: #28a745;
            --info-color: #17a2b8;
            --warning-color: #ffc107;
            --error-color: #dc3545;
        }

        /* --- Overall background and text --- */
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }

        /* --- Sidebar --- */
        [data-testid="stSidebar"] {
            background-color: var(--secondary-background-color);
            border-right: 1px solid var(--border-color);
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #ffffff;
        }

        /* --- Button Styles --- */
        .stButton>button {
            border: 2px solid var(--primary-color);
            background-color: transparent;
            color: var(--primary-color);
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }
        .stButton>button:focus {
            box-shadow: 0 0 0 0.2rem rgba(26, 150, 145, 0.5) !important;
        }
        
        /* Special styles for the main button */
        .stButton>button[kind="primary"] {
            background-color: var(--primary-color);
            color: white;
        }
        .stButton>button[kind="primary"]:hover {
            opacity: 0.8;
        }

        /* --- Containers and Blocks --- */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: var(--secondary-background-color);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1.5rem;
        }
        [data-testid="stExpander"] {
            border-color: var(--border-color) !important;
        }

        /* --- Input Components --- */
        .stSelectbox, .stFileUploader {
            border-radius: 5px;
        }

        /* --- Message Boxes --- */
        [data-testid="stAlert"] {
            border-radius: 8px;
        }
        [data-testid="stAlert"][data-baseweb="alert-positive"] { 
            background-color: rgba(40, 167, 69, 0.1); 
            border-left-color: var(--success-color); 
        }
        [data-testid="stAlert"][data-baseweb="alert-negative"] { 
            background-color: rgba(220, 53, 69, 0.1); 
            border-left-color: var(--error-color); 
        }
        [data-testid="stAlert"][data-baseweb="alert-info"] { 
            background-color: rgba(23, 162, 184, 0.1); 
            border-left-color: var(--info-color); 
        }
        [data-testid="stAlert"][data-baseweb="alert-warning"] { 
            background-color: rgba(255, 193, 7, 0.1); 
            border-left-color: var(--warning-color); 
        }

        /* --- Mermaid Chart Styles --- */
        .mermaid {
            background-color: var(--secondary-background-color) !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

apply_custom_theme()

# --- Core Function Caching ---
@st.cache_resource
def get_mapping_engine():
    """Initialize and cache the core mapping engine"""
    try:
        engine = CoreMappingEngine()
        return engine
    except Exception as e:
        st.error(f"Core engine initialization failed: {e}")
        st.warning("Please check if you have set the GOOGLE_API_KEY in the .env file or environment variables.")
        return None

@st.cache_data 

def get_column_clusters(df: Optional[pd.DataFrame] = None):
    """Get column clustering results"""
    if df is None:
        # If no DataFrame is provided, try to load the default data
        try:
            default_path = 'data/source/FDIC_Insured_Banks.csv'
            if os.path.exists(default_path):
                df = pd.read_csv(default_path)
            else:
                st.error("Default data file not found, please upload a CSV file first")
                return []
        except Exception as e:
            st.error(f"Failed to load default data: {e}")
            return []
    
    return cluster_columns(df=df)

# --- Initialize session state ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'mappings' not in st.session_state:
    st.session_state.mappings = {}
if 'current_column' not in st.session_state:
    st.session_state.current_column = None
if 'current_suggestion' not in st.session_state:
    st.session_state.current_suggestion = None
if 'modification_mode' not in st.session_state:
    st.session_state.modification_mode = False
if 'rejected_columns' not in st.session_state:
    st.session_state.rejected_columns = set()
if 'model_entities' not in st.session_state:
    st.session_state.model_entities = []
    
# Initialize session state keys
def init_session_state():
    defaults = {
        'df': None,
        'mappings': {},
        'current_column': None,
        'generated_entities': [],
        'generated_relationships': [],
        'modification_mode': False,
        'rejected_columns': set(),
        'last_analyzed_column': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Main Application ---
def main():
    st.set_page_config(layout="wide", page_title="LingoMap Semantic Mapper")
    init_session_state()

    st.title("✈️ LingoMap Semantic Mapping Tool")
    st.markdown("### An Intelligent Workbench for Data Semanticization")

    # --- Sidebar for Data Upload and Global Settings ---
    with st.sidebar:
        st.header("📁 Data Input")
        uploaded_file = st.file_uploader(
            "Select a CSV file",
            type=['csv'],
            help="Upload the CSV file you wish to map to semantic ontologies."
        )

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                # Only update if the dataframe has changed
                if not df.equals(st.session_state.df):
                    init_session_state() # Reset state on new file upload
                    st.session_state.df = df
                    st.success(f"✅ Loaded {len(df)} rows.")
            except Exception as e:
                st.error(f"❌ File Load Error: {e}")
                st.session_state.df = None
        
        if st.session_state.df is not None:
            st.divider()
            st.header("⚙️ Global Settings")
            st.subheader("Entity Conception")
            st.slider(
                "Expected Number of Entities", 
                min_value=2, max_value=15, value=8, key='n_clusters',
                help="Controls how many groups the columns will be clustered into."
            )

    # --- Main Content Area ---
    if st.session_state.df is None:
        st.info("Please upload a CSV file using the sidebar to begin.")
        return

    df = st.session_state.df
    
    # --- STAGE 1: ENTITY CONCEPTION ---
    st.header("Stage 1: Entity Conception")
    st.write("Automatically group related columns and use an LLM to define a core business entity for each group.")
    
    if st.button("🚀 Generate Core Entities", use_container_width=True, type="primary"):
        engine = get_mapping_engine()
        if engine is not None:
            with st.spinner("Analyzing columns and communicating with Gemini... This may take a moment."):
                n_clusters = st.session_state.get('n_clusters', 8)
                # The integer n_clusters is passed directly here.
                generated_entities = engine.generate_semantic_entities(df, n_clusters)
                st.session_state.generated_entities = generated_entities
            st.success("Entity Conception complete! ✅")
        else:
            st.error("Core engine initialization failed. Please check your API key or environment settings.")


    st.header("Stage 2: Relationship Definition")
    st.write("Based on the entities defined above, use an LLM to infer the logical connections between them.")
        
    if st.session_state.generated_entities:
        st.subheader("🤖 AI-Generated Entity Definitions")
        entities = st.session_state.generated_entities
        
        if not entities:
            st.warning("No entities were generated. The model may not have been able to process the clusters.")
        else:
            # --- MODIFIED SECTION START ---
            
            # Iterate through each entity and display it as a self-contained row.
            for entity in entities:
                # Use a container with a border to create a distinct row for each entity.
                with st.container(border=True):
                    if "error" in entity:
                        st.error(f"Cluster '{entity.get('clusterName', 'N/A')}' failed.")
                        st.json(entity)
                    else:
                        # Use columns internally to structure the content within the row.
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.info(f"**{entity.get('entityLabel', 'Unknown Entity')}**")
                            st.markdown(f"**ID:** `{entity.get('entityId', 'N/A')}`")
                        
                        with col2:
                            st.markdown(f"**Maps to:** `{entity.get('mapsToClass', 'N/A')}`")
                            st.markdown(f"**Comment:**")
                            st.write(f"*{entity.get('entityComment', 'No comment.')}*")

                st.write("") # Adds a small vertical space between entity rows

            # --- MODIFIED SECTION END ---
            
        st.divider()
        
        if st.button("🚀 Define Entity Relationships", use_container_width=True, type="primary"):
            engine = get_mapping_engine()
            entities = st.session_state.generated_entities

            if engine is not None:
                with st.spinner("Analyzing entity connections with Gemini..."):
                    relationships = engine.generate_entity_relationships(entities)
                    st.session_state.generated_relationships = relationships
                st.success("Relationship definition complete! ✅")
            else:
                st.error("Core engine initialization failed. Please check your API key or environment settings.")

    # --- STAGE 2: DETAILED COLUMN MAPPING ---
    # (The rest of your UI for detailed mapping would go here)
    # st.header("Stage 2: Detailed Column Mapping")
    # if not st.session_state.generated_entities:
    #     st.info("Please run Stage 1 'Generate Core Entities' first.")
    # else:
    #     # Your previous logic for selecting columns from clusters and suggesting mappings
    #     # can be adapted here.
    #     st.write("Select a column from the groups below to get a detailed mapping suggestion.")
        
    #     clusters = cluster_columns(df)
    #     for i, cluster_info in enumerate(clusters):
    #         cluster_name = cluster_info.get('name', f"Group {i+1}")
    #         columns = cluster_info.get('columns', [])
            
    #         with st.expander(f"**{cluster_name}** ({len(columns)} columns)"):
    #             for col in columns:
    #                 if st.button(col, key=f"btn_{col}"):
    #                     st.session_state.current_column = col
    #                     # This would trigger the detailed mapping logic from your original file
    #                     st.info(f"Selected '{col}' for detailed mapping. (Logic to be implemented)")
    #                     st.rerun()
    
    if st.session_state.generated_relationships:
        st.subheader("🔗 AI-Generated Relationship Links")
        relationships = st.session_state.generated_relationships
        
        if not relationships:
            st.warning("No relationships were defined by the model.")
        else:
            for rel in relationships:
                with st.container(border=True):
                    # 格式化顯示: Source -> Property -> Target
                    source = rel.get('sourceEntity', 'N/A')
                    target = rel.get('targetEntity', 'N/A')
                    prop = rel.get('usingProperty', 'N/A').split(':')[-1] # 只顯示屬性名稱

                    st.markdown(f"##### {rel.get('associationId', 'N/A')}")
                    st.code(f"{source}  ->  ({prop})  ->  {target}", language="text")
                    
                    with st.expander("View Details and Justification"):
                        st.markdown(f"**Source Entity:** `{source}`")
                        st.markdown(f"**Target Entity:** `{target}`")
                        st.markdown(f"**Connecting Property:** `{rel.get('usingProperty', 'N/A')}`")
                        st.markdown(f"**Justification:** *{rel.get('justification', 'No justification provided.')}*")
                st.write("") # Add vertical space
        st.divider()

    # --- STAGE 3: DETAILED COLUMN MAPPING (Placeholder) ---
    st.header("Stage 3: Detailed Column Mapping")
    
    if not st.session_state.generated_relationships:
        st.info("Please run Stage 1 and 2 first to define entities and their relationships.")
    else:
        st.write("Select a column from the groups below to get a detailed mapping suggestion from the AI.")
        
        clusters = st.session_state.get('llm_clusters', []) # 我們將把叢集結果存儲在 session state 中
        if not clusters:
             # 如果尚未分群，則執行分群
            clusters = cluster_columns(df, n_clusters=st.session_state.get('n_clusters', 8))
            st.session_state['llm_clusters'] = clusters

        for cluster_info in clusters:
            cluster_name = cluster_info.get('name', 'Unnamed Group')
            columns = cluster_info.get('columns', [])
            
            with st.expander(f"**Entity Group: {cluster_name}** ({len(columns)} columns)"):
                # --- 新增：分析整個群組的按鈕 ---
                if st.button(f"Analyze All Columns in '{cluster_name}'", key=f"btn_group_{cluster_name}", type="primary"):
                    engine = get_mapping_engine()
                    
                    # 初始化存儲建議的地方
                    if 'all_suggestions' not in st.session_state:
                        st.session_state.all_suggestions = {}

                    with st.spinner(f"Analyzing {len(columns)} columns for '{cluster_name}'... This will take time due to rate limits."):
                        progress_bar = st.progress(0)
                        for i, col_name in enumerate(columns):
                            # 呼叫核心引擎進行分析 (引擎內部應包含延遲)
                            if engine is not None:
                                suggestion = engine.suggest_column_mapping(
                                    column_name=col_name,
                                    df=df,
                                    entities=st.session_state.generated_entities,
                                    clusters=st.session_state.llm_clusters
                                )
                            else:
                                suggestion = {
                                    "error": "Core engine initialization failed. Please check your API key or environment settings."
                                }
                            st.session_state.all_suggestions[col_name] = suggestion
                            progress_bar.progress((i + 1) / len(columns), text=f"Analyzed: {col_name}")

                    st.success(f"Successfully analyzed all columns in '{cluster_name}'!")
                # 為每個按鈕設置一個唯一的 key
                for col in columns:
                    if st.button(f"Analyze `{col}`", key=f"btn_{col}"):
                        st.session_state.current_column = col
                        st.session_state.current_suggestion = None # 清除舊的建議
                        st.rerun() # 重新整理以觸發分析

        # Display analysis and suggestion for the selected column
        if st.session_state.current_column:
            col_name = st.session_state.current_column
            
            st.subheader(f"🔬 Analysis for Column: `{col_name}`")

            # 觸發分析與建議生成
            if st.session_state.current_suggestion is None:
                engine = get_mapping_engine()
                if engine is not None:
                    with st.spinner(f"Generating detailed mapping for `{col_name}`..."):
                        suggestion = engine.suggest_column_mapping(
                            column_name=col_name,
                            df=df,
                            entities=st.session_state.generated_entities,
                            clusters=st.session_state.llm_clusters
                        )
                        st.session_state.current_suggestion = suggestion
                        st.rerun()
                else:
                    st.session_state.current_suggestion = {
                        "error": "Core engine initialization failed. Please check your API key or environment settings."
                    }

            # 顯示建議結果
            suggestion = st.session_state.current_suggestion
            if suggestion:
                if "error" in suggestion:
                    st.error(suggestion["error"])
                else:
                    with st.container(border=True):
                        st.markdown("##### AI Mapping Suggestion")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Belongs to Entity:**")
                            st.success(f"`{suggestion.get('partOfEntity', 'N/A')}`")
                        
                        with col2:
                             # 格式化信心分數
                            score = suggestion.get('confidenceScore', 0)
                            try:
                                score = int(float(score) * 100)
                            except:
                                score = 0
                            st.markdown(f"**Confidence Score:**")
                            st.progress(score, text=f"{score}%")

                        st.markdown(f"**Recommended Property:**")
                        st.info(f"`{suggestion.get('mapsToProperty', 'N/A')}`")
                        
                        with st.expander("View AI's Justification"):
                            st.write(f"*{suggestion.get('justification', 'No justification provided.')}*")
                            st.caption(f"Mapping Type: `{suggestion.get('mappingType', 'N/A')}`")

# --- Generate Turtle Mapping Rules ---
def generate_rules_ttl(mappings):
    """Generate TTL mapping rules"""
    # 1. Define all required namespace prefixes
    prefixes = """
@prefix fibo-be-le-fbo: <https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/FormalBusinessOrganizations/> .
@prefix fibo-fbc-fct-fse: <https://spec.edmcouncil.org/fibo/ontology/FBC/FunctionalEntities/FinancialServicesEntities/> .
@prefix fibo-fnd-plc-adr: <https://spec.edmcouncil.org/fibo/ontology/FND/Places/Addresses/> .
@prefix fibo-fnd-rel-rel: <https://spec.edmcouncil.org/fibo/ontology/FND/Relations/Relations/> .
@prefix fibo-fnd-dt-fd: <https://spec.edmcouncil.org/fibo/ontology/FND/DatesAndTimes/FinancialDates/> .
@prefix fibo-ind-ei-ei: <https://spec.edmcouncil.org/fibo/ontology/IND/EconomicIndicators/EconomicIndicators/> .
@prefix fibo-fbc-pas-caa: <https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/> .
@prefix cmns-cls: <https://www.omg.org/spec/Commons/Classifiers/> .
@prefix cmns-lng: <https://www.omg.org/spec/Commons/Languages/> .
@prefix cmns-av: <https://www.omg.org/spec/Commons/AnnotationVocabulary/> .
@prefix cmns-col: <https://www.omg.org/spec/Commons/Collections/> .
@prefix cmns-cxtdsg: <https://www.omg.org/spec/Commons/ContextualDesignators/> .
@prefix cmns-dt: <https://www.omg.org/spec/Commons/DatesAndTimes/> .
@prefix cmns-id: <https://www.omg.org/spec/Commons/Identifiers/> .
@prefix cmns-pt: <https://www.omg.org/spec/Commons/PartiesAndSituations/> .
@prefix cmns-qtu: <https://www.omg.org/spec/Commons/QuantitiesAndUnits/> .
@prefix cmns-txt: <https://www.omg.org/spec/Commons/TextDatatype/> .
@prefix cmns-utl: <https://www.omg.org/spec/Commons/Utilities/> .
@prefix mymap: <http://example.org/mapping/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

"""

    # 2. Iterate through all accepted mappings in session_state
    mapping_blocks = []
    for column_name, suggestion in mappings.items():
        # Skip rejected or invalid mappings
        if suggestion.get('status') == 'rejected' or not suggestion.get('maps_to_property'):
            continue
        
        mapping_type = suggestion.get('mapping_type', 'ColumnMapping')  # Default is ColumnMapping
        map_name = f"map_{column_name.replace(' ', '_')}"  # Create a valid turtle subject name
        
        block = f"<#{map_name}>\n"

        # 3. Generate different Turtle syntax blocks based on different mapping_types
        if mapping_type == "ColumnMapping":
            block += f"    a mymap:ColumnMapping ;\n"
            block += f'    mymap:sourceColumn "{column_name}" ;\n'
            block += f"    mymap:mapsToProperty <{suggestion.get('maps_to_property')}> ;\n"
            # Use the new part_of field
            if suggestion.get('part_of'):
                block += f"    mymap:partOf <{suggestion.get('part_of')}> ;\n"
            block += f"    mymap:confidenceScore {suggestion.get('confidence_score', 5)} ;\n"
            block += f'    mymap:justification "{suggestion.get("justification", "No justification provided")}" .\n'

        elif mapping_type == "IdentifierMapping":
            block += f"    a mymap:IdentifierMapping ;\n"
            block += f'    mymap:sourceColumn "{column_name}" ;\n'
            block += f"    mymap:mapsToIdentifier <{suggestion.get('maps_to_property')}> ;\n"
            if suggestion.get('part_of'):
                block += f"    mymap:partOf <{suggestion.get('part_of')}> ;\n"
            block += f"    mymap:confidenceScore {suggestion.get('confidence_score', 5)} ;\n"
            block += f'    mymap:justification "{suggestion.get("justification", "No justification provided")}" .\n'
        
        elif mapping_type == "ClassificationMapping":
            block += f"    a mymap:ClassificationMapping ;\n"
            block += f'    mymap:sourceColumn "{column_name}" ;\n'
            block += f"    mymap:mapsToClassification <{suggestion.get('maps_to_property')}> ;\n"
            if suggestion.get('part_of'):
                block += f"    mymap:partOf <{suggestion.get('part_of')}> ;\n"
            block += f"    mymap:confidenceScore {suggestion.get('confidence_score', 5)} ;\n"
            block += f'    mymap:justification "{suggestion.get("justification", "No justification provided")}" .\n'
        
        else:  # Default
            block += f"    a mymap:ColumnMapping ;\n"
            block += f'    mymap:sourceColumn "{column_name}" ;\n'
            block += f"    mymap:mapsToProperty <{suggestion.get('maps_to_property')}> ;\n"
            if suggestion.get('part_of'):
                block += f"    mymap:partOf <{suggestion.get('part_of')}> ;\n"
            block += f"    mymap:confidenceScore {suggestion.get('confidence_score', 5)} ;\n"
            block += f'    mymap:justification "{suggestion.get("justification", "No justification provided")}" .\n'
        
        mapping_blocks.append(block)
    
    # 4. Combine the final TTL content
    header = prefixes + "\n# LingoMap Generated Mapping Rules\n# Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n"
    return header + "\n".join(mapping_blocks)

if __name__ == "__main__":
    main()
