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
        'api_key': None,
        'engine_instance': None,
        'generated_entities': None,
        'current_column': None,
        'current_suggestion': None,
        'modification_mode': False,
        'mappings': {},
        'rejected_columns': set()
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

    # --- STAGE 2: DETAILED COLUMN MAPPING ---
    # (The rest of your UI for detailed mapping would go here)
    st.header("Stage 2: Detailed Column Mapping")
    if not st.session_state.generated_entities:
        st.info("Please run Stage 1 'Generate Core Entities' first.")
    else:
        # Your previous logic for selecting columns from clusters and suggesting mappings
        # can be adapted here.
        st.write("Select a column from the groups below to get a detailed mapping suggestion.")
        
        clusters = cluster_columns(df)
        for i, cluster_info in enumerate(clusters):
            cluster_name = cluster_info.get('name', f"Group {i+1}")
            columns = cluster_info.get('columns', [])
            
            with st.expander(f"**{cluster_name}** ({len(columns)} columns)"):
                for col in columns:
                    if st.button(col, key=f"btn_{col}"):
                        st.session_state.current_column = col
                        # This would trigger the detailed mapping logic from your original file
                        st.info(f"Selected '{col}' for detailed mapping. (Logic to be implemented)")
                        st.rerun()

    # (Further stages for mapping overview and TTL generation would follow)
# def main():
#     st.set_page_config(layout="wide", page_title="LingoMap Semantic Mapper")
#     init_session_state()

#     st.title("✈️ LingoMap Semantic Mapper")
#     st.markdown("An intelligent workbench for mapping CSV data to formal ontologies like FIBO.")
    
#     # Sidebar
#     with st.sidebar:
#         st.header("📁 Data Upload")
#         uploaded_file = st.file_uploader(
#             "Select CSV file",
#             type=['csv'],
#             help="Upload the CSV file you want to perform semantic mapping on"
#         )
#         st.markdown("---")
#         if uploaded_file:
#             try:
#                 df = pd.read_csv(uploaded_file)
#                 st.session_state.df = df
#             except Exception as e:
#                 st.error(f"❌ File Load Failed: {e}")
#                 st.session_state.df = None
        
#         if st.session_state.df is not None:
#             st.success(f"✅ Loaded {len(st.session_state.df)} rows.")
#             st.subheader("Clustering Settings")
#             n_clusters = st.slider(
#                 "Expected Number of Entities",
#                 min_value=2,
#                 max_value=15,
#                 value=8,
#                 help="Adjust this to match the number of core concepts in your data."
#             )
#             st.session_state.n_clusters = n_clusters
    
#     # Main content area
#         if st.session_state.df is None:
#             st.info("👋 Welcome to LingoMap! Please upload a CSV file and enter your API key to begin.")
#             return

#         df = st.session_state.df
        
#         # Step 1: Data Overview
#         st.header("📋 Step 1: Data Overview")
#         with st.expander("View data preview", expanded=True):
#             st.dataframe(df.head(), use_container_width=True)
        
#         # STEP 1: ENTITY CONCEPTION
#         st.header("Stage 1: AI-Powered Entity Conception")
#         st.markdown(
#             "In this stage, the system analyzes your data, groups related columns into clusters, "
#             "and uses AI to propose a core business entity for each cluster."
#         )
        
#         if st.button("🚀 Generate Core Entities", use_container_width=True, type="primary"):
#             engine = st.session_state.engine_instance
#             if engine:
#                 with st.spinner("Analyzing columns and consulting with Gemini... This may take a moment."):
#                     clustering_params = {'n_clusters': st.session_state.get('n_clusters', 8)}
#                     entities = engine.generate_semantic_entities(df, clustering_params)
#                     st.session_state.generated_entities = entities
#                 st.success("✅ Entity Conception complete!")
#                 st.rerun() # Rerun to display the results immediately
#             else:
#                 st.error("Engine not initialized.")
                
#         # Display generated entities
#         if st.session_state.generated_entities:
#             st.subheader("🤖 AI-Generated Entity Definitions")
#             entities_data = st.session_state.generated_entities
            
#             # Create a flexible grid layout
#             num_entities = len(entities_data)
#             cols = st.columns(num_entities)
            
#             for i, entity in enumerate(entities_data):
#                 with cols[i]:
#                     if "error" in entity:
#                         st.error(f"Cluster '{entity.get('cluster_name', 'Unknown')}' failed.")
#                         st.json(entity)
#                     else:
#                         with st.container(border=True):
#                             st.info(f"**{entity.get('entityLabel', 'N/A')}**")
#                             st.markdown(f"**ID:** `{entity.get('entityId', 'N/A')}`")
#                             st.markdown(f"**Maps to Class:**")
#                             st.code(entity.get('mapsToClass', 'N/A'), language='text')
#                             st.markdown(f"**Comment:** *{entity.get('entityComment', 'N/A')}*")
#                             with st.expander("View Clustered Columns"):
#                                 st.write(entity.get('cluster_columns', []))

#             st.divider()
#             # STEP 2: COLUMN MAPPING (FUTURE WORK)
#             st.header("Stage 2: Detailed Column-to-Property Mapping")
#             st.info("This next stage will allow you to map individual columns within each defined entity to specific ontology properties. (Coming soon!)")
            
#             # Step 2: Select Columns (Optimized Version)
#             st.subheader("2. Select Columns (Semantic Grouping)")
#             # Get clustering results
#             clustered_columns = get_column_clusters(df)

#             for i, cluster_info in enumerate(clustered_columns):
#                 cluster_name = cluster_info['name']
#                 columns = cluster_info['columns']
#                 # Create a title for the group, e.g. "Cluster 1: Address Fields"
#                 with st.expander(f"Group {i+1}: {cluster_name} ({len(columns)} columns)"):
#                     # Create a button or select box to handle this group
#                     for col in columns:
#                         if st.button(col, key=f"btn_{col}"):
#                             st.session_state.current_column = col
#                             st.rerun()
            
            
#             # Step 3: Field Analysis and Mapping
#             if st.session_state.current_column:
#                 col_name = st.session_state.current_column
                
#                 # Reset suggestion when column changes
#                 if 'last_analyzed_column' not in st.session_state or st.session_state.last_analyzed_column != col_name:
#                     st.session_state.current_suggestion = None
#                     st.session_state.modification_mode = False
#                     st.session_state.last_analyzed_column = col_name
                
#                 st.header(f"🔍 Step 3: Analyze Field `{col_name}`")
                
#                 # Display field basic information
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("Unique values", df[col_name].nunique())
#                 with col2:
#                     st.metric("Empty values", df[col_name].isna().sum())
#                 with col3:
#                     st.metric("Data type", str(df[col_name].dtype))
                
#                 # Display field preview
#                 with st.expander("Field data preview"):
#                     st.write(df[col_name].head(10).tolist())
                
#                 # AI suggestion button
#                 if st.button(f"Get AI suggestion for `{col_name}`", type="primary", use_container_width=True):
#                     st.session_state.modification_mode = False  # Ensure not in modification mode
#                     engine = get_mapping_engine()
#                     if engine:
#                         # --- Dynamic setting of Temperature ---
#                         # If this field has been rejected before, use a higher temperature (e.g. 0.7) to increase creativity
#                         # Otherwise, use a lower temperature (e.g. 0.2) to get more stable and consistent results
#                         temperature = 0.7 if col_name in st.session_state.rejected_columns else 0.2

#                         # Show current exploration mode in UI
#                         if temperature > 0.5:
#                             st.info("Mode: Deep exploration (AI will try to provide more diverse suggestions)")
                        
#                         named_clusters = get_column_clusters(df)
#                         cluster_name = next((c['name'] for c in named_clusters if col_name in c['columns']), "Unknown")
                        
#                         with st.spinner("AI core engine is starting and performing inference... This may take 10-20 seconds."):
#                             suggestion = engine.suggest_mapping(
#                                 column_name=col_name, 
#                                 series=df[col_name],
#                                 model_entities=st.session_state.model_entities,
#                                 cluster_name=cluster_name,
#                                 temperature=temperature
#                             )
#                             st.session_state.current_suggestion = suggestion
#                             st.session_state.last_analyzed_column = col_name
#                     else:
#                         st.session_state.current_suggestion = {
#                             "error": "Core engine initialization failed, please check API key settings."
#                         }
                
#                 # Display suggestion results
#                 if st.session_state.current_suggestion:
#                     suggestion = st.session_state.current_suggestion
                    
#                     if "error" in suggestion:
#                         st.error(suggestion["error"])
#                     else:
#                         with st.container(border=True):
#                             st.subheader(f"Suggestion for `{col_name}`")
#                             st.markdown(f"**Part of (belongs to entity)**: `{suggestion.get('part_of', 'N/A')}`")
#                             st.success(f"**Maps to Property**: `{suggestion.get('maps_to_property', 'N/A')}`")
                            
#                             # Automatically convert confidence index format
#                             confidence_score = suggestion.get('confidence_score', 'N/A')
#                             try:
#                                 score = float(confidence_score)
#                                 # If the score is between 0 and 1, multiply by 100
#                                 if 0 <= score <= 1:
#                                     score = int(round(score * 100))
#                                 # If the score is between 1 and 10, multiply by 10
#                                 elif 1 < score <= 10:
#                                     score = int(round(score * 10))
#                                 else:
#                                     score = int(round(score))
#                                 confidence_score = score
#                             except (ValueError, TypeError):
#                                 pass
                            
#                             st.warning(f"**Confidence index**: {confidence_score} / 100")
                            
#                             with st.expander("**View AI's justification**"):
#                                 st.write(suggestion.get('justification', 'No justification provided.'))
                        
#                         # Accept/reject buttons
#                         col1, col2, col3 = st.columns(3)
#                         with col1:
#                             if st.button("✅ Accept suggestion", type="primary", use_container_width=True):
#                                 st.session_state.mappings[col_name] = {
#                                     **suggestion,
#                                     'status': 'accepted',
#                                     'timestamp': datetime.now().isoformat()
#                                 }
#                                 st.success(f"✅ Accepted mapping suggestion for `{col_name}`")
#                                 st.rerun()
                        
#                         with col2:
#                             if st.button("❌ Reject suggestion", use_container_width=True):
#                                 st.session_state.rejected_columns.add(col_name)
#                                 st.session_state.mappings[col_name] = {
#                                     **suggestion,
#                                     'status': 'rejected',
#                                     'timestamp': datetime.now().isoformat()
#                                 }
#                                 st.error(f"❌ Rejected mapping suggestion for `{col_name}`")
#                                 st.rerun()
                        
#                         with col3:
#                             if st.button("🔧 Manual modification", use_container_width=True):
#                                 st.session_state.modification_mode = True
#                                 st.rerun()
                        
#                         # Manual modification mode
#                         if st.session_state.modification_mode:
#                             st.write("---")
#                             with st.container(border=True):
#                                 st.subheader("✏️ Manual Modification & Re-evaluation")
#                                 st.info("If you've found a better ontology term, paste its full URI below for AI re-evaluation.")
                                
#                                 user_uri = st.text_input("Paste new vocabulary URI:", key="user_provided_uri")
                                
#                                 if st.button("🤖 Re-evaluate this URI", use_container_width=True, type="primary"):
#                                     if user_uri and user_uri.startswith("http"):
#                                         engine = get_mapping_engine()
#                                         if engine:
#                                             with st.spinner("AI is re-evaluating your suggestion..."):
#                                                 profile = profile_column(col_name, df[col_name])
#                                                 new_suggestion = engine.reevaluate_mapping(
#                                                     profile=profile, 
#                                                     user_provided_uri=user_uri,
#                                                     model_entities=st.session_state.model_entities
#                                                 )
#                                                 st.session_state.current_suggestion = new_suggestion
#                                                 st.session_state.modification_mode = False
#                                                 st.rerun()
#                                     else:
#                                         st.error("Please enter a valid URI starting with http.")
        
#     # Step 4: Mapping Overview
#     if st.session_state.mappings:
#         st.header("📊 Step 4: Mapping Overview")
        
#         # Convert mappings dictionary to DataFrame for better display
#         overview_data = []
#         for col, map_data in st.session_state.mappings.items():
#             overview_data.append({
#                 "Column": col,
#                 "Status": "✅ Accepted" if map_data.get('status') != 'rejected' else "❌ Rejected",
#                 "Part of (belongs to entity)": map_data.get('part_of', 'N/A'),
#                 "Maps to Property": map_data.get('maps_to_property', 'N/A')
#             })
#         st.dataframe(pd.DataFrame(overview_data), use_container_width=True)
        
#         # Generate TTL rules
#         if st.button("🔧 Generate TTL rules", type="primary", use_container_width=True):
#             ttl_content = generate_rules_ttl(st.session_state.mappings)
#             st.text_area("Generated TTL rules", ttl_content, height=400)
            
#             # Download button
#             st.download_button(
#                 label="📥 Download TTL file",
#                 data=ttl_content,
#                 file_name=f"lingomap_rules_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ttl",
#                 mime="text/plain"
#             )

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
