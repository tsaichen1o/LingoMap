"""
LingoMap Semantic Mapping Tool
A smart tool that combines data analysis, RAG retrieval, and large language models to help you add standardized semantics to your data sets.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import sys
import streamlit.components.v1 as components

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
def get_column_clusters(df: pd.DataFrame = None):
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

# --- Main Application ---
def main():
    st.title("✈️ LingoMap Semantic Mapping Tool")
    st.markdown("### Smart Data Semanticization Tool - Based on FIBO Ontology")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Select CSV file",
            type=['csv'],
            help="Upload the CSV file you want to perform semantic mapping on"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success(f"✅ Successfully loaded {len(df)} rows, {len(df.columns)} columns")
                
                # Display basic statistics
                st.subheader("📊 Data Overview")
                st.write(f"**Number of rows**: {len(df)}")
                st.write(f"**Number of columns**: {len(df.columns)}")
                st.write(f"**Memory usage**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
            except Exception as e:
                st.error(f"❌ File loading failed: {e}")
                st.session_state.df = None
    
    # Main content area
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Step 1: Data Overview
        st.header("📋 Step 1: Data Overview")
        with st.expander("View data preview", expanded=True):
            st.dataframe(df.head(), use_container_width=True)
        
        # Step 2: Select Columns (Optimized Version)
        st.subheader("2. Select Columns (Semantic Grouping)")
        # Get clustering results
        clustered_columns = get_column_clusters(df)

        for i, cluster_info in enumerate(clustered_columns):
            cluster_name = cluster_info['name']
            columns = cluster_info['columns']
            # Create a title for the group, e.g. "Cluster 1: Address Fields"
            with st.expander(f"Group {i+1}: {cluster_name} ({len(columns)} columns)"):
                # Create a button or select box to handle this group
                for col in columns:
                    if st.button(col, key=f"btn_{col}"):
                        st.session_state.current_column = col
                        st.rerun()
        
        # Step 3: Field Analysis and Mapping
        if st.session_state.current_column:
            col_name = st.session_state.current_column
            
            # Reset suggestion when column changes
            if 'last_analyzed_column' not in st.session_state or st.session_state.last_analyzed_column != col_name:
                st.session_state.current_suggestion = None
                st.session_state.modification_mode = False
                st.session_state.last_analyzed_column = col_name
            
            st.header(f"🔍 Step 3: Analyze Field `{col_name}`")
            
            # Display field basic information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unique values", df[col_name].nunique())
            with col2:
                st.metric("Empty values", df[col_name].isna().sum())
            with col3:
                st.metric("Data type", str(df[col_name].dtype))
            
            # Display field preview
            with st.expander("Field data preview"):
                st.write(df[col_name].head(10).tolist())
            
            # AI suggestion button
            if st.button(f"Get AI suggestion for `{col_name}`", type="primary", use_container_width=True):
                st.session_state.modification_mode = False  # Ensure not in modification mode
                engine = get_mapping_engine()
                if engine:
                    # --- Dynamic setting of Temperature ---
                    # If this field has been rejected before, use a higher temperature (e.g. 0.7) to increase creativity
                    # Otherwise, use a lower temperature (e.g. 0.2) to get more stable and consistent results
                    temperature = 0.7 if col_name in st.session_state.rejected_columns else 0.2

                    # Show current exploration mode in UI
                    if temperature > 0.5:
                        st.info("Mode: Deep exploration (AI will try to provide more diverse suggestions)")
                    
                    named_clusters = get_column_clusters(df)
                    cluster_name = next((c['name'] for c in named_clusters if col_name in c['columns']), "Unknown")
                    
                    with st.spinner("AI core engine is starting and performing inference... This may take 10-20 seconds."):
                        suggestion = engine.suggest_mapping(
                            column_name=col_name, 
                            series=df[col_name],
                            model_entities=st.session_state.model_entities,
                            cluster_name=cluster_name,
                            temperature=temperature
                        )
                        st.session_state.current_suggestion = suggestion
                        st.session_state.last_analyzed_column = col_name
                else:
                    st.session_state.current_suggestion = {
                        "error": "Core engine initialization failed, please check API key settings."
                    }
            
            # Display suggestion results
            if st.session_state.current_suggestion:
                suggestion = st.session_state.current_suggestion
                
                if "error" in suggestion:
                    st.error(suggestion["error"])
                else:
                    with st.container(border=True):
                        st.subheader(f"Suggestion for `{col_name}`")
                        st.markdown(f"**Part of (belongs to entity)**: `{suggestion.get('part_of', 'N/A')}`")
                        st.success(f"**Maps to Property**: `{suggestion.get('maps_to_property', 'N/A')}`")
                        
                        # Automatically convert confidence index format
                        confidence_score = suggestion.get('confidence_score', 'N/A')
                        try:
                            score = float(confidence_score)
                            # If the score is between 0 and 1, multiply by 100
                            if 0 <= score <= 1:
                                score = int(round(score * 100))
                            # If the score is between 1 and 10, multiply by 10
                            elif 1 < score <= 10:
                                score = int(round(score * 10))
                            else:
                                score = int(round(score))
                            confidence_score = score
                        except (ValueError, TypeError):
                            pass
                        
                        st.warning(f"**Confidence index**: {confidence_score} / 100")
                        
                        with st.expander("**View AI's justification**"):
                            st.write(suggestion.get('justification', 'No justification provided.'))
                    
                    # Accept/reject buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("✅ Accept suggestion", type="primary", use_container_width=True):
                            st.session_state.mappings[col_name] = {
                                **suggestion,
                                'status': 'accepted',
                                'timestamp': datetime.now().isoformat()
                            }
                            st.success(f"✅ Accepted mapping suggestion for `{col_name}`")
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ Reject suggestion", use_container_width=True):
                            st.session_state.rejected_columns.add(col_name)
                            st.session_state.mappings[col_name] = {
                                **suggestion,
                                'status': 'rejected',
                                'timestamp': datetime.now().isoformat()
                            }
                            st.error(f"❌ Rejected mapping suggestion for `{col_name}`")
                            st.rerun()
                    
                    with col3:
                        if st.button("🔧 Manual modification", use_container_width=True):
                            st.session_state.modification_mode = True
                            st.rerun()
                    
                    # Manual modification mode
                    if st.session_state.modification_mode:
                        st.write("---")
                        with st.container(border=True):
                            st.subheader("✏️ Manual Modification & Re-evaluation")
                            st.info("If you've found a better ontology term, paste its full URI below for AI re-evaluation.")
                            
                            user_uri = st.text_input("Paste new vocabulary URI:", key="user_provided_uri")
                            
                            if st.button("🤖 Re-evaluate this URI", use_container_width=True, type="primary"):
                                if user_uri and user_uri.startswith("http"):
                                    engine = get_mapping_engine()
                                    if engine:
                                        with st.spinner("AI is re-evaluating your suggestion..."):
                                            profile = profile_column(col_name, df[col_name])
                                            new_suggestion = engine.reevaluate_mapping(
                                                profile=profile, 
                                                user_provided_uri=user_uri,
                                                model_entities=st.session_state.model_entities
                                            )
                                            st.session_state.current_suggestion = new_suggestion
                                            st.session_state.modification_mode = False
                                            st.rerun()
                                else:
                                    st.error("Please enter a valid URI starting with http.")
    
    # Step 4: Mapping Overview
    if st.session_state.mappings:
        st.header("📊 Step 4: Mapping Overview")
        
        # Convert mappings dictionary to DataFrame for better display
        overview_data = []
        for col, map_data in st.session_state.mappings.items():
            overview_data.append({
                "Column": col,
                "Status": "✅ Accepted" if map_data.get('status') != 'rejected' else "❌ Rejected",
                "Part of (belongs to entity)": map_data.get('part_of', 'N/A'),
                "Maps to Property": map_data.get('maps_to_property', 'N/A')
            })
        st.dataframe(pd.DataFrame(overview_data), use_container_width=True)
        
        # Generate TTL rules
        if st.button("🔧 Generate TTL rules", type="primary", use_container_width=True):
            ttl_content = generate_rules_ttl(st.session_state.mappings)
            st.text_area("Generated TTL rules", ttl_content, height=400)
            
            # Download button
            st.download_button(
                label="📥 Download TTL file",
                data=ttl_content,
                file_name=f"lingomap_rules_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ttl",
                mime="text/plain"
            )

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

def display_mermaid_chart(code: str):
    """Use components.html to reliably render Mermaid charts"""
    components.html(
        f"""
        <pre class="mermaid" style="text-align: center;">
            {code}
        </pre>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        height=650  # Adjust height to fit the chart
    )

if __name__ == "__main__":
    main()
