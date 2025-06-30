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
from contextlib import redirect_stdout
import io


sys.path.append(os.path.dirname(__file__))

try:
    from core_engine import CoreMappingEngine
    from column_clusterer import cluster_columns
    from ttl_generator import generate_semantic_blueprint_ttl
    from validate_mapping import MappingValidator
    from run_evaluation import Evaluator
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
                
                # --- THIS IS THE FIX ---
                # Correctly unpack the tuple returned by the function
                clusters, entities = engine.generate_semantic_entities(df, n_clusters)
                
                # Store them in separate session state variables
                st.session_state.llm_clusters = clusters
                st.session_state.generated_entities = entities
                # --------------------
            st.success("Entity Conception complete! ✅")
        else:
            st.error("Core engine initialization failed. Please check your API key or environment settings.")

    st.divider()
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
            
        # st.divider()
        
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

    # --- STAGE 3: DETAILED COLUMN MAPPING (Placeholder) ---
    st.divider()
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
                all_analyzed = all(col in st.session_state.get('all_suggestions', {}) for col in columns)
                group_icon = "🟢" if all_analyzed else "⚪️"
                # --- 新增：分析整個群組的按鈕 ---
                if st.button(f"{group_icon} Analyze All Columns in '{cluster_name}'", key=f"btn_group_{cluster_name}", type="primary"):
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
                    analyzed = col in st.session_state.get('all_suggestions', {})
                    icon = "🟢" if analyzed else "⚪️"
                    btn_label = f"{icon} Analyze `{col}`"
                    if st.button(btn_label, key=f"btn_{col}"):
                        st.session_state.current_column = col
                        st.session_state.current_suggestion = None
                        st.rerun()

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
                # --- NEW: Check if the suggestion is a rule or a mapping ---
                if suggestion.get("isRule"):
                    with st.container(border=True):
                        st.info(f"**Conditional Rule for `{suggestion.get('sourceColumn')}`**")
                        st.markdown(f"**Rule Type:** `{suggestion.get('ruleType', 'N/A')}`")
                        condition_data = suggestion.get('condition', {})
                        action_data = suggestion.get('action', {})
                        # --- FIX: Ensure condition and action are dictionaries ---
                        # Check if they are actually dictionaries before trying to access their items
                        if isinstance(condition_data, dict) and isinstance(action_data, dict):
                            # If they are dicts, we can safely use .get()
                            operator = condition_data.get('operator', '[op]')
                            condition_value = condition_data.get('value', '[val]')
                            action_type = action_data.get('type', '[type]')
                            action_value = action_data.get('value', '[val]')

                            st.code(
                                f"IF {suggestion.get('sourceColumn')} {operator} '{condition_value}' "
                                f"THEN {action_type} `{action_value}` "
                                f"ON `{suggestion.get('targetEntity')}`",
                                language="bash"
                            )
                        else:
                            # If they are not dicts, it indicates a malformed response from the LLM.
                            st.error("Rule format from AI is invalid. Could not parse 'condition' or 'action'.")
                            st.json({"expected_condition_format": {"operator": "...", "value": "..."} , "received_condition": condition_data})
                            st.json({"expected_action_format": {"type": "...", "value": "..."} , "received_action": action_data})
                        # --- END OF FIX ---

                        with st.expander("View Justification"):
                            st.write(f"*{suggestion.get('justification', 'No justification.')}*")
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
    
    # --- STAGE 4: FINAL BLUEPRINT GENERATION ---
    st.divider()
    st.header("Stage 4: Generate Final Semantic Blueprint")
    # 只有在至少有一些欄位映射建議產生後，才顯示這個階段
    if st.session_state.get('all_suggestions'):
        st.write("Click the button below to compile all AI-generated definitions, relationships, and mappings into a complete TTL file.")
        
        if st.button("🚀 Generate Full TTL Blueprint", use_container_width=True, type="primary"):
            with st.spinner("Compiling all stages into the final TTL file..."):
                # 從 session_state 中讀取所有需要的數據
                entities = st.session_state.get('generated_entities', [])
                relationships = st.session_state.get('generated_relationships', [])
                column_mappings = st.session_state.get('all_suggestions', {})
                
                # 呼叫 TTL 生成器
                ttl_content = generate_semantic_blueprint_ttl(
                    entities=entities,
                    relationships=relationships,
                    column_mappings=column_mappings
                )
                
                # 將生成的內容存儲在 session_state 中以便顯示和下載
                st.session_state.final_ttl_content = ttl_content
            st.success("TTL Blueprint generated successfully! ✅")

        # 如果 TTL 內容已生成，則顯示它和下載按鈕
        if 'final_ttl_content' in st.session_state:
            ttl_to_show = st.session_state.final_ttl_content
            
            st.subheader("📄 Generated TTL Preview")
            st.text_area("TTL Content", ttl_to_show, height=400)
            
            # 提供下載按鈕
            st.download_button(
                label="📥 Download .ttl File",
                data=ttl_to_show,
                file_name=f"LingoMap_Blueprint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ttl",
                mime="text/turtle"
            )
    else:
        st.info("Please analyze at least one column group in Stage 3 to enable blueprint generation.")
        
    # --- STAGE 5: VALIDATION & EVALUATION ---
    st.divider()
    st.header("Stage 5: Validation & Evaluation")

    # 只有在 TTL 檔案生成後才顯示此階段
    if 'final_ttl_content' in st.session_state and st.session_state.final_ttl_content:
        
        # --- 5A: Blueprint Validation ---
        st.subheader("5A. Validate TTL Blueprint Syntax & Structure")
        st.write("This check validates the generated TTL file itself for syntactical correctness and structural integrity against your custom schema.")

        if st.button("🔬 Validate Generated TTL Blueprint"):
            with st.spinner("Validating TTL blueprint..."):
                # 將 session state 中的 TTL 內容寫入一個暫存檔案
                temp_ttl_path = "temp_generated_rules.ttl"
                with open(temp_ttl_path, "w", encoding="utf-8") as f:
                    f.write(st.session_state.final_ttl_content)
                
                # 實例化驗證器並執行
                validator = MappingValidator(temp_ttl_path)
                
                # 捕獲 print 輸出以便在 Streamlit 中顯示
                log_stream = io.StringIO()
                with redirect_stdout(log_stream):
                    validator.run_all_checks()
                
                # 將驗證報告存儲在 session state 中
                st.session_state.validation_report = log_stream.getvalue()
                st.session_state.validation_summary = validator.results
            
            st.success("Blueprint validation complete!")

        # 顯示驗證報告
        if 'validation_report' in st.session_state:
            summary = st.session_state.validation_summary
            st.code(st.session_state.validation_report, language="log")

            errors = summary.get('errors', [])
            warnings = summary.get('warnings', [])
            
            if not errors:
                st.success("🎉 Congratulations! No critical errors found in the TTL blueprint.")
            else:
                st.error(f"🚨 Found {len(errors)} critical error(s). Please review the log above.")

        st.divider()

        # --- 5B: Performance Evaluation ---
        st.subheader("5B. Evaluate AI Accuracy vs. Golden Standard")
        st.write("This check compares the AI's mapping suggestions against your manually created `lingomap_rules.ttl` file to calculate performance metrics like accuracy.")

        # 指定您的黃金準則檔案路徑
        golden_standard_file = "lingomap_rules.ttl"

        if not os.path.exists(golden_standard_file):
            st.warning(f"Golden standard file `{golden_standard_file}` not found. Cannot perform evaluation.")
        elif st.button("🏆 Evaluate AI Performance"):
            with st.spinner("Comparing AI suggestions against the golden standard..."):
                try:
                    evaluator = Evaluator(golden_standard_file)
                    report = evaluator.evaluate(st.session_state.get('all_suggestions', {}))
                    st.session_state.evaluation_report = report
                except Exception as e:
                    st.error(f"An error occurred during evaluation: {e}")
            
            st.success("Performance evaluation complete!")

        # 顯示評估報告
        if 'evaluation_report' in st.session_state:
            report = st.session_state.evaluation_report
            st.markdown("#### Evaluation Summary")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{report['accuracy']:.2f}%")
            col2.metric("Correct Predictions", report['correct'])
            col3.metric("Incorrect Predictions", report['incorrect'])
            
            with st.expander("View Detailed Comparison Log"):
                st.dataframe(pd.DataFrame(report['log']))

    else:
        st.info("Please generate the final TTL blueprint in Stage 4 to enable validation and evaluation.")
# --- Run the main application ---

if __name__ == "__main__":
    main()
