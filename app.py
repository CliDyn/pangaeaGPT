#app.py

import streamlit as st
import os
import importlib
import logging
from io import StringIO           
import pandas as pd
import xarray as xr                
import json
import time                       
import uuid
import base64
from pathlib import Path

from src.ui.directory_explorer import render_directory_explorer
import src.config as config
from src.config import DEPLOYMENT_MODE
# ------------------------------------------------------------------
# Set Page Config first! (This must be the very first Streamlit command)
st.set_page_config(
    page_title="PANGAEA GPT",
    page_icon="img/pangaea-logo.png",
    layout="wide"
)
# ------------------------------------------------------------------

# === Step 1: Force OpenAI API Key ===
if "openai_api_key" not in st.session_state:
    if DEPLOYMENT_MODE == "local":
        try:
            st.session_state["openai_api_key"] = st.secrets["general"]["openai_api_key"]
        except KeyError:
            st.sidebar.warning("⚠️ OpenAI API key not found in .streamlit/secrets.toml. Please enter it below.")
            user_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
            if user_api_key:
                st.session_state["openai_api_key"] = user_api_key
                st.rerun()
            st.stop()
    else:  # Hugging Face mode
        st.sidebar.warning("⚠️ Please enter your OpenAI API key below to enable full functionality.")
        user_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if user_api_key:
            st.session_state["openai_api_key"] = user_api_key
            st.rerun()
        st.stop()

API_KEY = st.session_state["openai_api_key"]
os.environ["OPENAI_API_KEY"] = API_KEY

# === Step 2: Display LangSmith Fields in the Sidebar ===
def img_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

with st.sidebar:
    # Using the path we know works
    logo_path = "img/pangaea-logo.png"
    
    # Display the logo centered using base64 encoding
    if Path(logo_path).exists():
        img_base64 = img_to_base64(logo_path)
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center;">
                <img src="data:image/png;base64,{img_base64}" width="120">
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.title("Configuration")
    # Rest of the sidebar code...

    model_name = st.selectbox(
        "Select Model",
        ["gpt-4.1-mini", "gpt-4.1", "gpt-4.1-nano", "gpt-4o", "o3-mini", "o3", "o4-mini", "codex-mini-latest"],
        index=0,
        key="model_name"
    )
# Determine initial values for LangSmith inputs based on deployment mode
    if DEPLOYMENT_MODE == "local":
        try:
            initial_langsmith_api_key = st.secrets["general"]["langchain_api_key"]
        except KeyError:
            initial_langsmith_api_key = ""
        try:
            initial_langsmith_project_name = st.secrets["general"]["langchain_project_name"]
        except KeyError:
            initial_langsmith_project_name = ""
    else:
        initial_langsmith_api_key = ""
        initial_langsmith_project_name = ""

    # LangSmith API key input (pre-filled in local mode if available)
    langsmith_api_key = st.text_input(
        "LangSmith API Key (optional)",
        type="password",
        value=initial_langsmith_api_key,
        key="langsmith_api_key"
    )
    # LangSmith project name input (pre-filled in local mode if available)
    langsmith_project_name = st.text_input(
        "LangSmith Project Name (optional)",
        value=initial_langsmith_project_name,
        key="langsmith_project_name"
    )
    
    # Search Mode Toggle in Sidebar
    st.divider()
    st.markdown("### Search Settings")
    search_mode = st.radio(
        "Search Mode",
        options=["simple", "deep"],
        format_func=lambda x: "🔍 Simple Search" if x == "simple" else "🔬 Deep Search (Multi-Search)",
        key="search_mode_toggle",
        help="Simple: Quick single search | Deep: Multiple search variations for comprehensive results"
    )
    st.session_state.search_mode = search_mode
    
    # Visual indicator of current mode
    if st.session_state.search_mode == "deep":
        st.info("🔬 **Deep Search Mode**: Will perform multiple search variations and consolidate results for better coverage")
    else:
        st.success("🔍 **Simple Search Mode**: Quick single search for fast results")

# === Step 3: Update Environment Variables for LangSmith Keys ===
langchain_api_key = st.session_state.get("langsmith_api_key") or ""
langchain_project_name = st.session_state.get("langsmith_project_name") or ""

# Add region support
if DEPLOYMENT_MODE == "local":
    try:
        langchain_region = st.secrets["general"].get("langchain_region", "us")
    except KeyError:
        langchain_region = "us"  # Default to US region
else:
    langchain_region = "us"  # Default for non-local deployment

# Set the endpoint based on region
if langchain_region.lower() == "eu":
    langchain_endpoint = "https://eu.api.smith.langchain.com"
else:
    langchain_endpoint = "https://api.smith.langchain.com"

if langchain_api_key and langchain_project_name:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_PROJECT_NAME"] = langchain_project_name
    os.environ["LANGCHAIN_ENDPOINT"] = langchain_endpoint
    os.environ["LANGCHAIN_REGION"] = langchain_region

# === Step 4: Reload the Config Module so It Picks Up the Updated Environment Variables ===
import src.config as config
importlib.reload(config)

# === Step 5: Import Configuration Values, Styles, Utilities, and Agent Functions ===
from src.config import API_KEY, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT_NAME, DEPLOYMENT_MODE
from src.ui.styles import CUSTOM_UI
from src.utils import log_history_event
from langchain_core.messages import HumanMessage, AIMessage
from src.memory import CustomMemorySaver


from PIL import Image
SYSTEM_ICON = "img/11111111.png"
USER_ICON = "img/2222222.png"

# Import logic functions from main.py
from main import (
    initialize_session_state,
    get_search_agent,
    process_search_query,
    add_user_message_to_search,
    add_assistant_message_to_search,
    load_selected_datasets_into_cache,
    set_active_datasets_from_selection,
    get_datasets_info_for_active_datasets,
    add_user_message_to_data_agent,
    add_assistant_message_to_data_agent,
    create_and_invoke_supervisor_agent,
    convert_dataset_to_csv,
    has_new_plot,
    reset_new_plot_flag,
    get_dataset_csv_name,
    set_current_page,
    set_selected_text,
    set_show_dataset,
    set_dataset_for_data_agent,
    ensure_memory,
    ensure_thread_id,
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# === Step 6: Initialize Session State ===
initialize_session_state(st.session_state)

if "oceanographer_agent_used" not in st.session_state:
    st.session_state.oceanographer_agent_used = False
if "ecologist_agent_used" not in st.session_state:
    st.session_state.ecologist_agent_used = False  
if "dataframe_agent_used" not in st.session_state:
    st.session_state.dataframe_agent_used = False

# === Step 7: Load Custom UI Styles (for example, CSS) ===
st.markdown(CUSTOM_UI, unsafe_allow_html=True)

# Add this new initialization for search mode
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "simple"  # Default to simple search

# === Step 8: Create the Search Agent (which now uses the updated keys) ===
search_agent = get_search_agent(
    datasets_info=st.session_state.datasets_info,
    model_name=st.session_state["model_name"],
    api_key=API_KEY
)

# -------------------------
# SEARCH PAGE (Dataset Explorer)
# -------------------------
if st.session_state.current_page == "search":
    st.markdown("## Dataset Explorer")
    
    chat_placeholder = st.container()
    message_placeholder = st.empty()
    chat_input_container = st.container()

    predefined_queries = [
        "Search for data on gelatinous zooplankton in the Fram Strait?",
        "Continuous records of the atmospheric greenhouse gases",
        "Find datasets about renewable energy sources.",
        "Search for prokaryote abundance data on Hakon Mosby volcano",
        "Global distributions of coccolithophores abundance and biomass",
        "Shipboard acoustic doppler current profiling during POSEIDON cruise P414 (POS414)", 
        "Processed data of CTD buoys 2019O1 to 2019O8 MOSAiC", 
        "Retrieve Physical oceanography and current velocity data from mooring AK6-1; get it directly from 10.1594/PANGAEA.954299",
        "Get Processed TACE current meter mooring; retireve it directly from 10.1594/PANGAEA.946892",
        'Download Moored current and temperature measurements in the Faroe Bank Channel; get it using doi: 10.1594/PANGAEA.864110'
    ]
    selected_query = st.selectbox(
        "Select an example or write down your query:",
        [""] + predefined_queries,
        index=0,
        key="selected_query",
    )
    if selected_query != "":
        set_selected_text(st.session_state, selected_query)
    else:
        set_selected_text(st.session_state, st.session_state.get("selected_text", ""))

# Display chat messages (including any table with initial search results)
    with chat_placeholder:
        print(f"DEBUG: USER_ICON type: {type(USER_ICON)}, value: {USER_ICON}")
        for i, message in enumerate(st.session_state.messages_search):
            if message["role"] == "system":
                with st.chat_message(message["role"], avatar=SYSTEM_ICON):
                    st.markdown(message["content"])
            elif message["role"] == "user":
                with st.chat_message(message["role"], avatar=USER_ICON):
                    st.markdown(message["content"])
            else:  # assistant messages
                with st.chat_message(message["role"], avatar=SYSTEM_ICON):
                    # Skip displaying raw JSON arrays (debug output)
                    content = message["content"]
                    if content.strip().startswith("[") and content.strip().endswith("]"):
                        # This looks like a raw JSON array, skip it or show in debug expander
                        with st.expander("🐛 Debug output", expanded=False):
                            st.code(content, language="json")
                    elif "Direct access to DOIs:" in content:
                        # Show DOI access in a compact expander
                        with st.expander("📋 DOIs accessed", expanded=False):
                            st.markdown(content)
                    else:
                        st.markdown(content)
                    
            # Handle tables differently based on whether it's the final consolidated result
            if "table" in message:
                df = pd.read_json(StringIO(message["table"]), orient="split")
                
                # Check if this is the final consolidated result
                content = message.get("content", "")
                is_final_result = ("Search completed:" in content or 
                                 "Deep search completed:" in content or
                                 "search completed:" in content.lower())
                
                # For intermediate results, use an expander (collapsed by default)
                if not is_final_result:
                    # Extract info about this intermediate result
                    expander_title = "🔍 Show intermediate search results"
                    if "Datasets Information:" in message.get("content", ""):
                        expander_title = f"🔍 Intermediate results ({len(df)} datasets found)"
                    
                    with st.expander(expander_title, expanded=False):
                        for index, row in df.iterrows():
                            cols = st.columns([1, 2, 2, 4, 2, 1])
                            cols[0].write(f"**#{row['Number']}**")
                            cols[1].write(f"**Name:** {row['Name']}")
                            cols[2].write(f"**DOI:** [{row['DOI']}]({row['DOI']})")
                            # For intermediate results, show description directly (no nested expander)
                            with cols[3]:
                                st.caption("Description:")
                                st.text(row['Description'][:200] + "..." if len(row['Description']) > 200 else row['Description'])
                            parameters_list = row['Parameters'].split(", ")
                            if len(parameters_list) > 7:
                                parameters_list = parameters_list[:7] + ["..."]
                            cols[4].write(f"**Parameters:** {', '.join(parameters_list)}")
                            checkbox_key = f"select-intermediate-{i}-{index}"
                            with cols[5]:
                                selected = st.checkbox("Select", key=checkbox_key)
                            if selected:
                                st.session_state.selected_datasets.add(row['DOI'])
                            else:
                                st.session_state.selected_datasets.discard(row['DOI'])
                else:
                    # For final results, display normally (visible by default)
                    # Add select all checkbox at the top
                    cols = st.columns([1, 2, 2, 4, 2, 1])
                    with cols[5]:
                        all_dois_in_table = set(df['DOI'].tolist())
                        all_selected = all_dois_in_table.issubset(st.session_state.selected_datasets)
                        
                        select_all = st.checkbox("Select All", key=f"select_all_{i}", value=all_selected)
                        if select_all and not all_selected:
                            st.session_state.selected_datasets.update(all_dois_in_table)
                            st.rerun()
                        elif not select_all and all_selected:
                            st.session_state.selected_datasets -= all_dois_in_table
                            st.rerun()
                    
                    for index, row in df.iterrows():
                        cols = st.columns([1, 2, 2, 4, 2, 1])
                        cols[0].write(f"**#{row['Number']}**")
                        cols[1].write(f"**Name:** {row['Name']}")
                        cols[2].write(f"**DOI:** [{row['DOI']}]({row['DOI']})")
                        with cols[3].expander("See full description"):
                            st.write(row['Description'])
                        parameters_list = row['Parameters'].split(", ")
                        if len(parameters_list) > 7:
                            parameters_list = parameters_list[:7] + ["..."]
                        cols[4].write(f"**Parameters:** {', '.join(parameters_list)}")
                        checkbox_key = f"select-final-{i}-{index}"
                        with cols[5]:
                            selected = st.checkbox("Select", key=checkbox_key, value=row['DOI'] in st.session_state.selected_datasets)
                        if selected:
                            st.session_state.selected_datasets.add(row['DOI'])
                        else:
                            st.session_state.selected_datasets.discard(row['DOI'])

    # --- Single search input form ---
    with chat_input_container:
        st.markdown("<br>" * 2, unsafe_allow_html=True)
        st.markdown("""
            <style>
            .stFormSubmitButton > button {
                background-color: rgba(67, 163, 151, 0.6) !important; 
                color: white !important;
                border: none !important;
                border-radius: 6px !important;
                padding: 8px 14px !important;
                cursor: pointer !important;
                font-size: 14px !important;
                box-shadow: 0 0 15px rgba(67, 163, 151, 0.3) !important;
            }
            .stFormSubmitButton > button:hover {
                background-color: white !important;
                color: rgb(67, 163, 151) !important;
                border: 1px solid rgb(67, 163, 151) !important;
                box-shadow: 0 0 20px rgba(67, 163, 151, 0.5) !important;
            }
            .stFormSubmitButton > button:active {
                transform: translateY(1px) !important;
            }
            </style>
        """, unsafe_allow_html=True)
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input(
                "Enter your query:",
                value=st.session_state.get("selected_text", ""),
                key="chat_input",
            )
            submit_button = st.form_submit_button(label='Send')
        if submit_button and user_input:
            st.session_state.selected_text = ""
            
            # 1. ADD USER MESSAGE FIRST
            st.session_state.messages_search.append({"role": "user", "content": user_input})
            # Log user event
            log_history_event(st.session_state, "user_message", {"page": "search", "content": user_input})
            
            # Create a container for search progress
            with message_placeholder:
                search_container = st.container()
                with search_container:
                    if st.session_state.search_mode == "deep":
                        st.info("🔬 Initiating deep parallel multi-search analysis...")
                    else:
                        st.info("🔍 Performing quick search...")
                    
                    # Create placeholder for search updates
                    search_progress = st.empty()
                    
                    # Add a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Initialize parallel search progress tracking
                    if st.session_state.search_mode == "deep":
                        st.session_state.parallel_search_progress = {
                            'total_queries': 0,
                            'completed_queries': 0,
                            'current_query': '',
                            'results': []
                        }
                        status_text.text("🚀 Analyzing query and planning parallel search strategy...")
                    else:
                        status_text.text("🔍 Searching PANGAEA database...")
                    progress_bar.progress(0.1)
                    
                    # Create a placeholder for real-time parallel search updates
                    parallel_status = st.empty() if st.session_state.search_mode == "deep" else None
                    
                    # Set processing flag for deep search mode
                    if st.session_state.search_mode == "deep":
                        st.session_state.processing = True
                    
                    # 2. EXECUTE SEARCH (this will append the dataset table message)
                    ai_message = process_search_query(user_input, search_agent, st.session_state)
                    
                    # Stop processing flag
                    if st.session_state.search_mode == "deep":
                        st.session_state.processing = False
                    
                    # Complete progress
                    progress_bar.progress(1.0)
                    if st.session_state.search_mode == "deep":
                        # Show simple completion message for deep search
                        status_text.text("✅ Deep parallel search completed!")
                        if parallel_status:
                            # Check if we have parallel search progress data
                            if 'parallel_search_progress' in st.session_state:
                                progress_data = st.session_state.parallel_search_progress
                                results = progress_data.get('results', [])
                                if results:
                                    total_datasets = sum(r.get('count', 0) for r in results)
                                    successful_searches = sum(1 for r in results if r.get('success', False))
                                    parallel_status.success(f"🚀 Enhanced Performance: {successful_searches} parallel searches completed with intelligent metadata extraction (estimated 3-5x speedup)")
                                else:
                                    parallel_status.success(f"🚀 Performance: Parallel search execution completed (estimated 3-5x speedup)")
                            else:
                                parallel_status.success(f"🚀 Performance: Deep search completed with enhanced capabilities")
                    else:
                        status_text.text("✅ Search completed!")
                    
                    time.sleep(1.0)  # Brief pause to show completion
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    search_progress.empty()
                    if parallel_status:
                        parallel_status.empty()
            
            # 3. ADD FINAL ASSISTANT SUMMARY MESSAGE
            # The message with the table has already been added inside process_search_query.
            st.session_state.messages_search.append({"role": "assistant", "content": ai_message})
            
            # Log the assistant event
            log_history_event(st.session_state, "assistant_message", {"page": "search", "content": ai_message})
            
            # Update search tracking
            st.session_state.search_query = user_input
            st.session_state.search_offset = 10
            if st.session_state.datasets_info is not None:
                st.session_state.search_total = len(st.session_state.datasets_info)
            set_show_dataset(st.session_state, False)
            st.rerun()

    # --- Load More button ---
    if st.session_state.datasets_info is not None:
        total_hits = st.session_state.get("search_total", st.session_state.datasets_info.attrs.get('total', 0))
        current_count = st.session_state.datasets_info.shape[0]
        if current_count < total_hits:
            if st.button("Load More Datasets", key="load_more_button"):
                offset = st.session_state.get("search_offset", current_count)
                query = st.session_state.get("search_query", st.session_state.get("selected_text", ""))
                from src.search.search_pg_default import pg_search_default
                new_df = pg_search_default(query, count=10, from_idx=offset)
                if new_df.empty:
                    st.info("No more datasets available.")
                else:
                    st.session_state.datasets_info = pd.concat(
                        [st.session_state.datasets_info, new_df], ignore_index=True
                    )
                    st.session_state.search_offset = offset + new_df.shape[0]
                    st.rerun()

    # --- Send Selected Datasets button ---
    button_placeholder = st.empty()
    if len(st.session_state.selected_datasets) > 0:
        with button_placeholder:
            if st.button('Send Selected Datasets to Data Agent', key='send_datasets_button'):
                # 1. Force the creation of a new session ID to create a new, clean sandbox.
                old_thread_id = st.session_state.get("thread_id") # Remember old ID
                ensure_thread_id(st.session_state, force_new=True)
                new_thread_id = st.session_state.get("thread_id")
                logging.info(f"New Data Agent session started. Sandbox/Thread ID: {new_thread_id}")

                # 2. KILL OLD PERSISTENT REPL TO START WITH CLEAN SLATE
                if old_thread_id:
                    from src.tools.python_repl import REPLManager
                    REPLManager.cleanup_repl(old_thread_id)
                    logging.info(f"Cleaned up persistent REPL for previous session: {old_thread_id}")

                # 3. Clear state from any previous analysis to ensure a clean start.
                st.session_state.messages_data_agent = []
                st.session_state.intermediate_steps = []
                st.session_state.execution_history = []
                st.session_state.oceanographer_agent_used = False
                st.session_state.ecologist_agent_used = False
                st.session_state.visualization_agent_used = False
                st.session_state.dataframe_agent_used = False

                # 3. Load datasets into the newly defined sandbox.
                load_selected_datasets_into_cache(st.session_state.selected_datasets, st.session_state)
                set_active_datasets_from_selection(st.session_state)
                set_current_page(st.session_state, "data_agent")
                st.rerun()
        st.markdown("""
            <style>
            div[data-testid="stButton"] > button {
                position: fixed !important;
                bottom: 80px !important;
                right: 20px !important;
                z-index: 9999 !important;
                background-color: rgba(125, 209, 231, 0.8);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.3s ease;
                box-shadow: 0 0 15px rgba(125, 209, 231, 0.3);
                animation: subtle-glow 2s infinite;
            }
            div[data-testid="stButton"] > button:hover {
                transform: translateY(-2px);
                background-color: rgba(125, 209, 231, 1);
                box-shadow: 0 0 20px rgba(125, 209, 231, 0.5);
            }
            @keyframes subtle-glow {
                0% { box-shadow: 0 0 15px rgba(125, 209, 231, 0.3); }
                50% { box-shadow: 0 0 20px rgba(125, 209, 231, 0.5); }
                100% { box-shadow: 0 0 15px rgba(125, 209, 231, 0.3); }
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        button_placeholder.empty()

    # Display active datasets (if any) and allow downloads or sending one dataset to Data Agent
    if st.session_state.active_datasets:
        for doi in st.session_state.active_datasets:
            dataset, name = st.session_state.datasets_cache.get(doi, (None, None))
            if dataset is not None:
                csv_data = convert_dataset_to_csv(dataset)
                with st.expander(f"Current Dataset: {doi}", expanded=st.session_state.show_dataset):
                    if isinstance(dataset, pd.DataFrame):
                        st.dataframe(dataset)
                    elif isinstance(dataset, xr.Dataset):
                        st.write(dataset)
                    else:
                        st.write("Unsupported data type")
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col1:
                        if csv_data:
                            st.download_button(
                                label="Download data as CSV",
                                data=csv_data,
                                file_name=get_dataset_csv_name(doi),
                                mime='text/csv',
                                key=f"download-{doi.split('/')[-1]}",
                                use_container_width=True
                            )
                        else:
                            st.write("Download not available for this dataset type.")
                    with col3:
                        if st.button(f"Send {doi} to Data Agent", use_container_width=True):
                            set_dataset_for_data_agent(st.session_state, doi, csv_data, dataset, name)
    message_placeholder = st.empty()


# -------------------------
# DATA AGENT PAGE
# -------------------------
elif st.session_state.current_page == "data_agent":
    st.markdown("## Data Agent")
    message_placeholder = st.empty()

    # --- START: Restore Buttons ---
    logging.info("Entered Data Agent page") # Keep this log line

    # --- Add Button Row ---
    button_cols = st.columns([1, 1, 1, 3]) # Adjust ratios as needed

    with button_cols[0]:
        # Save history button
        if st.button("Export History", key="export_history_data_agent", use_container_width=True):
            history_data = st.session_state.get("execution_history", [])
            try:
                # Attempt to pretty-print JSON
                json_string = json.dumps(history_data, indent=4, ensure_ascii=False)
            except TypeError as e:
                # Fallback for objects that aren't JSON serializable
                logging.error(f"Error serializing history: {e}. Using simple string conversion.")
                json_string = str(history_data)

            # Use a unique key for the download button itself to avoid conflicts
            download_key = f"download_hist_{uuid.uuid4()}" 
            st.download_button(
                label="Download History (JSON)",
                data=json_string.encode('utf-8'),
                file_name="pangaea_session_history.json",
                mime="application/json",
                key=download_key
            )
            logging.info("Initiated session history download")

    with button_cols[1]:
        # Clear History button
        if st.button("Clear History", key="clear_history_data_agent", use_container_width=True):
            st.session_state.messages_data_agent = []
            st.session_state.intermediate_steps = []
            # Optionally clear execution history as well
            # st.session_state.execution_history = []
            logging.info("Cleared Data Agent history")
            st.rerun()

    with button_cols[2]:
         # Back to Search button
         if st.button("Back to Search", key="back_to_search_data_agent", use_container_width=True, type="secondary"):
             set_current_page(st.session_state, "search")
             logging.info("Navigated back to Search page")
             st.rerun()

    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 1rem;'>", unsafe_allow_html=True) # Add a separator line with adjusted margin
    # --- END: Restore Buttons ---

    # --- START: Display Agent Steps Title with Icon ---
    agent_icon_path = "img/agent_icon.png" 
    agent_steps_label = "Agent Steps"
    try:
        if Path(agent_icon_path).exists():
            agent_icon_base64 = img_to_base64(agent_icon_path)
            # Use markdown with HTML to display icon and text, styled like a subheader
            st.markdown(
                f'''<div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                       <img src="data:image/png;base64,{agent_icon_base64}" width="24" height="24" style="margin-right: 10px; vertical-align: middle;">
                       <h3 style="margin: 0; padding: 0; border: none; background: none; text-shadow: none; vertical-align: middle;">{agent_steps_label}</h3>
                   </div>''',
                unsafe_allow_html=True
            )
        else:
            # Fallback if icon not found
            st.subheader(f"🕵️ {agent_steps_label}")
            logging.warning(f"Agent icon not found at: {agent_icon_path}")
    except Exception as e:
        # Fallback on any error during icon loading/encoding
        st.subheader(f"🕵️ {agent_steps_label}")
        logging.error(f"Error loading agent icon: {e}")
    # --- END: Display Agent Steps Title with Icon ---

    # Define the container for the callback handler output
    callback_container = st.container()
    st.markdown("---") # Optional separator

    #Render Directory
    render_directory_explorer()

    # --- Existing Code Continues Below ---
    datasets_info = get_datasets_info_for_active_datasets(st.session_state)
    logging.info(f"Loaded {len(datasets_info)} datasets for Data Agent")
    ensure_memory(st.session_state)
    ensure_thread_id(st.session_state)
    user_input = st.chat_input("Enter your query:")
    if user_input:
        st.session_state.processing = True
        st.session_state.messages_data_agent.append({"role": "user", "content": f"{user_input}"})
        logging.info(f"User input in Data Agent: {user_input}")
        log_history_event(
            st.session_state,
            "user_message",
            {"page": "data_agent", "content": user_input}
        )
        user_query = user_input
        # Show "processing" message
        with message_placeholder:
            st.info("Your request is being processed. You can see the progress below in the Agent Steps panel.")
        # Instantiate the StreamlitCallbackHandler
        st_callback = StreamlitCallbackHandler(
            parent_container=callback_container,
            max_thought_containers=4,
            expand_new_thoughts=True,
            collapse_completed_thoughts=False
        )
        response = create_and_invoke_supervisor_agent(
            user_query,
            datasets_info,
            st.session_state["memory"],
            st.session_state,
            st_callback=st_callback
        )
        if response:
            try:
                new_content = response['messages'][-1].content
                
                # FIXED: Only get plot_images from the CURRENT execution
                plot_images = []
                
                # 1. Check if response has plot_images
                if 'plot_images' in response and response['plot_images']:
                    plot_images = response.get("plot_images", [])
                    logging.info(f"Found {len(plot_images)} plots in response")
                
                # 2. If not in response, check ONLY the last message
                if not plot_images:
                    last_msg = response['messages'][-1]
                    if hasattr(last_msg, 'additional_kwargs') and 'plot_images' in last_msg.additional_kwargs:
                        plot_images = last_msg.additional_kwargs.get("plot_images", [])
                        logging.info(f"Found {len(plot_images)} plots in last message")
                
                # 3. Validate that plot files actually exist
                plot_images = [img for img in plot_images if img and os.path.exists(img)]
                
                logging.info(f"Final plot_images for THIS response: {plot_images}")
                
                oceanographer_used = response.get("oceanographer_agent_used", False)
                ecologist_used = response.get("ecologist_agent_used", False)
                visualization_used = response.get("visualization_agent_used", False)
                
                # Extract from last message if not in response
                if hasattr(response['messages'][-1], 'additional_kwargs'):
                    last_kwargs = response['messages'][-1].additional_kwargs
                    oceanographer_used = oceanographer_used or last_kwargs.get("oceanographer_used", False)
                    ecologist_used = ecologist_used or last_kwargs.get("ecologist_used", False)
                    visualization_used = visualization_used or last_kwargs.get("visualization_used", False)
                
                st.session_state.messages_data_agent.append({
                    "role": "assistant",
                    "content": new_content,
                    "plot_images": plot_images,  # Only images from THIS execution
                    "oceanographer_used": oceanographer_used,
                    "ecologist_used": ecologist_used,
                    "visualization_agent_used": visualization_used
                })
                log_history_event(
                    st.session_state,
                    "assistant_message",
                    {"page": "data_agent", "content": new_content}
                )
                logging.info(f"Processed assistant response in Data Agent, content: {new_content}")
                logging.info(f"Added {len(plot_images)} plot images to THIS message")
                message_placeholder.empty()
                st.rerun()
            except Exception as e:
                logging.error(f"Error invoking graph: {e}")
                st.error(f"An error occurred: {e}")
            message_placeholder.empty()
    else:
        # Display existing messages
        for message in st.session_state.messages_data_agent:
            logging.info(f"Displaying message in Data Agent: {message['role']}")
            
            if message["role"] == "system":
                with st.chat_message(message["role"], avatar=SYSTEM_ICON):
                    st.markdown(message["content"])
            elif message["role"] == "user":
                with st.chat_message(message["role"], avatar=USER_ICON):
                    st.markdown(message["content"])
            else:
                with st.chat_message(message["role"], avatar=SYSTEM_ICON):
                    st.markdown(message["content"])
                    # Display ONLY the plot images that belong to THIS specific message
                    if "plot_images" in message and message["plot_images"]:
                        logging.info(f"Message has {len(message['plot_images'])} plot images")
                        for plot_info in message["plot_images"]:
                            if isinstance(plot_info, tuple):
                                plot_path, code_path = plot_info
                                if os.path.exists(plot_path):
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    with col2:
                                        st.image(plot_path, caption='Generated Plot', use_container_width=True)
                                    if os.path.exists(code_path):
                                        with open(code_path, 'r') as f:
                                            code = f.read()
                                        st.code(code, language='python')
                            else:
                                # Only show if the file still exists (handles sandbox switching)
                                if os.path.exists(plot_info):
                                    logging.info(f"Displaying plot: {plot_info}")
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    with col2:
                                        st.image(plot_info, caption='Generated Plot', use_container_width=True)
                                else:
                                    logging.warning(f"Plot file no longer exists (different sandbox?): {plot_info}")
                st.session_state.visualization_agent_used = False
        
        # Display datasets info
        for info in datasets_info:
            logging.info(f"Displaying dataset in Data Agent: DOI {info['doi']}, type: {info['data_type']}")
            with st.expander(f"Dataset: {info['name']}", expanded=False):
                # Add clickable DOI link
                st.markdown(f"**DOI:** [{info['doi']}]({info['doi']})")
                st.write(f"**Name:** {info['name']}")
                st.write(f"**Type:** {info['data_type']}")
                if info['data_type'] == "pandas DataFrame":
                    st.dataframe(info['dataset'])
                    if 'files' in info:
                        st.write(f"**Files:** {', '.join(info['files'])}")
                    logging.info(f"Displayed DataFrame for DOI {info['doi']}")
                elif info['data_type'] == "xarray Dataset":
                    st.write("**Variables:**")
                    st.write(list(info['dataset'].data_vars))
                    st.write("**Attributes:**")
                    st.write(info['dataset'].attrs)
                    if 'files' in info:
                        st.write(f"**Files:** {', '.join(info['files'])}")
                    logging.info(f"Displayed xarray Dataset for DOI {info['doi']}")
                elif info['data_type'] == "file folder" or info['data_type'] == "other":
                    if 'files' in info:
                        st.write(f"**Files:** {', '.join(info['files'])}")
                    else:
                        st.write(info['df_head'])
                    logging.info(f"Displayed {info['data_type']} for DOI {info['doi']}")
                else:
                    st.write("Unsupported data type")
                    if 'files' in info:
                        st.write(f"**Files:** {', '.join(info['files'])}")
                    logging.warning(f"Unsupported data type displayed for DOI {info['doi']}")
        if has_new_plot(st.session_state):
            reset_new_plot_flag(st.session_state)
else:
    if len(st.session_state.active_datasets) == 0 and st.session_state.current_page == "data_agent":
        st.warning("No datasets loaded. Please load a dataset first.")
        logging.warning("No datasets loaded in Data Agent")