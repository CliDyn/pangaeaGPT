# main.py
import logging
import uuid
import os
from typing import List
import streamlit as st

import pandas as pd
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.agents import (
    create_supervisor_agent,
    create_search_agent,
    create_visualization_agent,
    create_pandas_agent
)
from src.search.dataset_utils import fetch_dataset, convert_df_to_csv
from src.memory import CustomMemorySaver


def initialize_session_state(session_state: dict):
    session_state_defaults = {
        "messages_search": [],
        "messages_data_agent": [],
        "datasets_cache": {},
        "datasets_info": None,
        "active_datasets": [],
        "selected_datasets": set(),
        "show_dataset": True,
        "current_page": "search",
        "dataset_dfs": {},
        "dataset_names": {},
        "saved_plot_paths": {},
        "memory": MemorySaver(),
        "oceanographer_agent_used": False,
        "ecologist_agent_used": False,
        "visualization_agent_used": False,
        "dataframe_agent_used": False,
        "specialized_agent_used": False,
        "chat_history": ChatMessageHistory(session_id="search-agent-session"),
        "search_method": "PANGAEA Search (default)",
        "selected_text": "",
        "new_plot_generated": False,
        "execution_history": []
    }

    for key, value in session_state_defaults.items():
        if key not in session_state:
            session_state[key] = value


def get_search_agent(datasets_info, model_name, api_key):
    # Add search_mode parameter
    search_mode = st.session_state.get("search_mode", "simple")
    return create_search_agent(datasets_info=datasets_info, search_mode=search_mode)


def process_search_query(user_input: str, search_agent, session_data: dict):
    """
    Processes a user search query using the enhanced multi-search agent.
    """
    # Initialize or reset chat history
    session_data["chat_history"] = ChatMessageHistory(session_id="search-agent-session")
    
    # Populate chat history
    for message in session_data["messages_search"]:
        if message["role"] == "user":
            session_data["chat_history"].add_user_message(message["content"])
        elif message["role"] == "assistant":
            session_data["chat_history"].add_ai_message(message["content"])

    # Create truncated history function
    def get_truncated_chat_history(session_id):
        truncated_messages = session_data["chat_history"].messages[-20:]
        truncated_history = ChatMessageHistory(session_id=session_id)
        
        for msg in truncated_messages:
            if isinstance(msg, HumanMessage):
                truncated_history.add_user_message(msg.content)
            elif isinstance(msg, AIMessage):
                truncated_history.add_ai_message(msg.content)
            else:
                truncated_history.add_message(msg)
                
        return truncated_history

    # Create agent with memory
    search_agent_with_memory = RunnableWithMessageHistory(
        search_agent,
        get_truncated_chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # Log the search execution
    logging.info(f"Starting multi-search process for query: {user_input}")
    
    # Invoke agent
    response = search_agent_with_memory.invoke(
        {"input": user_input},
        {"configurable": {"session_id": "search-agent-session"}},
    )

    # Extract response and intermediate steps
    ai_message = response["output"]
    intermediate_steps = response.get("intermediate_steps", [])
    
    # Log search statistics
    search_count = sum(1 for step in intermediate_steps if step[0].tool == "search_pg_datasets")
    logging.info(f"Completed multi-search with {search_count} searches executed")
    
    # Check if consolidation was performed
    consolidation_performed = any(step[0].tool == "consolidate_search_results" for step in intermediate_steps)
    
    if consolidation_performed and session_data.get("datasets_info") is not None:
        # Add the final consolidated table to messages
        if session_data.get("search_mode") == "deep":
            message = f"**Deep search completed:** Executed {search_count} search variations and consolidated results."
        else:
            message = f"**Search completed:** Found {len(session_data['datasets_info'])} datasets."
            
        session_data["messages_search"].append({
            "role": "assistant",
            "content": message,
            "table": session_data["datasets_info"].to_json(orient="split")
        })
    
    return ai_message


def add_user_message_to_search(user_input: str, session_data: dict):
    session_data["messages_search"].append({"role": "user", "content": user_input})


def add_assistant_message_to_search(content: str, session_data: dict):
    session_data["messages_search"].append({"role": "assistant", "content": content})


def load_selected_datasets_into_cache(selected_datasets, session_data: dict):
    """
    Loads selected datasets into cache by fetching them into a single sandbox with subdirectories.
    
    Args:
        selected_datasets: List or set of DOIs to fetch.
        session_data: Dictionary containing session state.
    """
    logging.info(f"Starting load_selected_datasets_into_cache for {len(selected_datasets)} datasets")

    # Get the persistent thread_id for the session. It must have been created before this.
    thread_id = session_data.get("thread_id")
    if not thread_id:
        logging.error("CRITICAL: thread_id not found. Forcing creation of a new one.")
        ensure_thread_id(session_data, force_new=True)
        thread_id = session_data["thread_id"]

    # Define the main sandbox directory using the persistent thread_id
    sandbox_main = os.path.join("tmp", "sandbox", thread_id)
    os.makedirs(sandbox_main, exist_ok=True)
    logging.info(f"Using persistent main sandbox for session {thread_id}: {sandbox_main}")

    for i, doi in enumerate(selected_datasets, 1):
        logging.info(f"Processing DOI: {doi}")
        if doi not in session_data["datasets_cache"]:
            # Create a more descriptive subdirectory for this dataset
            sanitized_doi = doi.replace('/', '_').replace(':', '_')
            target_dir = os.path.join(sandbox_main, f"dataset_{i}_{sanitized_doi}")
            os.makedirs(target_dir, exist_ok=True)
            
            # Fetch dataset into the subdirectory
            from src.search.dataset_utils import fetch_dataset
            dataset_path, name = fetch_dataset(doi, target_dir=target_dir)
            if dataset_path is not None:
                session_data["datasets_cache"][doi] = (dataset_path, name)  # dataset_path is target_dir
                logging.info(f"Loaded and cached dataset for DOI {doi}, path: {dataset_path}, name: {name}")
            else:
                logging.warning(f"Failed to load dataset for DOI {doi}")
        else:
            logging.info(f"DOI {doi} already in cache, skipping fetch")


def set_active_datasets_from_selection(session_data: dict):
    session_data["active_datasets"] = list(session_data["selected_datasets"])


import pandas as pd
import xarray as xr

def get_datasets_info_for_active_datasets(session_data: dict):
    """
    Retrieves information about active datasets from the cache.
    
    Args:
        session_data: Dictionary containing session state.
    
    Returns:
        list: List of dictionaries with dataset info.
    """
    logging.info("Starting get_datasets_info_for_active_datasets")
    datasets_info = []
    for doi in session_data["active_datasets"]:
        dataset_path, name = session_data["datasets_cache"].get(doi, (None, None))
        description = "No description available"
        if session_data["datasets_info"] is not None:
            description_row = session_data["datasets_info"].loc[
                session_data["datasets_info"]["DOI"] == doi, "Short Description"
            ]
            description = description_row.values[0] if len(description_row) > 0 else description

        info = {'doi': doi, 'name': name, 'description': description}

        if dataset_path is None:
            logging.warning(f"DOI {doi}: No dataset loaded into cache")
            info.update({
                'df_head': "Failed to load",
                'dataset': None,
                'data_type': "failed"
            })
        elif isinstance(dataset_path, str) and os.path.isdir(dataset_path):
            # Handle dataset as a directory
            files = os.listdir(dataset_path)
            if not files:
                logging.warning(f"Directory for DOI {doi} is empty at path: {dataset_path}")
                info.update({
                    'df_head': "No files found",
                    'dataset': dataset_path,
                    'data_type': "sandbox (empty)",
                    'sandbox_path': dataset_path,
                    'files': []
                })
            else:
                logging.info(f"Directory for DOI {doi} contains {len(files)} files")
                # Try loading data.csv for DataFrames
                if "data.csv" in files:
                    try:
                        df = pd.read_csv(os.path.join(dataset_path, "data.csv"))
                        info.update({
                            'df_head': df.head().to_string(),
                            'dataset': df,
                            'data_type': "pandas DataFrame",
                            'sandbox_path': dataset_path,
                            'files': files
                        })
                        logging.info(f"DOI {doi}: Loaded data.csv as DataFrame")
                    except Exception as e:
                        logging.error(f"Failed to load data.csv for DOI {doi}: {e}")
                        file_list = ", ".join(files)
                        info.update({
                            'df_head': f"Files: {file_list}",
                            'dataset': dataset_path,
                            'data_type': "other",
                            'sandbox_path': dataset_path,
                            'files': files
                        })
                # Try loading netCDF files
                elif any(f.endswith(('.nc', '.cdf', '.netcdf')) for f in files):
                    try:
                        nc_file = next(f for f in files if f.endswith(('.nc', '.cdf', '.netcdf')))
                        xr_ds = xr.open_dataset(os.path.join(dataset_path, nc_file))
                        info.update({
                            'df_head': str(xr_ds),
                            'dataset': xr_ds,
                            'data_type': "xarray Dataset",
                            'sandbox_path': dataset_path,
                            'files': files
                        })
                        logging.info(f"DOI {doi}: Loaded {nc_file} as xarray Dataset")
                    except Exception as e:
                        logging.error(f"Failed to load netCDF for DOI {doi}: {e}")
                        file_list = ", ".join(files)
                        info.update({
                            'df_head': f"Files: {file_list}",
                            'dataset': dataset_path,
                            'data_type': "other",  # Changed type to "other"
                            'sandbox_path': dataset_path,
                            'files': files
                        })
                else:
                    # Treat as a generic file folder
                    file_info = [f"{f} ({os.path.getsize(os.path.join(dataset_path, f))/1024:.1f} KB)" for f in files[:10]]
                    info.update({
                        'df_head': "Files: " + ", ".join(file_info) + (", ..." if len(files) > 10 else ""),
                        'dataset': dataset_path,
                        'data_type': "file folder",
                        'sandbox_path': dataset_path,
                        'files': files
                    })
                    logging.info(f"DOI {doi}: Treated as file folder")
        else:
            logging.error(f"Unexpected dataset type for DOI {doi}: {type(dataset_path)}")
            info.update({
                'df_head': f"Unexpected dataset type: {type(dataset_path)}",
                'dataset': dataset_path,
                'data_type': "unknown"
            })

        datasets_info.append(info)
    
    logging.info(f"Processed {len(datasets_info)} datasets")
    return datasets_info


def create_and_invoke_supervisor_agent(user_query: str, datasets_info: list, memory, session_data: dict, st_callback=None):
    import time
    import uuid
    import logging
    import traceback
    
    session_data["processing"] = True
    
    # Prepare dataset_globals with sandbox paths
    dataset_globals = {}
    dataset_variables = []
    for i, info in enumerate(datasets_info):
        var_name = f"dataset_{i+1}"
        dataset_variables.append(var_name)
        if 'sandbox_path' in info:
            dataset_globals[var_name] = info['sandbox_path']
        elif info['data_type'] == "pandas DataFrame":
            dataset_globals[var_name] = info['dataset']
    
    graph = create_supervisor_agent(user_query, datasets_info, memory)
    
    if graph is None:
        session_data["processing"] = False
        return None

    messages = []
    for message in session_data["messages_data_agent"]:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"], name="User"))
        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"], name="Assistant"))
        else:
            messages.append(AIMessage(content=message["content"], name=message["role"]))

    # The user's query is already in the messages list from the session state.
    # The line below was adding it a second time, causing duplication. It has been removed.
    # messages.append(HumanMessage(content=user_query, name="User"))
    
    limited_messages = messages[-15:]  # Keep last 15 messages including the new query
    initial_state = {
        "messages": limited_messages,
        "next": "supervisor",
        "agent_scratchpad": [],
        "user_query": user_query,  # CRITICAL FIX: Use consistent 'user_query' key
        "plot_images": [],
        "last_agent_message": "",
        "plan": []
    }

    config = {
        "configurable": {"thread_id": session_data.get('thread_id', str(uuid.uuid4())), "recursion_limit": 5}
    }
    if st_callback:
        config["callbacks"] = [st_callback]
        logging.info("StreamlitCallbackHandler added to config.")
    else:
        logging.info("No StreamlitCallbackHandler provided.")

    try:
        response = graph.invoke(initial_state, config=config)
        session_data["processing"] = False
        return response
    except Exception as e:
        session_data["processing"] = False
        logging.error(f"Error during graph invocation: {e}", exc_info=True)
        raise e


def add_user_message_to_data_agent(user_input: str, session_data: dict):
    session_data["messages_data_agent"].append({"role": "user", "content": f"{user_input}"})


def add_assistant_message_to_data_agent(content: str, plot_images, agent_usage_flags, session_data: dict):
    """
    UPDATED: Now accepts agent_usage_flags dict instead of single visualization flag
    """
    new_message = {
        "role": "assistant",
        "content": content,
        "plot_images": plot_images if plot_images else [],
        "oceanographer_used": agent_usage_flags.get("oceanographer_used", False),
        "ecologist_used": agent_usage_flags.get("ecologist_used", False),
        "visualization_used": agent_usage_flags.get("visualization_used", False),
        "dataframe_used": agent_usage_flags.get("dataframe_used", False),
        "specialized_agent_used": agent_usage_flags.get("specialized_agent_used", False)
    }
    session_data["messages_data_agent"].append(new_message)


def convert_dataset_to_csv(dataset) -> bytes:
    """
    Convert a dataset to CSV format, handling various input types.
    
    Args:
        dataset: The dataset to convert (DataFrame, string path, xarray Dataset, etc.)
        
    Returns:
        bytes: The CSV data or empty bytes if conversion failed
    """
    # Simply pass the dataset to convert_df_to_csv which now handles all types
    return convert_df_to_csv(dataset)


def has_new_plot(session_data: dict) -> bool:
    return session_data.get("new_plot_generated", False)


def reset_new_plot_flag(session_data: dict):
    session_data["new_plot_generated"] = False


def get_dataset_csv_name(doi: str) -> str:
    return f"dataset_{doi.split('/')[-1]}.csv"


def set_current_page(session_data: dict, page_name: str):
    session_data["current_page"] = page_name


def set_selected_text(session_data: dict, text: str):
    session_data["selected_text"] = text


def set_show_dataset(session_data: dict, show: bool):
    session_data["show_dataset"] = show


def set_dataset_for_data_agent(session_data: dict, doi: str, csv_data: bytes, dataset: pd.DataFrame, name: str):
    session_data["dataset_csv"] = csv_data
    session_data["dataset_df"] = dataset
    session_data["dataset_name"] = name
    session_data["current_page"] = "data_agent"


def ensure_memory(session_data: dict):
    if "memory" not in session_data:
        session_data["memory"] = CustomMemorySaver()


def ensure_thread_id(session_data: dict, force_new: bool = False):
    """Ensures a thread_id exists for the session. Can force creation of a new one."""
    if force_new or "thread_id" not in session_data or session_data["thread_id"] is None:
        session_data["thread_id"] = str(uuid.uuid4())
        logging.info(f"Generated new thread_id: {session_data['thread_id']} (forced: {force_new})")