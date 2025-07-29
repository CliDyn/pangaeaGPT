# src/agents/visualization_agent.py 
import os
import logging
import streamlit as st
import pandas as pd
import xarray as xr
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor

from ..prompts import Prompts
from ..llm_factory import get_llm  # Use the new factory
from ..tools.python_repl import CustomPythonREPLTool
from ..tools.reflection_tools import reflect_tool
from ..tools.package_tools import install_package_tool
from ..tools.visualization_tools import (
    example_visualization_tool,
    list_plotting_data_files_tool,
    wise_agent_tool
)
# from ..tools.era5_retrieval_tool import era5_retrieval_tool
# from ..tools.copernicus_marine_tool import copernicus_marine_tool

def create_visualization_agent(user_query, datasets_info):
    """
    Creates a visualization agent for plotting and data visualization.
    
    Args:
        user_query (str): The user's query
        datasets_info (list): Information about available datasets
        
    Returns:
        AgentExecutor: The visualization agent executor
    """
    # Initialize variables
    datasets_text = ""
    dataset_variables = []
    datasets = {}
    
    # Extract the main UUID directory (parent directory of first dataset's sandbox path)
    uuid_main_dir = None
    for i, info in enumerate(datasets_info):
        sandbox_path = info.get('sandbox_path')
        if sandbox_path and isinstance(sandbox_path, str) and os.path.isdir(sandbox_path):
            uuid_main_dir = os.path.dirname(os.path.abspath(sandbox_path))
            logging.info(f"Found main UUID directory from sandbox_path: {uuid_main_dir}")
            break
    
    # List all files in the main UUID directory for reference
    uuid_dir_files = []
    if uuid_main_dir and os.path.exists(uuid_main_dir):
        try:
            uuid_dir_files = os.listdir(uuid_main_dir)
        except Exception as e:
            logging.error(f"Error listing UUID directory files: {str(e)}")
    
    # Add the UUID directory to the datasets dict
    datasets["uuid_main_dir"] = uuid_main_dir
    
    # Add the results directory for natural saving
    if uuid_main_dir:
        results_dir = os.path.join(uuid_main_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        datasets["results_dir"] = results_dir
        logging.info(f"Added results_dir to datasets: {results_dir}")
    
    # Continue with the path instructions...
    uuid_paths = "### ⚠️ CRITICAL: EXACT DATASET PATHS - MUST USE THESE EXACTLY AS SHOWN ⚠️\n"
    uuid_paths += "The following paths contain unique IDs that MUST be used with os.path.join():\n\n"
    
    # Add the main UUID directory first
    if uuid_main_dir:
        uuid_paths += f"# MAIN OUTPUT DIRECTORY - YOUR WORKSPACE:\n"
        uuid_paths += f"uuid_main_dir = r'{uuid_main_dir}'\n"
        uuid_paths += f"results_dir = r'{results_dir}'  # 📁 Save all plots here!\n\n"
        uuid_paths += f"# Files currently in main directory: {', '.join(uuid_dir_files) if uuid_dir_files else 'None'}\n\n"         
    
    # First list all datasets with their paths
    for i, info in enumerate(datasets_info):
        var_name = f"dataset_{i + 1}"
        datasets[var_name] = info['dataset']
        dataset_variables.append(var_name)
        
        # We use 'sandbox_path', which is always a string, not the DataFrame object itself
        sandbox_path = info.get('sandbox_path')
        
        # Now the logic is unified: we ALWAYS provide a path, even if the data is already in memory
        if sandbox_path and isinstance(sandbox_path, str) and os.path.isdir(sandbox_path):
            full_uuid_path = os.path.abspath(sandbox_path).replace('\\', '/')
            
            uuid_paths += f"# Dataset {i+1}: {info['name']}\n"
            uuid_paths += f"# This dataset is located in a directory. Use the path variable below.\n"
            uuid_paths += f"{var_name}_path = r'{full_uuid_path}'\n\n"
            
            if os.path.exists(full_uuid_path):
                try:
                    files = os.listdir(full_uuid_path)
                    uuid_paths += f"# Files available in {var_name}_path: {', '.join(files)}\n\n"
                except Exception as e:
                    uuid_paths += f"# Error listing files: {str(e)}\n\n"
        # THE 'ELSE' BLOCK IS COMPLETELY REMOVED. The contradiction is eliminated.
    
    # Global warning about path handling
    uuid_paths += "# ⚠️ CRITICAL WARNINGS ⚠️\n"
    uuid_paths += "# 1. NEVER use '/mnt/data/...' or similar paths - they DO NOT EXIST and WILL CAUSE ERRORS\n"
    uuid_paths += "# 2. ALWAYS use the exact dataset_X_path variables shown above\n"
    uuid_paths += "# 3. ALWAYS check which files exist before trying to read them\n\n"
    
    # Continue with standard dataset info
    for i, info in enumerate(datasets_info):
        #var_name = f"dataset_{i + 1}"
        datasets_text += (
            f"Dataset {i + 1}:\n"
            #f"Variable Name: {var_name}\n"
            f"Name: {info['name']}\n"
            f"Description: {info['description']}\n"
            f"Type: {info['data_type']}\n"
            f"Sample Data: {info['df_head']}\n\n"
        )
    
    # Put the UUID paths section at the beginning of the datasets_text
    datasets_text = uuid_paths + datasets_text
    
    # Store the dataset text in session state for other tools to access
    st.session_state["viz_datasets_text"] = datasets_text
    
    # Generate the prompt with the modified datasets_text
    prompt = Prompts.generate_visualization_agent_system_prompt(user_query, datasets_text, dataset_variables)
  
    # Initialize the LLM using the factory
    llm = get_llm(temperature=0.1)

    # Create the CustomPythonREPLTool with sandbox paths
    repl_tool = CustomPythonREPLTool(datasets=datasets)

    # Define the tools available to the agent
    tools_vis = [
        repl_tool,
        reflect_tool,
        install_package_tool,
        #example_visualization_tool,
        list_plotting_data_files_tool,
        wise_agent_tool
        # era5_retrieval_tool,  # Disabled for VisualizationAgent
        # copernicus_marine_tool  # Disabled for VisualizationAgent
    ]
    
    # Create the agent with the updated prompt and tools
    # IMPORTANT: Remove plot_path from the expected variables
    agent_visualization = create_openai_tools_agent(
        llm,
        tools=tools_vis,
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
    )

    # Create the agent executor
    return AgentExecutor(
        agent=agent_visualization,
        tools=tools_vis,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=25
    )

def initialize_agents(user_query, datasets_info):
    """
    Initializes all 4 specialized agents based on available dataset types.
    
    Args:
        user_query (str): The user's query
        datasets_info (list): Information about available datasets
        
    Returns:
        tuple: (oceanographer_agent, ecologist_agent, visualization_agent, dataframe_agent)
    """
    if datasets_info:
        # Create all 4 specialized agents
        
        # 1. Create OceanographerAgent (marine/climate data + ERA5 + Copernicus)
        from .oceanographer_agent import create_oceanographer_agent
        oceanographer_agent = create_oceanographer_agent(
            user_query=user_query,
            datasets_info=datasets_info
        )
        
        # 2. Create EcologistAgent (biodiversity/ecological data, no ERA5/Copernicus)
        from .ecologist_agent import create_ecologist_agent
        ecologist_agent = create_ecologist_agent(
            user_query=user_query,
            datasets_info=datasets_info
        )
        
        # 3. Create VisualizationAgent (general visualization, all tools)
        visualization_agent = create_visualization_agent(
            user_query=user_query,
            datasets_info=datasets_info
        )

        # 4. Create the new powerful DataFrameAgent
        # We no longer need to check for specific dataframe types, as the new agent is more generic.
        from .pandas_agent import create_pandas_agent
        dataframe_agent = create_pandas_agent(user_query, datasets_info)
        logging.info("DataFrameAgent initialized successfully with new implementation.")

        logging.info("All 4 agents initialized successfully")
        return oceanographer_agent, ecologist_agent, visualization_agent, dataframe_agent
    else:
        st.warning("No datasets loaded. Please load datasets first.")
        logging.warning("No datasets provided to initialize_agents")
        return None, None, None, None