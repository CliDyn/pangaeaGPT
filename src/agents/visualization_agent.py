# src/agents/visualization_agent.py
import os
import logging
import streamlit as st
import pandas as pd
import xarray as xr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor

from ..prompts import Prompts
from ..config import API_KEY
from ..tools.python_repl import CustomPythonREPLTool
from ..tools.reflection_tools import reflect_tool
from ..tools.package_tools import install_package_tool
from ..tools.visualization_tools import (
    example_visualization_tool,
    list_plotting_data_files_tool,
    wise_agent_tool
)
from ..tools.era5_retrieval_tool import era5_retrieval_tool
from ..tools.copernicus_marine_tool import copernicus_marine_tool

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
    # This is the common parent directory all datasets are stored in
    uuid_main_dir = None
    for i, info in enumerate(datasets_info):
        if isinstance(info.get('dataset'), str) and os.path.isdir(info.get('dataset')):
            # Get the parent directory (UUID directory)
            uuid_main_dir = os.path.dirname(os.path.abspath(info.get('dataset')))
            break
    
    # List all files in the main UUID directory for reference
    uuid_dir_files = []
    if uuid_main_dir and os.path.exists(uuid_main_dir):
        try:
            uuid_dir_files = os.listdir(uuid_main_dir)
        except Exception as e:
            logging.error(f"Error listing UUID directory files: {str(e)}")
    
    # Add the UUID directory to the datasets dict so it can be accessed
    datasets["uuid_main_dir"] = uuid_main_dir
    
    # Create a section with crystal clear path instructions
    uuid_paths = "### ⚠️ CRITICAL: EXACT DATASET PATHS - MUST USE THESE EXACTLY AS SHOWN ⚠️\n"
    uuid_paths += "The following paths contain unique IDs that MUST be used with os.path.join():\n\n"
    
    # Add the main UUID directory first
    if uuid_main_dir:
        uuid_paths += f"# MAIN OUTPUT DIRECTORY - SAVE RESULTS HERE:\n"
        uuid_paths += f"uuid_main_dir = r'{uuid_main_dir}'\n\n"
        uuid_paths += f"# Files currently in main directory: {', '.join(uuid_dir_files) if uuid_dir_files else 'None'}\n\n"
         
    # First list all datasets with their paths
    for i, info in enumerate(datasets_info):
        var_name = f"dataset_{i + 1}"
        datasets[var_name] = info['dataset']  # Sandbox path or dataset object
        dataset_variables.append(var_name)
        
        # Add the FULL UUID PATH for each dataset
        if isinstance(info['dataset'], str) and os.path.isdir(info['dataset']):
            # Get the absolute path and ensure it has forward slashes for consistency
            full_uuid_path = os.path.abspath(info['dataset']).replace('\\', '/')
            
            # Create the path variable assignment with proper escaping
            uuid_paths += f"# Dataset {i+1}: {info['name']}\n"
            uuid_paths += f"{var_name}_path = r'{full_uuid_path}'\n\n"
            
            # Check if the directory exists and list its contents
            if os.path.exists(full_uuid_path):
                try:
                    files = os.listdir(full_uuid_path)
                    uuid_paths += f"# Files available in {var_name}_path:\n"
                    uuid_paths += f"# {', '.join(files)}\n\n"
                    
                    # Now add dataset-specific examples based on actual files
                    uuid_paths += f"# DATASET-SPECIFIC WORKING CODE EXAMPLES FOR {var_name}:\n"
                    
                    # Example for listing files
                    uuid_paths += f"# Example: List all files in {var_name}_path\n"
                    uuid_paths += f"import os\n"
                    uuid_paths += f"files = os.listdir({var_name}_path)\n"
                    uuid_paths += f"print(f\"Files in {var_name}:\")\n"
                    uuid_paths += f"for file in files:\n"
                    uuid_paths += f"    print(f\"  - {{file}}\")\n\n"
                    
                    # Add file-specific examples
                    if any(f.endswith('.csv') for f in files):
                        csv_file = next(f for f in files if f.endswith('.csv'))
                        uuid_paths += f"# Example: Load CSV file '{csv_file}' from {var_name}_path\n"
                        uuid_paths += f"import pandas as pd\n"
                        uuid_paths += f"csv_path = os.path.join({var_name}_path, '{csv_file}')\n"
                        uuid_paths += f"df = pd.read_csv(csv_path)\n"
                        uuid_paths += f"print(df.head())\n\n"
                    
                    if any(f.endswith(('.nc', '.cdf', '.netcdf')) for f in files):
                        nc_file = next(f for f in files if f.endswith(('.nc', '.cdf', '.netcdf')))
                        uuid_paths += f"# Example: Load netCDF file '{nc_file}' from {var_name}_path\n"
                        uuid_paths += f"import xarray as xr\n"
                        uuid_paths += f"nc_path = os.path.join({var_name}_path, '{nc_file}')\n"
                        uuid_paths += f"ds = xr.open_dataset(nc_path)\n"
                        uuid_paths += f"print(ds)\n\n"
                    
                    # Plotting example specific to this dataset
                    if any(f.endswith('.csv') for f in files):
                        csv_file = next(f for f in files if f.endswith('.csv'))
                        uuid_paths += f"# Example: Create plot using data from {var_name}_path\n"
                        uuid_paths += f"import pandas as pd\n"
                        uuid_paths += f"import matplotlib.pyplot as plt\n"
                        uuid_paths += f"csv_path = os.path.join({var_name}_path, '{csv_file}')\n"
                        uuid_paths += f"df = pd.read_csv(csv_path)\n"
                        uuid_paths += f"plt.figure(figsize=(10, 6))\n"
                        uuid_paths += f"# Replace 'column_name' with an actual column from your data\n"
                        uuid_paths += f"plt.plot(df.index, df.iloc[:, 0])\n"
                        uuid_paths += f"plt.title('Data from {info['name']}')\n"
                        uuid_paths += f"plt.savefig(plot_path)  # ALWAYS use this plot_path variable\n\n"
                    
                except Exception as e:
                    uuid_paths += f"# Error listing files: {str(e)}\n\n"
            
            uuid_paths += f"# ⚠️ WARNING: ALWAYS USE THE EXACT PATH WITH os.path.join({var_name}_path, 'filename')! ⚠️\n\n"
        else:
            # For non-directory datasets (like pandas DataFrames)
            uuid_paths += f"# Dataset {i+1}: {info['name']} (in-memory dataset, not a directory)\n"
            uuid_paths += f"# Access this dataset directly with the variable name '{var_name}'\n\n"
    
    # Global warning about path handling
    uuid_paths += "# ⚠️ CRITICAL WARNINGS ⚠️\n"
    uuid_paths += "# 1. NEVER use '/mnt/data/...' or similar paths - they DO NOT EXIST and WILL CAUSE ERRORS\n"
    uuid_paths += "# 2. ALWAYS use the exact dataset_X_path variables shown above\n"
    uuid_paths += "# 3. ALWAYS check which files exist before trying to read them\n\n"
    
    # Continue with standard dataset info after the path examples
    for i, info in enumerate(datasets_info):
        var_name = f"dataset_{i + 1}"
        datasets_text += (
            f"Dataset {i + 1}:\n"
            f"Variable Name: {var_name}\n"
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
  
    # Initialize the LLM
    llm = ChatOpenAI(api_key=API_KEY, model_name=st.session_state.model_name)

    # Create the CustomPythonREPLTool with sandbox paths
    repl_tool = CustomPythonREPLTool(datasets=datasets)

    # Define the tools available to the agent
    tools_vis = [
        repl_tool,
        reflect_tool,
        install_package_tool,
        example_visualization_tool,
        list_plotting_data_files_tool,
        wise_agent_tool,
        era5_retrieval_tool,
        copernicus_marine_tool
    ]
    
    # Create the agent with the updated prompt and tools
    agent_visualization = create_openai_tools_agent(
        llm,
        tools=tools_vis,
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                ("user", "{input}"),  # THIS LINE IS KEY - ensures task info is passed
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
        return_intermediate_steps=True
    )

def initialize_agents(user_query, datasets_info):
    """
    Initializes agents based on available dataset types.
    
    Args:
        user_query (str): The user's query
        datasets_info (list): Information about available datasets
        
    Returns:
        tuple: (visualization_agent, dataframe_agent)
    """
    if datasets_info:
        # Create visualization agent
        visualization_agent = create_visualization_agent(
            user_query=user_query,
            datasets_info=datasets_info
        )

        # Only create DataFrameAgent if there are pandas DataFrames
        dataframe_agent = None
        dataframe_datasets = [info for info in datasets_info if isinstance(info['dataset'], pd.DataFrame)]
        if dataframe_datasets:
            from .pandas_agent import create_pandas_agent
            dataframe_agent = create_pandas_agent(
                user_query=user_query,
                datasets_info=dataframe_datasets
            )
            logging.info(f"DataFrameAgent initialized with {len(dataframe_datasets)} DataFrames")
        else:
            logging.info("No pandas DataFrames available; skipping DataFrameAgent initialization")

        return visualization_agent, dataframe_agent
    else:
        st.warning("No datasets loaded. Please load datasets first.")
        logging.warning("No datasets provided to initialize_agents")
        return None, None