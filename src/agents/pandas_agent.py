# src/agents/pandas_agent.py
# REBUILT to use create_openai_tools_agent while keeping the original name.

import os
import logging
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor

from ..prompts import Prompts
from ..llm_factory import get_llm
from ..tools.python_repl import CustomPythonREPLTool
from ..tools.visualization_tools import list_plotting_data_files_tool

def create_pandas_agent(user_query, datasets_info):
    """
    Creates a data analysis agent (named DataFrameAgent for compatibility)
    that only has access to a Python REPL and a file listing tool.
    This version uses create_openai_tools_agent for more robust, general-purpose code execution.

    Args:
        user_query (str): The user's query.
        datasets_info (list): Information about available datasets.

    Returns:
        AgentExecutor: The DataFrame agent executor.
    """
    # Initialize variables
    datasets_text = ""
    dataset_variables = []
    datasets_for_repl = {} # This will hold the actual data/paths for the REPL tool

    # --- This context setup is borrowed from the visualization agents ---
    # It ensures the agent knows about the sandbox paths, which is crucial for loading files.
    uuid_main_dir = None
    for i, info in enumerate(datasets_info):
        sandbox_path = info.get('sandbox_path')
        if sandbox_path and isinstance(sandbox_path, str) and os.path.isdir(sandbox_path):
            uuid_main_dir = os.path.dirname(os.path.abspath(sandbox_path))
            break
            
    datasets_for_repl["uuid_main_dir"] = uuid_main_dir
    if uuid_main_dir:
        results_dir = os.path.join(uuid_main_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        datasets_for_repl["results_dir"] = results_dir

    # Provide the agent with the direct path variables it needs to construct file paths
    uuid_paths_info = "### ⚠️ CRITICAL: Use these exact path variables to access data files ⚠️\n"
    for i, info in enumerate(datasets_info):
        var_name = f"dataset_{i + 1}"
        datasets_for_repl[var_name] = info['dataset']
        dataset_variables.append(var_name)
        
        if isinstance(info['dataset'], str) and os.path.isdir(info['dataset']):
            full_uuid_path = os.path.abspath(info['dataset']).replace('\\', '/')
            uuid_paths_info += f"# Path for Dataset {i+1} ({info['name']})\n"
            uuid_paths_info += f"{var_name}_path = r'{full_uuid_path}'\n"
    
    # Provide info on the data itself
    datasets_summary = ""
    for i, info in enumerate(datasets_info):
        datasets_summary += (
            f"Dataset {i + 1}:\n"
            f"Name: {info['name']}\n"
            f"Description: {info['description']}\n"
            f"Type: {info['data_type']}\n"
            f"Sample Data/Info: {info['df_head']}\n\n"
        )
    
    datasets_text = uuid_paths_info + "\n" + datasets_summary
    # --- End of context setup ---

    # Get our new, specific prompt for the DataFrameAgent
    prompt_template = Prompts.generate_pandas_agent_system_prompt(
        user_query, datasets_text, dataset_variables
    )

    # Use the LLM factory. Low temperature is good for precise code generation.
    llm = get_llm(temperature=0.0)

    # Define the agent's very specific and limited toolset
    repl_tool = CustomPythonREPLTool(datasets=datasets_for_repl)
    tools = [
        repl_tool,
        list_plotting_data_files_tool
    ]
    
    # Create the agent using the standard 'create_openai_tools_agent'
    agent = create_openai_tools_agent(
        llm,
        tools=tools,
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
    )

    # Create the agent executor
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=25
    )