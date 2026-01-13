# src/agents/visualization_agent.py (Refactored)
import logging
import streamlit as st
# Removed redundant imports (os, pandas, xarray, langchain_core.prompts, langchain.agents)

# Import the new helper functions from base.py
from .base import prepare_visualization_environment, create_standard_agent_executor
from ..prompts import Prompts
from ..llm_factory import get_llm
from ..tools.python_repl import CustomPythonREPLTool
from ..tools.reflection_tools import reflect_tool
from ..tools.package_tools import install_package_tool
from ..tools.visualization_tools import (
    list_plotting_data_files_tool,
    wise_agent_tool
)

def create_visualization_agent(user_query, datasets_info):
    """
    Creates a general visualization agent.
    """
    # Use the helper function to prepare the environment
    datasets, datasets_text, dataset_variables = prepare_visualization_environment(datasets_info)

    # Generate the specific prompt for this agent
    prompt = Prompts.generate_visualization_agent_system_prompt(user_query, datasets_text, dataset_variables)

    # Initialize the LLM
    llm = get_llm(temperature=0.1)

    # Create the CustomPythonREPLTool
    repl_tool = CustomPythonREPLTool(
        datasets=datasets,
        results_dir=datasets.get("results_dir"),
        session_key=st.session_state.get("thread_id", "default")
    )

    # Define the tools
    tools = [
        repl_tool,
        reflect_tool,
        install_package_tool,
        list_plotting_data_files_tool,
        wise_agent_tool
    ]

    # Use the helper function to create the executor
    return create_standard_agent_executor(llm, tools, prompt)

# Centralized initialize_agents function (remains here as __init__.py depends on it)
def initialize_agents(user_query, datasets_info):
    """
    Initializes all 4 specialized agents.
    """
    if datasets_info:
        # We must import here to avoid circular dependencies
        from .oceanographer_agent import create_oceanographer_agent
        from .ecologist_agent import create_ecologist_agent
        from .pandas_agent import create_pandas_agent

        oceanographer_agent = create_oceanographer_agent(user_query, datasets_info)
        ecologist_agent = create_ecologist_agent(user_query, datasets_info)
        visualization_agent = create_visualization_agent(user_query, datasets_info)
        dataframe_agent = create_pandas_agent(user_query, datasets_info)

        logging.info("All 4 agents initialized successfully")
        return oceanographer_agent, ecologist_agent, visualization_agent, dataframe_agent
    else:
        st.warning("No datasets loaded. Please load datasets first.")
        logging.warning("No datasets provided to initialize_agents")
        return None, None, None, None