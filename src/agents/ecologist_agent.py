# src/agents/ecologist_agent.py (Refactored)
import logging
# Removed redundant imports

# Import the new helper functions from base.py
from .base import prepare_visualization_environment, create_standard_agent_executor
from ..prompts import Prompts
# We standardize using the LLM factory
from ..llm_factory import get_llm 
from ..tools.python_repl import CustomPythonREPLTool
from ..tools.reflection_tools import reflect_tool
from ..tools.package_tools import install_package_tool
from ..tools.visualization_tools import (
    list_plotting_data_files_tool,
    wise_agent_tool
)

def create_ecologist_agent(user_query, datasets_info, session_id="default"):
    """
    Creates the Ecologist agent specialized in biodiversity data visualization.
    """
    # Use the helper function to prepare the environment
    datasets, datasets_text, dataset_variables = prepare_visualization_environment(datasets_info, session_id=session_id)

    # Generate the specific prompt for this agent
    prompt = Prompts.generate_ecologist_agent_system_prompt(user_query, datasets_text, dataset_variables)

    # Initialize the LLM using the factory (standardizing the approach)
    llm = get_llm(temperature=0.1)

    # Create the CustomPythonREPLTool
    repl_tool = CustomPythonREPLTool(
        datasets=datasets,
        results_dir=datasets.get("results_dir"),
        session_key=session_id
    )

    # Define the tools (Ecologist does NOT include ERA5/Copernicus)
    tools = [
        repl_tool,
        reflect_tool,
        install_package_tool,
        list_plotting_data_files_tool,
        wise_agent_tool
    ]

    # Use the helper function to create the executor
    return create_standard_agent_executor(llm, tools, prompt)