# src/agents/oceanographer_agent.py (Refactored)
import logging
# Removed redundant imports

# Import the new helper functions from base.py
from .base import prepare_visualization_environment, create_standard_agent_executor
from ..prompts import Prompts
from ..llm_factory import get_llm
from ..tools.python_repl import CustomPythonREPLTool
# (Import tools specific to this agent)
from ..tools.reflection_tools import reflect_tool
from ..tools.package_tools import install_package_tool
from ..tools.visualization_tools import (
    list_plotting_data_files_tool,
    wise_agent_tool
)
from ..tools.era5_retrieval_tool import era5_retrieval_tool
from ..tools.copernicus_marine_tool import copernicus_marine_tool

def create_oceanographer_agent(user_query, datasets_info, session_id="default"):
    """
    Creates the Oceanographer agent specialized in marine data visualization.
    """
    # Use the helper function to prepare the environment
    datasets, datasets_text, dataset_variables = prepare_visualization_environment(datasets_info, session_id=session_id)

    # Generate the specific prompt for this agent
    prompt = Prompts.generate_oceanographer_agent_system_prompt(user_query, datasets_text, dataset_variables)

    # Initialize the LLM
    llm = get_llm(temperature=0.1)

    # Create the CustomPythonREPLTool
    repl_tool = CustomPythonREPLTool(
        datasets=datasets,
        results_dir=datasets.get("results_dir"),
        session_key=session_id
    )

    # Define the tools (Oceanographer includes ERA5 and Copernicus)
    tools = [
        repl_tool,
        reflect_tool,
        install_package_tool,
        list_plotting_data_files_tool,
        wise_agent_tool,
        era5_retrieval_tool,
        copernicus_marine_tool
    ]

    # Use the helper function to create the executor
    return create_standard_agent_executor(llm, tools, prompt)