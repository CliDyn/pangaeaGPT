# src/agents.py
# ==========================================
# TRANSITIONAL FILE - This file re-exports components from the new modular structure
# This maintains backward compatibility while the codebase transitions
# Eventually all imports should be updated to use the new module paths directly
# ==========================================

# Re-export tools from their new modular locations
from src.tools.python_repl import CustomPythonREPLTool
from src.tools.reflection_tools import reflect_tool, reflect_on_image
from src.tools.planning_tools import planning_tool
from src.tools.package_tools import install_package_tool
from src.tools.visualization_tools import (
    example_visualization_tool,
    list_plotting_data_files_tool,
    wise_agent_tool
)
from src.tools.era5_retrieval_tool import era5_retrieval_tool

# Re-export agent creation functions from their new modular locations
from src.agents.search_agent import create_search_agent
from src.agents.pandas_agent import create_pandas_agent
from src.agents.visualization_agent import create_visualization_agent, initialize_agents
from src.agents.supervisor_agent import create_supervisor_agent, create_and_invoke_supervisor_agent
from src.agents.base import agent_node
from src.agents.oceanographer_agent import create_oceanographer_agent
from src.agents.ecologist_agent import create_ecologist_agent
from src.tools.copernicus_marine_tool import copernicus_marine_tool
from src.tools.era5_retrieval_tool import era5_retrieval_tool

# All code that was previously in this file has been moved to appropriate modules.
# This file now only re-exports symbols to maintain backward compatibility.
# For new code, please import directly from the specific modules.