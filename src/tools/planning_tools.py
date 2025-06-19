# src/tools/planning_tools.py
import logging
import json
import streamlit as st
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from ..config import API_KEY

class PlanningToolArgs(BaseModel):
    user_query: str = Field(description="The user's original query")
    conversation_history: str = Field(description="The conversation history so far")
    available_agents: List[str] = Field(description="List of available agent types")
    current_plan: str = Field(description="The current plan, if any exists")
    datasets_info: str = Field(description="Information about available datasets")

def planning_tool(user_query: str, conversation_history: str, available_agents: List[str], 
                  current_plan: str, datasets_info: str) -> str:
    """
    A tool for creating or updating a plan based on user query and conversation.
    Returns a JSON string containing task steps with assigned agents and status.
    """
    # Check if we're in CLI mode
    from ..config import IS_CLI_MODE
    if IS_CLI_MODE:
        model_name = "gpt-4.1"  # Default model for CLI
    else:
        model_name = st.session_state.get("model_name", "gpt-4.1")
    llm = ChatOpenAI(api_key=API_KEY, model_name=model_name)
    
    # Optimized system message to create simpler, more efficient plans with data type awareness
    system_message = """
    You are a Planning Tool that creates MINIMAL, EFFICIENT task plans for data analysis workflows.
    Based on the user query, conversation history, available agents, and dataset information, create a plan with these components:
    1. A list of tasks needed to address the user query
    2. Assignment of each task to the appropriate agent type
    3. Status tracking (pending, in_progress, completed, failed)
    
    CRITICAL PLANNING GUIDELINES:
    - SIMPLICITY IS KEY: Create the MINIMUM number of tasks needed - usually 1-2 tasks, but complex workflows may require more.
    - **WRITER AGENT RULE**: If the workflow requires more than one specialist agent (e.g., OceanographerAgent and EcologistAgent), you MUST add a final task: `{"task": "Synthesize the findings from all agents into a cohesive final report.", "agent": "WriterAgent", "status": "pending"}`. This should always be the last task.
    - AVOID TASK SPLITTING: For simple queries that can be solved in one step, create JUST ONE task
    - DIRECT IMPLEMENTATION: For basic operations like counting, finding maximums, or calculating statistics, use a SINGLE task
    
    **MANDATORY DATA TYPE ROUTING RULES:**
    YOU MUST examine the datasets info provided and apply these rules:
    - **NetCDF/xarray Datasets (.nc, .cdf, .netcdf files)**: NEVER assign to DataFrameAgent. Use OceanographerAgent, EcologistAgent, or VisualizationAgent.
    - **pandas DataFrames (.csv files, data.csv)**: Can be assigned to any agent including DataFrameAgent.
    - **File folders with unknown formats**: Assign to VisualizationAgent or domain-specific agents, NEVER to DataFrameAgent.
    - **Failed datasets**: Do not create tasks, the supervisor should respond directly about the issue.
    
    Examples of queries that should be ONE TASK (DATA TYPE AWARE):
    - "What is the most common species?" → ONE task for DataFrameAgent (ONLY if CSV/DataFrame data)
    - "What is the most common species?" → ONE task for EcologistAgent (if NetCDF/xarray data)
    - "Calculate the average temperature" → ONE task for DataFrameAgent (ONLY if CSV/DataFrame data)
    - "Calculate the average temperature" → ONE task for OceanographerAgent (if NetCDF/xarray data)
    - "Count how many records are in the dataset" → ONE task for DataFrameAgent (ONLY if CSV/DataFrame data)
    - "Count how many records are in the dataset" → ONE task for OceanographerAgent (if NetCDF/xarray data)
    - "Show the distribution of species" → ONE task for EcologistAgent
    - "Plot ocean temperature data" → ONE task for OceanographerAgent
    - "Create a scatter plot" → ONE task for VisualizationAgent
    - "Plot sampling station map" → ONE task for VisualizationAgent
    - "Create a geographic map" → ONE task for VisualizationAgent
    - "Map the sampling locations" → ONE task for VisualizationAgent
    - "Show station locations on a map" → ONE task for VisualizationAgent
    
    Examples of queries that should be SPLIT INTO MULTIPLE TASKS:
    - "Create an SDM (Species Distribution Model) map" → TWO tasks:
    Task 1: OceanographerAgent - "Retrieve current ocean environmental data (temperature, salinity, chlorophyll) from Copernicus Marine for the study region"
    Task 2: EcologistAgent - "Create SDM map using MaxEnt or sdmpredictors package with species occurrence data and environmental layers"

    - "Perform advanced oceanographic statistical analysis with climate indices" → TWO tasks:
    Task 1: OceanographerAgent - "Download ERA5 atmospheric and Copernicus ocean data, calculate climate indices (NAO, AMO, PDO)"
    Task 2: OceanographerAgent - "Perform wavelet coherence analysis and cross-correlation with scipy.signal to identify teleconnections" (NOTE: Changed from DataFrameAgent to OceanographerAgent for NetCDF data)

    SPLIT TASK RULES:
    - Data retrieval (ERA5/Copernicus) → OceanographerAgent first
    - Complex analysis requiring the data → Appropriate specialist agent second (consider data type!)
    - Each task should be self-contained with clear outputs for the next task

    FORMAT YOUR RESPONSE AS A VALID JSON ARRAY where each item has:
    - "task": task description (be specific and include the complete action needed)
    - "agent": agent name (must be one from the available_agents list)
    - "status": "pending" (for new tasks), "in_progress", "completed", or "failed"
    
    AGENT SELECTION GUIDELINES (DATA TYPE AWARE):
    - OceanographerAgent: Use for marine/ocean data visualization, climate analysis, physical oceanography, temperature, salinity, currents, sea level data, and when working with ERA5 or Copernicus Marine data. **CAN HANDLE NetCDF/xarray datasets**.
    - EcologistAgent: Use for biodiversity data visualization, species analysis, ecological patterns, biological/environmental studies. Does NOT have access to ERA5/Copernicus Marine tools. **CAN HANDLE NetCDF/xarray datasets**.
    - VisualizationAgent: Use for MAPPING tasks, geographic plots, sampling station maps, general plotting and visualization tasks that don't specifically fall into oceanography or ecology categories. ALWAYS use for queries containing: "map", "plot", "geographic", "station locations", "sampling stations". **CAN HANDLE NetCDF/xarray datasets**.
    - DataFrameAgent: Use ONLY for data analysis, filtering, counting, statistics, finding patterns, and basic operations on **TABULAR DATA (pandas DataFrames)**. **CANNOT HANDLE NetCDF/xarray datasets**. DO NOT assign NetCDF files or visualization tasks here.
    """
    
    # Set current_plan to empty array if it's not provided
    if not current_plan or current_plan.strip() == "":
        current_plan = "[]"
    
    human_message = f"""
    USER QUERY: {user_query}
    
    AVAILABLE AGENTS: {available_agents}
    
    DATASETS INFO: {datasets_info}
    
    CURRENT PLAN (if any): {current_plan}
    
    CONVERSATION HISTORY: {conversation_history}
    
    **CRITICAL**: Before creating any plan, examine the DATASETS INFO above to identify data types:
    - Look for "Type:" field in each dataset
    - Check for file extensions (.nc, .cdf, .netcdf indicate NetCDF/xarray)
    - Apply the mandatory data type routing rules
    
    Create a MINIMAL EFFICIENT plan that addresses the user's query using the available agents.
    For simple analysis like finding frequency, counting occurrences, or basic statistics:
    - Use DataFrameAgent ONLY if the data type is "pandas DataFrame" or contains CSV files
    - Use OceanographerAgent/EcologistAgent/VisualizationAgent if the data type is "xarray Dataset" or contains NetCDF files
    
    Return only the JSON array with no additional text.
    """
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message)
    ]
    
    response = llm.invoke(messages)
    return response.content

# Create the structured tool
planning_tool = StructuredTool.from_function(
    func=planning_tool,
    name="create_or_update_plan",
    description="Creates or updates a plan for addressing the user's query with a sequence of tasks assigned to specific agents",
    args_schema=PlanningToolArgs
)