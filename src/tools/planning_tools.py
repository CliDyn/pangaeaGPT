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
    
    # Optimized system message to create simpler, more efficient plans
    system_message = """
    You are a Planning Tool that creates MINIMAL, EFFICIENT task plans for data analysis workflows.
    Based on the user query, conversation history, and available agents, create a plan with these components:
    1. A list of tasks needed to address the user query
    2. Assignment of each task to the appropriate agent type
    3. Status tracking (pending, in_progress, completed, failed)
    
    CRITICAL PLANNING GUIDELINES:
    - SIMPLICITY IS KEY: Create the MINIMUM number of tasks needed - usually 1-2 tasks, but complex workflows may require more
    - AVOID TASK SPLITTING: For simple queries that can be solved in one step, create JUST ONE task
    - DIRECT IMPLEMENTATION: For basic operations like counting, finding maximums, or calculating statistics, use a SINGLE task
    
    Examples of queries that should be ONE TASK:
    - "What is the most common species?" → ONE task for DataFrameAgent
    - "Calculate the average temperature" → ONE task for DataFrameAgent
    - "Count how many records are in the dataset" → ONE task for DataFrameAgent
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
    Task 2: DataFrameAgent - "Perform wavelet coherence analysis and cross-correlation with scipy.signal to identify teleconnections"

    SPLIT TASK RULES:
    - Data retrieval (ERA5/Copernicus) → OceanographerAgent first
    - Complex analysis requiring the data → Appropriate specialist agent second
    - Each task should be self-contained with clear outputs for the next task

    FORMAT YOUR RESPONSE AS A VALID JSON ARRAY where each item has:
    - "task": task description (be specific and include the complete action needed)
    - "agent": agent name (must be one from the available_agents list)
    - "status": "pending" (for new tasks), "in_progress", "completed", or "failed"
    
    AGENT SELECTION GUIDELINES:
    - OceanographerAgent: Use for marine/ocean data visualization, climate analysis, physical oceanography, temperature, salinity, currents, sea level data, and when working with ERA5 or Copernicus Marine data
    - EcologistAgent: Use for biodiversity data visualization, species analysis, ecological patterns, biological/environmental studies. Does NOT have access to ERA5/Copernicus Marine tools
    - VisualizationAgent: Use for MAPPING tasks, geographic plots, sampling station maps, general plotting and visualization tasks that don't specifically fall into oceanography or ecology categories. ALWAYS use for queries containing: "map", "plot", "geographic", "station locations", "sampling stations"
    - DataFrameAgent: Use ONLY for data analysis, filtering, counting, statistics, finding patterns, and basic operations on tabular data. DO NOT use for visualization or plotting tasks
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
    
    Create a MINIMAL EFFICIENT plan that addresses the user's query using the available agents.
    For simple analysis like finding frequency, counting occurrences, or basic statistics, use JUST ONE TASK with the specific action.
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