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
    model_name = st.session_state.get("model_name", "gpt-4o")
    llm = ChatOpenAI(api_key=API_KEY, model_name=model_name)
    
    # Optimized system message to create simpler, more efficient plans
    system_message = """
    You are a Planning Tool that creates MINIMAL, EFFICIENT task plans for data analysis workflows.
    Based on the user query, conversation history, and available agents, create a plan with these components:
    1. A list of tasks needed to address the user query
    2. Assignment of each task to the appropriate agent type
    3. Status tracking (pending, in_progress, completed, failed)
    
    CRITICAL PLANNING GUIDELINES:
    - SIMPLICITY IS KEY: Create the MINIMUM number of tasks needed (1-2 tasks if possible)
    - AVOID TASK SPLITTING: For simple queries that can be solved in one step, create JUST ONE task
    - DIRECT IMPLEMENTATION: For basic operations like counting, finding maximums, or calculating statistics, use a SINGLE task
    
    Examples of queries that should be ONE TASK:
    - "What is the most common species?" → ONE task for DataFrameAgent
    - "Calculate the average temperature" → ONE task for DataFrameAgent
    - "Count how many records are in the dataset" → ONE task for DataFrameAgent
    - "Show the distribution of species" → ONE task for VisualizationAgent
    
    FORMAT YOUR RESPONSE AS A VALID JSON ARRAY where each item has:
    - "task": task description (be specific and include the complete action needed)
    - "agent": agent name (must be one from the available_agents list)
    - "status": "pending" (for new tasks), "in_progress", "completed", or "failed"
    
    AGENT SELECTION GUIDELINES:
    - DataFrameAgent: Use for data analysis, filtering, counting, statistics, and identifying patterns
    - VisualizationAgent: Use ONLY when the user explicitly requests a plot or when visualization would significantly enhance understanding
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