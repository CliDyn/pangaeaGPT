# src/agents/supervisor_agent.py
import logging
import uuid
import json
import functools
import traceback
import os
from typing import List, TypedDict, Dict, Sequence
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langgraph.graph import StateGraph, END

from .base import agent_node
from .visualization_agent import initialize_agents
from .writer_agent import create_writer_agent
from ..tools.planning_tools import planning_tool
from ..llm_factory import get_llm  # Use the new factory

def create_supervisor_agent(user_query, datasets_info, memory):
    """
    Creates a supervisor agent to coordinate between different specialized agents.
    
    Args:
        user_query (str): The user's query
        datasets_info (list): Information about available datasets
        memory: The memory checkpoint saver
        
    Returns:
        Graph: The compiled supervisor agent workflow
    """
    # Initialize agents
    oceanographer_agent, ecologist_agent, visualization_agent, dataframe_agent = initialize_agents(user_query, datasets_info)
    writer_agent = create_writer_agent(datasets_info)
    logging.info("Initialized all agents, including WriterAgent")

    # members list building:
    members = []
    if oceanographer_agent:
        members.append("OceanographerAgent")
    if ecologist_agent:
        members.append("EcologistAgent")
    if visualization_agent:
        members.append("VisualizationAgent")
    if dataframe_agent:
        members.append("DataFrameAgent")
    
    # Add the WriterAgent to the list of members the supervisor knows about.
    if writer_agent:
        members.append("WriterAgent")

    logging.info(f"Supervisor managing agent types: {members}")

    # Format dataset information for both planning and direct responses
    datasets_text = ""
    if datasets_info:
        for i, info in enumerate(datasets_info, 1):
            # Include file information for better routing decisions
            file_info = ""
            if 'files' in info and info['files']:
                file_list = info['files'][:5]  # Show first 5 files
                file_info = f"Files: {', '.join(file_list)}"
                if len(info['files']) > 5:
                    file_info += f" (and {len(info['files']) - 5} more)"
                file_info += "\n"
            
            datasets_text += (
                f"Dataset {i}:\n"
                f"Name: {info['name']}\n"
                f"DOI: {info.get('doi', 'Not available')}\n"
                f"Description: {info['description']}\n"
                f"Type: {info['data_type']}\n"
                f"{file_info}\n"
            )
    else:
        datasets_text = "No datasets available."

    system_prompt_supervisor = f"""
You are a supervisor managing conversations and tasks. Your primary role is to delegate tasks to the correct specialized agents.

### 🚨 CRITICAL ROUTING RULES - NON-NEGOTIABLE 🚨

1.  **NEVER, under any circumstances, route a visualization or plotting task to the `DataFrameAgent`.** This agent is incapable of creating plots and will reject the task. Any query containing words like 'plot', 'map', 'chart', 'figure', 'graph', 'visualize', or 'distribution' is a visualization task.

2.  **ALWAYS route visualization tasks to `VisualizationAgent`, `OceanographerAgent`, or `EcologistAgent`.**

3.  The `DataFrameAgent`'s ONLY purpose is for **non-visual computation**: calculating numbers, filtering tables, performing statistical analysis, and returning text or numerical results.

### Available Agents and Their Strict Capabilities
- **OceanographerAgent / EcologistAgent / VisualizationAgent**: Your agents for ALL VISUAL tasks. They create plots, maps, and figures.
- **DataFrameAgent**: Your agent for ALL COMPUTATIONAL tasks. It performs calculations and data manipulation. **IT DOES NOT PLOT.**

### Examples of Correct Routing:
- "Plot the depth distribution" -> **"VisualizationAgent"**
- "Calculate statistics for the depth distribution" -> **"DataFrameAgent"**
- "Create a scatter plot of temperature vs salinity" -> **"OceanographerAgent"**
- "What is the correlation between temperature and salinity?" -> **"DataFrameAgent"**

### CRITICAL ROUTING INSTRUCTIONS
Your ONLY options for the "next" field are:
- "RESPOND" - Use this when you want to answer directly without delegating
- "FINISH" - Use this when all tasks are complete
- {', '.join(members)} - These are the agent names you can route to

You must NEVER set "next" to "create_or_update_plan" or any other tool name!

### DATA TYPE AWARE ROUTING - CRITICAL
Before routing any task, examine the dataset information below to understand the data types:
{datasets_text}

**MANDATORY DATA TYPE ROUTING RULES:**
- **NetCDF/xarray Datasets (.nc, .cdf, .netcdf files)**: NEVER assign to DataFrameAgent. Use OceanographerAgent, EcologistAgent, or VisualizationAgent.
- **pandas DataFrames (.csv files, data.csv)**: Can be routed to any agent. DataFrameAgent is the specialist for non-visual analysis.
- **File folders with unknown formats**: Route to VisualizationAgent for initial exploration.
- **Failed datasets**: Route to RESPOND to explain the issue.

### Direct Response Instructions
For the following types of queries, respond directly using the 'RESPOND' option:
- **Questions about conversation history**, such as 'what we discussed before,' 'summarize our conversation,' or 'remind me of previous topics.' Use the provided conversation history to answer accurately.
- **Simple questions about datasets**, such as 'What's the description of the first dataset?', 'How many datasets are loaded?', or 'What are the parameters of dataset 2?' Use the dataset information above to answer directly.
- **Data type incompatibility issues**, when user requests operations that are incompatible with the loaded data types.

When responding directly, provide a concise and accurate answer using the conversation history and dataset information, without involving other agents. Skip the planning phase to save time and improve efficiency.

### Complex Task Instructions
For visualization and data analysis requests (like plotting, data manipulation, etc.):
1. **FIRST**: Check the data types in the dataset information above
2. **SECOND**: Apply the mandatory data type routing rules
3. Review the current plan in the "plan" field
4. Choose the appropriate agent from: {', '.join(members)}
5. Set "next" ONLY to one of the agent names or "FINISH" or "RESPOND"

### Agent Capabilities Details
- **OceanographerAgent**: Use for marine/ocean data visualization, climate analysis, physical oceanography, and when working with ERA5 or Copernicus Marine data. Specializes in temperature, salinity, currents, sea level data. **CAN HANDLE NetCDF/xarray datasets**.
- **EcologistAgent**: Use for biodiversity data visualization, species analysis, ecological patterns, and biological/environmental studies. Does NOT have access to ERA5/Copernicus Marine tools. **CAN HANDLE NetCDF/xarray datasets**.
- **VisualizationAgent**: Use for MAPPING tasks, geographic plots, sampling station maps, and general plotting/visualization tasks. ALWAYS route queries containing "map", "plot", "geographic", "station locations", "sampling stations" here. **CAN HANDLE NetCDF/xarray datasets**.
- **DataFrameAgent**: Your primary agent for **non-visual data analysis**. Use for filtering, counting, statistics, finding patterns, and answering questions that require running Python code on the data. It can load data from any file path provided in the context. **Route all computational and data manipulation tasks here.**

### Examples of Correct Data-Type Aware Routing
- User asks: "Summarize our conversation" → Set "next" to "RESPOND"
- "Plot ocean temperature data" (NetCDF dataset) → "OceanographerAgent" (NOT DataFrameAgent)
- "Analyze species distribution" (CSV dataset) → "EcologistAgent" or "DataFrameAgent"
- "Create a scatter plot" → "VisualizationAgent"
- "Plot sampling station map" → "VisualizationAgent"
- **"Calculate statistics on the CSV file" → "DataFrameAgent"**
- **"What is the maximum depth in the NetCDF file?" → "DataFrameAgent" (It will write code to open the file and find the max)**
- **"Count how many records are in the dataset" (any file type) → "DataFrameAgent"**
- All tasks completed → Set "next" to "FINISH"

### REMEMBER
- **ALWAYS check data types before routing**
- Tools like 'create_or_update_plan' are NOT valid options for the "next" field!
- The planning process happens automatically - your job is only to decide which agent should handle the task next.
- Only use agent names that are actually available: {', '.join(members)}
"""

    # Define the routing function schema
    function_def = {
        "name": "route",
        "description": "Select the next step in the workflow.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "type": "string",
                    "enum": ["FINISH", "RESPOND"] + members,
                    "description": "MUST be one of: 'RESPOND', 'FINISH', or an agent name. NEVER use tool names here."
                },
                "plan": {
                    "title": "Plan",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "agent": {"type": "string", "enum": members},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "failed"]}
                        },
                        "required": ["task", "agent", "status"]
                    }
                }
            },
            "required": ["next", "plan"],
        },
    }

    # Create the supervisor prompt template
    prompt_supervisor = ChatPromptTemplate.from_messages([
        ("system", system_prompt_supervisor),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Current plan: {plan}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("system", "Based on the user query, conversation, and current plan, decide the next step.")
    ])

    # Initialize the LLM using the factory. Supervisor should be deterministic.
    llm_supervisor = get_llm(temperature=0.0)

    # Bind the planning tool to the LLM
    tools = [planning_tool]
    
    # Create the supervisor chain with forced function calling
    llm_with_tools = llm_supervisor.bind_functions(
        functions=[function_def], 
        function_call={"name": "route"}  # Force it to use the route function
    )

    # Update chain definition to handle plan correctly
    supervisor_chain = (
        {
            "messages": lambda x: x["messages"],
            "agent_scratchpad": lambda x: x["agent_scratchpad"],
            "plan": lambda x: json.dumps(x["plan"] if x.get("plan") else [])
        }
        | prompt_supervisor
        | llm_with_tools
        | JsonOutputFunctionsParser()
    )

    # Define the supervisor node function
    def supervisor_with_planning(state):
        # Safety net for infinite loops, just in case
        if "iteration_count" not in state:
            state["iteration_count"] = 0
        state["iteration_count"] += 1
        if state["iteration_count"] > 50:
            logging.error("Max iterations reached. Forcing FINISH.")
            state["next"] = "FINISH"
            state["messages"].append(AIMessage(content="Error: Maximum workflow iterations reached. Aborting.", name="Supervisor"))
            return state

        # Mark the last agent's task as completed
        last_message = state["messages"][-1]
        if hasattr(last_message, 'name') and last_message.name in members:
            agent_name = last_message.name
            for task in state.get("plan", []):
                # Find the task that was just in progress by this agent
                if task.get("agent") == agent_name and task.get("status") == "in_progress":
                    task["status"] = "completed"
                    logging.info(f"Marked task '{task.get('task')}' as 'completed' for agent {agent_name}.")
                    break
        
        # Create the plan if it doesn't exist
        if not state.get("plan"):
            conversation_history = "\n".join([f"{msg.name if hasattr(msg, 'name') else 'User'}: {msg.content}" 
                                            for msg in state["messages"][-5:] if hasattr(msg, "content")])
            try:
                user_query = state.get("user_query", state['messages'][-1].content)
                plan_result_str = planning_tool.invoke({
                    "user_query": user_query,
                    "conversation_history": conversation_history,
                    "available_agents": members,
                    "current_plan": "[]",
                    "datasets_info": datasets_text
                })
                state["plan"] = json.loads(plan_result_str)
                logging.info(f"Successfully created initial plan: {json.dumps(state['plan'])}")
            except Exception as e:
                logging.error(f"Error creating plan: {e}", exc_info=True)
                state["messages"].append(AIMessage(content=f"Error: Could not create a valid execution plan. {e}", name="Supervisor"))
                state["next"] = "FINISH"
                return state
        
        # Find the next pending task in the plan
        next_task = None
        for task in state.get("plan", []):
            if task.get("status") == "pending":
                next_task = task
                break
        
        # Decide the next step based on the plan
        if next_task:
            agent_to_call = next_task["agent"]
            # Mark the task as in_progress and route to the agent
            next_task["status"] = "in_progress"
            state["next"] = agent_to_call
            logging.info(f"Routing to next agent in plan: {agent_to_call} for task: '{next_task['task']}'")
        else:
            # If there are no more pending tasks, the plan is complete.
            state["next"] = "FINISH"
            logging.info("No pending tasks in plan. Routing to FINISH.")

        return state

    def supervisor_response(state):
        """
        Responsible for generating the final response after all agents have contributed.
        Crucially maintains access to the conversation history and previous agent outputs.
        Includes extensive debugging to troubleshoot dataset parameter passing issues.
        """
        from main import get_datasets_info_for_active_datasets
        
        # Begin debugging
        logging.info("\n==================== SUPERVISOR RESPONSE DEBUGGING ====================")
        logging.info(f"Entering supervisor_response with state keys: {list(state.keys())}")
        logging.info(f"Messages count: {len(state.get('messages', []))}")
        logging.info(f"Last agent message: {state.get('last_agent_message', 'None')}")
        logging.info(f"Current plot_images: {state.get('plot_images', [])}")
        
        # Trace stack to find call origin
        logging.info("Call stack trace:")
        stack_trace = traceback.format_stack()
        for line in stack_trace[:-1]:  # Exclude the current frame
            logging.info(line.strip())
        
        # Get model name using the factory
        llm = get_llm()
        logging.info(f"Using model: {llm.model_name}")
        
        # Debug session state keys
        session_state_keys = list(st.session_state.keys())
        logging.info(f"Available session_state keys: {session_state_keys}")
        
        # Extract datasets info with extensive debugging
        try:
            # Debug the datasets retrieval function
            logging.info("Attempting to get active datasets info...")
            active_datasets_info = get_datasets_info_for_active_datasets(st.session_state)
            
            # Verify what we received
            logging.info(f"Type of active_datasets_info: {type(active_datasets_info)}")
            logging.info(f"Length of active_datasets_info: {len(active_datasets_info) if isinstance(active_datasets_info, (list, dict)) else 'Not a collection'}")
            
            # Try to sample the first dataset if available
            if isinstance(active_datasets_info, list) and active_datasets_info:
                sample_keys = list(active_datasets_info[0].keys())
                logging.info(f"Sample dataset keys: {sample_keys}")
        except Exception as e:
            logging.error(f"CRITICAL ERROR retrieving datasets: {str(e)}")
            logging.error(traceback.format_exc())
            active_datasets_info = []  # Fallback to empty list
        
        # Extract and validate all critical components
        
        # 1. Get conversation history with validation FIRST (moved up to fix bug)
        all_messages = state.get("messages", [])
        logging.info(f"Total message count: {len(all_messages)}")
        
        # 2. Extract visualization agent outputs and plot information
        visualization_used = state.get("visualization_agent_used", False)
        plot_images = state.get("plot_images", [])
        logging.info(f"Visualization used: {visualization_used}, Plot images count: {len(plot_images)}")

        for msg in all_messages:
            if hasattr(msg, 'additional_kwargs') and 'plot_images' in msg.additional_kwargs:
                agent_plot_images = msg.additional_kwargs.get('plot_images', [])
                for img in agent_plot_images:
                    if img and os.path.exists(img) and img not in plot_images:
                        plot_images.append(img)
                        logging.info(f"Found plot image from agent message: {img}")
        
        # Log the final collected plot images
        logging.info(f"Total plot images after extraction: {len(plot_images)}, paths: {plot_images}")
        
        # Debug message types
        message_types = {}
        for msg in all_messages:
            msg_type = type(msg).__name__
            if msg_type not in message_types:
                message_types[msg_type] = 0
            message_types[msg_type] += 1
        logging.info(f"Message types: {message_types}")
        
        # 3. Extract visualization messages specifically
        oceanographer_messages = [msg for msg in all_messages 
                                if hasattr(msg, 'name') and msg.name == "OceanographerAgent"]
        ecologist_messages = [msg for msg in all_messages 
                            if hasattr(msg, 'name') and msg.name == "EcologistAgent"]
        visualization_messages = [msg for msg in all_messages 
                                if hasattr(msg, 'name') and msg.name == "VisualizationAgent"]
        dataframe_messages = [msg for msg in all_messages 
                            if hasattr(msg, 'name') and msg.name == "DataFrameAgent"]
        logging.info(f"Visualization messages count: {len(visualization_messages)}")

        agent_messages = oceanographer_messages + ecologist_messages + visualization_messages + dataframe_messages
        
        # 4. Extract latest user query with validation
        user_messages = [msg for msg in all_messages if not hasattr(msg, 'name') or not msg.name]
        if user_messages:
            latest_user_query = user_messages[-1].content
            logging.info(f"Latest user query: '{latest_user_query[:50]}...'")
        else:
            latest_user_query = state.get("user_query", "No query found")
            logging.info(f"No user messages found, using state input: '{latest_user_query[:50]}...'")
        
        # Format datasets information with deep validation and file information
        datasets_text = ""
        if active_datasets_info:
            logging.info("Formatting datasets information...")
            for i, info in enumerate(active_datasets_info, 1):
                # Validate each required field exists
                required_fields = ['name', 'description', 'data_type']
                missing_fields = [field for field in required_fields if field not in info]
                
                if missing_fields:
                    logging.warning(f"Dataset {i} missing fields: {missing_fields}")
                
                # Get DOI with fallbacks and validation
                doi = info.get('doi', 'Not available')
                
                # Include file information for routing decisions
                file_info = ""
                if 'files' in info and info['files']:
                    file_list = info['files'][:5]  # Show first 5 files
                    file_info = f"Files: {', '.join(file_list)}"
                    if len(info['files']) > 5:
                        file_info += f" (and {len(info['files']) - 5} more)"
                    file_info += "\n"
                
                # Build dataset text with explicit format including file info
                current_dataset = (
                    f"Dataset {i}:\n"
                    f"Name: {info.get('name', 'Unknown')}\n"
                    f"DOI: {doi}\n"
                    f"Description: {info.get('description', 'No description available')}\n"
                    f"Type: {info.get('data_type', 'Unknown type')}\n"
                    f"{file_info}"
                )
                datasets_text += current_dataset
                
                # Log what we're adding with data type info
                logging.info(f"Added dataset {i}: {info.get('name', 'Unknown')} (Type: {info.get('data_type', 'Unknown type')})")
        else:
            datasets_text = "No active dataset selected."
            logging.warning("No active datasets found - using empty placeholder")
        
        # Log the final datasets_text for debugging
        logging.info(f"Final datasets_text length: {len(datasets_text)}")
        logging.info(f"datasets_text preview: {datasets_text[:200]}...")
        
        # Format visualization information
        agent_info = ""
        agents_used = []

        # Check which agents were used
        if oceanographer_messages:
            agents_used.append("OceanographerAgent")
        if ecologist_messages:
            agents_used.append("EcologistAgent") 
        if visualization_messages:
            agents_used.append("VisualizationAgent")
        if dataframe_messages:
            agents_used.append("DataFrameAgent")

        # Build agent info string
        if agents_used:
            agent_info = f"The following specialized agents were used in this conversation: {', '.join(agents_used)}.\n"
            
            # Get the most recent agent message from ANY agent
            if agent_messages:  # NOW WE USE agent_messages!
                recent_agent_content = agent_messages[-1].content
                agent_info += f"Most recent agent analysis: {recent_agent_content[:100]}...\n"
            
            # Add plot information
            if plot_images:
                agent_info += f"Plots were generated and are available.\n"
        
        # Get agent context
        last_agent_message = state.get("last_agent_message", "")
        logging.info(f"Last agent message length: {len(last_agent_message)}")
        
        # Construct system message with debugging
        try:
            # Create a direct substitution in the system message with data type awareness
            system_message = f"""You are a supervisor capable of answering simple questions directly. If the user's query is basic (e.g., about available analysis), answer using the selected dataset context below:

{datasets_text}

{agent_info}

**IMPORTANT DATA TYPE CONTEXT:**
When providing responses about data analysis capabilities, always consider the data types shown above:
- NetCDF/xarray datasets require specialized tools and cannot be processed by standard DataFrame operations
- pandas DataFrames can be processed with standard tabular data operations
- File folders may contain various formats requiring different approaches

Always address the latest user query directly, even if it is similar to previous queries. You can acknowledge repetition (e.g., 'As I mentioned earlier...') and reference previous answers if relevant, but ensure each query receives a clear and complete response. For complex queries, follow these agent guidelines:

Format any code in markdown and keep responses concise.

The last agent that processed this request said: {last_agent_message}

Latest user query: {latest_user_query}

Please provide a response to the latest user query, taking into account the conversation history and data type constraints."""
            
            logging.info(f"System message constructed successfully, length: {len(system_message)}")
            
            # Check for template literals that weren't substituted
            if "{datasets_text}" in system_message:
                logging.error("ERROR: '{datasets_text}' template literal found in system message!")
            if "{agent_info}" in system_message:
                logging.error("ERROR: '{agent_info}' template literal found in system message!")
            
            # Log the first part of the system message
            logging.info(f"System message preview: {system_message[:300]}...")
            
        except Exception as e:
            logging.error(f"Error building system message: {str(e)}")
            logging.error(traceback.format_exc())
            # Emergency fallback
            system_message = "You are a supervisor assistant. Please respond to the latest query as best you can."
        
        # Build conversation history for context
        try:
            full_history = "\n".join([
                f"{msg.name if hasattr(msg, 'name') else 'User'}: {msg.content}" 
                for msg in all_messages 
                if hasattr(msg, "content")
            ])
            
            logging.info(f"Built conversation history, length: {len(full_history)}")
            logging.info(f"History preview: {full_history[:200]}...")
            
        except Exception as e:
            logging.error(f"Error building conversation history: {str(e)}")
            logging.error(traceback.format_exc())
            full_history = "Error retrieving conversation history."
        
        # Prepare final prompt and invoke LLM
        prompt = f"{system_message}\n\nConversation history:\n{full_history}"
        logging.info(f"Final prompt length: {len(prompt)}")
        
        # Log key statistics about the prompt to diagnose issues
        prompt_components = {
            "system_message": len(system_message),
            "conversation_history": len(full_history),
            "datasets_text": len(datasets_text),
            "agent_info": len(agent_info),
            "last_agent_message": len(last_agent_message),
            "latest_user_query": len(latest_user_query)
        }
        logging.info(f"Prompt component lengths: {json.dumps(prompt_components)}")
        
        # Invoke the LLM with the full context
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            logging.info(f"LLM response received, length: {len(response.content)}")
            logging.info(f"Response preview: {response.content[:200]}...")
        except Exception as e:
            logging.error(f"Error invoking LLM: {str(e)}")
            logging.error(traceback.format_exc())
            response_content = "I encountered an error processing your request. Please try again."
            response = type('obj', (object,), {'content': response_content})
        
        # Update state and return
        try:
            # IMPORTANT: Create the final message WITH plot_images included
            final_message = AIMessage(
                content=response.content, 
                name="Supervisor",
                additional_kwargs={
                    "plot_images": plot_images,  # Include the collected plot images
                    "oceanographer_used": state.get("oceanographer_agent_used", False),
                    "ecologist_used": state.get("ecologist_agent_used", False),
                    "visualization_used": state.get("visualization_agent_used", False)
                }
            )
            
            state["messages"] = state.get("messages", []) + [final_message]
            state["next"] = "FINISH"
            
            # Ensure plot_images are preserved in state
            state["plot_images"] = plot_images
            
            # Final state validation
            logging.info(f"Final state keys: {list(state.keys())}")
            logging.info(f"Final messages count: {len(state['messages'])}")
            logging.info(f"Final plot_images: {state.get('plot_images', [])}")
            logging.info("==================== END SUPERVISOR RESPONSE DEBUGGING ====================\n")
            
        except Exception as e:
            logging.error(f"Error updating state: {str(e)}")
            logging.error(traceback.format_exc())
            # Emergency recovery - create minimal valid state
            state["messages"] = state.get("messages", []) + [
                AIMessage(content="Error processing your request. Technical details have been logged.", name="Supervisor")
            ]
            state["next"] = "FINISH"
        
        return state

    # Define the agent state
    class AgentState(TypedDict):
        messages: Sequence[BaseMessage]
        next: str
        agent_scratchpad: Sequence[BaseMessage]
        user_query: str
        last_agent_message: str
        plot_images: List[str]
        model_name: str
        plan: List[Dict[str, str]]

    # Initialize the workflow
    workflow = StateGraph(AgentState)

    # Add nodes
    if oceanographer_agent:
        workflow.add_node("OceanographerAgent", 
                         functools.partial(agent_node, agent=oceanographer_agent, name="OceanographerAgent"))
    if ecologist_agent:
        workflow.add_node("EcologistAgent", 
                         functools.partial(agent_node, agent=ecologist_agent, name="EcologistAgent"))
    if visualization_agent:
        workflow.add_node("VisualizationAgent", 
                         functools.partial(agent_node, agent=visualization_agent, name="VisualizationAgent"))
    if dataframe_agent:
        workflow.add_node("DataFrameAgent", 
                         functools.partial(agent_node, agent=dataframe_agent, name="DataFrameAgent"))
    # Add the new WriterAgent node
    if writer_agent:
        workflow.add_node("WriterAgent", functools.partial(agent_node, agent=writer_agent, name="WriterAgent"))
    workflow.add_node("supervisor", supervisor_with_planning)
    workflow.add_node("supervisor_response", supervisor_response)

    # Configure edges
    for member in members:
        workflow.add_edge(member, "supervisor")
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    conditional_map["RESPOND"] = "supervisor_response"
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    workflow.set_entry_point("supervisor")

    # Compile and return the graph
    graph = workflow.compile(checkpointer=memory)
    logging.info("Compiled workflow graph")
    return graph

def create_and_invoke_supervisor_agent(user_query: str, datasets_info: list, memory, session_data: dict, st_callback=None):
    """
    Creates and invokes the supervisor agent workflow.
    
    Args:
        user_query (str): The user's query
        datasets_info (list): Information about available datasets
        memory: The memory checkpoint saver
        session_data (dict): The session state data
        st_callback: Optional Streamlit callback handler
        
    Returns:
        dict: The response state from the agent workflow
    """
    import time
    import traceback
    
    session_data["processing"] = True
    
    # Prepare dataset_globals with sandbox paths
    dataset_globals = {}
    dataset_variables = []
    for i, info in enumerate(datasets_info):
        var_name = f"dataset_{i+1}"
        dataset_variables.append(var_name)
        if 'sandbox_path' in info:
            dataset_globals[var_name] = info['sandbox_path']
        elif info['data_type'] == "pandas DataFrame":
            dataset_globals[var_name] = info['dataset']
    
    graph = create_supervisor_agent(user_query, datasets_info, memory)
    
    if graph is None:
        session_data["processing"] = False
        return None

    messages = []
    for message in session_data["messages_data_agent"]:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"], name="User"))
        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"], name="Assistant"))
        else:
            messages.append(AIMessage(content=message["content"], name=message["role"]))

    limited_messages = messages[-10:]
    initial_state = {
        "messages": limited_messages,
        "next": "supervisor",
        "agent_scratchpad": [],
        "input": user_query,
        "plot_images": [],
        "last_agent_message": "",
        "plan": []  # Added plan to initial state
    }

    config = {
        "configurable": {"thread_id": session_data.get('thread_id', str(uuid.uuid4())), "recursion_limit": 7}
    }
    if st_callback:
        config["callbacks"] = [st_callback]
        logging.info("StreamlitCallbackHandler added to config.")
    else:
        logging.info("No StreamlitCallbackHandler provided.")

    try:
        response = graph.invoke(initial_state, config=config)
        session_data["processing"] = False
        
        # CRITICAL: Ensure plot_images are passed to the response
        if "plot_images" not in response:
            response["plot_images"] = []
        
        # Extract plot_images from the final message if needed
        if response.get("messages"):
            last_msg = response["messages"][-1]
            if hasattr(last_msg, 'additional_kwargs') and 'plot_images' in last_msg.additional_kwargs:
                response["plot_images"] = last_msg.additional_kwargs.get("plot_images", [])
                logging.info(f"Extracted {len(response['plot_images'])} plot images from final message")
        
        return response
    except Exception as e:
        session_data["processing"] = False
        logging.error(f"Error during graph invocation: {e}", exc_info=True)
        raise e