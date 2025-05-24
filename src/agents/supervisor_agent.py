# src/agents/supervisor_agent.py
import logging
import uuid
import json
import functools
import traceback
from typing import List, TypedDict, Dict, Sequence
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langgraph.graph import StateGraph, END

from .base import agent_node
from .visualization_agent import initialize_agents
from ..tools.planning_tools import planning_tool
from ..config import API_KEY

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
    visualization_agent, dataframe_agent = initialize_agents(user_query, datasets_info)
    logging.info("Initialized agents for workflow")

    # Dynamically build the members list
    members = []
    if visualization_agent:
        members.append("VisualizationAgent")
    if dataframe_agent:
        members.append("DataFrameAgent")
    logging.info(f"Supervisor managing members: {members}")

    # Format dataset information for both planning and direct responses
    datasets_text = ""
    if datasets_info:
        for i, info in enumerate(datasets_info, 1):
            datasets_text += (
                f"Dataset {i}:\n"
                f"Name: {info['name']}\n"
                f"DOI: {info.get('doi', 'Not available')}\n"
                f"Description: {info['description']}\n"
                f"Type: {info['data_type']}\n\n"
            )
    else:
        datasets_text = "No datasets available."

    # Define the updated supervisor system prompt
    system_prompt_supervisor = f"""
You are a supervisor managing conversations and tasks. Your primary role is to efficiently handle user queries by either responding directly or delegating tasks to specialized agents.

### CRITICAL ROUTING INSTRUCTIONS
Your ONLY options for the "next" field are:
- "RESPOND" - Use this when you want to answer directly without delegating
- "FINISH" - Use this when all tasks are complete
- {', '.join(members)} - These are the agent names you can route to

You must NEVER set "next" to "create_or_update_plan" or any other tool name!

### Direct Response Instructions
For the following types of queries, respond directly using the 'RESPOND' option:
- **Questions about conversation history**, such as 'what we discussed before,' 'summarize our conversation,' or 'remind me of previous topics.' Use the provided conversation history to answer accurately.
- **Simple questions about datasets**, such as 'What's the description of the first dataset?', 'How many datasets are loaded?', or 'What are the parameters of dataset 2?' Use the dataset information below to answer directly:
  {datasets_text}

When responding directly, provide a concise and accurate answer using the conversation history and dataset information, without involving other agents. Skip the planning phase to save time and improve efficiency.

### Complex Task Instructions
For visualization and data analysis requests (like plotting, data manipulation, etc.):
1. Review the current plan in the "plan" field
2. Choose the appropriate agent from: {', '.join(members)}
3. Set "next" ONLY to one of the agent names or "FINISH" or "RESPOND"

### Available Agents and Their Capabilities
{', '.join(members)}
- VisualizationAgent: Use for all plotting and visualization tasks. Always call it when request is related to the visualization.
- DataFrameAgent: Use for simple data analysis, filtering, and manipulation

### Examples of Correct Routing
- User asks: "Summarize our conversation" → Set "next" to "RESPOND"
- User asks: "Plot the data" → Set "next" to "VisualizationAgent"
- User asks: "Analyze trends" → Set "next" to "DataFrameAgent"
- All tasks completed → Set "next" to "FINISH"

### REMEMBER
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

    # Initialize the LLM
    llm_supervisor = ChatOpenAI(api_key=API_KEY, model_name=st.session_state.get("model_name", "gpt-4o"))

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
        # Safety net for infinite loops
        if "iteration_count" not in state:
            state["iteration_count"] = 0
        state["iteration_count"] += 1
        if state["iteration_count"] > 10:
            state["next"] = "FINISH"
            state["messages"].append(AIMessage(content="Max iterations reached. Stopping.", name="Supervisor"))
            return state

        # First create/update the plan
        if not state.get("plan") or len(state.get("plan", [])) == 0:
            # Create a plan first before making routing decisions
            conversation_history = "\n".join([f"{msg.name if hasattr(msg, 'name') else 'User'}: {msg.content}" 
                                            for msg in state["messages"][-5:] if hasattr(msg, "content")])
            try:
                # Get user query from the state
                user_query = state.get("input", "")
                plan_result = planning_tool.invoke({
                    "user_query": user_query,
                    "conversation_history": conversation_history,
                    "available_agents": members,
                    "current_plan": json.dumps([]),
                    "datasets_info": datasets_text
                })
                
                # Improved plan extraction and validation
                if isinstance(plan_result, str):
                    try:
                        # Try direct JSON parsing
                        plan = json.loads(plan_result)
                    except json.JSONDecodeError:
                        # Try extracting JSON from structured response
                        import re
                        json_match = re.search(r'(\[.*\])', plan_result, re.DOTALL)
                        if json_match:
                            extracted_json = json_match.group(1)
                            logging.info(f"Extracted JSON from string: {extracted_json}")
                            plan = json.loads(extracted_json)
                        else:
                            plan = []
                else:
                    # If already a Python object, use directly
                    plan = plan_result
                    
                # Ensure plan is stored properly
                state["plan"] = plan
                logging.info(f"Successfully created initial plan: {json.dumps(plan)}")
            except Exception as e:
                logging.error(f"Error creating plan: {str(e)}")
                state["plan"] = []
        
        # Explicitly copy plan into a new variable for the chain
        plan_for_chain = state.get("plan", [])
        
        # Explicitly convert plan to string when logging
        logging.info(f"Plan before supervisor chain: {json.dumps(plan_for_chain)}")
        
        # Modify how the chain is invoked to ensure plan is passed correctly
        result = supervisor_chain.invoke({
            "messages": state["messages"],
            "agent_scratchpad": state.get("agent_scratchpad", []),
            "plan": plan_for_chain  # Pass explicitly rather than relying on state
        })
        
        logging.info(f"Supervisor chain output: {result}")
        
        # Extract next step from the result
        next_step = result.get("next")
        
        # Add validation for the returned plan
        updated_plan = result.get("plan", state.get("plan", []))
        
        # Save the updated plan back with validation
        if isinstance(updated_plan, list):
            state["plan"] = updated_plan
        elif isinstance(updated_plan, str):
            try:
                state["plan"] = json.loads(updated_plan)
            except Exception as e:
                # Keep existing plan if parsing fails
                logging.error(f"Failed to parse updated plan: {updated_plan}, error: {str(e)}")
        
        # Critical validation step
        valid_next_steps = ["RESPOND", "FINISH"] + members
        logging.info(f"next_step type: {type(next_step)}, value: {repr(next_step)}, valid_steps: {valid_next_steps}")
        
        if next_step not in valid_next_steps:
            # Handle the case where LLM incorrectly outputs a tool name instead of a routing option
            if next_step == "create_or_update_plan":
                # If LLM tries to route to the planning tool, find the first pending task in the plan
                # and route to that agent instead
                if state["plan"]:
                    for task in state["plan"]:
                        if task.get("status") == "pending":
                            next_step = task.get("agent")
                            logging.info(f"Corrected routing from 'create_or_update_plan' to '{next_step}'")
                            break
                    else:
                        # If no pending tasks, finish
                        next_step = "FINISH"
                        logging.info("No pending tasks in plan, routing to FINISH")
                else:
                    # No plan, route to first available agent
                    next_step = members[0] if members else "FINISH"
                    logging.info(f"No plan available, routing to {next_step}")
            else:
                # For any other invalid routing, default to FINISH
                original_next_step = next_step
                next_step = "FINISH"
                logging.error(f"Invalid next_step '{original_next_step}' received. Defaulting to 'FINISH'.")
                state["messages"].append(AIMessage(content=f"Error: Invalid step '{original_next_step}'. Finishing.", name="Supervisor"))

        # Handle special case for RESPOND option
        if next_step == "RESPOND":
            state["next"] = "RESPOND"
            return state

        # Update task statuses based on last agent message
        last_agent_message = next((msg for msg in reversed(state["messages"]) 
                                if hasattr(msg, "name") and msg.name in members), None)
        if last_agent_message and state["plan"]:
            agent_name = last_agent_message.name
            for task in state["plan"]:
                if task.get("agent") == agent_name and task.get("status") in ["pending", "in_progress"]:
                    task["status"] = "completed"
                    logging.info(f"Marked task '{task.get('task')}' as completed for {agent_name}")
                    break

        # Check if all tasks are done
        if state["plan"] and all(task.get("status") == "completed" for task in state["plan"]):
            state["next"] = "FINISH"
            logging.info("All tasks completed; setting next to FINISH")
            return state

        # Set final routing decision
        state["next"] = next_step
        
        # If routing to an agent, mark the corresponding task as in_progress
        if next_step in members and state["plan"]:
            for task in state["plan"]:
                if task.get("agent") == next_step and task.get("status") == "pending":
                    task["status"] = "in_progress"
                    logging.info(f"Setting task '{task.get('task')}' as in_progress for {next_step}")
                    break

        # Log the final plan and routing decision
        logging.info(f"Final plan: {json.dumps(state.get('plan', []))}")
        logging.info(f"Final routing decision: {next_step}")
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
        
        # Get model name from session state
        model_name = st.session_state.get("model_name", "gpt-4o")
        logging.info(f"Using model: {model_name}")
        
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
        
        # 1. Extract visualization agent outputs and plot information
        visualization_used = state.get("visualization_agent_used", False)
        plot_images = state.get("plot_images", [])
        logging.info(f"Visualization used: {visualization_used}, Plot images count: {len(plot_images)}")
        
        # 2. Get conversation history with validation
        all_messages = state.get("messages", [])
        logging.info(f"Total message count: {len(all_messages)}")
        
        # Debug message types
        message_types = {}
        for msg in all_messages:
            msg_type = type(msg).__name__
            if msg_type not in message_types:
                message_types[msg_type] = 0
            message_types[msg_type] += 1
        logging.info(f"Message types: {message_types}")
        
        # 3. Extract visualization messages specifically
        visualization_messages = [msg for msg in all_messages 
                                if hasattr(msg, 'name') and msg.name == "VisualizationAgent"]
        logging.info(f"Visualization messages count: {len(visualization_messages)}")
        
        # 4. Extract latest user query with validation
        user_messages = [msg for msg in all_messages if not hasattr(msg, 'name') or not msg.name]
        if user_messages:
            latest_user_query = user_messages[-1].content
            logging.info(f"Latest user query: '{latest_user_query[:50]}...'")
        else:
            latest_user_query = state.get("input", "No query found")
            logging.info(f"No user messages found, using state input: '{latest_user_query[:50]}...'")
        
        # Format datasets information with deep validation
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
                
                # Build dataset text with explicit format
                current_dataset = (
                    f"Dataset {i}:\n"
                    f"Name: {info.get('name', 'Unknown')}\n"
                    f"DOI: {doi}\n"
                    f"Description: {info.get('description', 'No description available')}\n"
                    f"Type: {info.get('data_type', 'Unknown type')}\n"
                )
                datasets_text += current_dataset
                
                # Log what we're adding
                logging.info(f"Added dataset {i}: {info.get('name', 'Unknown')}")
        else:
            datasets_text = "No active dataset selected."
            logging.warning("No active datasets found - using empty placeholder")
        
        # Log the final datasets_text for debugging
        logging.info(f"Final datasets_text length: {len(datasets_text)}")
        logging.info(f"datasets_text preview: {datasets_text[:200]}...")
        
        # Format visualization information
        visualization_info = ""
        if visualization_used or visualization_messages:
            visualization_info = "The VisualizationAgent was used in this conversation.\n"
            if visualization_messages:
                recent_vis_content = visualization_messages[-1].content
                visualization_info += f"Most recent visualization analysis: {recent_vis_content[:100]}...\n"
            if plot_images:
                visualization_info += f"Plots were generated and are available at: {', '.join(plot_images)}\n"
        
        logging.info(f"Visualization info length: {len(visualization_info)}")
        
        # Get agent context
        last_agent_message = state.get("last_agent_message", "")
        logging.info(f"Last agent message length: {len(last_agent_message)}")
        
        # Construct system message with debugging
        try:
            # Create a direct substitution in the system message (not using f-string with placeholders)
            system_message = f"""You are a supervisor capable of answering simple questions directly. If the user's query is basic (e.g., about available analysis), answer using the selected dataset context below:

{datasets_text}

{visualization_info}
Always address the latest user query directly, even if it is similar to previous queries. You can acknowledge repetition (e.g., 'As I mentioned earlier...') and reference previous answers if relevant, but ensure each query receives a clear and complete response. For complex queries, follow these agent guidelines:
- Use VisualizationAgent for general plotting
Format any code in markdown and keep responses concise.

The last agent that processed this request said: {last_agent_message}

Latest user query: {latest_user_query}

Please provide a response to the latest user query, taking into account the conversation history."""
            
            logging.info(f"System message constructed successfully, length: {len(system_message)}")
            
            # Check for template literals that weren't substituted
            if "{datasets_text}" in system_message:
                logging.error("ERROR: '{datasets_text}' template literal found in system message!")
            if "{visualization_info}" in system_message:
                logging.error("ERROR: '{visualization_info}' template literal found in system message!")
            
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
            "visualization_info": len(visualization_info),
            "last_agent_message": len(last_agent_message),
            "latest_user_query": len(latest_user_query)
        }
        logging.info(f"Prompt component lengths: {json.dumps(prompt_components)}")
        
        # Invoke the LLM with the full context
        try:
            llm = ChatOpenAI(api_key=API_KEY, model_name=model_name)
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
            # IMPORTANT: Instead of creating a new state, modify the existing one
            # This preserves all the existing information including plot_images
            state["messages"] = state.get("messages", []) + [
                AIMessage(content=response.content, name="Supervisor")
            ]
            state["next"] = "FINISH"
            
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
    if visualization_agent:
        workflow.add_node("VisualizationAgent", 
                         functools.partial(agent_node, agent=visualization_agent, name="VisualizationAgent"))
    if dataframe_agent:
        workflow.add_node("DataFrameAgent", 
                         functools.partial(agent_node, agent=dataframe_agent, name="DataFrameAgent"))
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

    limited_messages = messages[-15:]
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
        "configurable": {"thread_id": session_data.get('thread_id', str(uuid.uuid4())), "recursion_limit": 5}
    }
    if st_callback:
        config["callbacks"] = [st_callback]
        logging.info("StreamlitCallbackHandler added to config.")
    else:
        logging.info("No StreamlitCallbackHandler provided.")

    try:
        response = graph.invoke(initial_state, config=config)
        session_data["processing"] = False
        return response
    except Exception as e:
        session_data["processing"] = False
        logging.error(f"Error during graph invocation: {e}", exc_info=True)
        raise e