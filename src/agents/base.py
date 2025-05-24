# src/agents/base.py
import logging
import os
import time
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from ..utils import log_history_event
from ..tools.package_tools import install_package_tool

def agent_node(state, agent, name):
    """
    Common agent node function used by all agents in the workflow.
    
    Args:
        state: The current state of the workflow
        agent: The agent to execute
        name: The name of the agent
        
    Returns:
        dict: The updated state after agent execution
    """
    import time
    import os
    logging.info(f"Entering agent_node for {name}")
    
    # Generate unique IDs for this agent execution
    agent_id = str(uuid.uuid4())
    agent_start_time = time.time()
    
    # Add thinking log entry for agent start
    logging.info(f"Starting processing with {name}")
    st.session_state.processing = True
    
    if 'agent_scratchpad' not in state or not isinstance(state['agent_scratchpad'], list):
        state['agent_scratchpad'] = []

    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if user_messages:
        last_user_message = user_messages[-1].content
        state['input'] = last_user_message
    else:
        state['input'] = state.get('input', '')
        
    # Find and include the task from the plan for this agent
    current_task = None
    for task in state.get('plan', []):
        if task.get('agent') == name and task.get('status') in ['in_progress', 'pending']:
            current_task = task
            break
            
    if current_task:
        task_description = current_task.get('task', '')
        # Only use the task description without the original user query
        state['input'] = f"TASK: {task_description}"
        logging.info(f"Modified input for {name} to include ONLY task: {task_description}")

    if 'plot_images' not in state or not isinstance(state['plot_images'], list):
        state['plot_images'] = []

    # Log that the agent is thinking
    thinking_id = str(uuid.uuid4())
    logging.info(f"Thinking... for {name}")
    
    # Invoke the agent
    # Add a placeholder for 'file' if the agent is VisualizationAgent
    if name == "VisualizationAgent" and 'file' not in state:
        state['file'] = None  # or provide an appropriate default value
    llm_start_time = time.time()
    result = agent.invoke(state)
    llm_end_time = time.time()
    
    # Log thinking completion
    logging.info(f"Completed thinking for {name}")
    
    last_message_content = result.get("output", "")
    intermediate_steps = result.get("intermediate_steps", [])
    returned_plot_images = result.get("plot_images", [])

    if 'intermediate_steps' not in st.session_state:
        st.session_state['intermediate_steps'] = []
    st.session_state['intermediate_steps'].extend(intermediate_steps)
    logging.info(f"Stored {len(intermediate_steps)} intermediate steps for {name}")

    # Log each step to the thinking log with hierarchy
    for i, step in enumerate(intermediate_steps):
        action = step[0]
        observation = step[1]
        tool_name = action.tool
        tool_input = action.tool_input
        
        # Create a unique ID for this tool execution
        tool_id = str(uuid.uuid4())
        tool_start_time = time.time() 
        
        # Add to thinking log
        input_summary = str(tool_input)
        if len(input_summary) > 100:
            input_summary = input_summary[:100] + "..."
            
        logging.info(f"Input: {input_summary} for {tool_name}")
        
        # Summarize the observation for the thinking log
        obs_summary = str(observation)
        if len(obs_summary) > 150:
            obs_summary = obs_summary[:150] + "..."
        
        # Add small delay to simulate execution time
        tool_end_time = time.time()
        
        logging.info(f"Result: {obs_summary} for {tool_name}")
        
        log_history_event(
            st.session_state,
            "tool_usage",
            {
                "agent_name": name,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "observation": observation
            }
        )
        logging.info(f"Logged tool usage for {name}, tool: {tool_name}")

    # Handle special cases like missing modules
    if name == "VisualizationAgent":
        if isinstance(last_message_content, dict):
            if last_message_content.get("error") == "ModuleNotFoundError":
                missing_module = last_message_content.get("missing_module")
                logging.info(f"Detected missing module: {missing_module}")
                
                install_id = str(uuid.uuid4())
                install_start = time.time()
                
                logging.info(f"Installing missing module: {missing_module}")
                
                install_result = install_package_tool.run({"package_name": missing_module})
                install_end = time.time()
                
                logging.info(f"Install result: {install_result}")
                if "successfully" in install_result:
                    logging.info(f"Successfully installed {missing_module}, retrying")
                    retry_result = agent.invoke(state)
                    last_message_content = retry_result.get("output", "")
                else:
                    logging.info(f"Failed to install {missing_module}")
                    last_message_content = f"Failed to install the missing package '{missing_module}'. Please install it manually."

    # Handle plot generation - UPDATED SECTION FOR FIXING FIGURE SAVING/DISPLAY
    new_plot_path = st.session_state.get("new_plot_path")
    logging.info(f"New plot path from session state: {new_plot_path}")
    
    # Add validation to ensure the plot path is valid and exists
    if new_plot_path and os.path.exists(new_plot_path):
        plot_id = str(uuid.uuid4())
        plot_time = time.time()
        
        # Create a log for the plot generation
        logging.info(f"Generated plot: {os.path.basename(new_plot_path)}")
        
        # Ensure the plot is added to the state's plot_images if not already there
        if new_plot_path not in state["plot_images"]:
            state["plot_images"].append(new_plot_path)
            st.session_state.new_plot_generated = True
        
        # Log the event for history tracking
        log_history_event(
            st.session_state,
            "plot_generated",
            {
                "plot_path": new_plot_path,
                "agent_name": name,
                "description": f"Plot generated by {name}"
            }
        )
        logging.info(f"Logged plot generation for {name}, path: {new_plot_path}")
    
    # Clear the new_plot_path to prevent duplicating the same plot
    st.session_state.new_plot_path = None
    
    # Finalize the agent execution
    # Combine plots from returned_plot_images and state["plot_images"], ensuring no duplicates
    all_plot_images = []
    for img_path in returned_plot_images:
        if os.path.exists(img_path) and img_path not in all_plot_images:
            all_plot_images.append(img_path)
    
    for img_path in state["plot_images"]:
        if os.path.exists(img_path) and img_path not in all_plot_images:
            all_plot_images.append(img_path)
    
    # Create the AI message with plot info
    ai_message = AIMessage(
        content=last_message_content,
        name=name,
        additional_kwargs={
            "plot_images": all_plot_images,
            "plot": all_plot_images[0] if all_plot_images else None
        }
    )
    state["messages"].append(ai_message)
    logging.info(f"Appended AI message for {name} with {len(all_plot_images)} plot images")

    # Trim messages if needed
    state["messages"] = state["messages"][-10:]
    
    # Store visualization state clearly
    if name == "VisualizationAgent":
        state["visualization_agent_used"] = True
        
    # Ensure last agent message is clearly stored
    state["last_agent_message"] = last_message_content
    
    # Add completion note to thinking log
    agent_end_time = time.time()
    logging.info(f"Completed processing with {name}")
    
    logging.info(f"Completed agent_node for {name}")
    return state