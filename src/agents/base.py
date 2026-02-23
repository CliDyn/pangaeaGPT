# src/agents/base.py
import os
import time
import logging
import uuid
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain_core.runnables import RunnableConfig

import sys
from typing import Dict, List, Any, Tuple

from ..utils import log_history_event
from ..utils.workspace import WorkspaceManager
from ..tools.package_tools import install_package_tool
from ..utils.session_manager import SessionManager


# =============================================================================
# SMART COMPRESSION MEMORY SYSTEM
# =============================================================================

def compress_messages_to_summary(messages: List[Any], existing_summary: str = "") -> str:
    """
    Compresses a list of messages into a concise summary using a fast LLM.
    This is the "Long-Term Memory" component of the Smart Compression system.

    Args:
        messages: List of messages to compress
        existing_summary: Any existing summary to incorporate

    Returns:
        str: A compressed summary of the conversation context
    """
    from langchain_openai import ChatOpenAI
    from ..config import API_KEY

    if not messages:
        return existing_summary

    # Format messages for compression
    message_text = ""
    for msg in messages:
        if hasattr(msg, 'name') and msg.name:
            speaker = msg.name
        elif isinstance(msg, HumanMessage):
            speaker = "User"
        elif isinstance(msg, AIMessage):
            speaker = "Assistant"
        else:
            speaker = "System"

        content = msg.content if hasattr(msg, 'content') else str(msg)
        # Truncate very long individual messages
        if len(content) > 1500:
            content = content[:1500] + "... [truncated]"
        message_text += f"[{speaker}]: {content}\n\n"

    # Use gpt-4.1-mini for fast, cheap compression
    compression_llm = ChatOpenAI(
        api_key=API_KEY,
        model_name="gpt-4.1-mini",
        temperature=0.0
    )

    compression_prompt = f"""You are a scientific analysis session summarizer. Your task is to compress conversation history into a dense, information-preserving summary.

CRITICAL: Preserve these elements with EXACT precision:
1. **Variable names** (e.g., df_drift, dataset_1_path, merged_data)
2. **File paths** (e.g., /path/to/era5_data/temperature.nc)
3. **Data transformations** (filtering logic, merge keys, aggregations)
4. **Key findings** (statistics, correlations, anomalies found)
5. **Plot/output locations** (saved files, generated figures)

EXISTING CONTEXT (incorporate this):
{existing_summary if existing_summary else "None yet."}

NEW MESSAGES TO COMPRESS:
{message_text}

OUTPUT FORMAT - Be structured and dense:
## Data State
- Variables created: [list with types]
- Files loaded/saved: [paths]

## Operations Performed
- [Step-by-step what was done]

## Key Results
- [Numbers, findings, conclusions]

## Current Status
- [What's ready for next steps]

Compress now:"""

    try:
        response = compression_llm.invoke([HumanMessage(content=compression_prompt)])
        logging.info(f"Compressed {len(messages)} messages into summary of {len(response.content)} chars")
        return response.content
    except Exception as e:
        logging.error(f"Error compressing messages: {e}")
        # Fallback: simple concatenation
        return existing_summary + "\n\n[Compression failed - raw context]: " + message_text[:2000]


def get_memory_context_for_agent(state: Dict) -> str:
    """
    Builds the complete memory context to inject into an agent's input.
    Combines the compressed chat_summary with recent messages.

    Args:
        state: The workflow state containing chat_summary and messages

    Returns:
        str: Formatted context string for injection
    """
    context_parts = []

    # Part 1: Compressed long-term memory
    chat_summary = state.get("chat_summary", "")
    if chat_summary:
        context_parts.append(
            "=" * 60 + "\n"
            "📓 SESSION CONTEXT (Previous Steps Summary)\n"
            "=" * 60 + "\n"
            f"{chat_summary}\n"
            "=" * 60
        )

    # Part 2: Recent messages (short-term memory) - last 3 for immediate context
    recent_messages = state.get("messages", [])[-3:]
    if recent_messages:
        recent_text = "\n".join([
            f"[{msg.name if hasattr(msg, 'name') and msg.name else 'User'}]: {msg.content[:500]}..."
            if len(msg.content) > 500 else
            f"[{msg.name if hasattr(msg, 'name') and msg.name else 'User'}]: {msg.content}"
            for msg in recent_messages if hasattr(msg, 'content')
        ])
        context_parts.append(
            "\n📌 RECENT CONTEXT (Last 3 messages):\n"
            f"{recent_text}"
        )

    return "\n\n".join(context_parts) if context_parts else ""

def agent_node(state, agent, name, config: RunnableConfig = None):
    """
    Common agent node function used by all agents in the workflow.

    Enhanced with SMART COMPRESSION MEMORY:
    - Injects chat_summary (long-term memory) into agent context
    - Preserves variable names, file paths, and key results across steps

    Args:
        state: The current state of the workflow
        agent: The agent to execute
        name: The name of the agent
        config: RunnableConfig from LangGraph containing thread_id

    Returns:
        dict: The updated state after agent execution
    """
    import time
    import os
    logging.info(f"Entering agent_node for {name}")

    session_id = config.get("configurable", {}).get("thread_id", "default") if config else "default"
    session_data = SessionManager.get_session(session_id)

    # Generate unique IDs for this agent execution
    agent_id = str(uuid.uuid4())
    agent_start_time = time.time()

    # Add thinking log entry for agent start
    logging.info(f"Starting processing with {name}")
    session_data["processing"] = True

    if 'agent_scratchpad' not in state or not isinstance(state['agent_scratchpad'], list):
        state['agent_scratchpad'] = []

    # Keep original logic for backward compatibility, but we'll override it if there's a task
    # DON'T modify shared state - use local variables instead
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    local_input = state.get('input', '')
    if user_messages:
        last_user_message = user_messages[-1].content
        local_input = last_user_message

    # Find and include the task from the plan for this agent
    current_task = None
    for task in state.get('plan', []):
        if task.get('agent') == name and task.get('status') in ['in_progress', 'pending']:
            current_task = task
            break

    if current_task:
        task_description = current_task.get('task', '')
        # Only use the task description without the original user query
        local_input = f"TASK: {task_description}"
        logging.info(f"Modified input for {name} to include ONLY task: {task_description}")

    # ==========================================================================
    # SMART COMPRESSION MEMORY INJECTION
    # Inject the compressed session context so this agent knows what happened before
    # ==========================================================================
    memory_context = get_memory_context_for_agent(state)
    if memory_context:
        # Prepend the memory context to the agent's input
        local_input = f"{memory_context}\n\n---\n\n🎯 CURRENT TASK:\n{local_input}"
        logging.info(f"Injected {len(memory_context)} chars of memory context into {name}'s input")

    if 'plot_images' not in state or not isinstance(state['plot_images'], list):
        state['plot_images'] = []

    # Log that the agent is thinking
    thinking_id = str(uuid.uuid4())
    logging.info(f"Thinking... for {name}")
    
    # MODIFICATION: Only filter messages if we have a task (normal supervisor flow)
    # This preserves backward compatibility for any direct agent calls
    if current_task:
        # Filter messages to exclude direct user messages when operating under supervisor
        filtered_messages = []
        for msg in state.get('messages', []):
            # Skip HumanMessage objects (direct user input)
            if isinstance(msg, HumanMessage):
                logging.debug(f"Filtering out HumanMessage from {name}'s message history")
                continue
            # Include AI messages from supervisor or other agents
            if isinstance(msg, AIMessage):
                filtered_messages.append(msg)
            # Include system messages if any
            if isinstance(msg, SystemMessage):
                filtered_messages.append(msg)
        
        logging.info(f"Filtered messages for {name}: {len(filtered_messages)} messages (from original {len(state.get('messages', []))})")
        messages_to_use = filtered_messages
    else:
        # If no task (edge case), use all messages for backward compatibility
        messages_to_use = state.get('messages', [])
        logging.info(f"No task found for {name}, using all {len(messages_to_use)} messages")
    
    # Prepare the state for agent invocation
    # Remove any plot_path references that might be expected
    # Ensure we're only passing what the agent expects
    agent_state = {
        'messages': messages_to_use,  # Use either filtered or all messages based on context
        'input': local_input,  # Use the local variable instead of modifying shared state
        'agent_scratchpad': state.get('agent_scratchpad', []),
        'plot_path': '',  # Empty string to satisfy template while transitioning to results_dir
        'file': ''
    }
    
    # Add any additional state items
    for key, value in state.items():
        if key not in agent_state:
            agent_state[key] = value
    
    # Invoke the agent with cleaned state
    llm_start_time = time.time()
    result = agent.invoke(agent_state)
    llm_end_time = time.time()
    
    # Log thinking completion
    logging.info(f"Completed thinking for {name}")
    
    last_message_content = result.get("output", "")
    intermediate_steps = result.get("intermediate_steps", [])
    returned_plot_images = result.get("plot_images", [])

    if "intermediate_steps" not in session_data:
        session_data["intermediate_steps"] = []
    session_data["intermediate_steps"].extend(intermediate_steps)
    logging.info(f"Stored {len(intermediate_steps)} intermediate steps for {name}")

    # Extract plot_images from the agent's result
    agent_plot_images = []
    
    # Check if the agent returned plot_images in its result
    if isinstance(result, dict) and 'plot_images' in result:
        agent_plot_images = result.get('plot_images', [])
        logging.info(f"Agent {name} returned {len(agent_plot_images)} plot images")
    
    # Also check intermediate steps for any plots generated by Python_REPL
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
        
        # Check for plot_images in Python_REPL observations
        if tool_name == 'Python_REPL' and isinstance(observation, dict):
            step_plot_images = observation.get('plot_images', [])
            for img in step_plot_images:
                if img not in agent_plot_images:
                    agent_plot_images.append(img)
                    logging.info(f"Found plot from Python_REPL: {img}")
        
        log_history_event(
            session_data,
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
    if name in ["OceanographerAgent", "EcologistAgent", "VisualizationAgent"]:
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
                    retry_result = agent.invoke(agent_state)
                    last_message_content = retry_result.get("output", "")
                else:
                    logging.info(f"Failed to install {missing_module}")
                    last_message_content = f"Failed to install the missing package '{missing_module}'. Please install it manually."

    # Validate that plot files exist
    valid_plot_images = []
    for img_path in agent_plot_images:
        if os.path.exists(img_path):
            valid_plot_images.append(img_path)
            if img_path not in state["plot_images"]:
                state["plot_images"].append(img_path)
            session_data["new_plot_generated"] = True
            
            # Log the plot generation
            plot_id = str(uuid.uuid4())
            logging.info(f"Found valid plot: {os.path.basename(img_path)}")
            
            log_history_event(
                session_data,
                "plot_generated",
                {
                    "plot_path": img_path,
                    "agent_name": name,
                    "description": f"Plot generated by {name}"
                }
            )
        else:
            logging.warning(f"Plot file not found: {img_path}")
    
    # Clear any legacy plot path references
    session_data["new_plot_path"] = None
    
    # Create the AI message with plot info
    ai_message = AIMessage(
        content=last_message_content,
        name=name,
        additional_kwargs={
            "plot_images": valid_plot_images,  # Use the validated plot images
            "plot": valid_plot_images[0] if valid_plot_images else None
        }
    )
    
    # Also ensure the state includes these plot images
    state["plot_images"] = valid_plot_images
    
    state["messages"].append(ai_message)
    logging.info(f"Appended AI message for {name} with {len(valid_plot_images)} plot images")

    # ==========================================================================
    # SMART COMPRESSION MEMORY MANAGEMENT
    # Instead of crude truncation, we compress old messages into chat_summary
    # This is handled by the supervisor, but we track message count here
    # ==========================================================================
    message_count = len(state.get("messages", []))
    logging.info(f"Current message count after {name}: {message_count}")
    # NOTE: Compression is triggered by supervisor_with_planning when count > threshold
    # We do NOT truncate here - that would lose context!

    # Store visualization state clearly
    if name == "OceanographerAgent":
        state["oceanographer_agent_used"] = True
        state["specialized_agent_used"] = True
    elif name == "EcologistAgent":
        state["ecologist_agent_used"] = True
        state["specialized_agent_used"] = True
    elif name == "VisualizationAgent":
        state["visualization_agent_used"] = True
        state["specialized_agent_used"] = True
    elif name == "DataFrameAgent":
        state["dataframe_agent_used"] = True
        
    # Ensure last agent message is clearly stored
    state["last_agent_message"] = last_message_content
    
    # Add completion note to thinking log
    agent_end_time = time.time()
    logging.info(f"Completed processing with {name}")
    
    logging.info(f"Completed agent_node for {name}")
    return state

def prepare_visualization_environment(datasets_info: List[Dict[str, Any]], session_id: str = "default") -> Tuple[Dict[str, Any], str, List[str]]:
    """
    Prepares dataset information using WorkspaceManager.
    Enhanced with full sandbox file scanning for accumulated datasets (Shopping Cart strategy).
    """
    # Initialize variables
    datasets_text = ""
    dataset_variables = []
    datasets = {}

    # 1. Get Main Paths via Manager
    thread_id = session_id

    uuid_main_dir = WorkspaceManager.get_sandbox_path(thread_id)
    results_dir = WorkspaceManager.get_results_dir(thread_id)

    datasets["uuid_main_dir"] = uuid_main_dir
    datasets["results_dir"] = results_dir

    logging.info(f"Environment Prep: Sandbox={uuid_main_dir}")

    # --- DEEP SCAN SANDBOX FOR ALL ACCUMULATED FILES (Shopping Cart strategy) ---
    sandbox_files = []
    if os.path.exists(uuid_main_dir):
        for root, dirs, files in os.walk(uuid_main_dir):
            # Skip results folder (outputs, not inputs)
            if 'results' in root:
                continue
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in files:
                if not f.startswith('.'):
                    # Create clean relative path from sandbox root
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, uuid_main_dir).replace('\\', '/')
                    sandbox_files.append(rel_path)

    logging.info(f"Environment Prep: Found {len(sandbox_files)} files in sandbox (accumulated data)")
    # -----------------------------------------------

    # 2. Build Prompt Instructions (Using forward slashes for cross-platform python safety)
    clean_main = uuid_main_dir.replace(os.sep, '/')
    clean_results = results_dir.replace(os.sep, '/')

    uuid_paths = f"### ACCUMULATED WORKSPACE FILES (The Shopping Cart)\n"
    uuid_paths += f"Root Dir: `{clean_main}`\n"
    uuid_paths += f"Results Dir: `{clean_results}`\n\n"

    # --- LIST ALL ACCUMULATED FILES ---
    if sandbox_files:
        uuid_paths += "**Available Files for Analysis:**\n"
        # Show first 60 files with relative paths
        for f in sandbox_files[:60]:
            uuid_paths += f"- `{f}`\n"
        if len(sandbox_files) > 60:
            uuid_paths += f"... ({len(sandbox_files) - 60} more files)\n"
        uuid_paths += "\n**TIP:** Load files using: `pd.read_csv(os.path.join(uuid_main_dir, 'folder/file.csv'))`\n\n"
    else:
        uuid_paths += "No files in sandbox yet.\n\n"
    # ----------------------------------

    # 3. Process individual datasets (from metadata)
    for i, info in enumerate(datasets_info):
        var_name = f"dataset_{i + 1}"

        # FIX: Prioritize sandbox_path (string) over dataset (which might be a DataFrame)
        # This ensures the Python REPL tool receives string paths to create dataset_X_path variables
        ds_path = info.get('sandbox_path') or info.get('dataset')
        if isinstance(ds_path, str):
            datasets[var_name] = ds_path  # Pass string path for REPL variable generation
        else:
            datasets[var_name] = info['dataset']  # Fallback for backward compatibility

        dataset_variables.append(var_name)

        if ds_path and isinstance(ds_path, str):
            full_uuid_path = os.path.abspath(ds_path).replace(os.sep, '/')
            uuid_paths += f"# Dataset {i+1}: {info['name']}\n"
            uuid_paths += f"{var_name}_path = r'{full_uuid_path}'\n\n"

            # List files if directory exists
            if os.path.isdir(full_uuid_path):
                try:
                    files = os.listdir(full_uuid_path)
                    # Don't list too many files (e.g. for Zarr)
                    if len(files) < 50:
                        uuid_paths += f"# Files available: {', '.join(files)}\n\n"
                    else:
                        uuid_paths += f"# Files available: {len(files)} items (Zarr/folder)\n\n"
                except Exception:
                    pass

    # 4. Global warnings
    uuid_paths += "# CRITICAL WARNINGS\n"
    uuid_paths += "# 1. NEVER use '/mnt/data/...' paths\n"
    uuid_paths += "# 2. ALWAYS use the exact path variables shown above\n"
    uuid_paths += "# 3. ALWAYS check which files exist before reading\n"
    uuid_paths += "# 4. All accumulated files are available - use the file list above!\n\n"

    # 5. Summary
    datasets_summary = ""
    for i, info in enumerate(datasets_info):
        datasets_summary += (
            f"Dataset {i + 1}:\n"
            f"Name: {info['name']}\n"
            f"Description: {info['description']}\n"
            f"Type: {info.get('data_type', 'pandas DataFrame')}\n"
            f"Sample Data: {info.get('df_head', 'Not available')}\n\n"
        )

    datasets_text = uuid_paths + datasets_summary

    session_data = SessionManager.get_session(session_id)
    session_data["viz_datasets_text"] = datasets_text

    return datasets, datasets_text, dataset_variables

def create_standard_agent_executor(llm, tools, prompt_template, max_iterations=25) -> AgentExecutor:
    """
    Common function to create a standard AgentExecutor using create_openai_tools_agent.
    """
    agent = create_openai_tools_agent(
        llm,
        tools=tools,
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=max_iterations
    )