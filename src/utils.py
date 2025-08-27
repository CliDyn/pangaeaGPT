# src/utils.py - Updated generate_unique_image_path function

import os
import uuid
import re
import logging
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import time
import json

# Generate a unique image path for saving plots
def generate_unique_image_path(sandbox_path=None):
    """
    Generate a unique image path for saving plots.
    
    Args:
        sandbox_path: Optional path to the current sandbox directory.
                     If provided, saves to sandbox/results/ instead of tmp/figs/
    
    Returns:
        str: Full path to save the image
    """
    unique_filename = f'fig_{uuid.uuid4()}.png'
    logging.info(f"DEBUG generate_unique_image_path called with sandbox_path: {sandbox_path}")
    if sandbox_path and os.path.exists(sandbox_path):
        # Save to sandbox/results/ directory
        results_dir = os.path.join(sandbox_path, 'results')
        os.makedirs(results_dir, exist_ok=True)
        unique_path = os.path.join(results_dir, unique_filename)
        logging.debug(f"Generated sandbox image path: {unique_path}")
    else:
        # Fallback to original behavior
        figs_dir = os.path.join('tmp', 'figs')
        os.makedirs(figs_dir, exist_ok=True)
        unique_path = os.path.join(figs_dir, unique_filename)
        logging.debug(f"Generated default image path: {unique_path}")
    
    return unique_path


# Function to sanitize input
def sanitize_input(query: str) -> str:
    return query.strip()

# Define the function to extract the last Python REPL command
def get_last_python_repl_command():
    import streamlit as st  # Ensure Streamlit is imported
    if 'intermediate_steps' not in st.session_state:
        logging.warning("No intermediate steps found in session state.")
        return None

    intermediate_steps = st.session_state['intermediate_steps']
    python_repl_commands = []
    for step in intermediate_steps:
        action = step[0]
        observation = step[1]
        if action.get('tool') == 'Python_REPL':
            python_repl_commands.append(action)

    if python_repl_commands:
        last_command_action = python_repl_commands[-1]
        command = last_command_action.get('tool_input', '')
        logging.debug(f"Extracted last Python REPL command: {command}")
        return command
    else:
        logging.warning("No Python_REPL commands found in intermediate steps.")
        return None


def make_json_serializable(obj):
    """
    Recursively convert any object to a JSON-serializable format.
    Handles nested structures, pandas objects, numpy arrays, etc.
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, date
    
    # Handle None
    if obj is None:
        return None
    
    # Handle basic JSON-serializable types
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Handle datetime objects
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    
    # Handle pandas Series
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    
    # Handle pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    
    # Handle numpy arrays and numpy scalars
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    
    # Handle dictionaries recursively
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    
    # Handle lists and tuples recursively
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    
    # Handle sets
    if isinstance(obj, set):
        return list(obj)
    
    # For any other object, try to convert to dict or string
    try:
        # Try to get __dict__ attribute
        if hasattr(obj, '__dict__'):
            return make_json_serializable(obj.__dict__)
    except:
        pass
    
    # Last resort: convert to string
    return str(obj)


def log_history_event(session_data: dict, event_type: str, details: dict):
    if "execution_history" not in session_data:
        session_data["execution_history"] = []  # fallback

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    event = {
        "type": event_type,
        "timestamp": timestamp
    }
    
    # Convert details to JSON-serializable format BEFORE adding to event
    try:
        serializable_details = make_json_serializable(details)
        event.update(serializable_details)
        logging.debug(f"Successfully serialized event details for type: {event_type}")
    except Exception as e:
        logging.error(f"Failed to serialize event details for type {event_type}: {e}")
        # Fallback: store a simplified version
        event.update({
            "serialization_error": str(e),
            "original_keys": list(details.keys()) if isinstance(details, dict) else "not_dict"
        })

    session_data["execution_history"].append(event)

def list_directory_contents(path):
    """
    Generate a formatted string listing all files in a directory and its subdirectories.
    """
    result = []
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * level
        result.append(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            result.append(f"{sub_indent}{file}")
    return "\n".join(result)


def escape_curly_braces(text):
    if isinstance(text, str):
        return text.replace('{', '{{').replace('}', '}}')
    return str(text)  # Convert non-strings to strings safely