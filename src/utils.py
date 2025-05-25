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


def log_history_event(session_data: dict, event_type: str, details: dict):
    if "execution_history" not in session_data:
        session_data["execution_history"] = []  # fallback

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    event = {
        "type": event_type,
        "timestamp": timestamp
    }
    event.update(details)  # merges in content from details

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