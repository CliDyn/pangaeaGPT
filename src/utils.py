# src/utils.py


import os
import uuid
import re
import logging
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Generate a unique image path for saving plots
def generate_unique_image_path():
    figs_dir = os.path.join('tmp', 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    unique_filename = f'fig_{uuid.uuid4()}.png'
    unique_path = os.path.join(figs_dir, unique_filename)
    logging.debug(f"Generated unique image path: {unique_path}")
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
