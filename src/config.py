# src/config.py

import os
import streamlit as st
import yaml
from datetime import datetime
import logging

# --- Load Central Configuration ---
CONFIG_FILE = os.path.join(os.getcwd(), "config.yaml")
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        app_config = yaml.safe_load(f)
else:
    app_config = {}

# Export deployment mode for use in other modules
# Check if running in CLI mode
IS_CLI_MODE = os.environ.get("PANGAEA_CLI_MODE", "false").lower() == "true"

# Export deployment mode for use in other modules
if IS_CLI_MODE:
    DEPLOYMENT_MODE = "cli"
else:
    DEPLOYMENT_MODE = app_config.get("deployment_mode", "huggingface")  # default to local if not specified

# --- Set API Keys based on Deployment Mode ---
if DEPLOYMENT_MODE == "local":
    # In local mode, use st.secrets
    try:
        API_KEY = st.secrets["general"]["openai_api_key"]
        LANGCHAIN_API_KEY = st.secrets["general"]["langchain_api_key"]
        # Use the environment variable if it exists; otherwise, fall back to st.secrets
        LANGCHAIN_PROJECT_NAME = os.environ.get("LANGCHAIN_PROJECT_NAME", st.secrets["general"]["langchain_project_name"])
        
        # Get LangChain region from secrets or default to 'us'
        LANGCHAIN_REGION = os.environ.get("LANGCHAIN_REGION", 
                                          st.secrets["general"].get("langchain_region", "us"))
    except (KeyError, AttributeError):
        # Fallback to environment variables if secrets not available
        API_KEY = os.environ.get("OPENAI_API_KEY", "")
        LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")
        LANGCHAIN_PROJECT_NAME = os.environ.get("LANGCHAIN_PROJECT_NAME", "")
        LANGCHAIN_REGION = os.environ.get("LANGCHAIN_REGION", "us")
elif DEPLOYMENT_MODE == "cli":
    # In CLI mode, use environment variables only
    API_KEY = os.environ.get("OPENAI_API_KEY", "")
    LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT_NAME = os.environ.get("LANGCHAIN_PROJECT_NAME", "")
    LANGCHAIN_REGION = os.environ.get("LANGCHAIN_REGION", "us")
else:
    # Huggingface or other modes
    API_KEY = os.environ.get("OPENAI_API_KEY", "")
    LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT_NAME = os.environ.get("LANGCHAIN_PROJECT_NAME", "")
    LANGCHAIN_REGION = os.environ.get("LANGCHAIN_REGION", "us")  # default to 'us' if not specified

# Set the appropriate LangChain endpoint based on region
if LANGCHAIN_REGION.lower() == "eu":
    LANGCHAIN_ENDPOINT = "https://eu.api.smith.langchain.com"
else:  # Default to US
    LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"

# --- Logging Setup (unchanged) ---
logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)

log_filename = f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_filepath = os.path.join(logs_dir, log_filename)
logging.basicConfig(
    filename=log_filepath,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)