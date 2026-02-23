# src/config.py

import os
import yaml
from datetime import datetime
import logging

# Attempt to import Streamlit — it may not be available in API/CLI mode
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except Exception:
    _HAS_STREAMLIT = False

# --- Load Environment Variables ---
from dotenv import load_dotenv
load_dotenv(override=True)

# --- Load Central Configuration ---
CONFIG_FILE = os.path.join(os.getcwd(), "config.yaml")
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        app_config = yaml.safe_load(f)
else:
    app_config = {}

# Export deployment mode for use in other modules
IS_CLI_MODE = os.environ.get("PANGAEA_CLI_MODE", "false").lower() == "true"

if IS_CLI_MODE:
    DEPLOYMENT_MODE = "cli"
else:
    DEPLOYMENT_MODE = app_config.get("deployment_mode", "huggingface")

# --- Set API Keys ---
# We always read from os.environ (which is populated by dotenv from the .env file)
API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("openai_api_key", ""))
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", os.environ.get("langchain_api_key", ""))
LANGCHAIN_PROJECT_NAME = os.environ.get("LANGCHAIN_PROJECT_NAME", os.environ.get("langchain_project_name", "test"))
LANGCHAIN_REGION = os.environ.get("LANGCHAIN_REGION", os.environ.get("langchain_region", "us"))
ARRAYLAKE_API_KEY = os.environ.get("ARRAYLAKE_API_KEY", os.environ.get("arraylake_api_key", ""))

if LANGCHAIN_REGION.lower() == "eu":
    LANGCHAIN_ENDPOINT = "https://eu.api.smith.langchain.com"
else:
    LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"

# --- Inject Keys Back into os.environ for Auto-Tracers ---
os.environ["OPENAI_API_KEY"] = API_KEY
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT_NAME
    os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT

# --- Logging Setup ---
logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)

log_filename = f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_filepath = os.path.join(logs_dir, log_filename)
logging.basicConfig(
    filename=log_filepath,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- ERA5 Configuration ---
from collections import namedtuple

DATA_DIR = "data"
PLOTS_DIR = "results"

VariableInfo = namedtuple("VariableInfo", ["short_name", "long_name", "units"])

ERA5_VARIABLES_DICT = {
    "sst": VariableInfo("sst", "Sea Surface Temperature", "K"),
    "t2": VariableInfo("t2", "2m Air Temperature", "K"),
    "u10": VariableInfo("u10", "10m u-component of wind", "m/s"),
    "v10": VariableInfo("v10", "10m v-component of wind", "m/s"),
    "mslp": VariableInfo("mslp", "Mean Sea Level Pressure", "Pa"),
    "tcc": VariableInfo("tcc", "Total Cloud Cover", "0-1"),
    "tp": VariableInfo("tp", "Total Precipitation", "m"),
    "sp": VariableInfo("sp", "Surface Pressure", "Pa"),
    "cp": VariableInfo("cp", "Convective Precipitation", "m"),
    "lsp": VariableInfo("lsp", "Large-scale Precipitation", "m"),
}

ERA5_VARIABLES = list(ERA5_VARIABLES_DICT.keys())

GEOGRAPHIC_REGIONS = {
    "north_atlantic": {"min_lat": 0.0, "max_lat": 60.0, "min_lon": 260.0, "max_lon": 360.0},
    "north_pacific": {"min_lat": 0.0, "max_lat": 60.0, "min_lon": 120.0, "max_lon": 260.0},
    "california_coast": {"min_lat": 32.0, "max_lat": 42.0, "min_lon": 235.0, "max_lon": 245.0},
    "mediterranean": {"min_lat": 30.0, "max_lat": 46.0, "min_lon": 354.0, "max_lon": 42.0},
    "gulf_of_mexico": {"min_lat": 18.0, "max_lat": 31.0, "min_lon": 260.0, "max_lon": 280.0},
    "caribbean": {"min_lat": 9.0, "max_lat": 22.0, "min_lon": 270.0, "max_lon": 300.0},
    "nino34": {"min_lat": -5.0, "max_lat": 5.0, "min_lon": 190.0, "max_lon": 240.0},
    "arctic": {"min_lat": 60.0, "max_lat": 90.0, "min_lon": 0.0, "max_lon": 360.0},
    "antarctic": {"min_lat": -90.0, "max_lat": -60.0, "min_lon": 0.0, "max_lon": 360.0},
    "global": {"min_lat": -90.0, "max_lat": 90.0, "min_lon": 0.0, "max_lon": 360.0},
}

def get_variable_info(var_id):
    short = get_short_name(var_id)
    return ERA5_VARIABLES_DICT.get(short)

def get_short_name(var_id):
    return var_id.lower()

def list_available_variables():
    return "\n".join([f"{k}: {v.long_name} ({v.units})" for k, v in ERA5_VARIABLES_DICT.items()])

def format_file_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"