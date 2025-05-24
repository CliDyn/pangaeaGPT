# src/tools/visualization_tools.py
import os
import logging
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
import streamlit as st

class ExampleVisualizationArgs(BaseModel):
    query: str = Field(description="The user's query about plotting.")

def get_example_of_visualizations(query: str) -> str:
    """
    Retrieves example visualizations related to the query.

    Parameters:
    - query (str): The user's query about plotting.

    Returns:
    - str: The content of the most relevant example file.
    """
    # Initialize embeddings from session state or config
    from ..config import API_KEY
    embeddings = OpenAIEmbeddings(api_key=API_KEY)

    # Load the existing vector store
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=os.path.join('data', 'examples_database', 'chroma_langchain_notebooks')
    )

    # Perform a similarity search
    results = vector_store.similarity_search_with_score(query, k=1)

    # Extract the most relevant document
    doc, score = results[0]

    # Construct the full path to the txt file
    file_name = doc.metadata['source'].lstrip('./')
    full_path = os.path.join('data', 'examples_database', file_name)

    # Read and return the content of the txt file
    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except Exception as e:
        logging.error(f"An error occurred while reading the file: {str(e)}")
        return ""  # Return empty string if error occurs

# Create the example visualization tool
example_visualization_tool = StructuredTool.from_function(
    func=get_example_of_visualizations,
    name="get_example_of_visualizations",
    description="Retrieves example visualization code related to the user's query.",
    args_schema=ExampleVisualizationArgs
)

# File listing tool definition
class ListPlottingDataFilesArgs(BaseModel):
    dummy_arg: str = Field(default="", description="(No arguments needed)")  

def list_plotting_data_files(dummy_arg: str = "") -> str:
    """
    Lists ALL files recursively from two sources:
    1. The data/plotting_data directory (static resources)
    2. All files in the current UUID sandbox directories (active datasets)
    
    Returns a flat list of all available file paths using relative paths.
    """
    import os
    import streamlit as st
    
    all_files = []
    cwd = os.getcwd()
    
    # Part 1: List files from data/plotting_data
    plotting_data_dir = os.path.join("data", "plotting_data")
    if os.path.exists(plotting_data_dir):
        for root, dirs, files in os.walk(plotting_data_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                # Keep this as a relative path
                all_files.append(f"STATIC: {full_path}")
    
    # Part 2: List all files from UUID sandbox directories
    if "active_datasets" in st.session_state and st.session_state["active_datasets"]:
        # Find all unique UUID sandbox parent directories
        sandbox_dirs = set()
        for doi in st.session_state["active_datasets"]:
            dataset_path, _ = st.session_state["datasets_cache"].get(doi, (None, None))
            if isinstance(dataset_path, str) and os.path.isdir(dataset_path):
                # Get the parent sandbox UUID directory
                parent_dir = os.path.dirname(os.path.abspath(dataset_path))
                sandbox_dirs.add(parent_dir)
        
        # List all files from each UUID directory recursively
        for sandbox_dir in sandbox_dirs:
            if os.path.exists(sandbox_dir):
                for root, dirs, files in os.walk(sandbox_dir):
                    for filename in files:
                        full_path = os.path.join(root, filename)
                        # Convert to relative path by removing the cwd prefix
                        if full_path.startswith(cwd):
                            rel_path = full_path[len(cwd)+1:]  # +1 to remove leading slash
                        else:
                            rel_path = full_path
                            
                        # Use consistent forward slashes
                        rel_path = rel_path.replace('\\', '/')
                        
                        # Include a prefix to distinguish ERA5 files
                        if "era5_data" in rel_path:
                            all_files.append(f"ERA5: {rel_path}")
                        else:
                            all_files.append(f"DATA: {rel_path}")
    
    # Return a simple list of all available files
    if all_files:
        return "Available files:\n" + "\n".join(all_files)
    else:
        return "No files found in plotting_data or active datasets."

# Create the list plotting data files tool
list_plotting_data_files_tool = StructuredTool.from_function(
    func=list_plotting_data_files,
    name="list_plotting_data_files",
    description="Lists ALL available files recursively, including plotting resources, dataset files, and ERA5 data. Use this to see exactly what files you can work with.",
    args_schema=ListPlottingDataFilesArgs
)

class WiseAgentToolArgs(BaseModel):
    query: str = Field(description="The query about visualization to send to Claude for advice. Include details about your dataset structure, variables, and visualization goals.")

def wise_agent(query: str) -> str:
    """
    A tool that uses Anthropic's Claude 3.7 Sonnet model to provide visualization advice.
    
    Args:
        query: The query about visualization to send to Claude
        
    Returns:
        str: Claude's advice on visualization
    """
    import streamlit as st
    import logging
    
    # Get Anthropic API key from secrets
    try:
        anthropic_api_key = st.secrets["general"]["anthropic_api_key"]
        logging.info("Successfully retrieved Anthropic API key from secrets")
    except KeyError:
        logging.error("Anthropic API key not found in .streamlit/secrets.toml")
        return "Error: Anthropic API key not found in .streamlit/secrets.toml. Please add it to use WISE_AGENT."
    
    # Get dataset information from viz_datasets_text in session state (set by create_visualization_agent)
    datasets_text = st.session_state.get("viz_datasets_text", "")
    
    if not datasets_text:
        # Try to extract dataset info from active datasets if viz_datasets_text not available
        try:
            from main import get_datasets_info_for_active_datasets
            datasets_info = get_datasets_info_for_active_datasets(st.session_state)
            
            # Format dataset information
            datasets_text = ""
            for i, info in enumerate(datasets_info, 1):
                datasets_text += (
                    f"Dataset {i}:\n"
                    f"Name: {info.get('name', 'Unknown')}\n"
                    f"DOI: {info.get('doi', 'Not available')}\n"
                    f"Description: {info.get('description', 'No description available')}\n"
                    f"Type: {info.get('data_type', 'Unknown type')}\n"
                    f"Sample Data: {info.get('df_head', 'No sample available')}\n\n"
                )
            logging.info("Successfully extracted dataset information from active datasets")
        except Exception as e:
            logging.error(f"Error retrieving dataset information: {str(e)}")
            datasets_text = "No dataset information available"
    
    # Get the list of available plotting data files
    try:
        available_files = list_plotting_data_files("")
        logging.info("Successfully retrieved available plotting data files")
    except Exception as e:
        logging.error(f"Error retrieving available files: {str(e)}")
        available_files = f"Error retrieving available files: {str(e)}"
    
    # Create the system prompt for Claude
    system_prompt = """You are WISE_AGENT, a scientific visualization expert specializing in data visualization for research datasets.

Your role is to provide specific, actionable advice on how to create the most effective visualizations for scientific data.

When giving visualization advice:
0. Provide superb visualizations! That's your life goal! 
1. ANALYZE THE DATA STRUCTURE first - recommend plot types based on the actual data dimensions and variables
2. Consider the SCIENTIFIC DOMAIN (oceanography, climate science, biodiversity) and its standard visualization practices
3. Recommend specific matplotlib/seaborn/plotly code strategies tailored to the data
4. Suggest appropriate color schemes that follow scientific conventions (e.g., sequential for continuous variables, categorical for discrete)
5. Provide precise advice on layouts, axes, legends, and annotations
6. For spatial/geographic data, recommend appropriate projections and map types
7. For time series, recommend appropriate temporal visualizations
8. Always prioritize clarity, accuracy, and scientific information density

Your advice should be specifically tailored to the datasets the user is working with. Be concise but thorough in your recommendations.
"""
    
    # Enhance the query with dataset information and available files
    enhanced_query = f"""
DATASET INFORMATION:
{datasets_text}

AVAILABLE PLOTTING DATA FILES:
{available_files}

USER QUERY:
{query}

Please provide visualization advice based on this information.
"""
    
    try:
        # Initialize the ChatAnthropic client with the specified model
        llm = ChatAnthropic(
            model="claude-3-7-sonnet-20250219",
            anthropic_api_key=anthropic_api_key,
            temperature=0.2,  # Low temperature for more precise advice
        )
        
        logging.info("Making request to Claude 3.7 Sonnet model with enhanced context")
        
        # Generate the response with the enhanced query
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enhanced_query}
            ]
        )
        
        logging.info("Successfully received response from Claude")
        return response.content
    except Exception as e:
        logging.error(f"Error using ChatAnthropic: {str(e)}")
        return f"Error using WISE_AGENT: {str(e)}"

# Create the wise agent tool
wise_agent_tool = StructuredTool.from_function(
    func=wise_agent,
    name="wise_agent",
    description="A tool that provides expert visualization advice using Anthropic's Claude 3.7 Sonnet model. Use this tool FIRST when planning complex visualizations or when you need guidance on best visualization practices for scientific data. Provide a detailed description of the data structure and visualization goals.",
    args_schema=WiseAgentToolArgs
)