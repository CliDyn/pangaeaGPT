# src/agents/search_agent.py
import logging
import streamlit as st
from pydantic import BaseModel, Field
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from ..search.search_pg_default import pg_search_default, direct_access_doi
from ..search.publication_qa_tool import answer_publication_questions, PublicationQAArgs
from ..config import API_KEY

# Define the arguments schema for the search tool
class SearchPangaeaArgs(BaseModel):
    query: str = Field(description="The search query string.")
    mindate: Optional[str] = Field(default=None, description="The minimum date in 'YYYY-MM-DD' format.")
    maxdate: Optional[str] = Field(default=None, description="The maximum date in 'YYYY-MM-DD' format.")
    minlat: Optional[float] = Field(default=None, description="The minimum latitude in decimal degrees.")
    maxlat: Optional[float] = Field(default=None, description="The maximum latitude in decimal degrees.")
    minlon: Optional[float] = Field(default=None, description="The minimum longitude in decimal degrees.")
    maxlon: Optional[float] = Field(default=None, description="The maximum longitude in decimal degrees.")

class DoiDatasetAccess(BaseModel):
    doi: str = Field(description="One or more DOIs separated by commas. You can use formats like: full URLs (https://doi.pangaea.de/10.1594/PANGAEA.******), IDs (PANGAEA.******), or just numbers (******).")

# Function to search PANGAEA datasets
def search_pg_datasets_tool(query: str, mindate: Optional[str] = None, maxdate: Optional[str] = None,
                           minlat: Optional[float] = None, maxlat: Optional[float] = None,
                           minlon: Optional[float] = None, maxlon: Optional[float] = None):
    global prompt_search

    # Log the parameters for debugging
    logging.info(f"Searching with query: {query}, mindate: {mindate}, maxdate: {maxdate}, "
                 f"minlat: {minlat}, maxlat: {maxlat}, minlon: {minlon}, maxlon: {maxlon}")

    # Call the search function with all parameters
    datasets_info = pg_search_default(query, mindate=mindate, maxdate=maxdate,
                                     minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon)

    logging.debug("Datasets info: %s", datasets_info)

    if not datasets_info.empty:
        st.session_state.datasets_info = datasets_info
        st.session_state.messages_search.append({
            "role": "assistant",
            "content": f"**Search query:** {query} (mindate: {mindate}, maxdate: {maxdate}, "
                       f"minlat: {minlat}, maxlat: {maxlat}, minlon: {minlon}, maxlon: {maxlon})"
        })
        st.session_state.messages_search.append({
            "role": "assistant",
            "content": "**Datasets Information:**",
            "table": datasets_info.to_json(orient="split")
        })

        datasets_description = ""
        for i, row in datasets_info.iterrows():
            datasets_description += (
                f"Dataset {i + 1}:\n"
                f"Name: {row['Name']}\n"
                f"Description: {row['Short Description']}\n"
                f"Parameters: {row['Parameters']}\n\n"
            )

        prompt_search = (
            f"The user has provided the following query: {query}\n"
            f"Available datasets:\n{datasets_description}\n"
            "Please identify the top two datasets that best match the user's query and explain why they are the most relevant. "
            "Do not suggest datasets without values in the Parameters field, because they cannot be directly downloaded.\n"
            "Respond using the following schema:\n"
            "{dataset name}\n{reason why relevant}\n{propose some short analysis and further questions to answer}"
        )

    return datasets_info, prompt_search

def create_search_agent(datasets_info=None):
    """
    Creates a search agent for PANGAEA dataset discovery.
    
    Args:
        datasets_info: Optional information about existing datasets
        
    Returns:
        AgentExecutor: The search agent executor
    """
    model_name = st.session_state.get("model_name", "gpt-3.5-turbo")
    if model_name == "o3-mini":
        llm = ChatOpenAI(api_key=API_KEY, model_name=model_name)
    else:
        llm = ChatOpenAI(api_key=API_KEY, model_name=model_name)

    # Generate dataset description string
    datasets_description = ""
    if datasets_info is not None:
        for i, row in datasets_info.iterrows():
            datasets_description += f"Dataset {i + 1}:\nName: {row['Name']}\nDOI: {row['DOI']}\nDescription: {row['Short Description']}\nParameters: {row['Parameters']}\n\n"
    
    # Create the system prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
            """
            You're a dataset search assistant for PANGAEA. You have access to three tools:  
            1. **search_pg_datasets**: Your primary tool for searching datasets based on user queries.  
            2. **direct_access_doi**: Use this only when the user provides specific DOI links to load datasets directly.  
            3. **answer_publication_questions**: Use this only when the user asks about publications related to a specific dataset they've identified.  

            **Search Parameters (for search_pg_datasets):**  
            **Dates:**  
            - If the user gives dates (e.g., 'from 2015', '2000s', 'before 2020'), set `mindate` and/or `maxdate` in 'YYYY-MM-DD'.  
            - Single year (e.g., '2021') → `mindate='2021-01-01'`, `maxdate='2021-12-31'`.  
            - No dates? Leave `mindate` and `maxdate` blank.  

            **Spatial:**  
            - If the user names a region (e.g., 'Laptev Sea', 'Fram Strait') or coords (e.g., 'north of 60N'), *always* set `minlat`, `maxlat`, `minlon`, `maxlon` in decimal degrees.  
            - Named regions? Use rough coords and extend by ±3 degrees to capture a broader area (e.g., 'Fram Strait' → `minlat=74.0`, `maxlat=84.0`, `minlon=-13.0`, `maxlon=13.0` from typical 77-81°N, -10 to 10°E).  
            - Specific coords (e.g., 'between 40N and 50N')? Use the exact values without extension.  
            - No location? Leave spatial params blank.  

            **Examples (for search_pg_datasets):**  
            - 'Temperature salinity data from Laptev Sea in 2000s' → `query='temperature salinity'`, `mindate='2000-01-01'`, `maxdate='2009-12-31'`, `minlat=67.0`, `maxlat=83.0`, `minlon=87.0`, `maxlon=143.0` (extended from 70-80°N, 90-140°E)  
            - 'Ocean data Fram Strait 2020' → `query='ocean data'`, `mindate='2020-01-01'`, `maxdate='2020-12-31'`, `minlat=74.0`, `maxlat=84.0`, `minlon=-13.0`, `maxlon=13.0` (extended from 77-81°N, -10 to 10°E)  
            - 'Zooplankton data' → `query='zooplankton'` (no dates or spatial)  
            - 'Data between 40N and 50N' → `query=''`, `minlat=40.0`, `maxlat=50.0`, `minlon` and `maxlon` blank (exact values, no extension)  

            **Rules (for search_pg_datasets):**  
            - For named regions, extend the coordinate range by ±3 degrees to account for sampling variations.  
            - For specific coordinates, use the exact values provided.  

            **Direct DOI Access (for direct_access_doi):**  
            - If the user provides one or more DOI links (e.g., 'https://doi.pangaea.de/10.1594/PANGAEA.123456'), use the 'direct_access_doi' tool to load the datasets directly and switch to the Data Agent page.  
            - The 'direct_access_doi' tool accepts a list of DOI strings. Extract all DOIs from the user's message (whether full URLs or just the DOI identifier, e.g., '10.1594/PANGAEA.123456') and pass them as a list to the tool.  
            - Ensure DOIs are valid PANGAEA DOIs (starting with '10.1594/PANGAEA'). If a DOI doesn't match this format, ask the user to confirm it's a PANGAEA dataset before proceeding.  
            - Example 1: User says "Load this dataset: https://doi.pangaea.de/10.1594/PANGAEA.123456" → Use 'direct_access_doi' with ["https://doi.pangaea.de/10.1594/PANGAEA.123456"]  
            - Example 2: User says "Load these datasets: https://doi.pangaea.de/10.1594/PANGAEA.123456 and 10.1594/PANGAEA.789012" → Use 'direct_access_doi' with ["https://doi.pangaea.de/10.1594/PANGAEA.123456", "10.1594/PANGAEA.789012"]  
            - If the user provides a DOI not hosted by PANGAEA (e.g., '10.1000/xyz'), respond with: "This DOI doesn't appear to be a PANGAEA dataset. Please provide a PANGAEA DOI (e.g., '10.1594/PANGAEA.******') or clarify your request."  

            **Publication Questions (for answer_publication_questions):**  
            - Only use this tool when the user specifically asks about publications or research findings related to a dataset they've already identified.  
            - Ensure you correctly pass the DOI to the tool. It should be the DOI retrieved after the search, as specified by the user.  
            - Do not generate DOIs; use only what is provided in the conversation history.  
            - If needed, ask the user to clarify which dataset they're referring to before using the tool.  

            **Remember:**  
            1. Prioritize dataset searches using the **search_pg_datasets** tool. Rephrase the query to optimize elastic search results, avoiding words like 'search'.  
            2. Use **direct_access_doi** only when the user provides DOI links.  
            3. Use **answer_publication_questions** only when the user asks about publications or research findings related to a specific dataset.  
            4. Always provide accurate, helpful, and concise responses to user queries.  
            """
             
             ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create the search tool
    search_tool = StructuredTool.from_function(
        func=search_pg_datasets_tool,
        name="search_pg_datasets",
        description="List datasets from PANGAEA based on a query, with optional date and spatial filters.",
        args_schema=SearchPangaeaArgs
    )

    # Create the publication QA tool
    publication_qa_tool = StructuredTool.from_function(
        func=answer_publication_questions,
        name="answer_publication_questions",
        description="A tool to answer questions about articles published from this dataset. This will be a journal article for which you should provide the tool with an already structured question about what the user wants. The input should be the DOI of the dataset (e.g. 'https://doi.org/10.1594/PANGAEA.xxxxxx') and the question. The question should be reworded to specifically send it to RAG. E.g. the hypothetical user's question 'Are there any related articles to the first dataset? If so what these articles are about?' will be re-worded for this tool as 'What is this article is about?'",
        args_schema=PublicationQAArgs
    )

    # Create the direct DOI access tool
    direct_doi_access_tool = StructuredTool.from_function(
        func=direct_access_doi, 
        name="direct_access_doi",
        description="Tool to access datasets directly bypassing search. Use this when user provides specific DOI links or dataset IDs (can be comma-separated). Examples: https://doi.pangaea.de/10.1594/PANGAEA.936254, PANGAEA.936254, or just 936254.",
        args_schema=DoiDatasetAccess
    )
    
    # Define the tools
    tools = [search_tool, publication_qa_tool, direct_doi_access_tool]

    # Bind the tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # Create the agent
    agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x.get("chat_history", []),
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
    )

    # Create the agent executor
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

def process_search_query(user_input: str, search_agent, session_data: dict):
    """
    Processes a user search query using the search agent.
    
    Args:
        user_input (str): The user's query
        search_agent: The search agent executor
        session_data (dict): The session state data
        
    Returns:
        str: The AI response message
    """
    # Initialize or reset chat history
    session_data["chat_history"] = ChatMessageHistory(session_id="search-agent-session")
    
    # Populate the chat history with previous messages
    for message in session_data["messages_search"]:
        if message["role"] == "user":
            session_data["chat_history"].add_user_message(message["content"])
        elif message["role"] == "assistant":
            session_data["chat_history"].add_ai_message(message["content"])

    # Create a function to get truncated chat history
    def get_truncated_chat_history(session_id):
        truncated_messages = session_data["chat_history"].messages[-20:]
        truncated_history = ChatMessageHistory(session_id=session_id)
        
        for msg in truncated_messages:
            if isinstance(msg, HumanMessage):
                truncated_history.add_user_message(msg.content)
            elif isinstance(msg, AIMessage):
                truncated_history.add_ai_message(msg.content)
            else:
                truncated_history.add_message(msg)
                
        return truncated_history

    # Create the agent with message history
    search_agent_with_memory = RunnableWithMessageHistory(
        search_agent,
        get_truncated_chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # Invoke the agent with the user's query
    response = search_agent_with_memory.invoke(
        {"input": user_input},
        {"configurable": {"session_id": "search-agent-session"}},
    )

    # Extract the AI message
    ai_message = response["output"]
    return ai_message