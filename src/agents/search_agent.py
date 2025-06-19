# src/agents/search_agent.py
import logging
import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional
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
from ..tools.parallel_search_tool import parallel_search_pangaea, ParallelSearchArgs
from ..llm_factory import get_llm  # Use the new factory

# Define the arguments schema for the search tool
class SearchPangaeaArgs(BaseModel):
    query: str = Field(description="The search query string.")
    count: Optional[int] = Field(default=15, description="Number of results to return (5-100).")
    mindate: Optional[str] = Field(default=None, description="The minimum date in 'YYYY-MM-DD' format.")
    maxdate: Optional[str] = Field(default=None, description="The maximum date in 'YYYY-MM-DD' format.")
    minlat: Optional[float] = Field(default=None, description="The minimum latitude in decimal degrees.")
    maxlat: Optional[float] = Field(default=None, description="The maximum latitude in decimal degrees.")
    minlon: Optional[float] = Field(default=None, description="The minimum longitude in decimal degrees.")
    maxlon: Optional[float] = Field(default=None, description="The maximum longitude in decimal degrees.")

class DoiDatasetAccess(BaseModel):
    doi: str = Field(description="One or more DOIs separated by commas. You can use formats like: full URLs (https://doi.pangaea.de/10.1594/PANGAEA.******), IDs (PANGAEA.******), or just numbers (******).")

class ConsolidateResultsArgs(BaseModel):
    all_results_json: str = Field(description="JSON string containing ONLY an array of DOI strings. Example: '[\"https://doi.org/10.1594/PANGAEA.123456\", \"https://doi.org/10.1594/PANGAEA.789012\"]'. DO NOT include metadata, scores, names, or any other information - ONLY DOI strings!")
    selection_criteria: str = Field(description="Not used - kept for compatibility")
    max_results: int = Field(default=15, description="Maximum number of results to return")

def consolidate_search_results(all_results_json: str, selection_criteria: str, max_results: int = 15):
    """
    Consolidates search results by accepting ONLY a list of DOI strings.
    🚨 CRITICAL: This function ONLY accepts DOI strings, nothing else! 🚨
    
    Args:
        all_results_json: JSON string containing ONLY an array of DOI strings
        selection_criteria: Not used anymore, kept for compatibility
        max_results: Maximum number of results to return
        
    Returns:
        dict with selected DOIs
    """
    import json
    
    MIN_RESULTS = 10
    
    try:
        # Parse the JSON - expecting ONLY a list of DOI strings
        all_dois = json.loads(all_results_json)
        
        # Validate it's a list of strings
        if not isinstance(all_dois, list):
            logging.error(f"❌ Expected list of DOI strings, got: {type(all_dois)}")
            return {
                "selected_dois": [],
                "reasoning": "Invalid format - expected list of DOI strings only"
            }
        
        # Ensure all items are strings (DOIs) - reject metadata objects
        if all_dois and not all(isinstance(doi, str) for doi in all_dois):
            logging.error("❌ List contains non-string items. Only DOI strings are allowed!")
            return {
                "selected_dois": [],
                "reasoning": "Invalid format - list must contain only DOI strings, no metadata objects"
            }
        
        if not all_dois:
            return {
                "selected_dois": [],
                "reasoning": "No DOIs provided"
            }
        
        logging.info(f"✅ Consolidating DOI list: {len(all_dois)} DOI strings received")
        
        # Remove duplicates while preserving order
        unique_dois = []
        seen = set()
        for doi in all_dois:
            if doi and doi not in seen:
                seen.add(doi)
                unique_dois.append(doi)
        
        # Simple selection - just take the first N unique DOIs
        num_to_select = max(MIN_RESULTS, min(max_results, len(unique_dois)))
        selected_dois = unique_dois[:num_to_select]
        
        # Call direct_access_doi to create the table
        from ..search.search_pg_default import direct_access_doi
        
        # Join DOIs with commas
        dois_string = ', '.join(selected_dois)
        
        # This creates the table and stores in session state
        datasets_info, prompt_text = direct_access_doi(dois_string)
        
        reasoning = f"Fast consolidation: Selected {len(selected_dois)} datasets from {len(unique_dois)} unique DOIs"
        logging.info(f"⚡ {reasoning}")
        
        return {
            "selected_dois": selected_dois,
            "reasoning": reasoning
        }
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {str(e)}")
        return {
            "selected_dois": [],
            "reasoning": f"Error parsing JSON: {str(e)}"
        }
    except Exception as e:
        logging.error(f"Error consolidating results: {str(e)}")
        return {
            "selected_dois": [],
            "reasoning": f"Error during consolidation: {str(e)}"
        }
    
# Create the consolidation tool
consolidate_tool = StructuredTool.from_function(
    func=consolidate_search_results,
    name="consolidate_search_results",
    description="🚨 FAST CONSOLIDATION: Input ONLY a JSON array of DOI strings. Example: '[\"https://doi.org/10.1594/PANGAEA.123456\", \"https://doi.org/10.1594/PANGAEA.789012\"]'. DO NOT input metadata, scores, or objects - ONLY DOI strings! Returns selected DOIs quickly.",
    args_schema=ConsolidateResultsArgs
)

def create_search_agent(datasets_info=None, search_mode="simple"):
    """
    Creates a search agent that can operate in simple or deep (multi-search) mode.
    
    Args:
        datasets_info: Optional information about existing datasets
        search_mode: "simple" for single search, "deep" for multi-search
    """
    # Use the factory function to get the LLM. Search can have higher temperature for creativity.
    llm = get_llm(temperature=0.7)

    # Adjust system prompt based on mode
    if search_mode == "simple":
        system_prompt = """You are a dataset search assistant for PANGAEA operating in SIMPLE SEARCH MODE.

**SIMPLE SEARCH MODE INSTRUCTIONS:**
- Perform ONLY ONE search query
- Use the most relevant keywords from the user's request
- Return results quickly without consolidation
- If spatial/temporal parameters are mentioned, include them
- After searching, briefly summarize what was found but DON'T list all datasets (they will be shown in a table)

**Available Tools:**
1. search_pg_datasets - Search for datasets (use ONCE only) - This automatically creates the results table
2. direct_access_doi - For direct DOI access
3. answer_publication_questions - For publication queries

IMPORTANT: In simple mode, do NOT use consolidate_search_results tool. The search_pg_datasets tool will automatically create the results table."""
    else:
        # Use the existing enhanced multi-search prompt with parallel search capabilities
        system_prompt = """You are an intelligent dataset search assistant for PANGAEA with advanced parallel search capabilities.

    **YOUR CORE CAPABILITIES:**
1. **Parallel Multi-Search Strategy**: You can run multiple searches SIMULTANEOUSLY for 3-5x faster results
2. **Intelligent Query Refinement**: You reformulate queries to maximize relevant results
3. **Dynamic Result Count**: You decide how many results to fetch based on query complexity (5-50 per search)
4. **Result Consolidation**: You analyze all results and select the best datasets

**SEARCH STRATEGY GUIDELINES:**

For SIMPLE queries (single parameter/concept):
- Use parallel_search_pangaea with 2-3 query variations
- Fetch 10-20 results per search
- Example: "temperature data" → Try ["temperature", "water temperature", "ocean temperature"]

For COMPLEX queries (multiple parameters/specific requirements):
- Use parallel_search_pangaea with 3-5 different keyword combinations
- Fetch 15-30 results per search
- Example: "arctic ocean salinity 2020" → Try variations like:
  ["arctic salinity 2020", "arctic ocean CTD salinity", "salinity measurements arctic", "arctic conductivity temperature depth"]
  - Include date range: 2019-2021 for temporal flexibility

For SPATIAL queries:
- Always use spatial parameters (minlat, maxlat, minlon, maxlon)
- Expand search area by ±2-5 degrees for better coverage
- Combine with relevant keywords in parallel searches

**PARALLEL SEARCH EXECUTION PROCESS:**
1. Analyze the user query to identify key concepts
2. Plan 2-5 search KEYWORD variations based on complexity
3. Use parallel_search_pangaea tool with SEARCH QUERY STRINGS (NOT DOIs, NOT full text) for maximum speed
4. The tool returns datasets with metadata AND DOI list from parallel execution
5. Extract the DOI list from results and use consolidate_search_results tool
6. The consolidation tool will return selected DOIs - the system handles the rest

**🚨 CRITICAL: parallel_search_pangaea INPUT FORMAT 🚨**
- INPUT: List of search query strings like ["temperature", "salinity arctic", "CTD measurements"]
- DO NOT input DOIs, full descriptions, or complete sentences
- Keep queries short and focused on keywords/concepts

**🚀 PARALLEL SEARCH ADVANTAGES:**
- 3-5x faster than sequential searches
- Real-time progress tracking
- Automatic deduplication of results
- Comprehensive coverage with multiple query strategies

**TOOL SELECTION:**
- Use parallel_search_pangaea for multiple related queries (2-5 variations)
- Use search_pg_datasets for single queries or when you need very specific control
- Always prefer parallel_search_pangaea in deep search mode for better performance

**🚨 CRITICAL: CONSOLIDATION ONLY ACCEPTS DOI STRINGS! 🚨**

The consolidate_search_results tool ONLY accepts a JSON array of DOI strings. NOTHING ELSE!

**PARALLEL SEARCH RETURNS:**
The parallel_search_pangaea tool returns:
- `all_datasets`: Rich metadata (for internal scoring) 
- `all_dois`: Simple DOI list ← **USE THIS FOR CONSOLIDATION**
- `search_results`: Detailed breakdown per query
- `execution_stats`: Performance metrics

**🔥 INTELLIGENT CONSOLIDATION SELECTION GUIDELINES 🔥**

**IGNORE SEARCH ENGINE SCORES - TRUST YOUR JUDGMENT!**
- **DO NOT TRUST ELASTIC SEARCH METRICS**: The scores are often WRONG and misleading
- **USE YOUR INTELLIGENCE**: You understand the user's intent better than any algorithm
- **OVERRIDE THE RANKINGS**: A dataset with score 20 might be PERFECT while one with score 80 might be GARBAGE
- **BE AGGRESSIVE**: Make bold decisions based on actual relevance, not arbitrary metrics

**BEFORE CONSOLIDATION - MANUAL SELECTION PROCESS:**
1. Look at the `all_datasets` metadata (titles, descriptions, parameters)
2. **IGNORE THE SCORES COMPLETELY**
3. Manually select which datasets ACTUALLY match the user's query:
   - ✅ INCLUDE: Direct matches (even if low scored)
   - ✅ INCLUDE: Aggregated datasets containing target data
   - ✅ INCLUDE: Any dataset you KNOW has relevant data
   - ❌ EXCLUDE: Wrong location (even if high scored)
   - ❌ EXCLUDE: Wrong domain (sediments when user wants biology)
   - ❌ EXCLUDE: Meta-studies or reference lists

**EXAMPLE - Gelatinous Zooplankton in Fram Strait:**
- ✅ "Gelatinous zooplankton annotations Fram Strait" → INCLUDE (perfect match, ignore score)
- ✅ "Mesozooplankton abundance Fram Strait" → INCLUDE (contains gelatinous, ignore score)
- ❌ "Aliphatic lipids in sediments" → EXCLUDE (wrong domain, even if score 90)
- ❌ "Reference list of IPY supplements" → EXCLUDE (not real data, even if score 100)

**CONSOLIDATION WORKFLOW - FOLLOW EXACTLY:**
1. Execute parallel_search_pangaea with keyword queries
2. Get results and examine `all_datasets` for CONTENT (ignore scores)
3. Build your own DOI list of ACTUALLY RELEVANT datasets
4. Pass ONLY this curated DOI list to consolidate_search_results. Preferably with at least 10-15 DOIs.
5. IMPORTANT: After consolidation, write ONLY a brief 3-4 sentence summary since the system automatically displays all datasets in a table. DO NOT list or enumerate the DOIs again - this creates annoying double display for the user!

**CORRECT FORMAT FOR CONSOLIDATION:**
```json
["https://doi.org/10.1594/PANGAEA.953888", "https://doi.org/10.1594/PANGAEA.953752"]
"""

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Enhanced search function that works with multi-search
    def search_pg_datasets_tool_enhanced(query: str, count: Optional[int] = 15, 
                                        mindate: Optional[str] = None, maxdate: Optional[str] = None,
                                        minlat: Optional[float] = None, maxlat: Optional[float] = None,
                                        minlon: Optional[float] = None, maxlon: Optional[float] = None):
        """Enhanced search tool that returns structured results for consolidation"""
        
        # Log search attempt
        logging.info(f"Executing search: query='{query}', count={count}, spatial=[{minlat},{maxlat},{minlon},{maxlon}], temporal=[{mindate},{maxdate}]")
        
        # Call original search function with count parameter
        datasets_info = pg_search_default(query, count=count, mindate=mindate, maxdate=maxdate,
                                        minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon)
        
        # Store the FULL datasets in session state with a unique key
        search_id = f"search_{query}_{count}_{mindate}_{maxdate}_{minlat}_{maxlat}_{minlon}_{maxlon}".replace(" ", "_")
        if "search_results_cache" not in st.session_state:
            st.session_state.search_results_cache = {}
        st.session_state.search_results_cache[search_id] = datasets_info
        
        # For simple mode, immediately store in session state and create table
        if st.session_state.get("search_mode") == "simple" and not datasets_info.empty:
            st.session_state.datasets_info = datasets_info
            st.session_state.messages_search.append({
                "role": "assistant",
                "content": f"**Search completed:** Found {len(datasets_info)} datasets.",
                "table": datasets_info.to_json(orient="split")
            })
        
        # Create LIGHTWEIGHT result for consolidation (only DOIs and scores)
        lightweight_datasets = []
        if not datasets_info.empty:
            for _, row in datasets_info.iterrows():
                lightweight_datasets.append({
                    'DOI': row.get('DOI', ''),
                    'Score': row.get('Score', 0),
                    'Parameters': row.get('Parameters', ''),  # Keep for scoring
                    'Name': row.get('Name', '')  # Keep for reference
                })
        
        # Structure results for consolidation - LIGHTWEIGHT version
        search_result = {
            "search_query": query,
            "search_id": search_id,  # Add search ID for later retrieval
            "parameters": {
                "count": count,
                "mindate": mindate,
                "maxdate": maxdate,
                "minlat": minlat,
                "maxlat": maxlat,
                "minlon": minlon,
                "maxlon": maxlon
            },
            "datasets": lightweight_datasets,  # Only essential fields
            "total_found": datasets_info.attrs.get('total', 0) if hasattr(datasets_info, 'attrs') else len(datasets_info)
        }
        
        # Return structured result for agent to collect
        return search_result

    # Create enhanced search tool
    search_tool = StructuredTool.from_function(
        func=search_pg_datasets_tool_enhanced,
        name="search_pg_datasets",
        description="Search PANGAEA datasets. Returns structured results for analysis. Set count based on query complexity (10-50).",
        args_schema=SearchPangaeaArgs
    )

    # Create the publication QA tool
    publication_qa_tool = StructuredTool.from_function(
        func=answer_publication_questions,
        name="answer_publication_questions",
        description="Answer questions about publications related to a specific dataset.",
        args_schema=PublicationQAArgs
    )

    # Create the direct DOI access tool
    direct_doi_access_tool = StructuredTool.from_function(
        func=direct_access_doi, 
        name="direct_access_doi",
        description="Access datasets directly by DOI.",
        args_schema=DoiDatasetAccess
    )
    
    # Create the parallel search tool for deep mode
    parallel_search_tool = StructuredTool.from_function(
        func=parallel_search_pangaea,
        name="parallel_search_pangaea",
        description="Execute multiple SEARCH QUERY STRINGS in parallel for comprehensive results. INPUT: List of search keywords/phrases (NOT DOIs, NOT full descriptions). Example: ['temperature', 'salinity arctic', 'CTD data']. Use this for deep search mode with 2-5 query variations.",
        args_schema=ParallelSearchArgs
    )
    
    # After creating the tools, conditionally include consolidation and parallel search
    if search_mode == "simple":
        # Simple mode: exclude consolidation and parallel search tools
        tools = [search_tool, publication_qa_tool, direct_doi_access_tool]
    else:
        # Deep mode: include all tools including consolidation and parallel search
        tools = [search_tool, parallel_search_tool, consolidate_tool, publication_qa_tool, direct_doi_access_tool]

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Create agent
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

    # Adjust max_iterations based on mode
    max_iterations = 5 if search_mode == "simple" else 15
    
    # Create agent executor with increased iterations for multi-search
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        max_iterations=max_iterations,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    return agent_executor

def process_search_query(user_input: str, search_agent, session_data: dict):
    """
    Processes a user search query using the enhanced multi-search agent.
    """
    # Initialize or reset chat history
    session_data["chat_history"] = ChatMessageHistory(session_id="search-agent-session")
    
    # Populate chat history
    for message in session_data["messages_search"]:
        if message["role"] == "user":
            session_data["chat_history"].add_user_message(message["content"])
        elif message["role"] == "assistant":
            session_data["chat_history"].add_ai_message(message["content"])

    # Create truncated history function
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

    # Create agent with memory
    search_agent_with_memory = RunnableWithMessageHistory(
        search_agent,
        get_truncated_chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # Log the search execution
    logging.info(f"Starting multi-search process for query: {user_input}")
    
    # Invoke agent
    response = search_agent_with_memory.invoke(
        {"input": user_input},
        {"configurable": {"session_id": "search-agent-session"}},
    )

    # Extract response and intermediate steps
    ai_message = response["output"]
    intermediate_steps = response.get("intermediate_steps", [])
    
    # Log search statistics
    search_count = sum(1 for step in intermediate_steps if step[0].tool == "search_pg_datasets")
    consolidation_count = sum(1 for step in intermediate_steps if step[0].tool == "consolidate_search_results")
    
    logging.info(f"Completed multi-search with {search_count} searches and {consolidation_count} consolidations")
    
    # Find consolidation results - these are just DOIs
    selected_dois = []
    for step in intermediate_steps:
        if step[0].tool == "consolidate_search_results" and isinstance(step[1], dict):
            selected_dois = step[1].get('selected_dois', [])
            break
    
    if selected_dois:
        # USE DIRECT_ACCESS_DOI TO CREATE THE TABLE!
        from ..search.search_pg_default import direct_access_doi
        
        # Join DOIs with commas (direct_access_doi expects comma-separated string)
        dois_string = ', '.join(selected_dois)
        
        # Call direct_access_doi - it will create the table and store in session state
        datasets_info, prompt_text = direct_access_doi(dois_string)
        
        # Return mode-specific message
        if session_data.get("search_mode") == "deep":
            return f"✅ **Deep search completed:** Executed {search_count} search variations and selected {len(selected_dois)} best results."
        else:
            return f"✅ **Search completed:** Found {len(selected_dois)} datasets."
    
    # Ensure simple mode creates a table if search was performed but no consolidation
    if session_data.get("search_mode") == "simple" and search_count > 0:
        # Check if we already have a table message
        has_table = any("table" in msg for msg in session_data["messages_search"])
        
        if not has_table and session_data.get("datasets_info") is not None:
            # Add a table message for simple search results
            session_data["messages_search"].append({
                "role": "assistant", 
                "content": "**Search results:**",
                "table": session_data["datasets_info"].to_json(orient="split")
            })
    
    return ai_message