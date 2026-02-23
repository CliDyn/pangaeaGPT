# src/agents/search_agent.py
import logging
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_tools_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from ..search.search_pg_default import pg_search_default, direct_access_doi
from ..search.publication_qa_tool import answer_publication_questions, PublicationQAArgs
from ..tools.parallel_search_tool import parallel_search_pangaea, ParallelSearchArgs
from ..llm_factory import get_llm
from ..utils.session_manager import SessionManager

# Define the arguments schema for the search tool
class SearchPangaeaArgs(BaseModel):
    query: str = Field(description="The search query string.")
    count: Optional[int] = Field(default=30, description="Results to return. Default 30 (for benchmark comparability). Max 50.")
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
    max_results: int = Field(default=30, description="Maximum number of results to return")

def consolidate_search_results(all_results_json: str, selection_criteria: str, max_results: int = 15, config: RunnableConfig = None):
    """
    FINAL STEP: Adds collected DOIs to the workspace using the accumulative loader.
    Uses Shopping Cart strategy - datasets are ADDED to existing workspace.

    Args:
        all_results_json: JSON list of ALL relevant DOIs collected during search
        selection_criteria: Not used - kept for compatibility
        max_results: Maximum number of results to return

    Returns:
        str: Success message with count
    """
    import json
    from ..search.search_pg_default import direct_access_doi

    try:
        # Robust JSON parsing - handle both string and list
        if isinstance(all_results_json, list):
            all_dois = all_results_json
        else:
            all_dois = json.loads(all_results_json)

        # Filter to valid DOI strings only
        valid_dois = [d for d in all_dois if isinstance(d, str) and d.strip()]

        if not valid_dois:
            return "No valid DOIs provided."

        logging.info(f"Consolidating {len(valid_dois)} DOIs into Shopping Cart")

        # Call backend accumulator - THIS IS THE KEY
        # direct_access_doi now ADDS to the session state (Shopping Cart)
        session_data = SessionManager.get_session(session_id)
        
        datasets_info, msg = direct_access_doi(', '.join(valid_dois), session_data=session_data)

        # Get total count from session
        total_count = len(session_data.get('datasets_info', [])) if session_data.get('datasets_info') is not None else len(valid_dois)

        return f"Success! Added {len(valid_dois)} datasets to Workspace. Total active: {total_count}."

    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {str(e)}")
        return f"Error parsing JSON: {str(e)}"
    except Exception as e:
        logging.error(f"Error consolidating results: {str(e)}")
        return f"Error during consolidation: {str(e)}"
    
# Create the consolidation tool (Shopping Cart "Checkout")
consolidate_tool = StructuredTool.from_function(
    func=consolidate_search_results,
    name="consolidate_search_results",
    description="FINAL STEP (Checkout): Call this when you have found ALL necessary components (e.g., Track + Wind + Ice). Input is a JSON list of all collected DOIs. They will be ADDED to the workspace (Shopping Cart).",
    args_schema=ConsolidateResultsArgs
)


# === SMART FILTER TOOL ===
class FilterWorkspaceArgs(BaseModel):
    user_query: str = Field(description="The original user query/request to filter datasets against")
    max_keep: int = Field(default=30, description="Maximum number of datasets to keep after filtering (default 30 for benchmark comparability)")

def filter_workspace_datasets(user_query: str, max_keep: int = 30, config: RunnableConfig = None):
    """
    FINAL FILTER: Analyzes ALL accumulated datasets and keeps only the most relevant ones.
    Uses LLM to intelligently score datasets based on Name, Parameters, and Description.

    Args:
        user_query: The user's original request
        max_keep: Maximum datasets to keep

    Returns:
        str: Summary of filtering results
    """
    import pandas as pd

    session_id = config.get("configurable", {}).get("session_id", "default") if config else "default"
    session_data = SessionManager.get_session(session_id)

    # Check if we have datasets
    if "datasets_info" not in session_data or session_data["datasets_info"] is None:
        return "No datasets in workspace to filter."

    df = session_data["datasets_info"]
    if df.empty:
        return "Workspace is empty."

    total_before = len(df)

    if total_before <= max_keep:
        return f"Workspace has {total_before} datasets (≤{max_keep}). No filtering needed."

    logging.info(f"Filtering {total_before} datasets down to {max_keep} based on: '{user_query}'")

    # Build scoring prompt for LLM
    llm = get_llm(temperature=0.0)  # Deterministic for scoring

    # Create dataset summaries for scoring
    dataset_summaries = []
    for idx, row in df.iterrows():
        summary = f"[{idx}] Name: {row.get('Name', 'Unknown')[:100]}"
        params = row.get('Parameters', '')
        if params:
            summary += f" | Params: {str(params)[:80]}"
        desc = row.get('Short Description', row.get('Description', ''))
        if desc:
            summary += f" | Desc: {str(desc)[:100]}"
        dataset_summaries.append(summary)

    datasets_text = "\n".join(dataset_summaries)

    scoring_prompt = f"""You are a dataset relevance scorer.

USER REQUEST: "{user_query}"

DATASETS IN WORKSPACE:
{datasets_text}

TASK: Select up to {max_keep} MOST RELEVANT datasets for the user's request.
Consider: dataset name, parameters, and description.

**TARGET: Select the best matching datasets (up to {max_keep}). More results = better for comprehensive analysis.**

RESPOND WITH ONLY a JSON array of the dataset indices (the numbers in brackets) to KEEP, **ORDERED BY RELEVANCE (most relevant first)**.
Example: [0, 3, 5, 7, 12]

Keep datasets that:
- Directly match the topic/region/timeframe requested
- Have relevant parameters (e.g., if user wants temperature, keep datasets with temperature params)
- Are primary data (not meta-studies or reference lists)
- **PRIORITIZE directly downloadable formats (tab-delimited text, CSV, NetCDF) over ZIP archives.**

Remove datasets that:
- Are unrelated to the request
- Are duplicates or very similar to better ones
- Are meta-studies without actual data
- **ZIP archives/compressed files (unless ABSOLUTELY NO other data exists). PENALIZE ZIPs HEAVILY.**

JSON array of indices to keep (sorted by relevance):"""

    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=scoring_prompt)])

        # Parse the response to get indices
        import json
        import re

        # Extract JSON array from response
        content = response.content.strip()
        # Find array pattern
        match = re.search(r'\[[\d,\s]+\]', content)
        if match:
            indices_to_keep = json.loads(match.group())
        else:
            # Fallback: try to parse the whole response
            indices_to_keep = json.loads(content)

        # Validate indices
        valid_indices = [i for i in indices_to_keep if 0 <= i < len(df)][:max_keep]

        if not valid_indices:
            logging.warning("LLM returned no valid indices, keeping first N datasets")
            valid_indices = list(range(min(max_keep, len(df))))

        filtered_df['Number'] = filtered_df.index + 1

        # Update session state
        SessionManager.update_session(session_id, "datasets_info", filtered_df)
        session_data = SessionManager.get_session(session_id)

        removed_count = total_before - len(filtered_df)

        # Update UI message
        session_data.setdefault("messages_search", []).append({
            "role": "assistant",
            "content": f"**Filtered & Sorted:** Kept {len(filtered_df)} most relevant datasets (removed {removed_count}).",
            "table": filtered_df.to_json(orient="split")
        })

        logging.info(f"Filtering complete: {total_before} -> {len(filtered_df)} datasets")

        return f"Filtered and SORTED workspace from {total_before} to {len(filtered_df)} datasets. Removed {removed_count} less relevant datasets."

    except Exception as e:
        logging.error(f"Error during filtering: {e}")
        # Fallback: just keep first N
        filtered_df = df.head(max_keep).reset_index(drop=True)
        filtered_df['Number'] = filtered_df.index + 1
        SessionManager.update_session(session_id, "datasets_info", filtered_df)
        return f"Fallback filter: Kept first {max_keep} datasets (error in smart filtering: {e})"

filter_workspace_tool = StructuredTool.from_function(
    func=filter_workspace_datasets,
    name="filter_workspace",
    description="Final cleanup: auto-selects best datasets, removes ZIPs, ranks by relevance. ALWAYS call this after your 1-3 initial searches to finalize the workspace.",
    args_schema=FilterWorkspaceArgs
)

def create_search_agent(datasets_info=None, search_mode="simple", session_id="default"):
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
        system_prompt = """You are a SMART and EFFICIENT dataset search assistant for PANGAEA.

**YOUR MISSION:** Find the most relevant datasets for the user using 1 to 3 targeted searches, then filter the results.

**WORKFLOW - FOLLOW EXACTLY:**
1. Execute 1 to 3 broad, well-crafted searches using `search_pg_datasets`.
   - Use `count=30`.
   - Try to capture the core topic and region.
2. (Optional) If you need more datasets or a slightly different cut, do 1 or 2 more searches (maximum 5 searches total).
3. ✅ ONCE YOU HAVE ENOUGH DATASETS (typically after 1-3 searches), CALL: `filter_workspace(user_query="...", max_keep=30)` to automatically select and rank the best ones.
4. ✅ STOP - provide a brief text summary of what you found.

**BOOLEAN QUERY RULES:**
- ALWAYS use `count=30`
- AND combines terms: `topic AND region`
- OR expands options: `(term1 OR term2)`
- Quotes for exact phrasing: `"Fram Strait"`

**TERMINATION RULES:**
- ❌ DO NOT run more than 5 searches. It is a waste of time and API quota. 
- ✅ ALWAYS Call `filter_workspace` to finalize the list after searching.
- ✅ After `filter_workspace`, give a brief text summary and STOP.
"""
    else:
        # DEEP MODE: COMPREHENSIVE PARALLEL SEARCH
        system_prompt = """You are a COMPREHENSIVE Research Data Scout for PANGAEA.
Your goal is to find THE BEST datasets through EXHAUSTIVE parallel searching.

🔥 **CRITICAL: MULTIPLE ROUNDS OF PARALLEL SEARCHES** 🔥
Execute 3-4 ROUNDS of parallel searches (5 queries each) = 15-20 total searches!

**SEARCH STRATEGY - MULTIPLE ROUNDS:**

📋 **ROUND 1: Core Topic (5 queries)**
```
parallel_search_pangaea(
  search_queries=[
    "main_term AND region",
    "(synonym1 OR synonym2) AND region",
    "scientific_name AND region",
    "instrument AND parameter AND region",
    "method AND topic AND region"
  ],
  count_per_query=30
)
```

📋 **ROUND 2: Expanded Coverage (5 queries)**
```
parallel_search_pangaea(
  search_queries=[
    "related_parameter AND region",
    "nearby_region AND topic",
    "expedition_name AND parameter",
    "(data_type OR measurement) AND topic",
    "institution AND region AND parameter"
  ],
  count_per_query=30
)
```

📋 **ROUND 3: Deep Dive (5 queries)**
```
parallel_search_pangaea(
  search_queries=[
    "specific_depth AND parameter",
    "platform_type AND region",
    "time_series AND topic",
    "cruise_id AND parameter",
    "processed_data AND topic AND region"
  ],
  count_per_query=30
)
```

**AFTER EACH ROUND:**
- Review `all_datasets` metadata
- Select relevant DOIs for consolidation
- Run `consolidate_search_results` with curated DOI list

**WORKFLOW:**
1. Execute ROUND 1 parallel search → consolidate best DOIs
2. Execute ROUND 2 parallel search → consolidate best DOIs
3. Execute ROUND 3 parallel search → consolidate best DOIs
4. **MANDATORY**: Run `filter_workspace(user_query="...", max_keep=30)`
5. STOP and provide summary

**BOOLEAN QUERY RULES:**
- ALWAYS use `count_per_query=30`
- Use AND/OR/NOT and parentheses
- Use quotes for exact phrases
- Be creative with synonyms!

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
- **OVERRIDE THE RANKINGS**: A dataset with score 30 might be PERFECT while one with score 80 might be GARBAGE
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
   - ❌ **HARD EXCLUDE**: ZIP archives (unless completely unavoidable). We cannot easily process them. Look for 'tab-delimited', 'csv', 'text', or 'NetCDF' (.nc) versions instead.

**EXAMPLE - Gelatinous Zooplankton in Fram Strait:**
- ✅ "Gelatinous zooplankton annotations Fram Strait" → INCLUDE (perfect match, ignore score)
- ✅ "Mesozooplankton abundance Fram Strait" → INCLUDE (contains gelatinous, ignore score)
- ❌ "Aliphatic lipids in sediments" → EXCLUDE (wrong domain, even if score 90)
- ❌ "Reference list of IPY supplements" → EXCLUDE (not real data, even if score 100)

**CONSOLIDATION WORKFLOW - FOLLOW EXACTLY (Shopping Cart):**
1. Execute parallel_search_pangaea with your balanced Boolean keyword queries.
2. Get results and examine `all_datasets` for CONTENT (ignore scores)
3. Build your own DOI list of ACTUALLY RELEVANT datasets
4. Pass ONLY this curated DOI list to consolidate_search_results. Preferably with at least 10-15 DOIs.
5. **SHOPPING CART**: New datasets are ADDED to the workspace, not replacing old ones!
6. **SMART FILTER**: If workspace has >15-20 datasets, call `filter_workspace` to keep only the best!
7. IMPORTANT: After work complete, write ONLY a brief 3-4 sentence summary. DO NOT list DOIs - creates double display!

**SMART FILTERING (USE WHEN WORKSPACE HAS MANY DATASETS):**
- Call `filter_workspace(user_query="original request", max_keep=30)` at the END
- The filter uses AI to analyze Name, Parameters, Description
- Keeps most relevant, removes duplicates and irrelevant results
- ALWAYS use to finalize results!

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
    def search_pg_datasets_tool_enhanced(query: str, count: int = 30, 
                                        mindate: Optional[str] = None, maxdate: Optional[str] = None,
                                        minlat: Optional[float] = None, maxlat: Optional[float] = None,
                                        minlon: Optional[float] = None, maxlon: Optional[float] = None,
                                        config: RunnableConfig = None):
        """Enhanced search tool that returns structured results for consolidation"""
        
        # Log search attempt
        logging.info(f"Executing search: query='{query}', count={count}, spatial=[{minlat},{maxlat},{minlon},{maxlon}], temporal=[{mindate},{maxdate}]")
        
        # Use session_id from outer scope explicitly
        session_data = SessionManager.get_session(session_id)

        # Call original search function with count parameter
        datasets_info = pg_search_default(query, count=count, mindate=mindate, maxdate=maxdate,
                                        minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon)
        
        # Store the FULL datasets in session state with a unique key
        search_id = f"search_{query}_{count}_{mindate}_{maxdate}_{minlat}_{maxlat}_{minlon}_{maxlon}".replace(" ", "_")
        if "search_results_cache" not in session_data:
            session_data["search_results_cache"] = {}
        session_data["search_results_cache"][search_id] = datasets_info
        
        # For simple mode, use Shopping Cart accumulation logic
        # --- WORKSPACE ACCUMULATION (Shopping Cart) ---
        # Get existing DOIs to avoid duplicates
        existing_dois = []
        if "datasets_info" in session_data and session_data["datasets_info"] is not None:
            if isinstance(session_data["datasets_info"], pd.DataFrame) and not session_data["datasets_info"].empty:
                existing_dois = session_data["datasets_info"]['DOI'].tolist()

        if not datasets_info.empty:
            # Filter out duplicates from new results
            new_datasets = datasets_info[~datasets_info['DOI'].isin(existing_dois)]
            num_new = len(new_datasets)

            if existing_dois and num_new > 0:
                # Merge with existing
                combined = pd.concat([session_data["datasets_info"], new_datasets], ignore_index=True)
                combined['Number'] = combined.index + 1
                SessionManager.update_session(session_id, "datasets_info", combined)
                session_data = SessionManager.get_session(session_id)
                msg = f"**Search completed:** Added {num_new} new datasets. Total in workspace: {len(combined)}."
            elif existing_dois and num_new == 0:
                msg = f"**Search completed:** Found {len(datasets_info)} datasets (all already in workspace)."
            else:
                datasets_info['Number'] = datasets_info.index + 1
                SessionManager.update_session(session_id, "datasets_info", datasets_info)
                session_data = SessionManager.get_session(session_id)
                msg = f"**Search completed:** Found {len(datasets_info)} datasets."

            session_data.setdefault("messages_search", []).append({
                "role": "assistant",
                "content": msg,
                "table": session_data["datasets_info"].to_json(orient="split")
            })
        # ----------------------------------------------------
        
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
        description="Search PANGAEA with Boolean queries. ALWAYS count=30. Keep to 1-3 well-crafted searches total, then use filter_workspace.",
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
        # Simple mode: search + filter tools
        tools = [search_tool, filter_workspace_tool, publication_qa_tool, direct_doi_access_tool]
    else:
        # Deep mode: include all tools including consolidation, parallel search, and filter
        tools = [search_tool, parallel_search_tool, consolidate_tool, filter_workspace_tool, publication_qa_tool, direct_doi_access_tool]

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)

    # Efficient iterations - don't waste time
    max_iterations = 15 if search_mode == "simple" else 20
    
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

def process_search_query(user_input: str, search_agent: AgentExecutor, session_data: dict):
    """
    Processes a user search query using the search agent.
    Direct invocation like data page agents - NO RunnableWithMessageHistory!
    """
    logging.info(f"Starting search process for query: {user_input}")

    # Build chat history from session messages (last 10 messages for context)
    chat_history: List[HumanMessage | AIMessage] = []
    recent_messages = session_data.get("messages_search", [])[-10:]

    for message in recent_messages:
        if message["role"] == "user":
            chat_history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            # Skip table messages, only keep text content
            content = message.get("content", "")
            if content and not message.get("table"):
                chat_history.append(AIMessage(content=content))

    # Direct agent invocation - like data page agents!
    try:
        response = search_agent.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
    except Exception as e:
        logging.error(f"Search agent error: {e}")
        return f"Search error: {str(e)}"

    # Extract response and intermediate steps
    ai_message = response.get("output", "Search completed.")
    intermediate_steps = response.get("intermediate_steps", [])

    # Log search statistics and extract queries
    search_count = 0
    filter_count = 0
    executed_queries = []

    for step in intermediate_steps:
        if hasattr(step[0], 'tool'):
            tool_name = step[0].tool
            tool_input = step[0].tool_input
            
            if tool_name == "search_pg_datasets":
                search_count += 1
                query = tool_input.get('query', 'Unknown')
                executed_queries.append(f"'{query}'")
            elif tool_name == "parallel_search_pangaea":
                search_count += 1
                queries = tool_input.get('search_queries', [])
                if isinstance(queries, list):
                    executed_queries.extend([f"'{q}'" for q in queries])
                else:
                    executed_queries.append(f"'{str(queries)}'")
            elif tool_name == "filter_workspace":
                filter_count += 1

    logging.info(f"Search completed: {search_count} searches, {filter_count} filters")

    # Check workspace status
    workspace_count = 0
    if "datasets_info" in session_data and session_data["datasets_info"] is not None:
        if isinstance(session_data["datasets_info"], pd.DataFrame):
            workspace_count = len(session_data["datasets_info"])

    # Append executed queries to the message
    if executed_queries:
        queries_str = ", ".join(executed_queries)
        ai_message += f"\n\n**Executed Searches:** {queries_str}"

    # Add summary to response if we have datasets
    if workspace_count > 0:
        summary = f"\n\n**Workspace:** {workspace_count} datasets accumulated."
        if search_count > 1:
            summary = f"\n\n**Completed {search_count} searches.** Workspace: {workspace_count} datasets."
        if filter_count > 0:
            summary += f" (filtered)"

        # Don't duplicate if already in message
        if "Workspace" not in ai_message and "datasets" not in ai_message.lower():
            ai_message += summary

    return ai_message