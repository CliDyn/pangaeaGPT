# api/routes/search.py
"""
REST endpoints for the search workflow:
  - Execute search queries
  - Retrieve dataset tables
  - Select and load datasets into the sandbox
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.utils.session_manager import SessionManager
from main import (
    get_search_agent,
    process_search_query,
    add_user_message_to_search,
    add_assistant_message_to_search,
    load_selected_datasets_into_cache,
    set_active_datasets_from_selection,
    get_datasets_info_for_active_datasets,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str
    search_mode: Optional[str] = "simple"  # "simple" or "deep"


class SearchResponse(BaseModel):
    response: str
    datasets: Optional[list] = None
    dataset_count: int = 0


class DatasetSelectRequest(BaseModel):
    dois: List[str]


class DatasetInfo(BaseModel):
    doi: str
    name: Optional[str] = None
    description: Optional[str] = None
    data_type: Optional[str] = None
    files: Optional[list] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/sessions/{session_id}/search", response_model=SearchResponse)
async def execute_search(session_id: str, request: SearchRequest):
    """
    Execute a PANGAEA dataset search query.
    Returns the agent's response and optionally a dataset table.
    """
    session_data = SessionManager.get_session(session_id)

    if "thread_id" not in session_data:
        raise HTTPException(status_code=404, detail="Session not initialized. Create a session first.")

    # Set search mode
    session_data["search_mode"] = request.search_mode

    # Add user message to search history
    add_user_message_to_search(request.query, session_data)

    try:
        # Create and invoke the search agent
        search_agent = get_search_agent(
            datasets_info=session_data.get("datasets_info"),
            search_mode=request.search_mode,
            session_id=session_id
        )

        response_text = process_search_query(request.query, search_agent, session_data)
        add_assistant_message_to_search(response_text, session_data)

    except Exception as e:
        logging.error(f"Search error for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    # Re-fetch session data to ensure we have the latest updates from tools
    session_data = SessionManager.get_session(session_id)

    # Build dataset table if available
    datasets_list = None
    dataset_count = 0
    datasets_info = session_data.get("datasets_info")
    if datasets_info is not None and hasattr(datasets_info, "to_dict"):
        try:
            datasets_list = datasets_info.to_dict(orient="records")
            dataset_count = len(datasets_list)
        except Exception:
            datasets_list = None

    return SearchResponse(
        response=response_text,
        datasets=datasets_list,
        dataset_count=dataset_count,
    )


@router.get("/sessions/{session_id}/datasets")
async def get_datasets(session_id: str):
    """
    Get the current dataset search results table.
    """
    session_data = SessionManager.get_session(session_id)
    datasets_info = session_data.get("datasets_info")

    if datasets_info is None:
        return {"datasets": [], "count": 0}

    try:
        records = datasets_info.to_dict(orient="records")
    except Exception:
        records = []

    return {"datasets": records, "count": len(records)}


@router.post("/sessions/{session_id}/datasets/select")
async def select_datasets(session_id: str, request: DatasetSelectRequest):
    """
    Select DOIs from search results and load them into the sandbox.
    This fetches the actual data files and prepares them for agent analysis.
    """
    session_data = SessionManager.get_session(session_id)

    if "thread_id" not in session_data:
        raise HTTPException(status_code=404, detail="Session not initialized.")

    if not request.dois:
        raise HTTPException(status_code=400, detail="No DOIs provided.")

    # Update selected datasets
    session_data["selected_datasets"] = set(request.dois)

    try:
        # Load datasets into sandbox (this downloads files)
        load_selected_datasets_into_cache(request.dois, session_data)

        # Set as active
        set_active_datasets_from_selection(session_data)

        # Get info about the loaded datasets
        active_info = get_datasets_info_for_active_datasets(session_data)

        logging.info(f"Session {session_id}: Loaded {len(active_info)} datasets")

    except Exception as e:
        logging.error(f"Dataset loading error for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load datasets: {str(e)}")

    # Build response
    result = []
    for info in active_info:
        result.append(DatasetInfo(
            doi=info.get("doi", ""),
            name=info.get("name"),
            description=info.get("description"),
            data_type=info.get("data_type"),
            files=info.get("files"),
        ))

    return {"loaded": len(result), "datasets": [d.model_dump() for d in result]}


@router.get("/sessions/{session_id}/datasets/info")
async def get_active_datasets_info(session_id: str):
    """
    Get detailed info about the currently active (loaded) datasets.
    """
    session_data = SessionManager.get_session(session_id)

    if not session_data.get("active_datasets"):
        return {"datasets": [], "count": 0}

    try:
        active_info = get_datasets_info_for_active_datasets(session_data)
    except Exception as e:
        logging.error(f"Error getting dataset info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    result = []
    for info in active_info:
        result.append({
            "doi": info.get("doi", ""),
            "name": info.get("name"),
            "description": info.get("description"),
            "data_type": info.get("data_type"),
            "files": info.get("files"),
            "df_head": info.get("df_head"),
        })

    return {"datasets": result, "count": len(result)}


@router.get("/sessions/{session_id}/datasets/preview")
async def preview_dataset(session_id: str, doi: str):
    """
    Get a preview (head, columns, shape) for a cached dataset identified by DOI.
    """
    import pandas as pd
    import os

    session_data = SessionManager.get_session(session_id)
    cache = session_data.get("datasets_cache", {})

    if doi not in cache:
        raise HTTPException(status_code=404, detail=f"Dataset {doi} is not cached in this session.")

    dataset_path, name = cache[doi]

    # If it's a directory, try to find the main .tab or .csv file
    preview_data = {"doi": doi, "name": name}

    try:
        if isinstance(dataset_path, str) and os.path.isdir(dataset_path):
            # List files in the dataset directory
            files = os.listdir(os.path.join(dataset_path, "datasets")) if os.path.isdir(os.path.join(dataset_path, "datasets")) else os.listdir(dataset_path)
            preview_data["files"] = files

            # Try to load the first .tab or .csv file for preview
            target_dir = os.path.join(dataset_path, "datasets") if os.path.isdir(os.path.join(dataset_path, "datasets")) else dataset_path
            for f in sorted(files):
                if f.endswith((".tab", ".csv", ".tsv")):
                    fpath = os.path.join(target_dir, f)
                    try:
                        sep = "\t" if f.endswith((".tab", ".tsv")) else ","
                        # Try reading with comment skipping
                        df = pd.read_csv(fpath, sep=sep, comment="/*", nrows=20, on_bad_lines="skip")
                        if df.empty or len(df.columns) < 2:
                            # Try skipping header rows
                            for skip in range(5, 25):
                                df = pd.read_csv(fpath, sep=sep, skiprows=skip, nrows=10, on_bad_lines="skip")
                                if not df.empty and len(df.columns) >= 2:
                                    break
                        preview_data["preview_file"] = f
                        preview_data["columns"] = list(df.columns)
                        preview_data["shape"] = list(df.shape)
                        preview_data["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
                        preview_data["head"] = df.head(10).to_dict(orient="records")
                        break
                    except Exception as e:
                        logging.warning(f"Could not parse {f}: {e}")
                        continue
        elif hasattr(dataset_path, "head"):
            # It's a DataFrame directly
            df = dataset_path
            preview_data["columns"] = list(df.columns)
            preview_data["shape"] = list(df.shape)
            preview_data["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            preview_data["head"] = df.head(10).to_dict(orient="records")

    except Exception as e:
        logging.error(f"Preview error for {doi}: {e}", exc_info=True)
        preview_data["error"] = str(e)

    return preview_data
