# api/routes/sessions.py
"""
REST endpoints for session lifecycle management.
"""

import logging
import os
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.utils.session_manager import SessionManager
from src.tools.python_repl import REPLManager
from main import initialize_session_state, ensure_thread_id, ensure_memory

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
AVAILABLE_MODELS = [
    "gpt-5.2", "gpt-5", "gpt-4.1-mini", "gpt-4.1", "gpt-4.1-nano",
    "gpt-4o", "o3-mini", "o3", "o4-mini", "codex-mini-latest",
]


class SessionResponse(BaseModel):
    session_id: str
    thread_id: str


class SessionInfo(BaseModel):
    session_id: str
    thread_id: str
    search_message_count: int
    agent_message_count: int
    active_datasets: list
    has_datasets: bool
    model_name: Optional[str] = None


class ModelSetRequest(BaseModel):
    model_name: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/sessions", response_model=SessionResponse)
async def create_session():
    """
    Create a new session with a unique ID.
    Returns session_id and thread_id for subsequent requests.
    """
    session_id = str(uuid.uuid4())
    session_data = SessionManager.get_session(session_id)

    # Initialize with standard defaults from main.py
    initialize_session_state(session_data)
    ensure_thread_id(session_data)
    ensure_memory(session_data)

    # Set default model from env or fallback
    session_data["model_name"] = os.environ.get("PANGAEA_MODEL_NAME", "gpt-5.2")

    logging.info(f"Created new session: {session_id} (thread_id: {session_data['thread_id']})")

    return SessionResponse(
        session_id=session_id,
        thread_id=session_data["thread_id"],
    )


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """
    Get metadata about an existing session.
    """
    session_data = SessionManager.get_session(session_id)

    if "thread_id" not in session_data:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or not initialized.")

    datasets_info = session_data.get("datasets_info")
    has_datasets = datasets_info is not None and (
        hasattr(datasets_info, "__len__") and len(datasets_info) > 0
    )

    return SessionInfo(
        session_id=session_id,
        thread_id=session_data["thread_id"],
        search_message_count=len(session_data.get("messages_search", [])),
        agent_message_count=len(session_data.get("messages_data_agent", [])),
        active_datasets=session_data.get("active_datasets", []),
        has_datasets=has_datasets,
        model_name=session_data.get("model_name"),
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Cleanup a session: destroy Jupyter kernel, remove from SessionManager.
    """
    session_data = SessionManager.get_session(session_id)

    if "thread_id" not in session_data:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    # Cleanup the Jupyter kernel if one was started
    try:
        REPLManager.cleanup_repl(session_id)
    except Exception as e:
        logging.warning(f"Error cleaning up REPL for session {session_id}: {e}")

    # Remove session from the manager
    with SessionManager._lock:
        SessionManager._sessions.pop(session_id, None)

    logging.info(f"Deleted session: {session_id}")
    return {"status": "deleted", "session_id": session_id}


# ---------------------------------------------------------------------------
# Model Selection
# ---------------------------------------------------------------------------
@router.get("/models")
async def list_models():
    """Return the list of available LLM models."""
    default = os.environ.get("PANGAEA_MODEL_NAME", "gpt-5.2")
    return {"models": AVAILABLE_MODELS, "default": default}


@router.put("/sessions/{session_id}/model")
async def set_model(session_id: str, body: ModelSetRequest):
    """Set the LLM model for a specific session."""
    if body.model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {body.model_name}")

    session_data = SessionManager.get_session(session_id)
    if "thread_id" not in session_data:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    session_data["model_name"] = body.model_name
    os.environ["PANGAEA_MODEL_NAME"] = body.model_name
    logging.info(f"Session {session_id}: model set to {body.model_name}")
    return {"model_name": body.model_name}
