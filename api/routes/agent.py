# api/routes/agent.py
"""
WebSocket endpoint for real-time agent interaction.
Streams status updates, tool calls, and final responses to the client.
"""

import asyncio
import json
import logging
import traceback
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.utils.session_manager import SessionManager
from src.agents.supervisor_agent import create_and_invoke_supervisor_agent
from main import (
    get_datasets_info_for_active_datasets,
    add_user_message_to_data_agent,
    add_assistant_message_to_data_agent,
    ensure_memory,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def send_json(ws: WebSocket, msg_type: str, data: Any):
    """Send a typed JSON message over the WebSocket."""
    await ws.send_json({"type": msg_type, "data": data})


def _make_artifact_url(plot_path: str, session_id: str) -> str:
    """
    Convert an absolute plot path to a relative /artifacts/ URL.
    E.g. /abs/path/tmp/sandbox/<thread_id>/results/fig_xxx.png
      -> /artifacts/<thread_id>/results/fig_xxx.png
    """
    import os
    # Find the sandbox-relative portion
    marker = os.path.join("tmp", "sandbox")
    idx = plot_path.find(marker)
    if idx != -1:
        relative = plot_path[idx + len(marker):]
        # Normalise separators
        relative = relative.replace(os.sep, "/")
        if not relative.startswith("/"):
            relative = "/" + relative
        return f"/artifacts{relative}"
    # Fallback: just return the basename
    return f"/artifacts/{os.path.basename(plot_path)}"


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------
@router.websocket("/sessions/{session_id}/agent/ws")
async def agent_websocket(ws: WebSocket, session_id: str):
    """
    Bidirectional WebSocket for agent interaction.

    Client sends:
        {"query": "Plot temperature vs salinity"}

    Server streams:
        {"type": "status",    "data": {"message": "Processing..."}}
        {"type": "tool_call", "data": {"agent": "OceanographerAgent", "tool": "Python_REPL", ...}}
        {"type": "response",  "data": {"content": "...", "plot_urls": ["/artifacts/..."], ...}}
        {"type": "error",     "data": {"message": "..."}}
    """
    await ws.accept()
    logging.info(f"WebSocket connected: session {session_id}")

    session_data = SessionManager.get_session(session_id)
    if "thread_id" not in session_data:
        await send_json(ws, "error", {"message": "Session not initialized. Call POST /api/sessions first."})
        await ws.close()
        return

    try:
        while True:
            # Wait for the client to send a query
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await send_json(ws, "error", {"message": "Invalid JSON payload."})
                continue

            query = payload.get("query", "").strip()
            if not query:
                await send_json(ws, "error", {"message": "Empty query."})
                continue

            # Acknowledge receipt
            await send_json(ws, "status", {"message": "Processing query...", "query": query})

            # --- Prepare agent invocation ---
            ensure_memory(session_data)
            memory = session_data["memory"]

            # Get active datasets info
            active_datasets_info = []
            if session_data.get("active_datasets"):
                try:
                    active_datasets_info = get_datasets_info_for_active_datasets(session_data)
                except Exception as e:
                    logging.error(f"Error getting datasets info: {e}")

            if not active_datasets_info:
                await send_json(ws, "error", {
                    "message": "No datasets loaded. Please search and select datasets first."
                })
                continue

            # Add user message to data agent history
            add_user_message_to_data_agent(query, session_data)

            await send_json(ws, "status", {
                "message": f"Invoking supervisor agent with {len(active_datasets_info)} datasets..."
            })

            # --- Run the agent in a thread pool (blocking call) ---
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: create_and_invoke_supervisor_agent(
                        user_query=query,
                        datasets_info=active_datasets_info,
                        memory=memory,
                        session_data=session_data,
                        st_callback=None,
                    ),
                )
            except Exception as e:
                logging.error(f"Agent invocation error: {e}", exc_info=True)
                await send_json(ws, "error", {
                    "message": f"Agent error: {str(e)}",
                    "traceback": traceback.format_exc(),
                })
                continue

            if response is None:
                await send_json(ws, "error", {"message": "Agent returned no response."})
                continue

            # --- Extract results ---
            # Get the final message content
            messages = response.get("messages", [])
            final_content = ""
            agent_usage_flags = {}

            if messages:
                last_msg = messages[-1]
                final_content = getattr(last_msg, "content", str(last_msg))

                # Extract agent usage flags from additional_kwargs
                kwargs = getattr(last_msg, "additional_kwargs", {})
                agent_usage_flags = {
                    "oceanographer_used": kwargs.get("oceanographer_used", False),
                    "ecologist_used": kwargs.get("ecologist_used", False),
                    "visualization_used": kwargs.get("visualization_used", False),
                    "dataframe_used": kwargs.get("dataframe_used", False),
                    "specialized_agent_used": any([
                        kwargs.get("oceanographer_used", False),
                        kwargs.get("ecologist_used", False),
                        kwargs.get("visualization_used", False),
                    ]),
                }

            # Collect plot images and convert to URLs
            plot_images = response.get("plot_images", [])
            plot_urls = [_make_artifact_url(p, session_id) for p in plot_images if p]

            # Save to session history
            add_assistant_message_to_data_agent(
                content=final_content,
                plot_images=plot_images,
                agent_usage_flags=agent_usage_flags,
                session_data=session_data,
            )

            # Persist chat_summary if present
            if response.get("chat_summary"):
                session_data["chat_summary"] = response["chat_summary"]

            # --- Send final response ---
            await send_json(ws, "response", {
                "content": final_content,
                "plot_urls": plot_urls,
                "agent_usage": agent_usage_flags,
            })

    except WebSocketDisconnect:
        logging.info(f"WebSocket disconnected: session {session_id}")
    except Exception as e:
        logging.error(f"WebSocket fatal error for session {session_id}: {e}", exc_info=True)
        try:
            await send_json(ws, "error", {"message": f"Fatal error: {str(e)}"})
        except Exception:
            pass
        await ws.close()
