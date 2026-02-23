# api/main.py
"""
FastAPI backend for PangaeaGPT.
Replaces the Streamlit UI layer with REST + WebSocket endpoints.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import src.config  # Initialize environment variables (LangSmith, APIs) before anything else!
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.tools.python_repl import REPLManager
from .routes import sessions, search, agent


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logging.info("🚀 PangaeaGPT API starting up")
    # Ensure sandbox root exists
    os.makedirs("tmp/sandbox", exist_ok=True)
    yield
    # Cleanup all Jupyter kernels on shutdown
    logging.info("🛑 PangaeaGPT API shutting down – cleaning up kernels")
    REPLManager.cleanup_all()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PangaeaGPT API",
    version="0.1.0",
    description="Backend API for the PangaeaGPT multi-agent scientific data assistant.",
    lifespan=lifespan,
)

# CORS – allow the React dev server and any localhost origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",   # Vite default
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static files – serve generated plots & artifacts
# ---------------------------------------------------------------------------
SANDBOX_ROOT = os.path.join(os.getcwd(), "tmp", "sandbox")
os.makedirs(SANDBOX_ROOT, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=SANDBOX_ROOT), name="artifacts")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
app.include_router(sessions.router, prefix="/api", tags=["Sessions"])
app.include_router(search.router, prefix="/api", tags=["Search"])
app.include_router(agent.router, prefix="/api", tags=["Agent"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "pangaeagpt-api"}
