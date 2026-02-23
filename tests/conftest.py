import pytest
import os
from unittest.mock import MagicMock

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock environment variables typically required by the app."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

@pytest.fixture
def mock_streamlit(monkeypatch):
    """Mock Streamlit to prevent tests from failing due to Streamlit context."""
    import streamlit as st
    
    mock_st = MagicMock()
    mock_st.session_state = {}
    
    # We mainly need to avoid real imports of st breaking things if not in a Streamlit run
    monkeypatch.setattr("streamlit.session_state", mock_st.session_state)
    monkeypatch.setattr("streamlit.write", mock_st.write)
    monkeypatch.setattr("streamlit.markdown", mock_st.markdown)
    monkeypatch.setattr("streamlit.error", mock_st.error)
    monkeypatch.setattr("streamlit.warning", mock_st.warning)
    monkeypatch.setattr("streamlit.info", mock_st.info)
    monkeypatch.setattr("streamlit.success", mock_st.success)
    monkeypatch.setattr("streamlit.sidebar", mock_st.sidebar)
    
    return mock_st
