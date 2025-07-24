# src/llm_factory.py
"""
Centralized factory for creating and configuring language model instances.
This allows for model-specific settings, like handling the 'temperature' parameter.
"""
import streamlit as st
from langchain_openai import ChatOpenAI
from .config import API_KEY

def get_llm(model_name: str = None, temperature: float = 0.7):
    """
    Factory function to get a configured ChatOpenAI instance.

    This centralizes LLM creation and handles model-specific configurations,
    such as removing the 'temperature' parameter for models that don't support it.

    Args:
        model_name (str, optional): The name of the model to use. If None, it will be
                                    retrieved from the Streamlit session state.
        temperature (float, optional): The temperature to use for supported models.
                                       Defaults to 0.7.

    Returns:
        ChatOpenAI: A configured instance of the language model.
    """
    if model_name is None:
        # This works for both Streamlit and CLI mode, as CLI mocks session_state
        model_name = st.session_state.get("model_name", "gpt-4.1-mini")

    # List of models that are known to NOT support the 'temperature' parameter.
    # To add a new model with this restriction, just add its name to this list.
    models_without_temperature = [
        'o1-mini',
        'o3-mini', 'o3',
        'o4-mini',
        'codex-mini-latest'
    ]

    # Check if the selected model is in the list of models that don't support temperature
    if model_name in models_without_temperature:
        # For these models, instantiate without the temperature parameter to avoid errors
        return ChatOpenAI(
            api_key=API_KEY,
            model_name=model_name
        )
    else:
        # For all other standard models, include the temperature parameter
        return ChatOpenAI(
            api_key=API_KEY,
            model_name=model_name,
            temperature=temperature
        )