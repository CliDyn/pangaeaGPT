# src/agents/pandas_agent.py
import logging
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

from ..prompts import Prompts
from ..llm_factory import get_llm  # Use the new factory

def create_pandas_agent(user_query, datasets_info):
    """
    Creates a pandas DataFrame agent for data analysis.
    
    Args:
        user_query (str): The user's query
        datasets_info (list): Information about available datasets
        
    Returns:
        AgentExecutor: The pandas agent executor
    """
    # Use the factory function to get the correct LLM instance.
    # For a pandas agent, a low temperature is best for predictable, factual responses.
    llm = get_llm(temperature=0.0)

    # Assign unique variable names to each dataframe and collect dataframes
    dataset_variables = []
    dataframes = []
    datasets_text = ""  # Initialize datasets_text
    
    for i, info in enumerate(datasets_info, 1):  # Start enumeration at 1
        var_name = f"df{i}"  # Consistently name as df1, df2, etc.
        dataframes.append(info['dataset'])  # Collect dataframes into a list
        dataset_variables.append(var_name)
        
        # Build datasets_text
        datasets_text += (
            f"Dataset {i}:\n"  # Adjust index to match variable naming
            f"Variable Name: {var_name}\n"
            f"Name: {info['name']}\n"
            f"Description: {info['description']}\n"
            f"Head of DataFrame (use it only as an example):\n"
            f"{info['df_head']}\n\n"
        )

    # Create a custom system prompt that includes information about each dataframe
    system_prompt = Prompts.generate_pandas_agent_system_prompt(user_query, datasets_text, dataset_variables)

    # Create a ChatPromptTemplate with the system prompt
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create the pandas dataframe agent with the list of dataframes and the chat prompt
    agent_pandas = create_pandas_dataframe_agent(
        llm=llm,
        df=dataframes,  # Pass the list of dataframes
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=5,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        suffix=system_prompt,
        allow_dangerous_code=True,
        chat_prompt=chat_prompt
    )

    return agent_pandas