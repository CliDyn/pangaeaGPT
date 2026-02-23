import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict, List, Optional, Union
import logging

class SafeStreamlitCallbackHandler(BaseCallbackHandler):
    """
    A robust, safe implementation of a Streamlit callback handler.
    Designed to prevent RecursionErrors and provide clean UI feedback.
    Mimics the interface of langchain_community.callbacks.StreamlitCallbackHandler.
    """
    def __init__(self, parent_container: st.delta_generator.DeltaGenerator, 
                 max_thought_containers: int = 4, 
                 expand_new_thoughts: bool = True, 
                 collapse_completed_thoughts: bool = False):
        self.parent_container = parent_container
        self.max_thoughts = max_thought_containers
        self.expand_thoughts = expand_new_thoughts
        self.collapse_completed = collapse_completed_thoughts
        
        self.current_thought_container = None
        self.current_text_placeholder = None
        self.current_text = ""
        self.tokens_buffer = []
        self.step_counter = 0

    def _get_container(self):
        """Get or create the container for the current step."""
        if self.current_thought_container is None:
            self.step_counter += 1
            label = f"Step {self.step_counter}"
            with self.parent_container:
                # Use status for better grouping if available, else expander
                try:
                    self.current_thought_container = st.status(label, expanded=self.expand_thoughts)
                except AttributeError:
                    self.current_thought_container = st.expander(label, expanded=self.expand_thoughts)
        return self.current_thought_container

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when LLM starts running."""
        self.current_text = ""
        self.tokens_buffer = []
        
        container = self._get_container()
        with container:
            st.markdown("**Thinking...**")
            self.current_text_placeholder = st.empty()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token."""
        # Buffer tokens to avoid excessive Streamlit updates (which can cause lag/recursion issues)
        self.tokens_buffer.append(token)
        if len(self.tokens_buffer) >= 5:  # Update every 5 tokens
            chunk = "".join(self.tokens_buffer)
            self.current_text += chunk
            self.tokens_buffer = []
            
            if self.current_text_placeholder:
                try:
                    self.current_text_placeholder.markdown(self.current_text + "▌")
                except Exception:
                    pass # Ignore update errors to prevent crashes

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        # Flush buffer
        if self.tokens_buffer:
            self.current_text += "".join(self.tokens_buffer)
            self.tokens_buffer = []
            
        # Check for tool calls if text is empty (common with function-calling models)
        tool_calls_found = []
        try:
            # response is LLMResult
            if hasattr(response, 'generations') and response.generations:
                # usually generations[0] is the list of generations for the first prompt
                gen = response.generations[0][0]
                if hasattr(gen, 'message'):
                    msg = gen.message
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        tool_calls_found = [tc.get('name') for tc in msg.tool_calls]
                    elif hasattr(msg, 'additional_kwargs'):
                        fc = msg.additional_kwargs.get('function_call')
                        if fc:
                            tool_calls_found = [fc.get('name')]
        except Exception:
            pass
            
        if not self.current_text.strip() and tool_calls_found:
            self.current_text = f"**Decided to use tool(s):** `{', '.join(tool_calls_found)}`"
            
        if self.current_text_placeholder:
            try:
                self.current_text_placeholder.markdown(self.current_text)
            except Exception:
                pass
        
        # Reset for next potential LLM call in same chain (though usually one per step)
        # We don't close the container yet, tools might run.

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Run when tool starts running."""
        container = self._get_container()
        tool_name = serialized.get('name', 'Unknown Tool')
        
        with container:
            st.markdown(f"**🔧 Executing Tool:** `{tool_name}`")
            with st.expander("Input", expanded=False):
                st.code(input_str)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        container = self._get_container()
        with container:
            with st.expander("Output", expanded=False):
                # Truncate very long outputs to avoid UI performance issues
                if len(str(output)) > 2000:
                    display_output = str(output)[:2000] + "... [truncated]"
                else:
                    display_output = str(output)
                st.code(display_output)
            st.markdown("---")

    def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        """Run on agent action."""
        # Agent action often precedes tool execution
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends."""
        # If using st.status, update state
        if self.current_thought_container and hasattr(self.current_thought_container, 'update'):
            try:
                self.current_thought_container.update(state="complete", expanded=False if self.collapse_completed else True)
            except Exception:
                pass
        self.current_thought_container = None # Reset for next chain/agent