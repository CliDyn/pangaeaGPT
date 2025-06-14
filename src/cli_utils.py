"""
src/cli_utils.py - Utilities for CLI mode including Streamlit mocking
"""

import sys
from typing import Any, Optional, Dict, List
import logging


class MockStreamlit:
    """Mock Streamlit module for CLI mode"""
    
    def __init__(self):
        self.session_state = None
        self.secrets = MockSecrets()
    
    def set_page_config(self, **kwargs):
        """Mock set_page_config - does nothing in CLI mode"""
        pass
    
    def title(self, text):
        """Mock title - prints to console"""
        print(f"\n=== {text} ===\n")
    
    def markdown(self, text, unsafe_allow_html=False):
        """Mock markdown - prints to console"""
        # Remove HTML tags if present
        if unsafe_allow_html:
            import re
            text = re.sub('<[^<]+?>', '', text)
        print(text)
    
    def write(self, *args, **kwargs):
        """Mock write - prints to console"""
        for arg in args:
            print(arg)
    
    def info(self, text):
        """Mock info - prints with info prefix"""
        print(f"ℹ️ INFO: {text}")
    
    def warning(self, text):
        """Mock warning - prints with warning prefix"""
        print(f"⚠️ WARNING: {text}")
    
    def error(self, text):
        """Mock error - prints with error prefix"""
        print(f"❌ ERROR: {text}")
    
    def success(self, text):
        """Mock success - prints with success prefix"""
        print(f"✅ SUCCESS: {text}")
    
    def button(self, label, **kwargs):
        """Mock button - always returns False in CLI mode"""
        return False
    
    def checkbox(self, label, value=False, **kwargs):
        """Mock checkbox - returns the default value"""
        return value
    
    def selectbox(self, label, options, index=0, **kwargs):
        """Mock selectbox - returns the default option"""
        if options and 0 <= index < len(options):
            return options[index]
        return None
    
    def text_input(self, label, value="", **kwargs):
        """Mock text_input - returns the default value"""
        return value
    
    def text_area(self, label, value="", **kwargs):
        """Mock text_area - returns the default value"""
        return value
    
    def columns(self, spec):
        """Mock columns - returns mock column objects"""
        if isinstance(spec, int):
            return [MockColumn() for _ in range(spec)]
        else:
            return [MockColumn() for _ in spec]
    
    def container(self):
        """Mock container"""
        return MockContainer()
    
    def empty(self):
        """Mock empty"""
        return MockEmpty()
    
    def expander(self, label, expanded=False):
        """Mock expander"""
        return MockExpander(label, expanded)
    
    def progress(self, value):
        """Mock progress bar"""
        return MockProgress(value)
    
    def spinner(self, text):
        """Mock spinner"""
        return MockSpinner(text)
    
    def sidebar(self):
        """Mock sidebar"""
        return self  # Return self as sidebar has same methods
    
    def dataframe(self, data, **kwargs):
        """Mock dataframe display"""
        print(data)
    
    def download_button(self, label, data, file_name, mime, **kwargs):
        """Mock download button - saves file in CLI mode"""
        try:
            with open(file_name, 'wb' if isinstance(data, bytes) else 'w') as f:
                f.write(data)
            print(f"💾 File saved: {file_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to save file: {e}")
            return False
    
    def form(self, key, clear_on_submit=False):
        """Mock form"""
        return MockForm(key)
    
    def form_submit_button(self, label='Submit', **kwargs):
        """Mock form submit button"""
        return False
    
    def image(self, image, caption=None, **kwargs):
        """Mock image display"""
        if caption:
            print(f"🖼️ Image: {caption}")
        else:
            print(f"🖼️ Image displayed")
    
    def chat_message(self, role, avatar=None):
        """Mock chat message"""
        return MockChatMessage(role, avatar)
    
    def chat_input(self, placeholder=""):
        """Mock chat input - returns None in CLI mode"""
        return None
    
    def radio(self, label, options, index=0, **kwargs):
        """Mock radio - returns default option"""
        if options and 0 <= index < len(options):
            return options[index]
        return None
    
    def tabs(self, tabs):
        """Mock tabs"""
        return [MockTab(name) for name in tabs]
    
    def caption(self, text):
        """Mock caption"""
        print(f"   {text}")
    
    def divider(self):
        """Mock divider"""
        print("-" * 50)
    
    def rerun(self):
        """Mock rerun - does nothing in CLI mode"""
        pass
    
    def stop(self):
        """Mock stop - raises exception to stop execution"""
        raise StopExecution("Streamlit stop called")


class MockSecrets:
    """Mock streamlit.secrets"""
    def __init__(self):
        self.data = {
            "general": {}
        }
    
    def __getitem__(self, key):
        return self.data.get(key, {})


class MockColumn:
    """Mock column object"""
    def __getattr__(self, name):
        # Return the mock streamlit instance for any method call
        return lambda *args, **kwargs: None
    
    def write(self, *args):
        print(*args)
    
    def button(self, label, **kwargs):
        return False


class MockContainer:
    """Mock container object"""
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class MockEmpty:
    """Mock empty object"""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
    
    def empty(self):
        pass
    
    def write(self, text):
        print(text)
    
    def info(self, text):
        print(f"ℹ️ {text}")


class MockExpander:
    """Mock expander object"""
    def __init__(self, label, expanded):
        self.label = label
        self.expanded = expanded
    
    def __enter__(self):
        if self.expanded:
            print(f"\n▼ {self.label}")
        else:
            print(f"\n▶ {self.label} (collapsed)")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def write(self, text):
        print(f"  {text}")


class MockProgress:
    """Mock progress bar"""
    def __init__(self, value):
        self.value = value
        percentage = int(value * 100)
        print(f"Progress: [{'=' * (percentage // 10)}{' ' * (10 - percentage // 10)}] {percentage}%")
    
    def progress(self, value):
        self.value = value
        percentage = int(value * 100)
        print(f"Progress: [{'=' * (percentage // 10)}{' ' * (10 - percentage // 10)}] {percentage}%")
    
    def empty(self):
        pass


class MockSpinner:
    """Mock spinner context manager"""
    def __init__(self, text):
        self.text = text
    
    def __enter__(self):
        print(f"⏳ {self.text}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            print("✓ Done")


class MockForm:
    """Mock form context manager"""
    def __init__(self, key):
        self.key = key
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockChatMessage:
    """Mock chat message context manager"""
    def __init__(self, role, avatar):
        self.role = role
        self.avatar = avatar
    
    def __enter__(self):
        prefix = "👤" if self.role == "user" else "🤖"
        print(f"\n{prefix} {self.role.upper()}:")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def markdown(self, text):
        print(f"  {text}")
    
    def write(self, text):
        print(f"  {text}")


class MockTab:
    """Mock tab object"""
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        print(f"\n--- Tab: {self.name} ---")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def write(self, text):
        print(text)
    
    def button(self, label, **kwargs):
        return False


class StopExecution(Exception):
    """Exception to simulate st.stop()"""
    pass


def setup_cli_mode():
    """Set up CLI mode by injecting mock Streamlit module"""
    # Create mock streamlit module
    mock_st = MockStreamlit()
    
    # Create a proper mock session state
    from types import SimpleNamespace
    
    # We need to ensure session_state behaves like the real one
    # The real st.session_state allows both dict-like and attribute access
    class SessionStateMock(dict):
        def __getattr__(self, key):
            return self.get(key)
        
        def __setattr__(self, key, value):
            self[key] = value
        
        def __delattr__(self, key):
            if key in self:
                del self[key]
    
    # Initialize with empty session state that will be populated by CLIInterface
    mock_st.session_state = SessionStateMock()
    
    # Inject into sys.modules
    sys.modules['streamlit'] = mock_st
    
    # Also create common imports
    sys.modules['streamlit.components'] = type(sys)('streamlit.components')
    sys.modules['streamlit.components.v1'] = type(sys)('streamlit.components.v1')
    
    return mock_st


class CLICallbackHandler:
    """Callback handler for CLI mode to show agent progress"""
    
    def __init__(self):
        self.current_agent = None
        self.current_tool = None
    
    def on_agent_start(self, agent_name: str):
        """Called when an agent starts"""
        self.current_agent = agent_name
        print(f"\n🤖 {agent_name} is thinking...")
    
    def on_tool_start(self, tool_name: str, tool_input: Any):
        """Called when a tool starts"""
        self.current_tool = tool_name
        print(f"   🔧 Using tool: {tool_name}")
        if isinstance(tool_input, dict) and len(str(tool_input)) < 200:
            print(f"      Input: {tool_input}")
    
    def on_tool_end(self, output: Any):
        """Called when a tool ends"""
        if self.current_tool and len(str(output)) < 200:
            print(f"      Output: {output}")
    
    def on_agent_end(self, output: Any):
        """Called when an agent ends"""
        if self.current_agent:
            print(f"   ✓ {self.current_agent} completed")