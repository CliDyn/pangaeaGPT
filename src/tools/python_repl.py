# src/tools/python_repl.py

import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import streamlit as st
from io import StringIO
from pydantic import BaseModel, Field, PrivateAttr
from langchain_experimental.tools import PythonREPLTool
from typing import Any, Dict, Optional
import re
import threading
import subprocess
import json
import tempfile

from ..utils import log_history_event

# --- Start of new code for persistent REPL ---

class PersistentREPL:
    """
    Manages a persistent Python execution in SEPARATE subprocess.
    """
    def __init__(self, working_dir=None):
        self._working_dir = working_dir
        self._process = None
        self.is_initialized = False
        self._temp_script = None
        
        # Start separate Python process
        self._start_subprocess()
    
    def _start_subprocess(self):
        """Start a separate Python subprocess with the correct venv."""
        # Get the current Python executable (should be from venv if running in venv)
        python_executable = sys.executable
        
        # Create a script that will run in subprocess
        script_content = """
import sys
import json
import os
from io import StringIO

# Initialize globals/locals
exec_globals = {"__builtins__": __builtins__}
exec_locals = exec_globals

print("SUBPROCESS_READY", flush=True)

while True:
    try:
        # Read command as JSON
        line = input()
        if line == "EXIT_SUBPROCESS":
            break
        
        # Parse the command
        cmd = json.loads(line)
        
        if cmd["type"] == "exec":
            code = cmd["code"]
            
            # Capture stdout/stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            
            try:
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture
                
                # Execute code with persistent globals/locals
                exec(code, exec_globals, exec_locals)
                
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
                result = {
                    "status": "success",
                    "stdout": stdout_capture.getvalue(),
                    "stderr": stderr_capture.getvalue()
                }
            except Exception as e:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                result = {
                    "status": "error",
                    "error": f"Error: {repr(e)}\\n{type(e).__name__}: {str(e)}",
                    "stdout": stdout_capture.getvalue(),
                    "stderr": stderr_capture.getvalue()
                }
            
            print(json.dumps(result), flush=True)
            
        elif cmd["type"] == "chdir":
            os.chdir(cmd["path"])
            print(json.dumps({"status": "success", "cwd": os.getcwd()}), flush=True)
            
    except EOFError:
        break
    except Exception as e:
        print(json.dumps({"status": "fatal", "error": str(e)}), flush=True)
"""
        
        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            self._temp_script = f.name
        
        # Start subprocess
        self._process = subprocess.Popen(
            [python_executable, "-u", self._temp_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            env=os.environ.copy()
        )
        
        # Wait for ready signal
        ready_line = self._process.stdout.readline()
        if "SUBPROCESS_READY" not in ready_line:
            raise RuntimeError(f"Failed to start subprocess: {ready_line}")
        
        # Change directory if specified
        if self._working_dir and os.path.exists(self._working_dir):
            self._send_command({"type": "chdir", "path": self._working_dir})
            logging.info(f"Changed subprocess working directory to: {self._working_dir}")
        
        logging.info(f"Started separate Python subprocess (PID: {self._process.pid})")
    
    def _send_command(self, cmd: dict) -> dict:
        """Send command to subprocess and get response."""
        if not self._process or self._process.poll() is not None:
            self._start_subprocess()
        
        # Send command
        self._process.stdin.write(json.dumps(cmd) + "\n")
        self._process.stdin.flush()
        
        # Get response
        response_line = self._process.stdout.readline()
        if not response_line:
            return {"status": "error", "error": "No response from subprocess"}
        
        return json.loads(response_line)
    
    @staticmethod
    def sanitize_input(query: str) -> str:
        """Sanitize input to the python REPL (from LangChain)."""
        # Remove whitespace, backticks, and 'python' keyword
        query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
        query = re.sub(r"(\s|`)*$", "", query)
        return query
    
    def _sanitize_code(self, code: str) -> str:
        """Clean IPython/Jupyter specific syntax and apply LangChain sanitization."""
        # First apply LangChain's sanitization
        code = self.sanitize_input(code)
        
        # Handle IPython magic commands (!)
        lines = []
        for line in code.split('\n'):
            # Handle !{sys.executable} or similar
            if '!{' in line and '}' in line:
                # Extract the expression and convert to subprocess call
                import_stmt = "import subprocess, sys"
                if import_stmt not in lines:
                    lines.append(import_stmt)
                # Replace !{expr} with subprocess.run
                line = re.sub(
                    r'!\{([^}]+)\}(.*)', 
                    r'subprocess.run(str(\1) + r"\2", shell=True)', 
                    line
                )
                lines.append(line)
            # Handle simple !command
            elif line.strip().startswith('!'):
                import_stmt = "import subprocess"
                if import_stmt not in lines:
                    lines.append(import_stmt)
                cmd = line.strip()[1:]
                lines.append(f'subprocess.run("{cmd}", shell=True)')
            # Remove IPython magics like %matplotlib, %%time
            elif line.strip().startswith('%'):
                continue  # Skip IPython magic commands
            else:
                lines.append(line)
        
        return '\n'.join(lines)
    
    def run(self, code: str, timeout: int = 300) -> str:
        """Execute code in the separate subprocess."""
        # Clean the code
        cleaned_code = self._sanitize_code(code)
        
        # Send exec command
        result = self._send_command({"type": "exec", "code": cleaned_code})
        
        if result["status"] == "success":
            output = result["stdout"]
            if result.get("stderr"):
                if output:
                    output += "\n" + result["stderr"]
                else:
                    output = result["stderr"]
            return output
        elif result["status"] == "error":
            return result["error"]
        else:
            return f"Fatal error: {result.get('error', 'Unknown error')}"
    
    def close(self):
        """Cleanup and terminate subprocess."""
        if self._process:
            try:
                # Send exit command
                self._process.stdin.write("EXIT_SUBPROCESS\n")
                self._process.stdin.flush()
                self._process.wait(timeout=2)
            except:
                # Force terminate if needed
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except:
                    self._process.kill()
            
            logging.info(f"Terminated Python subprocess (PID: {self._process.pid})")
            self._process = None
        
        # Clean up temp script
        if self._temp_script and os.path.exists(self._temp_script):
            try:
                os.unlink(self._temp_script)
            except:
                pass

# REPLManager остается БЕЗ ИЗМЕНЕНИЙ
class REPLManager:
    """
    Singleton for managing PersistentREPL instances for each session (thread_id).
    """
    _instances: dict[str, PersistentREPL] = {}
    _lock = threading.Lock()

    @classmethod
    def get_repl(cls, session_id: str, sandbox_path: str = None) -> PersistentREPL:
        with cls._lock:
            if session_id not in cls._instances:
                logging.info(f"Creating new persistent REPL for session: {session_id}")
                if sandbox_path:
                    logging.info(f"Starting REPL in sandbox directory: {sandbox_path}")
                cls._instances[session_id] = PersistentREPL(working_dir=sandbox_path)
            return cls._instances[session_id]

    @classmethod
    def cleanup_repl(cls, session_id: str):
        with cls._lock:
            if session_id in cls._instances:
                logging.info(f"Closing and removing REPL for session: {session_id}")
                cls._instances[session_id].close()
                del cls._instances[session_id]
# --- End of new code ---


class CustomPythonREPLTool(PythonREPLTool):
    _datasets: dict = PrivateAttr()
    _results_dir: Optional[str] = PrivateAttr()
    _session_key: Optional[str] = PrivateAttr()

    def __init__(self, datasets, results_dir=None, session_key=None, **kwargs):
        super().__init__(**kwargs)
        self._datasets = datasets
        self._results_dir = results_dir
        self._session_key = session_key

    def _initialize_session(self, repl: PersistentREPL) -> None:
        """Prepare the REPL session by importing libraries and auto-loading data."""
        logging.info("Initializing persistent REPL session with auto-loading...")
        
        initialization_code = [
            "import os",
            "import sys", 
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import xarray as xr",
            "import logging",
            "from io import StringIO"
        ]

        # Add results_dir if available
        if self._results_dir:
            initialization_code.append(f"results_dir = r'{self._results_dir}'")
            initialization_code.append("os.makedirs(results_dir, exist_ok=True)")

        # Inject dataset variables by auto-loading from files
        for key, value in self._datasets.items():
            if key.startswith('dataset_') and isinstance(value, str) and os.path.isdir(value):
                # 1. Create the path variable (e.g., dataset_1_path)
                path_var_name = f"{key}_path"
                abs_path = os.path.abspath(value).replace('\\', '/')
                initialization_code.append(f"{path_var_name} = r'{abs_path}'")
                logging.info(f"Injected path variable {path_var_name} = {abs_path}")
                
                # 2. Attempt to find and auto-load data.csv into the main variable (e.g., dataset_1)
                csv_path = os.path.join(value, 'data.csv')
                if os.path.exists(csv_path):
                    initialization_code.append(f"# Auto-loading {key} from data.csv")
                    initialization_code.append(f"try:")
                    initialization_code.append(f"    {key} = pd.read_csv(os.path.join({path_var_name}, 'data.csv'))")
                    initialization_code.append(f"    print(f'Successfully auto-loaded {{os.path.join({path_var_name}, \"data.csv\")}} into `{key}` variable.')")
                    initialization_code.append(f"except Exception as e:")
                    initialization_code.append(f"    print(f'Could not auto-load {key}: {{e}}')")
                    logging.info(f"Added auto-loading code for {key} from {csv_path}")
            elif key == "uuid_main_dir" and isinstance(value, str):
                # Add the main UUID directory
                abs_path = os.path.abspath(value).replace('\\', '/')
                initialization_code.append(f"uuid_main_dir = r'{abs_path}'")
                logging.info(f"Injected uuid_main_dir = {abs_path}")

        # Execute all initialization code in one go
        full_init_code = "\n".join(initialization_code)
        logging.debug(f"REPL initialization code:\n{full_init_code}")
        result = repl.run(full_init_code)
        logging.info(f"REPL session initialization result: {result.strip()}")
        repl.is_initialized = True

    def _run(self, query: str, **kwargs) -> Any:
        """
        Execute code using the persistent REPL tied to the session.
        """
        # Get thread_id from session_state
        if 'thread_id' not in st.session_state or not st.session_state.thread_id:
            logging.error("CRITICAL ERROR: thread_id not found in session_state. Cannot run persistent REPL.")
            return {
                "result": "ERROR: Session ID is missing. Cannot continue.",
                "output": "ERROR: Session ID is missing. Cannot continue.",
                "plot_images": []
            }
        
        session_id = st.session_state.thread_id
        # Get sandbox path from datasets
        sandbox_path = self._datasets.get("uuid_main_dir")
        repl = REPLManager.get_repl(session_id, sandbox_path=sandbox_path)

        # "Warm up" the REPL on first call in the session
        if not repl.is_initialized:
            self._initialize_session(repl)
            repl.is_initialized = True

        logging.info(f"Executing code in persistent REPL for session {session_id}:\n{query}")
        
        # Execute user code
        output = repl.run(query)

        # Logic for detecting generated plots
        generated_plots = []
        image_extensions = ('.png', '.jpg', '.jpeg', '.svg', '.pdf')
        
        # Check for plot files in the output
        output_lines = output.strip().split('\n')
        for line in output_lines:
            potential_path = line.strip()
            if potential_path.lower().endswith(image_extensions) and os.path.exists(potential_path):
                if potential_path not in generated_plots:
                    generated_plots.append(potential_path)
                    logging.info(f"Found plot path in output: {potential_path}")

        # Also check results directory if it exists
        if self._results_dir and os.path.exists(self._results_dir):
            for file in os.listdir(self._results_dir):
                if file.lower().endswith(image_extensions):
                    full_path = os.path.join(self._results_dir, file)
                    if full_path not in generated_plots and os.path.exists(full_path):
                        generated_plots.append(full_path)

        if generated_plots:
            st.session_state.new_plot_generated = True
            log_history_event(
                st.session_state, "plot_generated",
                {"plot_paths": generated_plots, "agent": "PythonREPL", "content": query}
            )

        return {
            "result": output,
            "output": output,  # For backward compatibility
            "plot_images": generated_plots
        }