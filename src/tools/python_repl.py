# src/tools/python_repl.py

import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from io import StringIO
from pydantic import BaseModel, Field, PrivateAttr
from langchain_experimental.tools import PythonREPLTool
from typing import Any, Dict, Optional, List
import re
import threading
import json
import queue
import time
import atexit

# Import Jupyter Client
try:
    # We use KernelManager for lifecycle management and KernelClient for communication
    from jupyter_client import KernelManager
    from jupyter_client.client import KernelClient
except ImportError:
    logging.error("Jupyter client not installed. Please run 'pip install jupyter_client ipykernel'")
    # Raise error to ensure environment is correctly set up
    raise ImportError("Missing dependencies for persistent REPL. Install jupyter_client and ipykernel.")

from ..utils import log_history_event
from ..utils.workspace import WorkspaceManager
from ..utils.session_manager import SessionManager

# --- New Jupyter Kernel Executor ---

class JupyterKernelExecutor:
    """
    Manages a persistent Jupyter kernel for code execution.
    Replaces the previous subprocess-based PersistentREPL.
    """
    def __init__(self, working_dir=None):
        self._working_dir = working_dir
        # Initialize the KernelManager to use the standard python3 kernel
        self.km = KernelManager(kernel_name='python3')
        self.kc: Optional[KernelClient] = None
        self.is_initialized = False
        self._start_kernel()

    def _start_kernel(self):
        """Starts the Jupyter kernel."""
        logging.info("Starting Jupyter kernel...")
        
        # Ensure the ipykernel is available.
        try:
            # Accessing kernel_spec validates that the kernel is installed and found
            self.km.kernel_spec
        except Exception as e:
            logging.error(f"Could not find python3 kernel spec. Ensure ipykernel is installed. Error: {e}")
            # We might attempt installation here, but it's better to rely on environment setup
            raise RuntimeError("IPython kernel (python3) not available.")

        # Start the kernel in the specified working directory (CWD)
        if self._working_dir and os.path.exists(self._working_dir):
            self.km.start_kernel(cwd=self._working_dir)
        else:
            self.km.start_kernel()

        # Create the synchronous client
        self.kc = self.km.client()
        self.kc.start_channels()
        
        # Wait for the kernel to be ready
        try:
            self.kc.wait_for_ready(timeout=60)
            logging.info("Jupyter kernel started and ready.")
        except RuntimeError as e:
            logging.error(f"Kernel failed to start: {e}")
            self.km.shutdown_kernel()
            raise

    def _execute_code(self, code: str, timeout: float = 300.0) -> Dict[str, Any]:
        """Executes code synchronously in the kernel and captures outputs."""
        if not self.kc:
            return {"status": "error", "error": "Kernel client not available."}

        # Send the execution request
        msg_id = self.kc.execute(code)
        result = {
            "status": "success",
            "stdout": "",
            "stderr": "",
            "display_data": []
        }
        start_time = time.time()

        # Process messages until the kernel is idle or timeout occurs
        while True:
            if time.time() - start_time > timeout:
                logging.warning("Kernel execution timed out. Interrupting kernel.")
                self.km.interrupt_kernel()
                result["status"] = "error"
                result["error"] = f"Execution timed out after {timeout} seconds."
                break

            try:
                # Get messages from the IOPub channel (where outputs are published)
                # Use a small timeout to remain responsive
                msg = self.kc.get_iopub_msg(timeout=0.1)
            except queue.Empty:
                # Check if the kernel is still alive if the queue is empty
                if not self.km.is_alive():
                    result["status"] = "error"
                    result["error"] = "Kernel died unexpectedly."
                    break
                continue
            except Exception as e:
                logging.error(f"Error getting IOPub message: {e}")
                continue

            # Process the message if it relates to our execution request
            if msg['parent_header'].get('msg_id') == msg_id:
                msg_type = msg['msg_type']

                if msg_type == 'status':
                    if msg['content']['execution_state'] == 'idle':
                        break  # Execution finished
                elif msg_type == 'stream':
                    if msg['content']['name'] == 'stdout':
                        result["stdout"] += msg['content']['text']
                    elif msg['content']['name'] == 'stderr':
                        result["stderr"] += msg['content']['text']
                elif msg_type == 'display_data' or msg_type == 'execute_result':
                    # Capture rich display data (e.g., for interactive plots in the future)
                    result["display_data"].append(msg['content']['data'])
                    # Also capture text representation for the current agent output
                    if 'text/plain' in msg['content']['data']:
                        result["stdout"] += msg['content']['data']['text/plain'] + "\n"
                elif msg_type == 'error':
                    result["status"] = "error"
                    ename = msg['content']['ename']
                    evalue = msg['content']['evalue']
                    # Clean ANSI escape codes from traceback
                    traceback = "\n".join(msg['content']['traceback'])
                    traceback = re.sub(r'\x1b\[[0-9;]*m', '', traceback)
                    result["error"] = f"{ename}: {evalue}\n{traceback}"
                    break

        return result

    @staticmethod
    def sanitize_input(query: str) -> str:
        """Sanitize input (from LangChain)."""
        # Remove surrounding backticks, whitespace, and the 'python' keyword
        query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
        query = re.sub(r"(\s|`)*$", "", query)
        return query

    def _sanitize_code(self, code: str) -> str:
        """Clean specific syntax."""
        # Jupyter kernels handle magic commands natively, so we just need LangChain sanitization
        return self.sanitize_input(code)

    def run(self, code: str, timeout: int = 300) -> str:
        """Execute code in the Jupyter kernel and return formatted output."""
        cleaned_code = self._sanitize_code(code)

        result = self._execute_code(cleaned_code, timeout)

        if result["status"] == "success":
            output = result["stdout"]
            if result.get("stderr"):
                # Include stderr output (often contains warnings)
                if output:
                    output += "\n[STDERR]\n" + result["stderr"]
                else:
                    output = "[STDERR]\n" + result["stderr"]
            return output.strip()
        elif result["status"] == "error":
            return result["error"]
        else:
            return f"Fatal error: {result.get('error', 'Unknown error')}"

    def close(self):
        """Cleanup and terminate the kernel."""
        if self.kc:
            self.kc.stop_channels()
        if hasattr(self, 'km') and self.km.is_alive():
            logging.info("Shutting down Jupyter kernel...")
            self.km.shutdown_kernel(now=True)
            logging.info("Jupyter kernel shut down.")

# REPLManager updated to manage JupyterKernelExecutor instances
class REPLManager:
    """
    Singleton for managing JupyterKernelExecutor instances for each session (thread_id).
    """
    _instances: dict[str, JupyterKernelExecutor] = {}
    _lock = threading.Lock()

    @classmethod
    def get_repl(cls, session_id: str, sandbox_path: str = None) -> JupyterKernelExecutor:
        with cls._lock:
            if session_id not in cls._instances:
                logging.info(f"Creating new Jupyter Kernel for session: {session_id}")
                if sandbox_path:
                    logging.info(f"Starting Kernel in sandbox directory: {sandbox_path}")
                try:
                    cls._instances[session_id] = JupyterKernelExecutor(working_dir=sandbox_path)
                except Exception as e:
                    logging.error(f"Failed to create Jupyter Kernel for session {session_id}: {e}")
                    # Raise the exception so the calling agent can handle the failure
                    raise RuntimeError(f"Failed to initialize execution environment: {e}")
            return cls._instances[session_id]

    @classmethod
    def cleanup_repl(cls, session_id: str):
        with cls._lock:
            if session_id in cls._instances:
                logging.info(f"Closing and removing Kernel for session: {session_id}")
                cls._instances[session_id].close()
                del cls._instances[session_id]

    @classmethod
    def cleanup_all(cls):
        """Cleanup all running kernels."""
        logging.info("Cleaning up all kernels on exit...")
        sessions = list(cls._instances.keys())
        for session_id in sessions:
            cls.cleanup_repl(session_id)

# Ensure kernels are cleaned up when the application exits
atexit.register(REPLManager.cleanup_all)

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

    def _initialize_session(self, repl: JupyterKernelExecutor) -> None:
        """Prepare the Kernel session by importing libraries and auto-loading data."""
        logging.info("Initializing persistent Kernel session...")

        # --- WORKSPACE MANAGER SETUP ---
        # Get paths for the specific session
        thread_id = self._session_key or WorkspaceManager.get_thread_id()

        main_dir = WorkspaceManager.get_sandbox_path(thread_id).replace(os.sep, '/')
        results_dir = WorkspaceManager.get_results_dir(thread_id).replace(os.sep, '/')

        initialization_code = [
            "import os", "import sys", "import pandas as pd", "import numpy as np",
            "import matplotlib.pyplot as plt", "import xarray as xr", "import logging",
            "from io import StringIO",
            "import matplotlib", "matplotlib.use('Agg')",
            # Inject Paths
            f"uuid_main_dir = r'{main_dir}'",
            f"results_dir = r'{results_dir}'",
            "os.makedirs(results_dir, exist_ok=True)",
            # Force CWD to Sandbox
            f"os.chdir(uuid_main_dir)"
        ]
        # -------------------------------

        # Inject dataset variables (код загрузки датасетов оставляем как был)
        for key, value in self._datasets.items():
            if key.startswith('dataset_') and isinstance(value, str) and os.path.isdir(value):
                path_var_name = f"{key}_path"
                abs_path = os.path.abspath(value).replace(os.sep, '/')
                initialization_code.append(f"{path_var_name} = r'{abs_path}'")
                # Auto-load CSV attempt...
                if os.path.exists(os.path.join(value, 'data.csv')):
                    initialization_code.append(f"try: {key} = pd.read_csv(os.path.join({path_var_name}, 'data.csv'))\nexcept: pass")

        full_init_code = "\n".join(initialization_code)
        repl.run(full_init_code)
        repl.is_initialized = True

    def _run(self, query: str, **kwargs) -> Any:
        # Retrieve Session ID
        session_id = self._session_key or WorkspaceManager.get_thread_id()

        # Get Sandbox Path from Manager
        sandbox_path = WorkspaceManager.get_sandbox_path(session_id)

        try:
            # Pass explicit sandbox path to Kernel Executor
            repl = REPLManager.get_repl(session_id, sandbox_path=sandbox_path)
        except RuntimeError as e:
            return {"result": f"Error: {e}", "output": str(e), "plot_images": []}

        if not repl.is_initialized:
            self._initialize_session(repl)

        logging.info(f"Executing code for session {session_id}")

        # Execute user code
        output = repl.run(query)

        # --- ROBUST PLOT DETECTION ---
        generated_plots = []
        image_extensions = ('.png', '.jpg', '.jpeg', '.svg', '.pdf')

        # 1. Check actual results directory (File System Check)
        results_dir = WorkspaceManager.get_results_dir(session_id)
        if os.path.exists(results_dir):
            for file in os.listdir(results_dir):
                if file.lower().endswith(image_extensions):
                    full_path = os.path.join(results_dir, file)
                    # Check if modified in the last 15 seconds (fresh plot)
                    if time.time() - os.path.getmtime(full_path) < 15:
                        generated_plots.append(full_path)

        # 2. Check output for specific print statements (Fallback)
        matches = re.findall(r'([a-zA-Z0-9_\-/\\.]+\.(png|jpg|jpeg|svg|pdf))', output, re.IGNORECASE)
        for match in matches:
            potential_path = match[0].strip()
            # Use Manager to resolve path securely
            resolved = WorkspaceManager.resolve_path(potential_path, session_id)
            if resolved and os.path.exists(resolved) and resolved not in generated_plots:
                generated_plots.append(resolved)

        # -----------------------------

        if generated_plots:
             session_data = SessionManager.get_session(session_id)
             session_data["new_plot_generated"] = True
             log_history_event(session_data, "plot_generated",
                             {"plot_paths": generated_plots, "agent": "PythonREPL"})

        return {
            "result": output,
            "output": output,
            "plot_images": generated_plots
        }