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
from typing import Any
import subprocess
import threading
import queue

from ..utils import log_history_event

# --- Start of new code for persistent REPL ---

class PersistentREPL:
    """
    Manages a single long-lived interactive Python process.
    """
    def __init__(self, working_dir=None):
        self._working_dir = working_dir  # Store for restart purposes
        self._process = subprocess.Popen(
            [sys.executable, "-i", "-q", "-u"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=working_dir  # Start the subprocess in the specified directory
        )
        self._end_marker = f"---END_OF_OUTPUT_{os.urandom(8).hex()}---"
        self._output_queue = queue.Queue()
        self._stderr_queue = queue.Queue()

        # Threads for reading stdout and stderr without blocking
        self._stdout_thread = threading.Thread(target=self._enqueue_output, args=(self._process.stdout, self._output_queue))
        self._stderr_thread = threading.Thread(target=self._enqueue_output, args=(self._process.stderr, self._stderr_queue))
        self._stdout_thread.daemon = True
        self._stderr_thread.daemon = True
        self._stdout_thread.start()
        self._stderr_thread.start()
        
        # Flag to track if session has been initialized
        self.is_initialized = False

    def _enqueue_output(self, pipe, q):
        for line in iter(pipe.readline, ''):
            q.put(line)
        pipe.close()

    def run(self, code: str, timeout: int = 300) -> str:
        """Execute code in the persistent process."""
        # Clear queues before execution
        while not self._output_queue.empty(): 
            self._output_queue.get_nowait()
        while not self._stderr_queue.empty(): 
            self._stderr_queue.get_nowait()

        # Simply send the code directly followed by the end marker
        # Add a newline before the end marker to ensure it's on its own line
        command = f"{code}\n\nprint('{self._end_marker}')\n"
        
        try:
            self._process.stdin.write(command)
            self._process.stdin.flush()
        except BrokenPipeError:
            logging.error("REPL process died. Attempting restart.")
            self.__init__(working_dir=self._working_dir)
            return "ERROR: The Python REPL process died and was restarted. Please try your command again."

        output_lines = []
        stderr_lines = []
        
        while True:
            try:
                line = self._output_queue.get(timeout=timeout)
                if self._end_marker in line:
                    break
                output_lines.append(line)
            except queue.Empty:
                logging.error("Code execution timeout in REPL.")
                # Read stderr for diagnostics
                while not self._stderr_queue.empty():
                    stderr_lines.append(self._stderr_queue.get_nowait())
                error_info = "".join(stderr_lines)
                return f"TIMEOUT ERROR: Code execution exceeded {timeout} seconds. Stderr: {error_info}"

        # Collect remaining stderr
        while not self._stderr_queue.empty():
            stderr_lines.append(self._stderr_queue.get_nowait())

        return "".join(stderr_lines) + "".join(output_lines)

    def close(self):
        if self._process:
            self._process.terminate()
            self._process = None

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

    def __init__(self, datasets, **kwargs):
        super().__init__(**kwargs)
        self._datasets = datasets

    def _initialize_session(self, repl: PersistentREPL) -> None:
        """Prepare the REPL session by importing libraries and auto-loading data."""
        logging.info("Initializing persistent REPL session with auto-loading...")
        
        initialization_code = [
            "import os",
            "import sys", 
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import xarray as xr",
            "import logging",
            "from io import StringIO"
        ]

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
                    initialization_code.append(f"except Exception as e: print(f'Could not auto-load {key}: {{e}}')")
                    logging.info(f"Added auto-loading code for {key} from {csv_path}")

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
            return "ERROR: Session ID is missing. Cannot continue."
        
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

        # Logic for detecting generated plots (remains the same)
        generated_plots = []
        image_extensions = ('.png', '.jpg', '.jpeg', '.svg', '.pdf')
        output_lines = output.strip().split('\n')
        for line in output_lines:
            potential_path = line.strip()
            if potential_path.lower().endswith(image_extensions) and os.path.exists(potential_path):
                if potential_path not in generated_plots:
                    generated_plots.append(potential_path)
                    logging.info(f"Found plot path in output: {potential_path}")

        if generated_plots:
            st.session_state.new_plot_generated = True
            log_history_event(
                st.session_state, "plot_generated",
                {"plot_paths": generated_plots, "agent": "PythonREPL", "content": query}
            )

        return {
            "result": output,
            "output": output, # For backward compatibility
            "plot_images": generated_plots
        }