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
    def __init__(self):
        self._process = subprocess.Popen(
            [sys.executable, "-i", "-q", "-u"],  # -u for unbuffered output
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
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

    def run(self, code: str, timeout: int = 60) -> str:
        """Execute code in the persistent process."""
        # Clear queues before execution
        while not self._output_queue.empty(): self._output_queue.get_nowait()
        while not self._stderr_queue.empty(): self._stderr_queue.get_nowait()

        command = f"{code}\nprint('{self._end_marker}')\n"
        try:
            self._process.stdin.write(command)
            self._process.stdin.flush()
        except BrokenPipeError:
            logging.error("REPL process died. Attempting restart.")
            self.__init__() # Restart the process
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
    def get_repl(cls, session_id: str) -> PersistentREPL:
        with cls._lock:
            if session_id not in cls._instances:
                logging.info(f"Creating new persistent REPL for session: {session_id}")
                cls._instances[session_id] = PersistentREPL()
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
        """Prepare the REPL session by importing libraries and injecting variables."""
        logging.info("Initializing persistent REPL session...")

        # 1. Basic imports
        initialization_code = [
            "import os",
            "import sys",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import xarray as xr",
            "import logging",
            "from io import StringIO"
        ]

        # 2. Inject variables
        local_context = {}
        local_context.update(self._datasets)
        
        sandbox_path = local_context.get("uuid_main_dir")
        if sandbox_path:
            results_dir = os.path.join(sandbox_path, 'results')
            os.makedirs(results_dir, exist_ok=True)
            initialization_code.append(f"results_dir = r'{results_dir.replace('\\', '/')}'")
            logging.info(f"Injected variable results_dir: {results_dir}")
        else:
            logging.warning("uuid_main_dir not found, results_dir will not be defined.")

        for key, value in self._datasets.items():
            if isinstance(value, str) and os.path.isdir(value):
                path_var_name = f"{key}_path"
                abs_path = os.path.abspath(value).replace('\\', '/')
                initialization_code.append(f"{path_var_name} = r'{abs_path}'")
                logging.info(f"Injected path variable {path_var_name} = {abs_path}")
            # Note: We cannot directly inject large DataFrames,
            # the agent will need to load them using the injected paths.

        # 3. Execute initialization code
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
        repl = REPLManager.get_repl(session_id)

        # "Warm up" the REPL on first call in the session
        if not repl.is_initialized:
            self._initialize_session(repl)

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