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

from ..utils import log_history_event, generate_unique_image_path

class CustomPythonREPLTool(PythonREPLTool):
    _datasets: dict = PrivateAttr()

    def __init__(self, datasets, **kwargs):
        """
        Custom Python REPL tool that injects dataset variables and logs plot generation.
        :param datasets: Dictionary { "dataset_1": <DataFrame>, "dataset_2": <DataFrame>, ... }
        """
        super().__init__(**kwargs)
        self._datasets = datasets

    def _run(self, query: str, **kwargs) -> Any:
        """
        Execute the user-provided Python code in a local context containing:
        - st (Streamlit)
        - plt (Matplotlib Pyplot)
        - pd (Pandas)
        - All loaded dataset variables (self._datasets)
        - Path variables for each dataset (dataset_1_path, dataset_2_path, etc.)
        - A dynamically generated plot_path
        
        If a figure is saved to plot_path, a "plot_generated" event will be logged.
        
        Returns:
            dict: Results including output and paths to any generated plots
        """
        import streamlit as st
        import matplotlib.pyplot as plt
        import pandas as pd
        import xarray as xr
        import os
        import logging
        from io import StringIO
        from src.utils import log_history_event, generate_unique_image_path

        # Prepare local context with necessary packages
        local_context = {
            "st": st, 
            "plt": plt, 
            "pd": pd,
            "xr": xr,
            "os": os
        }

        # Inject the user's datasets
        local_context.update(self._datasets)
        
        # Add dataset path variables for any string paths (sandbox directories)
        for key, value in self._datasets.items():
            if isinstance(value, str) and os.path.isdir(value):
                # Create a path variable like dataset_1_path for dataset_1
                path_var_name = f"{key}_path"
                # Use the absolute path with consistent slash direction
                abs_path = os.path.abspath(value).replace('\\', '/')
                local_context[path_var_name] = abs_path
                
                # Also log which path variables are available
                logging.info(f"Added path variable {path_var_name} = {abs_path}")

        # Generate a unique file path for the plot
        plot_path = generate_unique_image_path()
        local_context['plot_path'] = plot_path
        
        # Log the code being executed for debugging
        logging.info(f"Executing code with available variables: {list(local_context.keys())}")
        logging.info(f"Code to execute:\n{query}")

        # Redirect stdout to capture output
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output

        plot_generated = False
        output = ""
        
        try:
            # Execute user code
            exec(query, local_context)
            output = redirected_output.getvalue()
            
            # Check if plt.savefig was used or figure was created
            if os.path.exists(plot_path):
                plot_generated = True
                
                # Store the path in session state for UI to display
                st.session_state.saved_plot_path = plot_path
                st.session_state.plot_image = plot_path
                st.session_state.new_plot_path = plot_path
                st.session_state.new_plot_generated = True
                
                # Log the plot generation
                status_message = f"Plot generated = True. Saved at: {plot_path}"
                logging.info(f"Plot generated successfully and saved to: {plot_path}")
                st.session_state.plot_generated_status = status_message
                
                log_history_event(
                    st.session_state,
                    "plot_generated",
                    {
                        "plot_path": plot_path.replace("sandbox:", ""),  # Remove sandbox prefix
                        "agent": "VisualizationAgent",
                        "description": "Python_REPL Generated Plot",
                        "content": query  # Store the actual code used
                    }
                )
            else:
                # If no plot was generated, check if we need to give feedback about path issues
                if '/mnt/data' in query:
                    error_msg = "ERROR: Detected invalid path '/mnt/data'. You must use the provided dataset path variables."
                    logging.warning(error_msg)
                    output += f"\n\n{error_msg}"
                
                if "plot_path" in query and "plt.savefig" not in query:
                    warning_msg = "WARNING: You defined plot_path but didn't use plt.savefig(plot_path). Plots won't be displayed."
                    logging.warning(warning_msg)
                    output += f"\n\n{warning_msg}"

            # KEY CHANGE: Include output in the result string
            result_message = f"Execution completed. Plot saved at: {plot_path if plot_generated else 'No plot generated'}"
            if output.strip():
                result_message += f"\n\nOutput:\n{output}"

            return {
                "result": result_message,  # Now includes the console output
                "output": output,
                "plot_images": [plot_path] if plot_generated else []
            }
        except Exception as e:
            logging.error(f"Error during code execution: {e}")
            console_output = redirected_output.getvalue()
            
            error_output = f"ERROR: {str(e)}\n\n"
            
            # Add helpful error diagnostics
            if "FileNotFoundError" in str(e):
                error_output += "This looks like a path problem. Please check:\n"
                error_output += "1. You're using the exact path variables (dataset_1_path, etc.)\n"
                error_output += "2. You're using os.path.join() to combine paths\n"
                error_output += "3. The file you're trying to access actually exists\n\n"
                error_output += "Available path variables:\n"
                for key in local_context:
                    if key.endswith('_path'):
                        error_output += f"- {key}: {local_context[key]}\n"
            
            elif "ModuleNotFoundError" in str(e):
                module_name = str(e).split("No module named ")[-1].strip("'")
                error_output += f"Missing module: {module_name}\n"
                error_output += "You can install it using the install_package tool."
            
            elif "name 'data' is not defined" in str(e) or "not defined" in str(e):
                var_name = str(e).split("'")[1] if "'" in str(e) else "unknown"
                error_output += f"Variable '{var_name}' is not defined. Available variables:\n"
                error_output += f"- Dataset variables: {[k for k in self._datasets.keys()]}\n"
                error_output += f"- Path variables: {[k for k in local_context.keys() if k.endswith('_path')]}\n"
                error_output += "Make sure you're using the correct variable names."
            
            # Add any output that was generated before the error occurred
            if console_output:
                error_output += f"\nOutput before error:\n{console_output}"
            
            return {
                "error": "ExecutionError",
                "message": error_output,  # Now includes both error details and console output
                "output": console_output
            }
        finally:
            # Restore stdout
            sys.stdout = old_stdout