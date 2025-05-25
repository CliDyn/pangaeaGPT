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
        import streamlit as st
        import matplotlib.pyplot as plt
        import pandas as pd
        import xarray as xr
        import os
        import logging
        from io import StringIO
        from src.utils import log_history_event

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
        
        # Extract the sandbox directory from the datasets
        sandbox_path = None
        
        # First check if uuid_main_dir is in datasets (it's added by visualization agents)
        if "uuid_main_dir" in self._datasets:
            sandbox_path = self._datasets["uuid_main_dir"]
            logging.info(f"Found uuid_main_dir for plots: {sandbox_path}")
        else:
            # Otherwise, try to find it from dataset paths
            for key, value in self._datasets.items():
                if isinstance(value, str) and os.path.isdir(value) and "sandbox" in value:
                    # Get the parent directory (sandbox UUID directory)
                    potential_sandbox = os.path.dirname(os.path.abspath(value))
                    if "sandbox" in potential_sandbox:
                        sandbox_path = potential_sandbox
                        logging.info(f"Extracted sandbox path from {key}: {sandbox_path}")
                        break
        
        # Create and provide the results directory
        if sandbox_path:
            results_dir = os.path.join(sandbox_path, 'results')
            os.makedirs(results_dir, exist_ok=True)
            local_context['results_dir'] = results_dir
            logging.info(f"Created/verified results directory: {results_dir}")
        else:
            # Fallback to tmp/figs if no sandbox
            results_dir = os.path.join('tmp', 'figs')
            os.makedirs(results_dir, exist_ok=True)
            local_context['results_dir'] = results_dir
            logging.warning(f"No sandbox found, using fallback results directory: {results_dir}")
        
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

        # Log the code being executed for debugging
        logging.info(f"Executing code with available variables: {list(local_context.keys())}")
        logging.info(f"Results directory available at: {results_dir}")
        logging.info(f"Code to execute:\n{query}")

        # Get list of image files BEFORE execution
        before_exec = set()
        if os.path.exists(results_dir):
            before_exec = set(os.listdir(results_dir))

        # Redirect stdout to capture output
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output

        plot_generated = False
        generated_plots = []
        output = ""
        
        try:
            # Execute user code
            exec(query, local_context)
            output = redirected_output.getvalue()
            
            # Check if any plots were saved to results_dir
            if os.path.exists(results_dir):
                # Re-check after execution
                import time
                time.sleep(0.1)  # Brief pause to ensure file is written
                after_exec = set(os.listdir(results_dir))
                
                # Find new files
                new_files = after_exec - before_exec
                
                # Check for new image files
                image_extensions = ('.png', '.jpg', '.jpeg', '.svg', '.pdf')
                new_images = [f for f in new_files if f.lower().endswith(image_extensions)]
                
                if new_images:
                    plot_generated = True
                    for img_file in new_images:
                        img_path = os.path.join(results_dir, img_file)
                        generated_plots.append(img_path)
                        logging.info(f"Found newly generated plot: {img_path}")
                    
                    # Store paths in session state
                    if generated_plots:
                        st.session_state.saved_plot_path = generated_plots[0]
                        st.session_state.plot_image = generated_plots[0]
                        st.session_state.new_plot_path = generated_plots[0]
                        st.session_state.new_plot_generated = True
                        
                        log_history_event(
                            st.session_state,
                            "plot_generated",
                            {
                                "plot_paths": generated_plots,
                                "agent": "VisualizationAgent",
                                "description": "Plots generated in results directory",
                                "content": query
                            }
                        )

            # Include output in the result string
            result_message = f"Execution completed. "
            if plot_generated:
                result_message += f"Plots saved to results directory: {', '.join([os.path.basename(p) for p in generated_plots])}"
            else:
                result_message += "No plots generated."
            
            if output.strip():
                result_message += f"\n\nOutput:\n{output}"

            return {
                "result": result_message,
                "output": output,
                "plot_images": generated_plots
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
                    if key.endswith('_path') or key == 'results_dir':
                        error_output += f"- {key}: {local_context[key]}\n"
            
            elif "ModuleNotFoundError" in str(e):
                module_name = str(e).split("No module named ")[-1].strip("'")
                error_output += f"Missing module: {module_name}\n"
                error_output += "You can install it using the install_package tool."
            
            # Add any output that was generated before the error occurred
            if console_output:
                error_output += f"\nOutput before error:\n{console_output}"
            
            return {
                "error": "ExecutionError",
                "message": error_output,
                "output": console_output,
                "plot_images": []  # Return empty list on error
            }
        finally:
            # Restore stdout
            sys.stdout = old_stdout