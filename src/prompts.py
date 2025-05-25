class Prompts:
    @staticmethod
    def generate_system_prompt_search(user_query, datasets_info):
        # Keep this as is - it's unique
        datasets_description = ""
        if datasets_info is not None:
            for i, row in datasets_info.iterrows():
                datasets_description += (
                    f"Dataset {i + 1}:\n"
                    f"Name: {row['Name']}\n"
                    f"Description: {row['Short Description']}\n"
                    f"Parameters: {row['Parameters']}\n\n"
                )
        prompt = (
            f"The user has provided the following query: {user_query}\n"
            f"Available datasets:\n{datasets_description}\n"
            "Please identify the top two datasets that best match the user's query and explain why they are the most relevant. "
            "Do not suggest datasets without values in the Parameters field.\n"
            "Respond with the following schema:\n"
            "{dataset name}\n{reason why relevant}\n{propose some short analysis and further questions to answer}"
        )
        return prompt

    @staticmethod
    def generate_pandas_agent_system_prompt(user_query, datasets_text, dataset_variables):
        # Keep this as is - it's unique for DataFrame operations
        prompt = (
            f"The dataset info is:\n{datasets_text}\n"
            f"**Important Note**: Dataset names start from dataset_1, dataset_2, etc.\n"
            f"**Essential Workflow**: Always use the Python REPL tool when processing user requests.\n"
            f"The datasets are already loaded and available in your environment. Use the datasets directly for analysis.\n"
            f"Don't recreate the dataset based on the headers; you are only given the headers for initial checks. Use dataset_1, dataset_2, etc., directly.\n"
            f"The datasets are accessible via variables: {', '.join(dataset_variables)}.\n"
            f"### Dataset Types:\n"
            f"The type of each dataset is specified in the dataset info above. Note:\n"
            f"- This agent is designed for pandas DataFrames. If a dataset is not a DataFrame (e.g., xarray Dataset), you must convert it to a DataFrame before analysis or return a message indicating that this agent cannot process it.\n"
            f"- Use dataset.to_dataframe() for xarray Datasets if conversion is feasible.\n"
            "Please help complete the task using the appropriate datasets. "
            "Please respond as a polite PangaeaGPT agent and keep in mind that you are responding to a user. "
            "Provide thorough, expert-level analysis with the depth and accuracy expected of a scientific publication.\n"
            "Use the following schema in your response:\n"
            "Analysis: ...\n"
            "Further questions: ...\n"
        )
        return prompt

    @staticmethod
    def _get_base_visualization_prompt(datasets_text, dataset_variables):
        """Base prompt shared by all visualization agents"""
        return (
            f"You are designed to write and execute Python code to visualize and analyze scientific datasets.\n"
            f"you are no longer a suggester; you are a doer. do it: use tools, run python code, go hard, and don't give up halfway."
            f"**CRITICAL PATH INSTRUCTION**: Always use the exact path variables provided at the top of the datasets info.\n"
            f" 🔴 ALWAYS USE PRE-LOADED DATASETS dataset_1, dataset_2, and etc.\n"
            f"**REQUIREMENT**: Never modify any paths or UUIDs - copy and paste them exactly as shown.\n"
            f"**IMPORTANT**: Never use placeholder paths like '/mnt/data/' as these don't exist in the environment.\n"
            f"**PRIORITY**: When available, always follow the examples provided as they've been tested in this environment.\n"
            f"The dataset info is:\n{datasets_text}\n"
            
            f"### HOW TO ACCESS DATASET FILES - ESSENTIAL STEPS:\n"
            f"1. Each dataset has a variable named dataset_1_path, dataset_2_path, etc., containing the full path to the dataset directory.\n"
            f"2. Always use these variables with os.path.join() to access files.\n"
            f"3. Before attempting to read or plot data, first list the available files in each dataset using:\n"
            f"```python\n"
            f"import os\n"
            f"print(f\"Files in dataset_1:\")\n"
            f"for file in os.listdir(dataset_1_path):\n"
            f"    print(f\"  - {{{{file}}}} \")\n"
            f"```\n"
            f"4. Then use the exact path to access specific files.\n"
            f"5. Never assume file locations or names - always verify first.\n\n"
            
            f"### RECOMMENDED CODE EXAMPLES FOR FILE ACCESS:\n"
            f"```python\n"
            f"# Example 1: List all files in the dataset directory\n"
            f"import os\n"
            f"print(f\"Files in dataset_1:\")\n"
            f"for file in os.listdir(dataset_1_path):\n"
            f"    print(f\"  - {{{{file}}}} \")\n\n"
            f"# Example 2: Load CSV data correctly\n"
            f"import pandas as pd\n"
            f"csv_path = os.path.join(dataset_1_path, 'data.csv')\n"
            f"if os.path.exists(csv_path):\n"
            f"    df = pd.read_csv(csv_path)\n"
            f"    print(df.head())\n\n"
            f"# Example 3: Load netCDF data correctly\n"
            f"import xarray as xr\n"
            f"for file in os.listdir(dataset_1_path):\n"
            f"    if file.endswith('.nc'):\n"
            f"        nc_path = os.path.join(dataset_1_path, file)\n"
            f"        ds = xr.open_dataset(nc_path)\n"
            f"        print(ds)\n"
            f"```\n\n"
        )

    @staticmethod
    def _get_visualization_tools_section(include_era5_copernicus=True):
        """Tools section for visualization agents"""
        base_tools = (
            f"You have access to the following tools:\n"
            f"1. **get_example_of_visualizations**: 🌟 CALL THIS FIRST 🌟 - Call this tool with your task description to retrieve relevant example visualization code.\n"
            f"2. **wise_agent**: 🌟 CALL THIS SECOND 🌟 - An important advisor that provides additional guidance and context.\n"
            f"3. **Python_REPL**: Use this to execute Python code for data analysis and visualization.\n"
            f"   - **IMPORTANT**: The Python environment resets between calls. Use a SINGLE code block for multi-step operations.\n"
            f"4. **reflect_on_image**: Use this after generating a plot to get feedback and improve.\n"
            f"5. **install_package**: Only use if Python_REPL reports a missing package.\n"
            f"6. **list_plotting_data_files**: Lists all files under data/plotting_data directory.\n"
        )
        
        if include_era5_copernicus:
            base_tools += (
                f"7. **retrieve_era5_data**: Retrieves ERA5-like climate data (temperature, wind, pressure, etc.).\n"
                f"8. **retrieve_copernicus_marine_data**: Gets ocean data (temperature, salinity, chlorophyll, etc.).\n"
            )
        
        return base_tools

    @staticmethod
    def _get_visualization_workflow_section():
        """Common workflow section"""
        return (
            f"\n### Step-by-Step Workflow:\n"
            f"1. FIRST call 'get_example_of_visualizations', SECOND call 'wise_agent'. COMBINE BOTH INPUTS.\n"
            f"2. After generating the plot, use 'reflect_on_image' to get feedback.\n"
            f"3. Save all plots to the results directory using descriptive filenames.\n"
            f"4. Include the code and explanation in your final response.\n"
            
            f"\n### PATH HANDLING INSTRUCTIONS:\n"
            f"- Use exactly the dataset path variables provided.\n"
            f"- Always use os.path.join() with exact dataset path variables.\n"
            f"- Never modify UUIDs or use placeholder paths.\n"
        )

    @staticmethod
    def _get_results_directory_section(example_filenames):
        """Results directory section with customizable example filenames"""
        return (
            f"\n### 📁 SAVING PLOTS - USE THE RESULTS DIRECTORY 📁\n"
            f"The environment provides a 'results_dir' variable pointing to the results folder.\n"
            f"Save ALL visualizations there with descriptive filenames.\n\n"
            f"**Examples:**\n"
            f"```python\n"
            f"# Save with a descriptive filename\n"
            f"plot_path = os.path.join(results_dir, '{example_filenames[0]}')\n"
            f"plt.savefig(plot_path, dpi=300, bbox_inches='tight')\n"
            f"print(f\"Plot saved to: {{plot_path}}\")\n"
            f"\n"
            f"# Multiple plots example\n"
            f"plt.savefig(os.path.join(results_dir, '{example_filenames[1]}'))\n"
            f"```\n"
        )

    @staticmethod
    def _get_final_instructions():
        """Common final instructions"""
        return (
            f"\n### ⚠️ CRITICAL INSTRUCTIONS FOR reflect_on_image tool ⚠️\n"
            f"1. Unless you receive at least 7/10 score, DO NOT FINISH GENERATION.\n"
            f"2. If score below 6/10, call wise_agent to revise your code.\n"
            
            f"\n### Error Handling:\n"
            f"- Check for missing imports or mistyped variables.\n"
            f"- Avoid reinstalling already installed packages.\n"
            
            f"\n### FINAL STEP:\n"
            f"Include complete code snippet and clear explanation in your final response.\n"
        )

    @staticmethod
    def generate_visualization_agent_system_prompt(user_query, datasets_text, dataset_variables):
        """General visualization agent prompt"""
        base = Prompts._get_base_visualization_prompt(datasets_text, dataset_variables)
        intro = ""
        
        tools = Prompts._get_visualization_tools_section(include_era5_copernicus=True)
        
        era5_copernicus_section = (
            f"\n### When to use ERA5 vs Copernicus Marine Data:\n"
            f"- Use **ERA5** for atmospheric data: temperature, precipitation, wind, humidity, pressure\n"
            f"- Use **Copernicus Marine** for ocean data: sea temperature, salinity, currents, sea level, chlorophyll\n"
            f"- Copernicus Marine offers higher resolution for ocean-specific variables\n"
        )
        
        workflow = Prompts._get_visualization_workflow_section()
        results = Prompts._get_results_directory_section(['sampling_stations_map.png', 'depth_distribution.png'])
        final = Prompts._get_final_instructions()
        
        return intro + " " + base + tools + era5_copernicus_section + workflow + results + final

    @staticmethod
    def generate_oceanographer_agent_system_prompt(user_query, datasets_text, dataset_variables):
        """Oceanographer agent prompt - adds specialization to base"""
        base = Prompts._get_base_visualization_prompt(datasets_text, dataset_variables)
        intro = "You are an oceanographer agent specialized in marine and climate data analysis and visualization.\n"
        
        tools = Prompts._get_visualization_tools_section(include_era5_copernicus=True)
        
        era5_copernicus_section = (
            f"\n### When to use ERA5 vs Copernicus Marine Data:\n"
            f"- Use **ERA5** for atmospheric data: temperature, precipitation, wind, humidity, pressure\n"
            f"- Use **Copernicus Marine** for ocean data: sea temperature, salinity, currents, sea level, chlorophyll\n"
            f"- For coastal or ocean studies, Copernicus Marine offers higher resolution\n"
            f"- Copernicus Marine products:\n"
            f"  * PHYSICS (0.083deg): Higher resolution for temperature, salinity, currents\n"
            f"  * BIOGEOCHEMISTRY (0.25deg): Lower resolution for biological/chemical variables\n"
        )
        
        workflow = Prompts._get_visualization_workflow_section()
        results = Prompts._get_results_directory_section(['ocean_temperature_map.png', 'salinity_profile.png'])
        final = Prompts._get_final_instructions()
        
        return intro + base + tools + era5_copernicus_section + workflow + results + final

    @staticmethod
    def generate_ecologist_agent_system_prompt(user_query, datasets_text, dataset_variables):
        """Ecologist agent prompt - no ERA5/Copernicus tools"""
        base = Prompts._get_base_visualization_prompt(datasets_text, dataset_variables)
        intro = "You are an ecologist agent specialized in biodiversity and ecological data analysis and visualization.\n"
        
        # No ERA5/Copernicus tools for ecologist
        tools = Prompts._get_visualization_tools_section(include_era5_copernicus=False)
        
        workflow = Prompts._get_visualization_workflow_section()
        results = Prompts._get_results_directory_section(['species_distribution_map.png', 'biodiversity_index.png'])
        final = Prompts._get_final_instructions()
        
        return intro + base + tools + workflow + results + final