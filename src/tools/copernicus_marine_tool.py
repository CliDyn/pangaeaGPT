"""
src/tools/copernicus_marine_tool.py

Copernicus Marine data retrieval tool for use in the visualization agent.
Retrieves oceanographic data from the Copernicus Marine Service.
"""

import os
import logging
import uuid
import xarray as xr
import pandas as pd
import numpy as np
import streamlit as st
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from langchain_core.tools import StructuredTool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CopernicusMarineRetrievalArgs(BaseModel):
    """Arguments for the Copernicus Marine data retrieval tool."""
    dataset_id: Literal[
        # Physics datasets (daily)
        "cmems_mod_glo_phy_my_0.083deg_P1D-m", 
        "cmems_mod_glo_phy_myint_0.083deg_P1D-m",
        # Biogeochemistry datasets (daily)
        "cmems_mod_glo_bgc_my_0.25deg_P1D-m",
        "cmems_mod_glo_bgc_myint_0.25deg_P1D-m"
    ] = Field(
        description="The Copernicus Marine dataset ID to retrieve. Available datasets include:\n"
        "- Physics datasets (1/12° resolution, ~8km):\n"
        "  * 'cmems_mod_glo_phy_my_0.083deg_P1D-m': Global Ocean Physics daily mean, 1992-2021 (reanalysis)\n"
        "  * 'cmems_mod_glo_phy_myint_0.083deg_P1D-m': Global Ocean Physics daily mean, 2021-present (interim)\n"
        "- Biogeochemistry datasets (1/4° resolution, ~25km):\n"
        "  * 'cmems_mod_glo_bgc_my_0.25deg_P1D-m': Global Ocean Biogeochemistry daily mean, 1993-2022 (reanalysis)\n"
        "  * 'cmems_mod_glo_bgc_myint_0.25deg_P1D-m': Global Ocean Biogeochemistry daily mean, 2023-present (interim)"
    )
    variables: List[str] = Field(
        description="List of variable names to extract, specific to each dataset:\n"
        "- Physics variables (cmems_mod_glo_phy_*_P1D-m):\n"
        "  * thetao: Potential temperature (°C)\n"
        "  * so: Salinity (PSU)\n"
        "  * uo: Eastward ocean current velocity (m/s)\n"
        "  * vo: Northward ocean current velocity (m/s)\n"
        "  * zos: Sea surface height (m)\n"
        "  * mlotst: Mixed layer thickness (m)\n"
        "  * bottomT: Sea floor potential temperature (°C)\n"
        "  * siconc: Sea ice concentration (fraction)\n"
        "  * sithick: Sea ice thickness (m)\n"
        "  * usi: Eastward sea ice velocity (m/s)\n"
        "  * vsi: Northward sea ice velocity (m/s)\n"
        "- Biogeochemistry variables (cmems_mod_glo_bgc_*_P1D-m):\n"
        "  * chl: Chlorophyll concentration (mg/m³)\n"
        "  * no3: Nitrate concentration (mmol/m³)\n"
        "  * po4: Phosphate concentration (mmol/m³)\n"
        "  * si: Silicate concentration (mmol/m³)\n"
        "  * nppv: Net primary production (mg/m³/day)\n"
        "  * o2: Dissolved oxygen (mmol/m³)"
    )
    start_datetime: str = Field(
        description="Start date in 'YYYY-MM-DD' format. Available range depends on dataset."
    )
    end_datetime: str = Field(
        description="End date in 'YYYY-MM-DD' format. Available range depends on dataset."
    )
    minimum_longitude: float = Field(
        description="Minimum longitude in decimal degrees (-180 to 180)."
    )
    maximum_longitude: float = Field(
        description="Maximum longitude in decimal degrees (-180 to 180)."
    )
    minimum_latitude: float = Field(
        description="Minimum latitude in decimal degrees (-90 to 90). Note: Physics datasets cover -80 to 90°N, Biogeochemistry datasets cover -89 to 90°N."
    )
    maximum_latitude: float = Field(
        description="Maximum latitude in decimal degrees (-90 to 90). Note: Physics datasets cover -80 to 90°N, Biogeochemistry datasets cover -89 to 90°N."
    )
    minimum_depth: Optional[float] = Field(
        None, description="Optional: Minimum depth in meters (positive value). Physics datasets have 50 vertical levels, Biogeochemistry datasets have 75 vertical levels."
    )
    maximum_depth: Optional[float] = Field(
        None, description="Optional: Maximum depth in meters (positive value). Physics datasets have 50 vertical levels (max ~5728m), Biogeochemistry datasets have 75 vertical levels (max ~5902m)."
    )
    vertical_axis: Optional[Literal['depth']] = Field(
        'depth', description="Vertical axis type (only 'depth' is supported)."
    )

def _generate_descriptive_filename(
    variables: List[str], 
    start_datetime: str, 
    end_datetime: str,
    minimum_depth: Optional[float] = None,
    maximum_depth: Optional[float] = None
) -> str:
    """
    Generate a descriptive filename based on request parameters.
    Format: {variables_joined}_{start_date}_to_{end_date}[_depth_{min_depth}_to_{max_depth}].nc
    
    Example: thetao_so_2020_01_01_to_2020_02_01_depth_0_to_100.nc
    """
    # Clean and join variables - limit to avoid overly long filenames
    clean_variables = []
    for var in variables[:3]:  # Limit to first 3 variables to avoid long filenames
        clean_var = var.replace('-', '_').replace('/', '_').replace(' ', '_')
        clean_variables.append(clean_var)
    
    if len(variables) > 3:
        variables_str = "_".join(clean_variables) + "_plus" + str(len(variables) - 3)
    else:
        variables_str = "_".join(clean_variables)
    
    # Convert dates to underscore format (YYYY-MM-DD -> YYYY_MM_DD)
    clean_start = start_datetime.split()[0].replace('-', '_')  # Handle datetime strings, take date part
    clean_end = end_datetime.split()[0].replace('-', '_')      # Handle datetime strings, take date part
    
    # Build filename
    filename_parts = [variables_str, clean_start, "to", clean_end]
    
    # Add depth range if specified
    if minimum_depth is not None or maximum_depth is not None:
        depth_min = str(int(minimum_depth)) if minimum_depth is not None else "surface"
        depth_max = str(int(maximum_depth)) if maximum_depth is not None else "bottom"
        filename_parts.extend(["depth", depth_min, "to", depth_max])
    
    # Join with underscores and add extension
    filename = "_".join(filename_parts) + ".nc"
    
    return filename

def retrieve_copernicus_marine_data(
    dataset_id: str,
    variables: List[str],
    start_datetime: str,
    end_datetime: str,
    minimum_longitude: float,
    maximum_longitude: float,
    minimum_latitude: float,
    maximum_latitude: float,
    minimum_depth: Optional[float] = None,
    maximum_depth: Optional[float] = None,
    vertical_axis: str = 'depth'
) -> dict:
    """
    Retrieves oceanographic data from the Copernicus Marine Service and saves it locally
    as NetCDF file.
    
    Args:
        dataset_id: The Copernicus Marine dataset ID
        variables: List of variable names to extract
        start_datetime: Start date in 'YYYY-MM-DD' format
        end_datetime: End date in 'YYYY-MM-DD' format
        minimum_longitude: Minimum longitude in decimal degrees
        maximum_longitude: Maximum longitude in decimal degrees
        minimum_latitude: Minimum latitude in decimal degrees
        maximum_latitude: Maximum latitude in decimal degrees
        minimum_depth: Optional minimum depth in meters
        maximum_depth: Optional maximum depth in meters
        vertical_axis: Vertical axis type ('depth' or 'elevation')
        
    Returns:
        dict: Results including success status and output file path
    """
    try:
        logging.info(
            f"Copernicus Marine retrieval: dataset={dataset_id}, "
            f"vars={variables}, "
            f"time={start_datetime}→{end_datetime}, "
            f"lon=[{minimum_longitude},{maximum_longitude}], "
            f"lat=[{minimum_latitude},{maximum_latitude}], "
            f"depth=[{minimum_depth},{maximum_depth}]"
        )
        
        # Check if the copernicusmarine package is available
        try:
            import copernicusmarine
            logging.info("Successfully imported copernicusmarine package")
        except ImportError:
            return {
                "success": False,
                "error": "The copernicusmarine package is not installed. Please install it with 'pip install copernicusmarine'.",
                "message": "Failed to retrieve Copernicus Marine data: copernicusmarine package not installed"
            }
        
        # 1) Determine or create a sandbox directory  
        main_dir = None
        if "active_datasets" in st.session_state and st.session_state["active_datasets"]:
            doi = next(iter(st.session_state["active_datasets"]))
            cached = st.session_state["datasets_cache"].get(doi)
            if cached:
                path = cached[0]
                if isinstance(path, str) and os.path.isdir(path):
                    main_dir = os.path.dirname(os.path.abspath(path))
        if not main_dir:
            main_dir = os.path.join("tmp", "sandbox", uuid.uuid4().hex)
            logging.info(f"Created new sandbox: {main_dir}")
        os.makedirs(main_dir, exist_ok=True)

        copernicus_dir = os.path.join(main_dir, "copernicus_data")
        os.makedirs(copernicus_dir, exist_ok=True)
        logging.info(f"Copernicus Marine output directory: {copernicus_dir}")
                
        # 2) Load the dataset from Copernicus Marine Service
        logging.info(f"Loading dataset: {dataset_id}")
        
        # Prepare parameters dictionary
        params = {
            "dataset_id": dataset_id,
            "variables": variables,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "minimum_longitude": minimum_longitude,
            "maximum_longitude": maximum_longitude,
            "minimum_latitude": minimum_latitude,
            "maximum_latitude": maximum_latitude,
            "vertical_axis": vertical_axis
        }
        
        # Add optional parameters only if they're provided
        if minimum_depth is not None:
            params["minimum_depth"] = minimum_depth
        if maximum_depth is not None:
            params["maximum_depth"] = maximum_depth
            
        # Load the dataset
        dataset = copernicusmarine.open_dataset(**params)
        
        # Get some information about the dataset
        variables_info = ", ".join(list(dataset.data_vars))
        
        # 3) Save to NetCDF file
        nc_filename = _generate_descriptive_filename(
            variables, start_datetime, end_datetime, minimum_depth, maximum_depth
        )
        nc_path = os.path.join(copernicus_dir, nc_filename)
        
        logging.info(f"Saving to NetCDF file: {nc_path}")
        dataset.to_netcdf(nc_path)
        logging.info(f"Successfully saved to NetCDF: {nc_path}")
        
        # 4) Return success with file path
        return {
            "success": True,
            "output_path": nc_path,
            "dataset_id": dataset_id,
            "variables": variables_info,
            "message": "Copernicus Marine data downloaded successfully in NetCDF format"
        }
        
    except Exception as e:
        logging.error(f"Error in Copernicus Marine retrieval: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to retrieve Copernicus Marine data: {str(e)}"
        }

# Register the tool for the agent system
copernicus_marine_tool = StructuredTool.from_function(
    func=retrieve_copernicus_marine_data,
    name="retrieve_copernicus_marine_data",
    description="Retrieves oceanographic data from the Copernicus Marine Service for a given dataset ID, variables, time range, and spatial bounds.",
    args_schema=CopernicusMarineRetrievalArgs
)