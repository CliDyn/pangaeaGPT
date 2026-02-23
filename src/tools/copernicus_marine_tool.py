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
import sys
import streamlit as st
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from langchain_core.tools import StructuredTool

from ..utils.workspace import WorkspaceManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CopernicusMarineRetrievalArgs(BaseModel):
    """Arguments for the Copernicus Marine data retrieval tool."""
    dataset_id: Literal[
        # ========== MULTI-YEAR REANALYSIS (1993-2021) ==========
        # All 3D variables in one dataset
        "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        # ========== ANALYSIS/FORECAST (2021-present) ==========
        # 3D variables are in SEPARATE datasets by variable type:
        "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",  # temperature only
        "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",      # salinity only
        "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",     # currents (uo, vo)
        "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",         # 2D surface vars only
        # ========== BIOGEOCHEMISTRY ==========
        "cmems_mod_glo_bgc_my_0.25deg_P1D-m",
        "cmems_mod_glo_bgc-plankton_anfc_0.25deg_P1D-m",
        "cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m",
    ] = Field(
        description="The Copernicus Marine dataset ID. IMPORTANT - choose based on time period AND variable:\n\n"
        "FOR HISTORICAL DATA (1993-2021):\n"
        "  * 'cmems_mod_glo_phy_my_0.083deg_P1D-m': ALL physics vars (thetao, so, uo, vo, zos, etc.)\n\n"
        "FOR RECENT DATA (2021-present) - variables are SPLIT into separate datasets:\n"
        "  * 'cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m': Temperature ONLY (thetao)\n"
        "  * 'cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m': Salinity ONLY (so)\n"
        "  * 'cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m': Currents ONLY (uo, vo)\n"
        "  * 'cmems_mod_glo_phy_anfc_0.083deg_P1D-m': 2D surface vars (zos, mlotst, siconc, etc.)\n\n"
        "FOR BIOGEOCHEMISTRY:\n"
        "  * 'cmems_mod_glo_bgc_my_0.25deg_P1D-m': All BGC vars, 1993-2022\n"
        "  * 'cmems_mod_glo_bgc-plankton_anfc_0.25deg_P1D-m': Chlorophyll (chl), 2022-present\n"
        "  * 'cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m': Nutrients (no3, po4, si), 2022-present"
    )
    variables: List[str] = Field(
        description="Variable names. Must match the dataset:\n"
        "- cmems_mod_glo_phy_my (1993-2021): thetao, so, uo, vo, zos, mlotst, bottomT, siconc, sithick, usi, vsi\n"
        "- cmems_mod_glo_phy-thetao_anfc (2021+): thetao ONLY\n"
        "- cmems_mod_glo_phy-so_anfc (2021+): so ONLY\n"
        "- cmems_mod_glo_phy-cur_anfc (2021+): uo, vo ONLY\n"
        "- cmems_mod_glo_phy_anfc (2021+): zos, mlotst, siconc, sithick, usi, vsi (2D surface vars)\n"
        "- cmems_mod_glo_bgc_my: chl, no3, po4, si, nppv, o2\n"
        "- cmems_mod_glo_bgc-plankton_anfc: chl\n"
        "- cmems_mod_glo_bgc-nut_anfc: no3, po4, si"
    )
    start_datetime: str = Field(
        description="Start date in 'YYYY-MM-DD' format."
    )
    end_datetime: str = Field(
        description="End date in 'YYYY-MM-DD' format."
    )
    minimum_longitude: float = Field(
        description="Minimum longitude (-180 to 180)."
    )
    maximum_longitude: float = Field(
        description="Maximum longitude (-180 to 180)."
    )
    minimum_latitude: float = Field(
        description="Minimum latitude (-90 to 90)."
    )
    maximum_latitude: float = Field(
        description="Maximum latitude (-90 to 90)."
    )
    minimum_depth: Optional[float] = Field(
        None, description="Minimum depth in meters. IMPORTANT: First depth level is ~0.5m, not 0. Use 0.5 for surface."
    )
    maximum_depth: Optional[float] = Field(
        None, description="Maximum depth in meters. Use same value as minimum_depth for single depth level (e.g., 0.5 for surface)."
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
        
        # 1) Determine directory
        copernicus_dir = WorkspaceManager.get_data_dir(subfolder="copernicus_data")
        logging.info(f"Copernicus output directory: {copernicus_dir}")

        # Fix depth values: minimum valid depth is ~0.5m, not 0
        if minimum_depth is not None and minimum_depth < 0.5:
            logging.info(f"Adjusting minimum_depth from {minimum_depth} to 0.5 (first valid depth level)")
            minimum_depth = 0.5
        if maximum_depth is not None and maximum_depth < 0.5:
            logging.info(f"Adjusting maximum_depth from {maximum_depth} to 0.5 (first valid depth level)")
            maximum_depth = 0.5

        # 2) Generate output filename and download data
        nc_filename = _generate_descriptive_filename(
            variables, start_datetime, end_datetime, minimum_depth, maximum_depth
        )
        nc_path = os.path.join(copernicus_dir, nc_filename)

        # Use subset() to download directly to file - avoids zarr store issues
        # that occur with open_dataset() in newer zarr versions
        logging.info(f"Downloading data directly to: {nc_path}")

        subset_params = {
            "dataset_id": dataset_id,
            "variables": variables,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "minimum_longitude": minimum_longitude,
            "maximum_longitude": maximum_longitude,
            "minimum_latitude": minimum_latitude,
            "maximum_latitude": maximum_latitude,
            "output_filename": nc_filename,
            "output_directory": copernicus_dir,
            "overwrite": True,
        }

        # Add optional depth parameters
        if minimum_depth is not None:
            subset_params["minimum_depth"] = minimum_depth
        if maximum_depth is not None:
            subset_params["maximum_depth"] = maximum_depth

        # Download the data using subset()
        copernicusmarine.subset(**subset_params)

        logging.info(f"Successfully downloaded to: {nc_path}")

        # Read back to get variable info
        dataset = xr.open_dataset(nc_path)
        variables_info = ", ".join(list(dataset.data_vars))
        dataset.close()

        relative_path = os.path.join('copernicus_data', nc_filename).replace("\\", "/")
        
        # 4) Return success with file path
        return {
            "success": True,
            "output_path": relative_path, # <--- FIXED: Now returns a clean, usable path
            "dataset_id": dataset_id,
            "variables": variables_info,
            "message": f"Copernicus Marine data downloaded successfully in NetCDF format to {relative_path}"
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