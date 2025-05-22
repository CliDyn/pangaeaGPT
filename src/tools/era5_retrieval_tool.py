# src/tools/era5_retrieval_tool.py
"""
src/tools/era5_retrieval_tool.py

ERA5 data retrieval tool for use in the visualization agent.
Retrieves climate data from the Google Cloud ARCO-ERA5 dataset.
Saves the retrieved data ONLY in Zarr format.
The Zarr filename is based on a hash of the request parameters.
Uses numcodecs for Zarr v3 compatibility.
"""

import os
import logging
import uuid
import xarray as xr
import pandas as pd
import numpy as np
import streamlit as st
from pydantic import BaseModel, Field
from typing import Optional, Literal
from langchain_core.tools import StructuredTool
import hashlib
import zarr
import shutil

# --- IMPORTS ---
try:
    import gcsfs
    from zarr.codecs import BloscCodec, GzipCodec # Reverted to numcodecs for Blosc and GZip classes
    from zarr.storage import LocalStore as DirectoryStore # Correct for Zarr v3+
    import blosc as blosc_clib

except ImportError as e:
    missing_module_message = str(e)
    if "gcsfs" in missing_module_message:
        errored_package = "gcsfs"
    elif "numcodecs" in missing_module_message:
        errored_package = "numcodecs"
    elif "zarr.storage" in missing_module_message: # Check if DirectoryStore import failed
        errored_package = "zarr.storage (problem with zarr installation or version)"
    elif "'blosc'" in missing_module_message or "No module named 'blosc'" in missing_module_message :
        errored_package = "blosc (Python wrapper for C-Blosc)"
    elif "zarr" in missing_module_message:
        errored_package = "zarr"
    else:
        errored_package = "A required library (gcsfs, zarr, numcodecs, or blosc)"

    install_command = "pip install --upgrade xarray zarr numcodecs gcsfs blosc streamlit pydantic langchain-core"
    detailed_message = (
        f"Error importing '{errored_package}': {missing_module_message}\n"
        f"One or more required libraries (gcsfs, zarr, numcodecs, blosc) are missing or could not be imported.\n"
        f"Please ensure all are installed, for example, by running: '{install_command}'.\n"
        f"If you specifically see an error related to 'blosc' or compression, "
        f"ensure the C-Blosc system library (e.g., 'libblosc-dev' on Debian/Ubuntu, or 'blosc' via Homebrew on macOS) is installed. "
        f"Then, try reinstalling the Python 'blosc' package: 'pip install --force-reinstall blosc'."
    )
    raise ImportError(detailed_message) from e

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ARCO_ERA5_MAIN_ZARR_STORE = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'

class ERA5RetrievalArgs(BaseModel):
    variable_id: Literal[
        "sea_surface_temperature", "surface_pressure", "total_cloud_cover", "total_precipitation",
        "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "2m_dewpoint_temperature",
        "geopotential", "specific_humidity", "temperature", "u_component_of_wind",
        "v_component_of_wind", "vertical_velocity"
    ] = Field(description="ERA5 variable to retrieve (must match Zarr store names).")
    start_date: str = Field(description="Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).")
    end_date: str = Field(description="End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).")
    min_latitude: float = Field(-90.0, description="Minimum latitude (–90 to 90).")
    max_latitude: float = Field(90.0, description="Maximum latitude (–90 to 90).")
    min_longitude: float = Field(0.0, description="Minimum longitude (0–360 or –180 to 360).")
    max_longitude: float = Field(359.75, description="Maximum longitude (0–360 or –180 to 360).")
    pressure_level: Optional[int] = Field(None, description="Pressure level in hPa for 3D variables.")

def _generate_descriptive_filename(
    variable_id: str, start_date: str, end_date: str,
    pressure_level: Optional[int] = None
) -> str:
    """
    Generate a descriptive filename based on request parameters.
    Format: {variable_id}_{start_date}_to_{end_date}[_level_{pressure}].zarr
    
    Example: 2m_temperature_2020_01_01_to_2020_02_01.zarr
    """
    # Clean variable_id - replace any special characters with underscores
    clean_variable = variable_id.replace('-', '_').replace('/', '_').replace(' ', '_')
    
    # Convert dates to underscore format (YYYY-MM-DD -> YYYY_MM_DD)
    clean_start = start_date.split()[0].replace('-', '_')  # Handle datetime strings, take date part
    clean_end = end_date.split()[0].replace('-', '_')      # Handle datetime strings, take date part
    
    # Build filename
    filename_parts = [clean_variable, clean_start, "to", clean_end]
    
    # Add pressure level if specified
    if pressure_level is not None:
        filename_parts.extend(["level", str(pressure_level)])
    
    # Join with underscores and add extension
    filename = "_".join(filename_parts) + ".zarr"
    
    return filename

def _generate_request_hash(
    variable_id: str, start_date: str, end_date: str,
    min_latitude: float, max_latitude: float,
    min_longitude: float, max_longitude: float,
    pressure_level: Optional[int]
) -> str:
    params_string = (
        f"{variable_id}-{start_date}-{end_date}-"
        f"{min_latitude:.2f}-{max_latitude:.2f}-"
        f"{min_longitude:.2f}-{max_longitude:.2f}-"
        f"{pressure_level if pressure_level is not None else 'None'}"
    )
    return hashlib.md5(params_string.encode('utf-8')).hexdigest()

def retrieve_era5_data(
    variable_id: str, start_date: str, end_date: str,
    min_latitude: float = -90.0, max_latitude: float = 90.0,
    min_longitude: float = 0.0, max_longitude: float = 359.75,
    pressure_level: Optional[int] = None
) -> dict:
    ds_arco = None
    zarr_group_obj = None
    directory_store_obj = None

    try:
        logging.info(f"ERA5 retrieval: var={variable_id}, time={start_date}→{end_date}, lat=[{min_latitude},{max_latitude}], lon=[{min_longitude},{max_longitude}], level={pressure_level}")

        main_dir = None
        if "streamlit" in sys.modules and hasattr(st, 'session_state') and \
           st.session_state and "active_datasets" in st.session_state and st.session_state.get("active_datasets"):
            try:
                doi = next(iter(st.session_state["active_datasets"]))
                cached = st.session_state["datasets_cache"].get(doi)
                if cached:
                    path = cached[0]
                    if isinstance(path, str) and os.path.isdir(path):
                        main_dir = os.path.dirname(os.path.abspath(path))
            except Exception as e_st:
                logging.warning(f"Could not determine main_dir from Streamlit session: {e_st}")

        if not main_dir:
            main_dir = os.path.join("tmp", "sandbox", uuid.uuid4().hex)
            logging.info(f"No active PANGAEA sandbox or not in Streamlit context. Created new general sandbox: {main_dir}")
        os.makedirs(main_dir, exist_ok=True)
        era5_specific_dir = os.path.join(main_dir, "era5_data")
        os.makedirs(era5_specific_dir, exist_ok=True)
        logging.info(f"ERA5 output directory: {era5_specific_dir}")

        logging.info(f"Opening ERA5 dataset: {ARCO_ERA5_MAIN_ZARR_STORE}")
        ds_arco = xr.open_zarr(ARCO_ERA5_MAIN_ZARR_STORE, consolidated=True, storage_options={'token': 'anon'})

        if variable_id not in ds_arco:
            raise ValueError(f"Variable '{variable_id}' not found. Available: {list(ds_arco.data_vars)}")

        var_data_from_arco = ds_arco[variable_id]
        logging.debug(f"Source encoding for '{variable_id}' from ARCO: {var_data_from_arco.encoding}")

        start_datetime_obj = pd.to_datetime(start_date)
        end_datetime_obj = pd.to_datetime(end_date)
        time_filtered_data = var_data_from_arco.sel(time=slice(start_datetime_obj, end_datetime_obj))

        lat_slice_coords = slice(max_latitude, min_latitude)
        lon_min_request_360 = min_longitude % 360
        lon_max_request_360 = max_longitude % 360

        if lon_min_request_360 > lon_max_request_360:
            part1 = time_filtered_data.sel(latitude=lat_slice_coords, longitude=slice(lon_min_request_360, 359.75))
            part2 = time_filtered_data.sel(latitude=lat_slice_coords, longitude=slice(0, lon_max_request_360))
            if part1.sizes.get('longitude', 0) > 0 and part2.sizes.get('longitude', 0) > 0:
                space_filtered_data = xr.concat([part1, part2], dim='longitude')
            elif part1.sizes.get('longitude', 0) > 0: space_filtered_data = part1
            elif part2.sizes.get('longitude', 0) > 0: space_filtered_data = part2
            else: space_filtered_data = time_filtered_data.sel(latitude=lat_slice_coords, longitude=slice(0,0))
        else:
            space_filtered_data = time_filtered_data.sel(
                latitude=lat_slice_coords,
                longitude=slice(lon_min_request_360, lon_max_request_360)
            )
        if pressure_level is not None and "level" in space_filtered_data.coords:
            space_filtered_data = space_filtered_data.sel(level=pressure_level)

        if not all(dim_size > 0 for dim_size in space_filtered_data.sizes.values()):
            msg = "Selected region/time period has zero size in one or more dimensions after subsetting."
            logging.warning(msg)
            return {"success": False, "error": msg, "message": f"Failed to retrieve ERA5 data: {msg}"}


        logging.info(f"Loading subsetted data into memory (shape: {space_filtered_data.shape})...")
        loaded_subset_data_array = space_filtered_data.load()
        logging.debug(f"Encoding of loaded_subset_data_array (after .load()): {loaded_subset_data_array.encoding}")

        clean_data_array = xr.DataArray(
            loaded_subset_data_array.data,
            coords=loaded_subset_data_array.coords,
            dims=loaded_subset_data_array.dims,
            name=variable_id,
            attrs=loaded_subset_data_array.attrs
        )
        logging.debug(f"Encoding of newly created clean_data_array for '{variable_id}': {clean_data_array.encoding}")
        final_subset_ds = xr.Dataset({variable_id: clean_data_array})

        compressor = None
        try:
            # Using Blosc from numcodecs
            compressor = BloscCodec(cname='lz4', clevel=5, shuffle=True)
            logging.info(f"Using numcodecs.Blosc: {compressor.get_config()}")
        except Exception as blosc_ex:
            logging.error(f"Could not configure Blosc from numcodecs: {blosc_ex}. Fallback to GZip from numcodecs.")
            try:
                compressor = GZip(level=5) # from numcodecs
                logging.info(f"Using numcodecs.GZip: {compressor.get_config()}")
            except Exception as gzip_ex:
                logging.error(f"Could not configure GZip from numcodecs either: {gzip_ex}. No compression will be used.")
                compressor = None

        fill_value = np.nan
        if not np.issubdtype(final_subset_ds[variable_id].dtype, np.floating):
            fill_value = None

        variable_specific_encoding_for_xarray = {
            "_FillValue": fill_value,
            "compressor": compressor, # xarray expects singular 'compressor'
            "filters": None,          # Explicitly no filters for Zarr v3 to avoid misinterpretation
        }

        output_encoding = {}
        for var_name_in_ds in final_subset_ds.variables:
            if var_name_in_ds == variable_id:
                output_encoding[var_name_in_ds] = variable_specific_encoding_for_xarray
            else: # For coordinates
                output_encoding[var_name_in_ds] = {"compressor": None, "filters": None, "_FillValue": None}

        request_hash_str = _generate_request_hash(
            variable_id, start_date, end_date, min_latitude, max_latitude,
            min_longitude, max_longitude, pressure_level
        )
        final_subset_ds.attrs.update({
            'title': f"ERA5 {variable_id} data subset",
            'description': f"Subset for time {start_date}-{end_date}, lat [{min_latitude},{max_latitude}], lon [{min_longitude},{max_longitude}], level {pressure_level or 'N/A'}.",
            'source_arco_era5': ARCO_ERA5_MAIN_ZARR_STORE,
            'retrieval_parameters_hash': request_hash_str
        })

        zarr_filename = _generate_descriptive_filename(
            variable_id, start_date, end_date, pressure_level
        )
        zarr_path = os.path.join(era5_specific_dir, zarr_filename)

        logging.info(f"Saving subset to Zarr store: {zarr_path} with encoding for {variable_id}: {output_encoding.get(variable_id)}")

        if os.path.exists(zarr_path):
            logging.warning(f"Zarr path {zarr_path} exists. Removing for a fresh Zarr v3 store creation.")
            shutil.rmtree(zarr_path)
        os.makedirs(zarr_path, exist_ok=True)

        #if zarr_group_obj.zarr_format != 3:
        #    raise RuntimeError(f"Failed to create a Zarr v3 store. Actual format: {zarr_group_obj.zarr_format}")

        final_subset_ds.to_zarr(
            store=zarr_path,  # Just use the path string
            mode='w',
            encoding=output_encoding,
            consolidated=True,
            zarr_format=3
        )

        logging.info(f"Successfully saved and consolidated Zarr store: {zarr_path}")

        return {"success": True, "output_path_zarr": zarr_path, "variable": variable_id, "message": f"ERA5 data saved to {zarr_path}"}

    except AttributeError as ae:
        logging.error(f"AttributeError in ERA5 retrieval: {ae}", exc_info=True)
        error_msg = str(ae)
        if "'module' object has no attribute 'DirectoryStore'" in error_msg:
            error_msg += " (Access DirectoryStore via zarr.storage.DirectoryStore for Zarr v3+)"
        # Check if the error is about Blosc not being found in zarr.codecs
        elif "cannot import name 'Blosc' from 'zarr.codecs'" in error_msg:
            error_msg += " (Blosc not found in zarr.codecs, ensure zarr & numcodecs are installed correctly and versions are compatible. Using numcodecs.Blosc instead.)"
        return {"success": False, "error": error_msg, "message": f"Failed to retrieve ERA5 data due to AttributeError or ImportError: {error_msg}"}

    except Exception as e:
        logging.error(f"Error in ERA5 retrieval: {e}", exc_info=True)
        error_msg = str(e)
        if "Expected a BytesBytesCodec" in error_msg or "must be an instance of Codec" in error_msg:
            error_msg += " (Zarr v3 codec error. Check codec instantiation (numcodecs) and ensure target store is Zarr v3.)"
        elif "Resource Exhausted" in error_msg or "Too many open files" in error_msg:
             error_msg += " (GCS access issue or system limits. Try again or check GCSFS config.)"
        elif "Unsupported type for store_like" in error_msg:
            error_msg += " (Issue with Zarr store path or type. Ensure gs:// URI is used directly with xr.open_zarr and storage_options.)"
        return {"success": False, "error": error_msg, "message": f"Failed to retrieve ERA5 data: {error_msg}"}
    finally:
        if ds_arco is not None:
            ds_arco.close()
            logging.info(f"Closed main ARCO ERA5 dataset: {ARCO_ERA5_MAIN_ZARR_STORE}")


import sys

era5_retrieval_tool = StructuredTool.from_function(
    func=retrieve_era5_data,
    name="retrieve_era5_data",
    description=(
        "Retrieves a subset of the ARCO-ERA5 Zarr climate reanalysis dataset "
        "for a given variable, time range, spatial bounds, and optional pressure level. "
        "Saves the data locally as a Zarr store (directory named with a hash of request parameters) "
        "within the 'era5_data' subdirectory of the current session's sandbox, and returns the path to this Zarr store."
    ),
    args_schema=ERA5RetrievalArgs
)