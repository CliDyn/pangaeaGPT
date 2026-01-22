import os
import sys
import logging
import shutil
import xarray as xr
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional, Literal
from langchain_core.tools import StructuredTool

from ..config import ARRAYLAKE_API_KEY

# --- IMPORTS & CONFIGURATION ---
try:
    import zarr
    from arraylake import Client
    try:
        import streamlit as st
    except ImportError:
        st = None
except ImportError as e:
    install_command = "pip install --upgrade xarray zarr arraylake pandas numpy pydantic langchain-core"
    raise ImportError(
        f"Required libraries missing. Please ensure arraylake is installed.\n"
        f"Try running: {install_command}"
    ) from e

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# EARTHMOVER (ARRAYLAKE) IMPLEMENTATION
# =============================================================================

# Variable Mapping: Maps old/friendly names to Earthmover short codes
VARIABLE_MAPPING = {
    # Temperature
    "sea_surface_temperature": "sst",
    "2m_temperature": "t2",
    "temperature": "t2",
    "skin_temperature": "skt",
    "dewpoint_temperature": "d2",
    
    # Wind
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "u_component_of_wind": "u10", 
    "v_component_of_wind": "v10",
    
    # Pressure
    "surface_pressure": "sp",
    "mean_sea_level_pressure": "mslp",
    
    # Clouds/Precip
    "total_cloud_cover": "tcc",
    "convective_precipitation": "cp",
    "large_scale_precipitation": "lsp",
    
    # Identity mappings (so short codes work)
    "t2": "t2", "sst": "sst", "mslp": "mslp", "u10": "u10", "v10": "v10",
    "sp": "sp", "tcc": "tcc", "cp": "cp", "lsp": "lsp", "sd": "sd"
}

class ERA5RetrievalArgs(BaseModel):
    query_type: Literal["spatial", "temporal"] = Field(
        description="CRITICAL OPTIMIZATION: Choose 'temporal' for TIME SERIES (fast for long history at specific points). Choose 'spatial' for MAPS (fast for large regions at specific times)."
    )
    variable_id: Literal[
        "t2", "sst", "mslp", "u10", "v10", "sp", "tcc", "cp", "lsp", "sd", "skt", "d2", "blh", "cape", "ssr", "ssrd", "tcw", "tcwv", "u100", "v100",
        "sea_surface_temperature", "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "total_cloud_cover", "surface_pressure"
    ] = Field(
        description="ERA5 variable to retrieve. Preferred short codes: 't2' (Air Temp), 'sst' (Sea Surface Temp), 'u10'/'v10' (Wind), 'mslp' (Pressure). Full names also accepted."
    )
    start_date: str = Field(description="Start date (YYYY-MM-DD). Data available 1975-2024.")
    end_date: str = Field(description="End date (YYYY-MM-DD).")
    min_latitude: float = Field(-90.0, description="Minimum latitude (-90 to 90).")
    max_latitude: float = Field(90.0, description="Maximum latitude (-90 to 90).")
    min_longitude: float = Field(0.0, description="Minimum longitude (0 to 360).")
    max_longitude: float = Field(359.75, description="Maximum longitude (0 to 360).")

def _generate_descriptive_filename(variable_id: str, query_type: str, start_date: str, end_date: str) -> str:
    """Generate a descriptive directory name for the Zarr store."""
    clean_var = variable_id.replace('_', '')
    clean_start = start_date.split()[0].replace('-', '')
    clean_end = end_date.split()[0].replace('-', '')
    # Changed extension to .zarr (it will be a directory)
    return f"era5_{clean_var}_{query_type}_{clean_start}_{clean_end}.zarr"

def retrieve_era5_data(
    query_type: str,
    variable_id: str, 
    start_date: str, 
    end_date: str,
    min_latitude: float = -90.0, 
    max_latitude: float = 90.0,
    min_longitude: float = 0.0, 
    max_longitude: float = 359.75
) -> dict:
    """
    Retrieves ERA5 Surface data from Earthmover (Arraylake).
    Optimized for either Spatial (Map) or Temporal (Time-series) queries.
    Saves as Zarr store (directory) to preserve original format efficiency.
    """
    local_zarr_path = None

    try:
        logging.info(f"🌍 Earthmover ERA5 Retrieval ({query_type}): {variable_id} | {start_date} to {end_date}")

        # 1. Validate API Key
        if not ARRAYLAKE_API_KEY or "..." in ARRAYLAKE_API_KEY:
             return {
                "success": False,
                "error": "Missing or Invalid Arraylake API Key.",
                "message": "Please set 'arraylake_api_key' in .streamlit/secrets.toml or environment variables."
            }

        # 2. Map Variable Name
        short_var = VARIABLE_MAPPING.get(variable_id.lower(), variable_id)
        logging.info(f"Mapped requested variable '{variable_id}' to '{short_var}'")

        # 3. Setup Sandbox Path
        main_dir = None
        if "streamlit" in sys.modules and hasattr(st, 'session_state'):
            thread_id = st.session_state.get("thread_id")
            if thread_id:
                main_dir = os.path.join("tmp", "sandbox", thread_id)
        
        if not main_dir:
            main_dir = os.path.join("tmp", "sandbox", "era5_earthmover")
        
        os.makedirs(main_dir, exist_ok=True)
        era5_dir = os.path.join(main_dir, "era5_data")
        os.makedirs(era5_dir, exist_ok=True)

        # 4. Generate Filename and Check Cache
        zarr_dirname = _generate_descriptive_filename(short_var, query_type, start_date, end_date)
        local_zarr_path = os.path.join(era5_dir, zarr_dirname)
        relative_path = os.path.join("era5_data", zarr_dirname) # For the agent to use

        if os.path.exists(local_zarr_path):
            logging.info(f"⚡ Cache hit: {local_zarr_path}")
            return {
                "success": True,
                "output_path": relative_path,
                "full_path": local_zarr_path,
                "variable": short_var,
                "format": "zarr",
                "message": f"Cached ERA5 data found at {relative_path}"
            }

        # 5. Connect to Earthmover
        logging.info("Connecting to Arraylake...")
        client = Client(token=ARRAYLAKE_API_KEY)
        repo_name = "earthmover-public/era5-surface-aws"
        repo = client.get_repo(repo_name)
        session = repo.readonly_session("main")

        # 6. Open Dataset based on Query Type
        ds = xr.open_dataset(
            session.store, 
            engine="zarr", 
            consolidated=False, 
            zarr_format=3, 
            chunks=None, # Let xarray handle dask chunks naturally
            group=query_type
        )

        if short_var not in ds:
            return {
                "success": False, 
                "error": f"Variable '{short_var}' not found in dataset.",
                "message": f"Available variables: {list(ds.data_vars)}"
            }

        # 7. Selection / Slicing
        # Handle Latitude slicing (decreasing order means slice must be Max -> Min)
        lat_slice = slice(max_latitude, min_latitude) 
        
        # Handle Longitude slicing
        req_min_lon = min_longitude % 360
        req_max_lon = max_longitude % 360
        
        if req_min_lon > req_max_lon:
             lon_slice = slice(req_min_lon, 359.75) 
             logging.warning(f"Longitude range crosses meridian. Clamping to {req_min_lon}-359.75.")
        else:
             lon_slice = slice(req_min_lon, req_max_lon)

        time_slice = slice(start_date, end_date)

        logging.info(f"Slicing: Lat({max_latitude}->{min_latitude}), Lon({req_min_lon}->{req_max_lon}), Time({start_date}->{end_date})")

        # Select and subset
        subset = ds[short_var].sel(
            time=time_slice,
            latitude=lat_slice,
            longitude=lon_slice
        )
        
        # 8. Download and Save to Zarr (Original Format)
        logging.info(f"Downloading data to {local_zarr_path} (Zarr format)...")
        
        # Convert to Dataset (required for to_zarr)
        ds_out = subset.to_dataset(name=short_var)
        
        # Clear encoding to prevent Zarr v3 conflicts from remote store
        for var in ds_out.variables:
            ds_out[var].encoding = {}

        # Remove directory if exists (to ensure clean write)
        if os.path.exists(local_zarr_path):
            shutil.rmtree(local_zarr_path)

        # Save as Zarr (Original format, optimized for chunking)
        ds_out.to_zarr(local_zarr_path, mode="w", consolidated=True, compute=True)
        logging.info("✅ Download complete.")

        return {
            "success": True,
            "output_path": relative_path,
            "full_path": local_zarr_path,
            "variable": short_var,
            "query_type": query_type,
            "format": "zarr",
            "message": f"ERA5 data ({query_type} optimized) retrieved and saved to {relative_path} (Zarr format)"
        }

    except Exception as e:
        logging.error(f"Error in ERA5 Earthmover retrieval: {e}", exc_info=True)
        # Cleanup partial data
        if local_zarr_path and os.path.exists(local_zarr_path):
            shutil.rmtree(local_zarr_path, ignore_errors=True)
        
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to retrieve ERA5 data: {str(e)}"
        }

era5_retrieval_tool = StructuredTool.from_function(
    func=retrieve_era5_data,
    name="retrieve_era5_data",
    description=(
        "Retrieves ERA5 Surface climate data from Earthmover (Arraylake). "
        "⚠️ CRITICAL: You MUST specify 'query_type' based on user intent. "
        "1. Use `query_type='temporal'` for TIME-SERIES graphs at specific locations (fast time slicing). "
        "2. Use `query_type='spatial'` for MAPS over regions (fast spatial slicing). "
        "Returns a Zarr directory path (use xr.open_dataset(path, engine='zarr') to open). "
        "Available vars: t2 (temp), sst, u10/v10 (wind), mslp (pressure), etc."
    ),
    args_schema=ERA5RetrievalArgs
)


# # src/tools/era5_retrieval_tool.py
# """
# src/tools/era5_retrieval_tool.py

# ERA5 data retrieval tool for use in the visualization agent.
# Retrieves climate data from the Google Cloud ARCO-ERA5 dataset.
# Saves the retrieved data ONLY in Zarr format.
# The Zarr filename is based on a hash of the request parameters.
# Optimized for memory efficiency using lazy loading (Dask) and streaming.
# """

# import os
# import sys
# import logging
# import uuid
# import hashlib
# import shutil
# import xarray as xr
# import pandas as pd
# import numpy as np
# from pydantic import BaseModel, Field
# from typing import Optional, Literal
# from langchain_core.tools import StructuredTool

# # --- IMPORTS & CONFIGURATION ---
# try:
#     import zarr
#     import gcsfs
#     # Optional: Check for Streamlit to support session state if available
#     try:
#         import streamlit as st
#     except ImportError:
#         st = None
# except ImportError as e:
#     install_command = "pip install --upgrade xarray zarr numcodecs gcsfs blosc pandas numpy pydantic langchain-core"
#     raise ImportError(
#         f"Required libraries missing. Please ensure zarr, gcsfs, and others are installed.\n"
#         f"Try running: {install_command}"
#     ) from e

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Target the Analysis-Ready (AR) Zarr store (Chunk-1 optimized)
# ARCO_ERA5_MAIN_ZARR_STORE = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'

# class ERA5RetrievalArgs(BaseModel):
#     variable_id: Literal[
#         "sea_surface_temperature", "surface_pressure", "total_cloud_cover", "total_precipitation",
#         "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "2m_dewpoint_temperature",
#         "geopotential", "specific_humidity", "temperature", "u_component_of_wind",
#         "v_component_of_wind", "vertical_velocity"
#     ] = Field(description="ERA5 variable to retrieve (must match Zarr store names).")
#     start_date: str = Field(description="Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).")
#     end_date: str = Field(description="End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).")
#     min_latitude: float = Field(-90.0, description="Minimum latitude (–90 to 90).")
#     max_latitude: float = Field(90.0, description="Maximum latitude (–90 to 90).")
#     min_longitude: float = Field(0.0, description="Minimum longitude (0–360 or –180 to 360).")
#     max_longitude: float = Field(359.75, description="Maximum longitude (0–360 or –180 to 360).")
#     pressure_level: Optional[int] = Field(None, description="Pressure level in hPa for 3D variables.")

# def _generate_descriptive_filename(
#     variable_id: str, start_date: str, end_date: str,
#     pressure_level: Optional[int] = None
# ) -> str:
#     """
#     Generate a descriptive filename based on request parameters.
#     """
#     clean_variable = variable_id.replace('-', '_').replace('/', '_').replace(' ', '_')
#     clean_start = start_date.split()[0].replace('-', '_')
#     clean_end = end_date.split()[0].replace('-', '_')
    
#     filename_parts = [clean_variable, clean_start, "to", clean_end]
#     if pressure_level is not None:
#         filename_parts.extend(["level", str(pressure_level)])
    
#     filename = "_".join(filename_parts) + ".zarr"
#     return filename

# def _generate_request_hash(
#     variable_id: str, start_date: str, end_date: str,
#     min_latitude: float, max_latitude: float,
#     min_longitude: float, max_longitude: float,
#     pressure_level: Optional[int]
# ) -> str:
#     params_string = (
#         f"{variable_id}-{start_date}-{end_date}-"
#         f"{min_latitude:.2f}-{max_latitude:.2f}-"
#         f"{min_longitude:.2f}-{max_longitude:.2f}-"
#         f"{pressure_level if pressure_level is not None else 'None'}"
#     )
#     return hashlib.md5(params_string.encode('utf-8')).hexdigest()

# def retrieve_era5_data(
#     variable_id: str, start_date: str, end_date: str,
#     min_latitude: float = -90.0, max_latitude: float = 90.0,
#     min_longitude: float = 0.0, max_longitude: float = 359.75,
#     pressure_level: Optional[int] = None
# ) -> dict:
#     ds_arco = None
#     zarr_path = None

#     try:
#         logging.info(f"ERA5 retrieval: var={variable_id}, time={start_date}→{end_date}, "
#                      f"lat=[{min_latitude},{max_latitude}], lon=[{min_longitude},{max_longitude}], "
#                      f"level={pressure_level}")

#         # --- 1. Sandbox / Path Logic ---
#         main_dir = None
#         if "streamlit" in sys.modules and st is not None and hasattr(st, 'session_state'):
#             try:
#                 thread_id = st.session_state.get("thread_id")
#                 if thread_id:
#                     main_dir = os.path.join("tmp", "sandbox", thread_id)
#                     logging.info(f"Found session thread_id. Using persistent sandbox: {main_dir}")
#             except Exception:
#                 pass
        
#         if not main_dir:
#             main_dir = os.path.join("tmp", "sandbox", "era5_data")
#             logging.info(f"Using general sandbox: {main_dir}")

#         os.makedirs(main_dir, exist_ok=True)
#         era5_specific_dir = os.path.join(main_dir, "era5_data")
#         os.makedirs(era5_specific_dir, exist_ok=True)

#         # Check cache
#         zarr_filename = _generate_descriptive_filename(variable_id, start_date, end_date, pressure_level)
#         zarr_path = os.path.join(era5_specific_dir, zarr_filename)
#         relative_zarr_path = os.path.join(os.path.basename(era5_specific_dir), zarr_filename)

#         if os.path.exists(zarr_path):
#              logging.info(f"Data already cached at {zarr_path}")
#              return {
#                 "success": True, 
#                 "output_path_zarr": relative_zarr_path, 
#                 "full_path": zarr_path,
#                 "variable": variable_id, 
#                 "message": f"Cached ERA5 data found at {relative_zarr_path}"
#             }

#         # --- 2. Open Dataset (Lazy / Dask) ---
#         logging.info(f"Opening ERA5 dataset: {ARCO_ERA5_MAIN_ZARR_STORE}")
        
#         # CRITICAL OPTIMIZATION: chunks={'time': 24}
#         # This prevents loading the whole dataset into RAM and creates an efficient download graph.
#         ds_arco = xr.open_zarr(
#             ARCO_ERA5_MAIN_ZARR_STORE, 
#             chunks={'time': 24}, 
#             consolidated=True, 
#             storage_options={'token': 'anon'}
#         )

#         if variable_id not in ds_arco:
#             raise ValueError(f"Variable '{variable_id}' not found. Available: {list(ds_arco.data_vars)}")

#         var_data_from_arco = ds_arco[variable_id]

#         # --- 3. Lazy Subsetting ---
#         start_datetime_obj = pd.to_datetime(start_date)
#         end_datetime_obj = pd.to_datetime(end_date)
#         time_filtered_data = var_data_from_arco.sel(time=slice(start_datetime_obj, end_datetime_obj))

#         lat_slice_coords = slice(max_latitude, min_latitude)
#         lon_min_request_360 = min_longitude % 360
#         lon_max_request_360 = max_longitude % 360

#         # Anti-Meridian / Date Line Logic
#         if lon_min_request_360 > lon_max_request_360:
#             part1 = time_filtered_data.sel(latitude=lat_slice_coords, longitude=slice(lon_min_request_360, 359.75))
#             part2 = time_filtered_data.sel(latitude=lat_slice_coords, longitude=slice(0, lon_max_request_360))
            
#             if part1.sizes.get('longitude', 0) > 0 and part2.sizes.get('longitude', 0) > 0:
#                 space_filtered_data = xr.concat([part1, part2], dim='longitude')
#             elif part1.sizes.get('longitude', 0) > 0: 
#                 space_filtered_data = part1
#             elif part2.sizes.get('longitude', 0) > 0: 
#                 space_filtered_data = part2
#             else: 
#                 space_filtered_data = time_filtered_data.sel(latitude=lat_slice_coords, longitude=slice(0,0))
#         else:
#             space_filtered_data = time_filtered_data.sel(
#                 latitude=lat_slice_coords,
#                 longitude=slice(lon_min_request_360, lon_max_request_360)
#             )

#         if pressure_level is not None and "level" in space_filtered_data.coords:
#             space_filtered_data = space_filtered_data.sel(level=pressure_level)

#         if not all(dim_size > 0 for dim_size in space_filtered_data.sizes.values()):
#             msg = "Selected region/time period has zero size."
#             logging.warning(msg)
#             return {"success": False, "error": msg, "message": f"Failed: {msg}"}

#         # --- 4. Prepare Output Dataset ---
#         final_subset_ds = space_filtered_data.to_dataset(name=variable_id)

#         # Clear encoding (fixes Zarr v3 codec conflicts)
#         for var in final_subset_ds.variables:
#             final_subset_ds[var].encoding = {}

#         # RE-CHUNK for safe streaming (24h blocks are manageable for Mac RAM)
#         final_subset_ds = final_subset_ds.chunk({'time': 24, 'latitude': -1, 'longitude': -1})

#         # Metadata
#         request_hash_str = _generate_request_hash(
#             variable_id, start_date, end_date, min_latitude, max_latitude,
#             min_longitude, max_longitude, pressure_level
#         )
#         final_subset_ds.attrs.update({
#             'title': f"ERA5 {variable_id} data subset",
#             'source_arco_era5': ARCO_ERA5_MAIN_ZARR_STORE,
#             'retrieval_parameters_hash': request_hash_str
#         })

#         if os.path.exists(zarr_path):
#             logging.warning(f"Zarr path {zarr_path} exists. Removing for freshness.")
#             shutil.rmtree(zarr_path)

#         # --- 5. Streaming Write (NO .load()) ---
#         logging.info(f"Streaming data to Zarr store: {zarr_path}")
        
#         # compute=True triggers the download/write graph chunk-by-chunk
#         final_subset_ds.to_zarr(
#             store=zarr_path,
#             mode='w',
#             consolidated=True,
#             compute=True
#         )

#         logging.info(f"Successfully saved: {zarr_path}")

#         return {
#             "success": True, 
#             "output_path_zarr": relative_zarr_path, 
#             "full_path": zarr_path,
#             "variable": variable_id, 
#             "message": f"ERA5 data saved to {relative_zarr_path}"
#         }
    
#     except Exception as e:
#         logging.error(f"Error in ERA5 retrieval: {e}", exc_info=True)
#         error_msg = str(e)
        
#         if "Resource Exhausted" in error_msg or "Too many open files" in error_msg:
#              error_msg += " (GCS access issue or system limits. Try again.)"
#         elif "Unsupported type for store_like" in error_msg:
#             error_msg += " (Issue with Zarr store path. Ensure gs:// URI is used.)"
        
#         # Cleanup partial data on failure
#         if zarr_path and os.path.exists(zarr_path):
#              shutil.rmtree(zarr_path, ignore_errors=True)

#         return {"success": False, "error": error_msg, "message": f"Failed: {error_msg}"}
    
#     finally:
#         if ds_arco is not None:
#             ds_arco.close()

# era5_retrieval_tool = StructuredTool.from_function(
#     func=retrieve_era5_data,
#     name="retrieve_era5_data",
#     description=(
#         "Retrieves a subset of the ARCO-ERA5 Zarr climate reanalysis dataset. "
#         "Saves the data locally as a Zarr store. "
#         "Uses efficient streaming to prevent memory overflow on large files."
#     ),
#     args_schema=ERA5RetrievalArgs
# )   