"""
ERA5 Data Retrieval Tool (Enhanced)
====================================
Cloud-optimized data retrieval from Earthmover's ERA5 archive.

Features:
- Intelligent caching with memory integration
- Retry logic with exponential backoff
- Progress tracking
- Rich variable metadata
- Validation and error handling
"""

import os
import sys
import time
import shutil
import logging
from pathlib import Path
from typing import Literal, Optional, Tuple
from datetime import datetime, timedelta

import xarray as xr
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import StructuredTool

from src.config import (
    ARRAYLAKE_API_KEY,
    get_variable_info, get_short_name, list_available_variables,
    ERA5_VARIABLES, GEOGRAPHIC_REGIONS, format_file_size
)
from src.memory import get_memory
from src.utils.workspace import WorkspaceManager

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# ARGUMENT SCHEMA
# ============================================================================

class ERA5RetrievalArgs(BaseModel):
    """Arguments for ERA5 data retrieval."""

    query_type: Literal["spatial", "temporal"] = Field(
        description=(
            "CRITICAL for performance optimization:\n"
            "- 'temporal': For TIME SERIES analysis - long time periods, focused geographic area\n"
            "- 'spatial': For SPATIAL MAPS - large geographic areas, short time periods"
        )
    )

    variable_id: str = Field(
        description=(
            "ERA5 variable to retrieve. Common options:\n"
            "- sst: Sea Surface Temperature (K)\n"
            "- t2: 2m Air Temperature (K)\n"
            "- u10, v10: 10m Wind Components (m/s)\n"
            "- mslp: Mean Sea Level Pressure (Pa)\n"
            "- tcc: Total Cloud Cover (0-1)\n"
            "- tp: Total Precipitation (m)"
        )
    )

    start_date: str = Field(
        description="Start date in YYYY-MM-DD format (e.g., '2020-01-01')"
    )

    end_date: str = Field(
        description="End date in YYYY-MM-DD format (e.g., '2020-12-31')"
    )

    min_latitude: float = Field(
        default=-90.0,
        ge=-90.0, le=90.0,
        description="Minimum latitude (southern bound, -90 to 90)"
    )

    max_latitude: float = Field(
        default=90.0,
        ge=-90.0, le=90.0,
        description="Maximum latitude (northern bound, -90 to 90)"
    )

    min_longitude: float = Field(
        default=0.0,
        ge=0.0, le=360.0,
        description="Minimum longitude (western bound, 0 to 360, East is positive)"
    )

    max_longitude: float = Field(
        default=359.75,
        ge=0.0, le=360.0,
        description="Maximum longitude (eastern bound, 0 to 360)"
    )

    region: Optional[str] = Field(
        default=None,
        description=(
            "Optional predefined region (overrides lat/lon if specified):\n"
            "north_atlantic, north_pacific, california_coast, mediterranean,\n"
            "gulf_of_mexico, caribbean, nino34, arctic, antarctic, global"
        )
    )

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}")
        return v

    @field_validator('variable_id')
    @classmethod
    def validate_variable(cls, v: str) -> str:
        short_name = get_short_name(v)
        if short_name not in ['sst', 't2', 'u10', 'v10', 'sp', 'mslp', 'tcc', 'cp', 'lsp', 'tp']:
            logger.warning(f"Variable '{v}' may not be available. Will attempt anyway.")
        return v


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_filename(variable: str, query_type: str, start: str, end: str) -> str:
    """Generate a descriptive filename for the dataset."""
    clean_var = variable.replace('_', '')
    clean_start = start.replace('-', '')
    clean_end = end.replace('-', '')
    return f"era5_{clean_var}_{query_type}_{clean_start}_{clean_end}.zarr"


def get_bounds_from_region(region: str) -> Optional[Tuple[float, float, float, float]]:
    """Get lat/lon bounds from a named region."""
    if region and region.lower() in GEOGRAPHIC_REGIONS:
        r = GEOGRAPHIC_REGIONS[region.lower()]
        return (r["min_lat"], r["max_lat"], r["min_lon"], r["max_lon"])
    return None


def estimate_download_size(
    start_date: str, end_date: str,
    lat_range: float, lon_range: float,
    query_type: str
) -> float:
    """Estimate download size in MB."""
    # ERA5 resolution: 0.25 degrees
    n_lat = int(lat_range / 0.25)
    n_lon = int(lon_range / 0.25)

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    if query_type == 'temporal':
        # Hourly data
        n_times = int((end - start).total_seconds() / 3600)
    else:
        # Daily data (spatial is usually aggregated)
        n_times = (end - start).days + 1

    # Estimate: 4 bytes per float32 value, plus overhead
    bytes_estimate = n_lat * n_lon * n_times * 4 * 1.2  # 20% overhead
    return bytes_estimate / (1024 * 1024)  # MB


# ============================================================================
# MAIN RETRIEVAL FUNCTION
# ============================================================================

def retrieve_era5_data(
    query_type: str,
    variable_id: str,
    start_date: str,
    end_date: str,
    min_latitude: float = -90.0,
    max_latitude: float = 90.0,
    min_longitude: float = 0.0,
    max_longitude: float = 359.75,
    region: Optional[str] = None
) -> str:
    """
    Retrieve ERA5 reanalysis data from Earthmover's cloud-optimized archive.

    Downloads data to ./data/ directory and registers it in memory for caching.

    Returns:
        Success message with file path, or error message.
    """
    memory = get_memory()

    # Get API key from config (which loads from Streamlit secrets or environment)
    api_key = ARRAYLAKE_API_KEY
    if not api_key:
        return "Error: ARRAYLAKE_API_KEY not found. Please set it in .streamlit/secrets.toml under [general] or in .env file."

    # Check icechunk is available
    try:
        import icechunk
    except ImportError:
        return (
            "Error: The 'icechunk' library is required but not installed.\n"
            "Please run: pip install icechunk"
        )

    # Apply region bounds if specified
    if region:
        bounds = get_bounds_from_region(region)
        if bounds:
            min_latitude, max_latitude, min_longitude, max_longitude = bounds
            logger.info(f"Using region '{region}': lat=[{min_latitude}, {max_latitude}], lon=[{min_longitude}, {max_longitude}]")
        else:
            logger.warning(f"Unknown region '{region}', using provided coordinates")

    # Resolve variable name
    short_var = get_short_name(variable_id)
    var_info = get_variable_info(variable_id)

    # Setup paths - use WorkspaceManager for sandbox-aware paths
    output_dir = Path(WorkspaceManager.get_data_dir("era5_data"))

    filename = generate_filename(short_var, query_type, start_date, end_date)
    local_path = str(output_dir / filename)
    relative_path = f"era5_data/{filename}"  # Relative to sandbox root

    # Check cache first
    if os.path.exists(local_path):
        existing = memory.get_dataset(local_path)
        if existing:
            logger.info(f"Cache hit: {local_path}")
            return (
                f"CACHE HIT - Data already downloaded\n"
                f"  Variable: {short_var} ({var_info.long_name if var_info else 'Unknown'})\n"
                f"  Period: {existing.start_date} to {existing.end_date}\n"
                f"  Path: {local_path}\n"
                f"  Relative path: {relative_path}\n\n"
                f"Load with: ds = xr.open_dataset('{relative_path}', engine='zarr')"
            )
        else:
            # File exists but not registered, register it
            memory.register_dataset(
                path=local_path,
                variable=short_var,
                query_type=query_type,
                start_date=start_date,
                end_date=end_date,
                lat_bounds=(min_latitude, max_latitude),
                lon_bounds=(min_longitude, max_longitude),
                file_size_bytes=sum(f.stat().st_size for f in Path(local_path).rglob('*') if f.is_file())
            )
            return (
                f"CACHE HIT - Found existing data\n"
                f"  Variable: {short_var}\n"
                f"  Path: {local_path}\n"
                f"  Relative path: {relative_path}\n\n"
                f"Load with: ds = xr.open_dataset('{relative_path}', engine='zarr')"
            )

    # Estimate download size
    lat_range = max_latitude - min_latitude
    lon_range = (max_longitude - min_longitude) % 360
    estimated_mb = estimate_download_size(start_date, end_date, lat_range, lon_range, query_type)

    logger.info(f"Estimated download size: {estimated_mb:.1f} MB")

    # Connect and download with retry logic
    max_retries = 3
    retry_delay = 2.0

    for attempt in range(max_retries):
        try:
            from arraylake import Client

            # Progress messages
            print(f"\n{'='*60}")
            print(f"DOWNLOADING ERA5 DATA")
            print(f"{'='*60}")
            print(f"  Variable: {short_var}" + (f" ({var_info.long_name})" if var_info else ""))
            print(f"  Period: {start_date} to {end_date}")
            print(f"  Region: lat=[{min_latitude:.2f}, {max_latitude:.2f}], lon=[{min_longitude:.2f}, {max_longitude:.2f}]")
            print(f"  Query Type: {query_type}")
            print(f"  Estimated Size: ~{estimated_mb:.1f} MB")
            print(f"{'='*60}")

            # Connect to Earthmover
            print("Connecting to Earthmover Arraylake...")
            client = Client(token=api_key)
            repo = client.get_repo("earthmover-public/era5-surface-aws")
            session = repo.readonly_session("main")
            print("Connected successfully!")

            # Open dataset with Zarr/Icechunk
            print(f"Opening {query_type} dataset...")
            ds = xr.open_dataset(
                session.store,
                engine="zarr",
                consolidated=False,
                zarr_format=3,  # Required for Icechunk
                chunks=None,
                group=query_type
            )

            # Validate variable exists
            if short_var not in ds:
                available = list(ds.data_vars)
                return (
                    f"Error: Variable '{short_var}' not found in dataset.\n"
                    f"Available variables: {', '.join(available)}\n\n"
                    f"Variable reference:\n{list_available_variables()}"
                )

            # ERA5 latitude is stored 90 -> -90 (descending)
            lat_slice = slice(max_latitude, min_latitude)

            # Handle longitude wrapping (ERA5 uses 0-360)
            req_min = min_longitude % 360
            req_max = max_longitude % 360

            if req_min > req_max:
                # Crosses prime meridian - for now just take the eastern part
                lon_slice = slice(req_min, 359.75)
                logger.warning("Region crosses prime meridian - taking eastern portion only")
            else:
                lon_slice = slice(req_min, req_max)

            # Subset the data
            print("Subsetting data...")
            subset = ds[short_var].sel(
                time=slice(start_date, end_date),
                latitude=lat_slice,
                longitude=lon_slice
            )

            # Convert to dataset
            print("Preparing data for download...")
            ds_out = subset.to_dataset(name=short_var)

            # Clear encoding for clean serialization
            for var in ds_out.variables:
                ds_out[var].encoding = {}

            # Add metadata
            ds_out.attrs['source'] = 'ERA5 Reanalysis via Earthmover Arraylake'
            ds_out.attrs['download_date'] = datetime.now().isoformat()
            ds_out.attrs['query_type'] = query_type
            if var_info:
                ds_out[short_var].attrs['long_name'] = var_info.long_name
                ds_out[short_var].attrs['units'] = var_info.units

            # Clean up existing file
            if os.path.exists(local_path):
                shutil.rmtree(local_path)

            # Save to Zarr
            print(f"Downloading and saving to {local_path}...")
            start_time = time.time()
            ds_out.to_zarr(local_path, mode="w", consolidated=True, compute=True)
            download_time = time.time() - start_time

            # Get actual file size
            file_size = sum(f.stat().st_size for f in Path(local_path).rglob('*') if f.is_file())

            # Register in memory
            shape = tuple(ds_out[short_var].shape)
            memory.register_dataset(
                path=local_path,
                variable=short_var,
                query_type=query_type,
                start_date=start_date,
                end_date=end_date,
                lat_bounds=(min_latitude, max_latitude),
                lon_bounds=(min_longitude, max_longitude),
                file_size_bytes=file_size,
                shape=shape
            )

            # Success message
            print(f"{'='*60}")
            print(f"DOWNLOAD COMPLETE")
            print(f"{'='*60}")

            result = (
                f"SUCCESS - Data downloaded\n"
                f"{'='*50}\n"
                f"  Variable: {short_var}"
            )
            if var_info:
                result += f" ({var_info.long_name})"
            result += (
                f"\n  Units: {var_info.units if var_info else 'Unknown'}\n"
                f"  Period: {start_date} to {end_date}\n"
                f"  Shape: {shape}\n"
                f"  Size: {format_file_size(file_size)}\n"
                f"  Time: {download_time:.1f}s\n"
                f"  Path: {local_path}\n"
                f"  Relative path: {relative_path}\n"
                f"{'='*50}\n\n"
                f"Load with:\n"
                f"  ds = xr.open_dataset('{relative_path}', engine='zarr')\n"
                f"  print(ds)\n"
                f"  data = ds['{short_var}']"
            )

            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {error_msg}")

            # Clean up partial download
            if os.path.exists(local_path):
                shutil.rmtree(local_path, ignore_errors=True)

            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                return (
                    f"Error: Failed after {max_retries} attempts.\n"
                    f"Last error: {error_msg}\n\n"
                    f"Troubleshooting:\n"
                    f"1. Check your ARRAYLAKE_API_KEY in .env\n"
                    f"2. Verify internet connection\n"
                    f"3. Try a smaller date range or region\n"
                    f"4. Check if variable '{short_var}' is available"
                )

    return "Error: Unexpected failure in retrieval logic."


# ============================================================================
# UTILITY FUNCTION: LIST CACHED DATA
# ============================================================================

def list_cached_data() -> str:
    """List all cached datasets."""
    memory = get_memory()
    return memory.list_datasets()


# ============================================================================
# LANGCHAIN TOOL CREATION
# ============================================================================

era5_retrieval_tool = StructuredTool.from_function(
    func=retrieve_era5_data,
    name="retrieve_era5_data",
    description=(
        "Retrieves ERA5 climate reanalysis data from Earthmover's cloud-optimized archive.\n\n"
        "USAGE:\n"
        "- Use query_type='temporal' for time series (long time, small area)\n"
        "- Use query_type='spatial' for maps (large area, short time)\n\n"
        "VARIABLES: sst (Sea Surface Temp), t2 (2m Temp), u10/v10 (Wind), "
        "mslp (Pressure), tcc (Cloud Cover), tp (Precipitation)\n\n"
        "REGIONS: north_atlantic, north_pacific, california_coast, mediterranean, "
        "gulf_of_mexico, nino34, arctic, antarctic, global\n\n"
        "Returns the file path. Load with: xr.open_dataset('PATH', engine='zarr')"
    ),
    args_schema=ERA5RetrievalArgs
)


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    # Test the tool
    from dotenv import load_dotenv
    load_dotenv()

    print("Testing ERA5 retrieval...")
    result = retrieve_era5_data(
        query_type="temporal",
        variable_id="sst",
        start_date="2023-01-01",
        end_date="2023-01-07",
        region="california_coast"
    )
    print(result)