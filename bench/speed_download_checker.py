#!/usr/bin/env python3
"""
speed_download_checker.py

Measure download+save time for ERA5 (as Zarr) and Copernicus Marine (as NetCDF) subsets,
and report resulting Zarr/NetCDF directory or file sizes—and show per-step timings.
"""

import os
import time
import logging
from pprint import pprint

# ———— SET UP "DELTA" LOGGING ————
class ElapsedFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self._last = time.time()
    def format(self, record):
        now = time.time()
        delta = now - self._last
        self._last = now
        record.delta = f"{delta:.3f}"
        return super().format(record)

handler = logging.StreamHandler()
handler.setFormatter(
    ElapsedFormatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s [ +%(delta)s s ]",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
)
root = logging.getLogger()
root.handlers = [handler]
root.setLevel(logging.INFO)

# — Add parent directory to Python path so we can import from src/ —
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.era5_retrieval_tool import retrieve_era5_data
from src.tools.copernicus_marine_tool import retrieve_copernicus_marine_data


def sizeof_file(path):
    """Return size of a file in MiB (float)."""
    if path and os.path.isfile(path):
        return os.path.getsize(path) / (1024**2)
    return 0.0


def sizeof_dir(path):
    """Return total size of directory (e.g., Zarr store) in MiB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024**2)


def test_era5():
    # month-long Fram Strait box: 74→81° N, –10→10° E
    params = {
        "variable_id": "2m_temperature",
        "start_date": "2020-01-01",
        "end_date":   "2020-02-01",
        "min_latitude": 78.0,
        "max_latitude": 80.0,
        "min_longitude": -5.0,
        "max_longitude": 5.0,
        "pressure_level": None
    }
    logging.info("=== ERA5 SPEED TEST START (Zarr only) ===")
    logging.info(f"Parameters: {params}")
    t0 = time.perf_counter()
    # retrieve & save as Zarr (retriever now writes only Zarr)
    result = retrieve_era5_data(**params)
    elapsed_total = time.perf_counter() - t0

    if result.get("success"):
        zarr_path = result["output_path_zarr"]
        logging.info(f"[ERA5] total elapsed: {elapsed_total:.1f} s")
        logging.info(f"Zarr store: {zarr_path} size: {sizeof_dir(zarr_path):.2f} MiB")
        print(f"\n[ERA5] total elapsed: {elapsed_total:.1f} s")
        print(f"  Zarr store → {zarr_path} ({sizeof_dir(zarr_path):.2f} MiB)")
    else:
        logging.error(f"ERA5 ERROR: {result.get('error') or result.get('message')}")

    logging.info("=== ERA5 SPEED TEST END ===")


def test_copernicus():
    # same month & box
    params = {
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "variables": ["thetao"],
        "start_datetime": "2020-01-01",
        "end_datetime":   "2020-01-05",
        "minimum_longitude": -5.0,
        "maximum_longitude":  5.0,
        "minimum_latitude":   74.0,
        "maximum_latitude":   81.0,
        "minimum_depth": None,
        "maximum_depth": None,
        "vertical_axis": "depth"
    }
    logging.info("=== COPERNICUS MARINE SPEED TEST START (NetCDF only) ===")
    logging.info(f"Parameters: {params}")
    t0 = time.perf_counter()
    # retrieve (saves NetCDF only now)
    result = retrieve_copernicus_marine_data(**params)
    elapsed_total = time.perf_counter() - t0

    if result.get("success"):
        nc = result["output_path"]  # Changed from "output_path_netcdf" to "output_path"
        logging.info(f"[Copernicus] total elapsed: {elapsed_total:.1f} s")
        logging.info(f"NetCDF file: {nc} size: {sizeof_file(nc):.2f} MiB")
        print(f"\n[Copernicus] total elapsed: {elapsed_total:.1f} s")
        print(f"  NetCDF → {nc} ({sizeof_file(nc):.2f} MiB)")
    else:
        logging.error(f"Copernicus ERROR: {result.get('error') or result.get('message')}")

    logging.info("=== COPERNICUS MARINE SPEED TEST END ===")


if __name__ == "__main__":
    test_era5()
    test_copernicus()