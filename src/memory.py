import os
import json
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

# --- Original CustomMemorySaver ---
class CustomMemorySaver(MemorySaver):
    def should_save(self, state: dict, key: str) -> bool:
        # Exclude 'messages' from being saved
        if key == 'messages':
            return False
        return True

# --- ERA5 Dataset Cache Memory ---
CACHE_FILE = "data/dataset_cache.json"

@dataclass
class DatasetInfo:
    path: str
    variable: str
    query_type: str
    start_date: str
    end_date: str
    lat_bounds: Tuple[float, float]
    lon_bounds: Tuple[float, float]
    file_size_bytes: int
    shape: Optional[Tuple[int, ...]] = None

class Memory:
    def __init__(self):
        self.cache_file = CACHE_FILE
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    # Convert dicts back to DatasetInfo
                    self.datasets = {k: DatasetInfo(**v) for k, v in data.items()}
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                self.datasets = {}
        else:
            self.datasets = {}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                # Convert DatasetInfo to dicts
                data = {k: asdict(v) for k, v in self.datasets.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get_dataset(self, path: str) -> Optional[DatasetInfo]:
        return self.datasets.get(path)

    def register_dataset(self, path, variable, query_type, start_date, end_date, lat_bounds, lon_bounds, file_size_bytes, shape=None):
        info = DatasetInfo(
            path=path,
            variable=variable,
            query_type=query_type,
            start_date=start_date,
            end_date=end_date,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            file_size_bytes=file_size_bytes,
            shape=shape
        )
        self.datasets[path] = info
        self._save_cache()
        logger.info(f"Registered dataset: {path}")

    def list_datasets(self) -> str:
        if not self.datasets:
            return "No datasets cached."
        
        lines = ["Cached Datasets:"]
        for path, info in self.datasets.items():
            lines.append(f"- {info.variable} ({info.query_type}): {info.start_date} to {info.end_date}, Path: {path}")
        return "\n".join(lines)

_memory_instance = None

def get_memory():
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = Memory()
    return _memory_instance