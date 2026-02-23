# src/utils/session_manager.py

import threading
from typing import Dict, Any

class SessionManager:
    _sessions: Dict[str, Dict[str, Any]] = {}
    _lock = threading.Lock()

    @classmethod
    def get_session(cls, thread_id: str) -> Dict[str, Any]:
        with cls._lock:
            if thread_id not in cls._sessions:
                # Initialize default session state mirroring app.py
                cls._sessions[thread_id] = {
                    "thread_id": thread_id,
                    "messages_search": [],
                    "messages_data_agent": [],
                    "datasets_cache": {},
                    "datasets_info": None, # pandas DataFrame
                    "active_datasets": [],
                    "selected_datasets": set(),
                    "execution_history": [],
                    "plot_images": []
                }
            return cls._sessions[thread_id]

    @classmethod
    def update_session(cls, thread_id: str, key: str, value: Any):
        with cls._lock:
            if thread_id in cls._sessions:
                cls._sessions[thread_id][key] = value

session_manager = SessionManager()
