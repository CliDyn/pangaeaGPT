# src/utils/workspace.py
"""
Централизованный менеджер рабочих директорий и путей (Sandbox Manager).
"""

import os
import logging
from pathlib import Path
from typing import Optional

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WorkspaceManager:
    """
    Централизованный менеджер путей и рабочих директорий (Sandbox).
    Единственный источник правды для путей типа 'tmp/sandbox/...'.
    """

    # Базовая директория. В Docker/Cloud можно переопределить через ENV.
    # Используем абсолютный путь для надежности.
    BASE_ROOT = os.environ.get(
        "PANGAEA_SANDBOX_ROOT",
        os.path.abspath(os.path.join(os.getcwd(), "tmp", "sandbox"))
    )

    # Стандартные подпапки
    RESULTS_DIR = "results"

    @staticmethod
    def get_thread_id() -> str:
        """
        Надежно получает текущий thread_id (ID сессии).
        Работает и в Streamlit, и в CLI/тестах.
        """
        # 1. Попробовать получить из Streamlit session_state
        try:
            import streamlit as st
            if hasattr(st, 'session_state'):
                if "thread_id" in st.session_state:
                    return st.session_state.get("thread_id", "default_session")
        except (ImportError, AttributeError):
            pass

        # 2. Если это CLI или тесты
        return os.environ.get("PANGAEA_THREAD_ID", "default_cli_session")

    @classmethod
    def get_sandbox_path(cls, thread_id: str = None) -> str:
        """Возвращает абсолютный путь к корню песочницы для текущей сессии."""
        if not thread_id:
            thread_id = cls.get_thread_id()

        path = os.path.join(cls.BASE_ROOT, thread_id)
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def get_results_dir(cls, thread_id: str = None) -> str:
        """Возвращает путь к папке results (куда сохранять картинки)."""
        sandbox = cls.get_sandbox_path(thread_id)
        results_path = os.path.join(sandbox, cls.RESULTS_DIR)
        os.makedirs(results_path, exist_ok=True)
        return results_path

    @classmethod
    def get_data_dir(cls, subfolder: str, thread_id: str = None) -> str:
        """
        Возвращает путь к специфической подпапке данных (например, 'era5_data').
        Создает её, если нет.
        """
        sandbox = cls.get_sandbox_path(thread_id)
        path = os.path.join(sandbox, subfolder)
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def resolve_path(cls, relative_path: str, thread_id: str = None) -> Optional[str]:
        """
        Безопасно превращает относительный путь (results/plot.png) в абсолютный.
        Защищает от Path Traversal (выхода за пределы песочницы).
        """
        sandbox = cls.get_sandbox_path(thread_id)
        sandbox_path = Path(sandbox).resolve()

        # Очищаем путь от ../ и ведущих слешей
        clean_rel = relative_path.lstrip(os.sep).replace("..", "")
        full_path = (sandbox_path / clean_rel).resolve()

        # Проверяем, что путь остался внутри песочницы
        if full_path.is_relative_to(sandbox_path):
            return str(full_path)

        logging.warning(f"Security Warning: Attempted path traversal outside sandbox: {relative_path}")
        return None
