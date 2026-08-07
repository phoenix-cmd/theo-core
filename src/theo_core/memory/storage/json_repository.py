"""JSONMemoryRepository — persistent JSON file-backed storage repository for memory entries.

Provides cross-session persistence for THEO's append-only memory entries.
"""

from __future__ import annotations

import json
import os
from typing import Any

from theo_core.domain.runtime.entities.memory_entry import MemoryEntry


class JSONMemoryRepository:
    """JSON file-backed memory repository.

    Persists memory entries to disk as a JSON array. Never deletes entries —
    supports append-only and status-update operations.
    """

    def __init__(self, file_path: str = "data/memory_store.json") -> None:
        """Initialize repository and ensure directory exists.

        Args:
            file_path: Relative or absolute path to JSON storage file.

        """
        self._file_path = file_path
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)

    def load_all(self) -> list[MemoryEntry]:
        """Load all memory entries from the JSON repository file.

        Returns:
            List of MemoryEntry objects.

        """
        if not os.path.exists(self._file_path):
            return []

        try:
            with open(self._file_path, encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                data: list[dict[str, Any]] = json.loads(content)
                return [MemoryEntry(**item) for item in data]
        except (json.JSONDecodeError, TypeError, KeyError) as err:
            msg = f"Corrupted memory repository file at {self._file_path!r}: {err}"
            raise RuntimeError(msg) from err

    def save_all(self, entries: list[MemoryEntry]) -> None:
        """Save all memory entries to the JSON repository file.

        Args:
            entries: List of MemoryEntry objects to serialize.

        """
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
        data = [e.model_dump(mode="json") for e in entries]
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
