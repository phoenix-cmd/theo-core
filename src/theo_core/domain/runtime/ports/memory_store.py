"""MemoryStorePort — interface for memory storage and retrieval.

Abstracts all memory operations including storage, retrieval, search,
and deletion. Implementations may use vector databases, file stores,
or in-memory structures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryStorePort(ABC):
    """Abstract interface for memory storage operations.

    Transport-agnostic: local dicts today, distributed vector DBs tomorrow.
    """

    @abstractmethod
    def store(self, key: str, value: Any, **metadata: Any) -> None:
        """Store a memory entry.

        Args:
            key: Unique key identifying this memory.
            value: The content to store.
            **metadata: Additional metadata to attach to the entry.

        """

    @abstractmethod
    def retrieve(self, key: str) -> Any | None:
        """Retrieve a memory entry by key.

        Args:
            key: The unique key of the memory entry.

        Returns:
            The stored value, or None if not found.

        """

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for memories matching a query.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            A list of matching memory entries with metadata.

        """

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a memory entry by key.

        Args:
            key: The unique key of the memory entry.

        Returns:
            True if the entry was deleted, False if not found.

        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check whether a memory entry exists.

        Args:
            key: The unique key to check.

        Returns:
            True if the entry exists.

        """
