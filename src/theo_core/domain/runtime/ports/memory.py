"""MemoryStorePort — abstract domain port for memory engine implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from theo_core.domain.runtime.entities.memory_entry import MemoryEntry


class MemoryStorePort(ABC):
    """Abstract domain port for memory engines."""

    @abstractmethod
    def store_fact(
        self,
        key: str,
        value: Any,
        category: str = "semantic",
        source: str = "user_statement",
    ) -> MemoryEntry:
        """Store a fact entry."""

    @abstractmethod
    def get_fact(self, key: str) -> MemoryEntry | None:
        """Retrieve active MemoryEntry for a key."""

    @abstractmethod
    def get_all_active(self) -> list[MemoryEntry]:
        """Return all active MemoryEntry items."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve memory items matching query."""
