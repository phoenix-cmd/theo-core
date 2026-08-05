"""RetrievalStrategy — abstract strategy interface for memory search engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theo_core.domain.runtime.entities.memory_entry import MemoryEntry
    from theo_core.domain.runtime.entities.retrieved_memory import RetrievedMemory


class RetrievalStrategy(ABC):
    """Abstract interface for memory retrieval strategies.

    Strategies evaluate active memory candidates against query input and return
    scored RetrievedMemory results.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name."""

    @abstractmethod
    def search(
        self,
        query: str,
        entries: list[MemoryEntry],
        top_k: int = 5,
    ) -> list[RetrievedMemory]:
        """Search memory entries for query matches.

        Args:
            query: User query or percept content text string.
            entries: List of active MemoryEntry items.
            top_k: Maximum number of results to return.

        Returns:
            List of scored RetrievedMemory instances sorted deterministically.

        """
