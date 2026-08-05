"""MemoryRetrievalEngine — memory search orchestrator using RetrievalStrategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from theo_core.memory.retrieval.strategies.keyword import KeywordRetrievalStrategy

if TYPE_CHECKING:
    from theo_core.domain.runtime.entities.memory_entry import MemoryEntry
    from theo_core.domain.runtime.entities.retrieved_memory import RetrievedMemory
    from theo_core.memory.retrieval.strategies.base import RetrievalStrategy


class MemoryRetrievalEngine:
    """Orchestrates memory search using the configured RetrievalStrategy."""

    def __init__(self, strategy: RetrievalStrategy | None = None) -> None:
        """Initialize retrieval engine with a strategy.

        Args:
            strategy: Optional RetrievalStrategy instance (defaults to KeywordRetrievalStrategy).

        """
        self._strategy = strategy or KeywordRetrievalStrategy()

    def search(
        self,
        query: str,
        entries: list[MemoryEntry],
        top_k: int = 5,
    ) -> list[RetrievedMemory]:
        """Execute memory search across active entries.

        Args:
            query: Search query text string.
            entries: List of active MemoryEntry candidates.
            top_k: Max results to return.

        Returns:
            List of scored RetrievedMemory objects.

        """
        return self._strategy.search(query, entries, top_k=top_k)

    @property
    def strategy_name(self) -> str:
        """Return the active strategy name."""
        return self._strategy.name
