"""RetrieverPort — interface for retrieving relevant information."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RetrieverPort(ABC):
    """Abstract interface for information retrieval.

    Retrieval spans memory, knowledge, and any other searchable store.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve entries relevant to the query.

        Args:
            query: The search query.
            top_k: Maximum number of results.
            filters: Optional filters to narrow results.

        Returns:
            A list of result dicts with content and metadata.

        """
