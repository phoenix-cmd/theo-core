"""KnowledgePort — interface for structured knowledge operations.

Knowledge is distinct from memory. Memory stores experiences;
Knowledge stores structured understanding (facts, concepts, relations).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class KnowledgePort(ABC):
    """Abstract interface for knowledge graph operations."""

    @abstractmethod
    def add_fact(self, subject: str, predicate: str, obj: str, **metadata: Any) -> str:
        """Add a fact to the knowledge graph.

        Args:
            subject: The subject entity.
            predicate: The relationship type.
            obj: The object entity.
            **metadata: Additional metadata.

        Returns:
            The unique ID of the created fact.

        """

    @abstractmethod
    def query(
        self, subject: str | None = None, predicate: str | None = None
    ) -> list[dict[str, Any]]:
        """Query facts from the knowledge graph.

        Args:
            subject: Optional subject to filter by.
            predicate: Optional predicate to filter by.

        Returns:
            A list of matching fact dicts.

        """

    @abstractmethod
    def traverse(self, start: str, relation: str, depth: int = 1) -> list[dict[str, Any]]:
        """Traverse the knowledge graph from a starting node.

        Args:
            start: The starting entity.
            relation: The relationship type to follow.
            depth: Maximum traversal depth.

        Returns:
            A list of connected entities with metadata.

        """
