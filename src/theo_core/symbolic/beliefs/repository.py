"""Belief persistence protocols and in-memory repository implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from theo_core.symbolic._primitives.identifiers import SymbolicId
    from theo_core.symbolic.beliefs.graph import BeliefGraph


@runtime_checkable
class BeliefRepository(Protocol):
    """Abstract persistence protocol for BeliefGraph.

    Behavioral guarantees:
    - save() preserves graph integrity.
    - load() reconstructs identical data.
    - Operations MUST NOT mutate the supplied graph.
    - Loading the same graph multiple times yields equivalent objects.
    """

    def save(self, graph_id: SymbolicId, graph: BeliefGraph) -> None:
        """Persist a BeliefGraph."""
        ...

    def load(self, graph_id: SymbolicId) -> BeliefGraph | None:
        """Load a BeliefGraph by graph_id."""
        ...

    def exists(self, graph_id: SymbolicId) -> bool:
        """Check if graph_id exists in storage."""
        ...

    def delete(self, graph_id: SymbolicId) -> None:
        """Remove stored graph by graph_id."""
        ...


class InMemoryBeliefRepository:
    """In-memory implementation of BeliefRepository."""

    def __init__(self) -> None:
        """Initialize empty repository storage."""
        self._store: dict[str, BeliefGraph] = {}

    def save(self, graph_id: SymbolicId, graph: BeliefGraph) -> None:
        """Store reference to BeliefGraph."""
        self._store[graph_id.value] = graph

    def load(self, graph_id: SymbolicId) -> BeliefGraph | None:
        """Retrieve stored BeliefGraph by graph_id."""
        return self._store.get(graph_id.value)

    def exists(self, graph_id: SymbolicId) -> bool:
        """Check if graph_id exists."""
        return graph_id.value in self._store

    def delete(self, graph_id: SymbolicId) -> None:
        """Delete graph_id from storage."""
        self._store.pop(graph_id.value, None)
