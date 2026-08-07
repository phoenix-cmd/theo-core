"""Generic graph persistence protocols and in-memory repository implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from theo_core.symbolic._graph.types import E, N

if TYPE_CHECKING:
    from theo_core.symbolic._graph.graph import Graph
    from theo_core.symbolic._graph.serialization import GraphLoader, GraphSerializer
    from theo_core.symbolic._primitives.identifiers import SymbolicId


@runtime_checkable
class GraphRepository(Protocol[N, E]):
    """Generic repository protocol for Graph[N, E].

    Behavioral guarantees:
    - save() preserves graph integrity.
    - load() reconstructs an identical graph.
    - Repository operations MUST NOT mutate the supplied graph.
    - Loading the same graph multiple times yields equivalent objects.
    """

    def save(self, graph_id: SymbolicId, graph: Graph[N, E]) -> None:
        """Persist a Graph instance.

        Args:
            graph_id: Unique identifier for the graph.
            graph: The Graph instance to persist.

        """
        ...

    def load(self, graph_id: SymbolicId) -> Graph[N, E] | None:
        """Load a Graph instance by graph_id.

        Args:
            graph_id: Unique identifier to look up.

        Returns:
            The reconstructed Graph if found, else None.

        """
        ...

    def exists(self, graph_id: SymbolicId) -> bool:
        """Check whether a graph exists in repository."""
        ...

    def delete(self, graph_id: SymbolicId) -> None:
        """Remove a stored graph."""
        ...


class InMemoryGraphRepository[N, E]:
    """In-memory GraphRepository implementation backed by serialized JSON snapshots.

    Stores JSON string snapshots to guarantee that loaded graphs are independent
    and operations do not mutate saved state.
    """

    def __init__(
        self,
        serializer: GraphSerializer[N, E],
        loader: GraphLoader[N, E],
    ) -> None:
        self.serializer = serializer
        self.loader = loader
        self._store: dict[str, str] = {}

    def save(self, graph_id: SymbolicId, graph: Graph[N, E]) -> None:
        """Serialize and store a snapshot of the graph."""
        snapshot_json = self.serializer.serialize(graph)
        self._store[graph_id.value] = snapshot_json

    def load(self, graph_id: SymbolicId) -> Graph[N, E] | None:
        """Deserialize and return a fresh Graph instance from snapshot."""
        snapshot_json = self._store.get(graph_id.value)
        if snapshot_json is None:
            return None
        return self.loader.deserialize(snapshot_json)

    def exists(self, graph_id: SymbolicId) -> bool:
        """Check if graph_id is present."""
        return graph_id.value in self._store

    def delete(self, graph_id: SymbolicId) -> None:
        """Remove graph_id from storage."""
        self._store.pop(graph_id.value, None)
