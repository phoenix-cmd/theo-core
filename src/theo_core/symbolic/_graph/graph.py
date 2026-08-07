"""Generic directed graph data structure. Strictly structural — no cognitive semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from theo_core.symbolic._primitives.errors import (
    DanglingEdgeError,
    DuplicateIdError,
)
from theo_core.symbolic._primitives.ordering import sorted_items, sorted_keys

if TYPE_CHECKING:
    from theo_core.symbolic._graph.types import E, EdgeKey, N, NodeId


class Graph[N, E]:
    """Generic directed graph storage container.

    Stores nodes keyed by NodeId (SymbolicId) and directed edges keyed by EdgeKey.
    Guarantees deterministic iteration ordering (sorted by NodeId/EdgeKey string representation).

    This class contains ONLY data storage and retrieval methods. Algorithms like BFS,
    DFS, cycle detection, or graph validation live in dedicated separate modules.
    """

    def __init__(self) -> None:
        self._nodes: dict[NodeId, N] = {}
        self._edges: dict[EdgeKey, E] = {}

    @property
    def node_count(self) -> int:
        """Return total number of nodes in graph."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Return total number of edges in graph."""
        return len(self._edges)

    def add_node(self, node_id: NodeId, node: N, overwrite: bool = False) -> None:
        """Add a node to the graph.

        Args:
            node_id: SymbolicId identifier for node.
            node: The node payload.
            overwrite: If True, overwrite existing node. Default False.

        Raises:
            DuplicateIdError: If a node with node_id already exists and overwrite is False.

        """
        if not overwrite and node_id in self._nodes:
            msg = f"Node with ID {node_id.value!r} already exists in graph."
            raise DuplicateIdError(msg)
        self._nodes[node_id] = node

    def get_node(self, node_id: NodeId) -> N | None:
        """Retrieve a node payload by node_id."""
        return self._nodes.get(node_id)

    def has_node(self, node_id: NodeId) -> bool:
        """Check if node exists in graph."""
        return node_id in self._nodes

    def remove_node(self, node_id: NodeId) -> None:
        """Remove a node and all connected incoming/outgoing edges.

        Args:
            node_id: NodeId to remove.

        """
        if node_id not in self._nodes:
            return
        del self._nodes[node_id]

        # Remove edges referencing this node
        edges_to_remove = [
            ek for ek in self._edges if ek.source == node_id or ek.target == node_id
        ]
        for ek in edges_to_remove:
            del self._edges[ek]

    def add_edge(self, edge_key: EdgeKey, edge: E) -> None:
        """Add a directed edge to the graph.

        Args:
            edge_key: EdgeKey specifying source, target, and relation.
            edge: The edge payload.

        Raises:
            DanglingEdgeError: If source or target node does not exist in graph.

        """
        if edge_key.source not in self._nodes:
            msg = f"Source node {edge_key.source.value!r} does not exist in graph."
            raise DanglingEdgeError(msg)
        if edge_key.target not in self._nodes:
            msg = f"Target node {edge_key.target.value!r} does not exist in graph."
            raise DanglingEdgeError(msg)

        self._edges[edge_key] = edge

    def get_edge(self, edge_key: EdgeKey) -> E | None:
        """Retrieve edge payload by EdgeKey."""
        return self._edges.get(edge_key)

    def has_edge(self, edge_key: EdgeKey) -> bool:
        """Check if edge exists in graph."""
        return edge_key in self._edges

    def remove_edge(self, edge_key: EdgeKey) -> None:
        """Remove an edge from graph if present."""
        self._edges.pop(edge_key, None)

    def get_nodes(self) -> list[tuple[NodeId, N]]:
        """Return all (NodeId, node) pairs in deterministic sorted order."""
        return sorted_items(self._nodes)

    def get_node_ids(self) -> list[NodeId]:
        """Return all node IDs in deterministic sorted order."""
        return sorted_keys(self._nodes)

    def get_edges(self) -> list[tuple[EdgeKey, E]]:
        """Return all (EdgeKey, edge) pairs in deterministic sorted order."""
        return sorted(self._edges.items(), key=lambda item: str(item[0]))

    def get_edges_from(self, source_id: NodeId) -> list[tuple[EdgeKey, E]]:
        """Return outgoing edges from source_id in deterministic sorted order."""
        matching = [
            (ek, edge) for ek, edge in self._edges.items() if ek.source == source_id
        ]
        return sorted(matching, key=lambda item: str(item[0]))

    def get_edges_to(self, target_id: NodeId) -> list[tuple[EdgeKey, E]]:
        """Return incoming edges to target_id in deterministic sorted order."""
        matching = [
            (ek, edge) for ek, edge in self._edges.items() if ek.target == target_id
        ]
        return sorted(matching, key=lambda item: str(item[0]))

    def neighbors(self, node_id: NodeId) -> list[NodeId]:
        """Return adjacent outgoing neighbor node IDs in deterministic sorted order."""
        outgoing_targets = {ek.target for ek in self._edges if ek.source == node_id}
        return sorted(outgoing_targets, key=lambda nid: nid.value)

    def clear(self) -> None:
        """Clear all nodes and edges."""
        self._nodes.clear()
        self._edges.clear()

    def copy(self) -> Graph[N, E]:
        """Return a shallow structural copy of this graph.

        Node and edge payloads are immutable value objects, so copying the
        storage dictionaries is sufficient to produce an independent graph.

        Returns:
            A new Graph instance with the same nodes and edges.

        """
        new_graph = Graph[N, E]()
        new_graph._nodes = dict(self._nodes)
        new_graph._edges = dict(self._edges)
        return new_graph
