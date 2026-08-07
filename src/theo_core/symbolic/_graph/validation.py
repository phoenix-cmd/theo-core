"""Structural graph validation layer.

Checks node IDs, dangling edges, orphan nodes, and schema integrity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from theo_core.symbolic._primitives.errors import (
    DanglingEdgeError,
    OrphanNodeError,
    ValidationError,
)

if TYPE_CHECKING:
    from theo_core.symbolic._graph.graph import Graph
    from theo_core.symbolic._graph.types import E, N, NodeId


class GraphValidator[N, E]:
    """Validates structural integrity of a Graph instance.

    All methods are deterministic and return list of ValidationError objects.
    """

    @staticmethod
    def validate(
        graph: Graph[N, E],
        allow_orphans: bool = True,
    ) -> list[ValidationError]:
        """Perform comprehensive structural validation on graph.

        Complexity Contract:
            Time: O(V + E)
            Memory: O(V)
            Deterministic: YES

        Args:
            graph: The Graph instance to validate.
            allow_orphans: If False, isolated nodes trigger OrphanNodeError.

        Returns:
            List of ValidationError instances found (empty list if valid).

        """
        errors: list[ValidationError] = []
        errors.extend(GraphValidator.validate_edges(graph))
        if not allow_orphans:
            errors.extend(GraphValidator.validate_no_orphans(graph))
        return errors

    @staticmethod
    def validate_edges(graph: Graph[N, E]) -> list[ValidationError]:
        """Verify that all edges reference existing nodes in graph."""
        errors: list[ValidationError] = []
        for ek, _ in graph.get_edges():
            if not graph.has_node(ek.source):
                msg = f"Edge source {ek.source.value!r} does not exist in graph."
                errors.append(DanglingEdgeError(msg))
            if not graph.has_node(ek.target):
                msg = f"Edge target {ek.target.value!r} does not exist in graph."
                errors.append(DanglingEdgeError(msg))
        return errors

    @staticmethod
    def validate_no_orphans(graph: Graph[N, E]) -> list[ValidationError]:
        """Verify that no node is isolated (has zero incoming and outgoing edges)."""
        errors: list[ValidationError] = []
        connected_nodes: set[NodeId] = set()

        for ek, _ in graph.get_edges():
            connected_nodes.add(ek.source)
            connected_nodes.add(ek.target)

        for nid in graph.get_node_ids():
            if nid not in connected_nodes:
                msg = f"Orphan node {nid.value!r} has zero connected edges."
                errors.append(OrphanNodeError(msg))

        return errors
