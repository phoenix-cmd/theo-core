"""Stateless graph traversal and cycle detection algorithms.

All algorithms include complexity contracts and are guaranteed to be deterministic and iterative.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from theo_core.symbolic._graph.graph import Graph
    from theo_core.symbolic._graph.types import E, EdgeKey, N, NodeId


class GraphTraversal:
    """Stateless graph traversal algorithms. All methods are static."""

    @staticmethod
    def bfs(
        graph: Graph[N, E],
        start_node: NodeId,
        edge_filter: Callable[[EdgeKey, E], bool] | None = None,
    ) -> list[NodeId]:
        """Breadth-First Search traversal starting from start_node.

        Complexity Contract:
            Time: O(V + E)
            Memory: O(V)
            Deterministic: YES

        Args:
            graph: The Graph to traverse.
            start_node: Starting NodeId.
            edge_filter: Optional predicate function to filter edges.

        Returns:
            List of visited NodeIds in BFS traversal order.

        """
        if not graph.has_node(start_node):
            return []

        visited: set[NodeId] = {start_node}
        queue: deque[NodeId] = deque([start_node])
        order: list[NodeId] = []

        while queue:
            current = queue.popleft()
            order.append(current)

            for ek, edge in graph.get_edges_from(current):
                if edge_filter is not None and not edge_filter(ek, edge):
                    continue
                neighbor = ek.target
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return order

    @staticmethod
    def dfs(
        graph: Graph[N, E],
        start_node: NodeId,
        edge_filter: Callable[[EdgeKey, E], bool] | None = None,
    ) -> list[NodeId]:
        """Depth-First Search traversal starting from start_node (iterative).

        Complexity Contract:
            Time: O(V + E)
            Memory: O(V)
            Deterministic: YES

        Args:
            graph: The Graph to traverse.
            start_node: Starting NodeId.
            edge_filter: Optional predicate function to filter edges.

        Returns:
            List of visited NodeIds in DFS traversal order.

        """
        if not graph.has_node(start_node):
            return []

        visited: set[NodeId] = set()
        stack: list[NodeId] = [start_node]
        order: list[NodeId] = []

        while stack:
            current = stack.pop()
            if current in visited:
                continue

            visited.add(current)
            order.append(current)

            # Get outgoing edges in reverse order so stack processing visits lowest key first
            edges = graph.get_edges_from(current)
            if edge_filter is not None:
                edges = [(ek, e) for ek, e in edges if edge_filter(ek, e)]

            for ek, _ in reversed(edges):
                if ek.target not in visited:
                    stack.append(ek.target)

        return order

    @staticmethod
    def depth_limited_search(
        graph: Graph[N, E],
        start_node: NodeId,
        max_depth: int,
        edge_filter: Callable[[EdgeKey, E], bool] | None = None,
    ) -> list[NodeId]:
        """Breadth-first traversal up to a specified maximum depth.

        Complexity Contract:
            Time: O(V + E)
            Memory: O(V)
            Deterministic: YES

        Args:
            graph: The Graph to traverse.
            start_node: Starting NodeId.
            max_depth: Maximum depth radius (0 returns only start_node).
            edge_filter: Optional predicate function to filter edges.

        Returns:
            List of visited NodeIds within max_depth.

        """
        if not graph.has_node(start_node) or max_depth < 0:
            return []

        visited: set[NodeId] = {start_node}
        queue: deque[tuple[NodeId, int]] = deque([(start_node, 0)])
        order: list[NodeId] = []

        while queue:
            current, depth = queue.popleft()
            order.append(current)

            if depth < max_depth:
                for ek, edge in graph.get_edges_from(current):
                    if edge_filter is not None and not edge_filter(ek, edge):
                        continue
                    neighbor = ek.target
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

        return order

    @staticmethod
    def ancestors(
        graph: Graph[N, E],
        node_id: NodeId,
        edge_filter: Callable[[EdgeKey, E], bool] | None = None,
    ) -> set[NodeId]:
        """Find all upstream ancestor nodes (nodes that can reach node_id via edges).

        Complexity Contract:
            Time: O(V + E)
            Memory: O(V)
            Deterministic: YES

        """
        if not graph.has_node(node_id):
            return set()

        ancestors_set: set[NodeId] = set()
        stack: list[NodeId] = [node_id]

        while stack:
            current = stack.pop()
            for ek, edge in graph.get_edges_to(current):
                if edge_filter is not None and not edge_filter(ek, edge):
                    continue
                parent = ek.source
                if parent not in ancestors_set:
                    ancestors_set.add(parent)
                    stack.append(parent)

        return ancestors_set

    @staticmethod
    def descendants(
        graph: Graph[N, E],
        node_id: NodeId,
        edge_filter: Callable[[EdgeKey, E], bool] | None = None,
    ) -> set[NodeId]:
        """Find all downstream descendant nodes reachable from node_id.

        Complexity Contract:
            Time: O(V + E)
            Memory: O(V)
            Deterministic: YES

        """
        if not graph.has_node(node_id):
            return set()

        visited = GraphTraversal.bfs(graph, node_id, edge_filter=edge_filter)
        return set(visited) - {node_id}

    @staticmethod
    def shortest_path(
        graph: Graph[N, E],
        start_node: NodeId,
        end_node: NodeId,
        edge_filter: Callable[[EdgeKey, E], bool] | None = None,
    ) -> list[NodeId] | None:
        """Find unweighted shortest path from start_node to end_node using BFS.

        Complexity Contract:
            Time: O(V + E)
            Memory: O(V)
            Deterministic: YES

        Returns:
            List of NodeIds representing the path from start to end inclusive,
            or None if unreachable.

        """
        if not graph.has_node(start_node) or not graph.has_node(end_node):
            return None

        if start_node == end_node:
            return [start_node]

        visited: set[NodeId] = {start_node}
        queue: deque[tuple[NodeId, list[NodeId]]] = deque([(start_node, [start_node])])

        while queue:
            current, path = queue.popleft()
            for ek, edge in graph.get_edges_from(current):
                if edge_filter is not None and not edge_filter(ek, edge):
                    continue
                neighbor = ek.target
                if neighbor == end_node:
                    return [*path, end_node]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, [*path, neighbor]))

        return None

    @staticmethod
    def expand_neighborhood(
        graph: Graph[N, E],
        start_node: NodeId,
        radius: int,
        edge_filter: Callable[[EdgeKey, E], bool] | None = None,
    ) -> set[NodeId]:
        """Expand neighborhood up to radius steps (bidirectional incoming/outgoing).

        Complexity Contract:
            Time: O(V + E)
            Memory: O(V)
            Deterministic: YES

        """
        if not graph.has_node(start_node) or radius < 0:
            return set()

        visited: set[NodeId] = {start_node}
        queue: deque[tuple[NodeId, int]] = deque([(start_node, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth < radius:
                # Outgoing neighbors
                for ek, edge in graph.get_edges_from(current):
                    if edge_filter is not None and not edge_filter(ek, edge):
                        continue
                    if ek.target not in visited:
                        visited.add(ek.target)
                        queue.append((ek.target, depth + 1))
                # Incoming neighbors
                for ek, edge in graph.get_edges_to(current):
                    if edge_filter is not None and not edge_filter(ek, edge):
                        continue
                    if ek.source not in visited:
                        visited.add(ek.source)
                        queue.append((ek.source, depth + 1))

        return visited


class CycleDetector:
    """Cycle detection algorithms for directed graphs.

    Uses Tarjan / Kahn / DFS coloring iteratively.
    """

    @staticmethod
    def has_cycle(graph: Graph[N, E]) -> bool:
        """Check if graph contains any directed cycle (iterative DFS coloring).

        Complexity Contract:
            Time: O(V + E)
            Memory: O(V)
            Deterministic: YES

        """
        # Node states: 0 = unvisited, 1 = visiting (in active path), 2 = visited
        state: dict[NodeId, int] = dict.fromkeys(graph.get_node_ids(), 0)

        for start_id in graph.get_node_ids():
            if state[start_id] != 0:
                continue

            # Stack entries: (node, child_index)
            stack: list[tuple[NodeId, int]] = [(start_id, 0)]
            state[start_id] = 1

            while stack:
                node, idx = stack[-1]
                edges = graph.get_edges_from(node)

                if idx < len(edges):
                    # Advance child index for current node
                    stack[-1] = (node, idx + 1)
                    target = edges[idx][0].target

                    if state[target] == 1:
                        return True
                    if state[target] == 0:
                        state[target] = 1
                        stack.append((target, 0))
                else:
                    # All children processed
                    state[node] = 2
                    stack.pop()

        return False
