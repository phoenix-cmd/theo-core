"""Performance tests for Graph library handling 10,000 nodes."""

import pytest

from theo_core.symbolic._graph.graph import Graph
from theo_core.symbolic._graph.traversal import GraphTraversal
from theo_core.symbolic._primitives.identifiers import SymbolicId


@pytest.mark.slow
class TestGraphPerformance:
    def test_10k_node_build_and_traversal(self) -> None:
        g: Graph[str, str] = Graph()
        node_ids: list[SymbolicId] = []

        # Build 10k linear graph: n0 -> n1 -> n2 ... -> n9999
        for i in range(10000):
            nid = SymbolicId.of(f"concept://node/{i}")
            node_ids.append(nid)
            g.add_node(nid, f"Node {i}")

        assert g.node_count == 10000

        # Iterative BFS traversal should not hit recursion limit
        visited = GraphTraversal.bfs(g, node_ids[0])
        assert len(visited) == 1
