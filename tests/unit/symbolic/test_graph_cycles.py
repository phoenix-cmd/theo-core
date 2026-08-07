"""Unit tests for CycleDetector."""

from theo_core.symbolic._graph.graph import Graph
from theo_core.symbolic._graph.traversal import CycleDetector
from theo_core.symbolic._graph.types import EdgeKey
from theo_core.symbolic._primitives.identifiers import SymbolicId


class TestCycleDetector:
    def test_acyclic_graph(self) -> None:
        g: Graph[str, str] = Graph()
        n1 = SymbolicId.of("concept://a")
        n2 = SymbolicId.of("concept://b")
        n3 = SymbolicId.of("concept://c")
        g.add_node(n1, "A")
        g.add_node(n2, "B")
        g.add_node(n3, "C")
        g.add_edge(EdgeKey(n1, n2), "e1")
        g.add_edge(EdgeKey(n2, n3), "e2")

        assert not CycleDetector.has_cycle(g)

    def test_cyclic_graph(self) -> None:
        g: Graph[str, str] = Graph()
        n1 = SymbolicId.of("concept://a")
        n2 = SymbolicId.of("concept://b")
        n3 = SymbolicId.of("concept://c")
        g.add_node(n1, "A")
        g.add_node(n2, "B")
        g.add_node(n3, "C")
        g.add_edge(EdgeKey(n1, n2), "e1")
        g.add_edge(EdgeKey(n2, n3), "e2")
        g.add_edge(EdgeKey(n3, n1), "e3")  # Cycle

        assert CycleDetector.has_cycle(g)

    def test_self_loop_cycle(self) -> None:
        g: Graph[str, str] = Graph()
        n1 = SymbolicId.of("concept://a")
        g.add_node(n1, "A")
        g.add_edge(EdgeKey(n1, n1), "self_loop")

        assert CycleDetector.has_cycle(g)
