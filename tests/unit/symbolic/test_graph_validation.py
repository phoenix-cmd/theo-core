"""Unit tests for GraphValidator."""

from theo_core.symbolic._graph.graph import Graph
from theo_core.symbolic._graph.types import EdgeKey
from theo_core.symbolic._graph.validation import GraphValidator
from theo_core.symbolic._primitives.errors import OrphanNodeError
from theo_core.symbolic._primitives.identifiers import SymbolicId


class TestGraphValidator:
    def test_valid_graph_returns_no_errors(self) -> None:
        g: Graph[str, str] = Graph()
        n1 = SymbolicId.of("concept://a")
        n2 = SymbolicId.of("concept://b")
        g.add_node(n1, "A")
        g.add_node(n2, "B")
        g.add_edge(EdgeKey(n1, n2), "e1")

        errors = GraphValidator.validate(g)
        assert len(errors) == 0

    def test_orphan_node_detection(self) -> None:
        g: Graph[str, str] = Graph()
        n1 = SymbolicId.of("concept://a")
        n2 = SymbolicId.of("concept://b")
        n3 = SymbolicId.of("concept://orphan")
        g.add_node(n1, "A")
        g.add_node(n2, "B")
        g.add_node(n3, "Orphan")
        g.add_edge(EdgeKey(n1, n2), "e1")

        errors = GraphValidator.validate(g, allow_orphans=False)
        assert len(errors) == 1
        assert isinstance(errors[0], OrphanNodeError)
