"""Unit tests for generic Graph container storage and retrieval operations."""

import pytest

from theo_core.symbolic._graph.graph import Graph
from theo_core.symbolic._graph.types import EdgeKey
from theo_core.symbolic._primitives.errors import DanglingEdgeError, DuplicateIdError
from theo_core.symbolic._primitives.identifiers import SymbolicId


class TestGraphCore:
    def test_add_and_get_node(self) -> None:
        g: Graph[str, str] = Graph()
        nid = SymbolicId.of("concept://a")
        g.add_node(nid, "Node A")

        assert g.node_count == 1
        assert g.has_node(nid)
        assert g.get_node(nid) == "Node A"

    def test_duplicate_node_id_raises(self) -> None:
        g: Graph[str, str] = Graph()
        nid = SymbolicId.of("concept://a")
        g.add_node(nid, "Node A")

        with pytest.raises(DuplicateIdError):
            g.add_node(nid, "Node A Duplicate")

    def test_add_and_get_edge(self) -> None:
        g: Graph[str, str] = Graph()
        n1 = SymbolicId.of("concept://a")
        n2 = SymbolicId.of("concept://b")
        g.add_node(n1, "A")
        g.add_node(n2, "B")

        ek = EdgeKey(source=n1, target=n2, relation="is_a")
        g.add_edge(ek, "Edge A->B")

        assert g.edge_count == 1
        assert g.has_edge(ek)
        assert g.get_edge(ek) == "Edge A->B"

    def test_dangling_edge_raises(self) -> None:
        g: Graph[str, str] = Graph()
        n1 = SymbolicId.of("concept://a")
        n2 = SymbolicId.of("concept://b")
        g.add_node(n1, "A")

        ek = EdgeKey(source=n1, target=n2, relation="is_a")
        with pytest.raises(DanglingEdgeError):
            g.add_edge(ek, "Edge")

    def test_remove_node_removes_connected_edges(self) -> None:
        g: Graph[str, str] = Graph()
        n1 = SymbolicId.of("concept://a")
        n2 = SymbolicId.of("concept://b")
        g.add_node(n1, "A")
        g.add_node(n2, "B")

        ek = EdgeKey(source=n1, target=n2, relation="is_a")
        g.add_edge(ek, "Edge")
        assert g.edge_count == 1

        g.remove_node(n1)
        assert g.node_count == 1
        assert g.edge_count == 0
        assert not g.has_node(n1)

    def test_deterministic_ordering(self) -> None:
        g: Graph[str, str] = Graph()
        n3 = SymbolicId.of("concept://c")
        n1 = SymbolicId.of("concept://a")
        n2 = SymbolicId.of("concept://b")

        g.add_node(n3, "C")
        g.add_node(n1, "A")
        g.add_node(n2, "B")

        node_ids = g.get_node_ids()
        assert [nid.value for nid in node_ids] == [
            "concept://a",
            "concept://b",
            "concept://c",
        ]
