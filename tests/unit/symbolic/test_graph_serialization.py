"""Unit tests for GraphSerializer and GraphLoader with checksum verification."""

import pytest

from theo_core.symbolic._graph.graph import Graph
from theo_core.symbolic._graph.serialization import GraphLoader, GraphSerializer
from theo_core.symbolic._graph.types import EdgeKey
from theo_core.symbolic._primitives.errors import ChecksumMismatchError
from theo_core.symbolic._primitives.identifiers import SymbolicId


def node_to_dict(n: str) -> dict:
    return {"label": n}


def dict_to_node(d: dict) -> str:
    return d["label"]


def edge_to_dict(e: str) -> dict:
    return {"name": e}


def dict_to_edge(d: dict) -> str:
    return d["name"]


class TestGraphSerialization:
    def test_roundtrip_serialization(self) -> None:
        g: Graph[str, str] = Graph()
        n1 = SymbolicId.of("concept://a")
        n2 = SymbolicId.of("concept://b")
        g.add_node(n1, "Node A")
        g.add_node(n2, "Node B")
        g.add_edge(EdgeKey(n1, n2, "is_a"), "Edge A->B")

        serializer = GraphSerializer[str, str](
            graph_type="test",
            node_to_dict=node_to_dict,
            edge_to_dict=edge_to_dict,
        )
        loader = GraphLoader[str, str](
            expected_graph_type="test",
            dict_to_node=dict_to_node,
            dict_to_edge=dict_to_edge,
        )

        json_str = serializer.serialize(g)
        g_reconstructed = loader.deserialize(json_str)

        assert g_reconstructed.node_count == g.node_count
        assert g_reconstructed.edge_count == g.edge_count
        assert g_reconstructed.get_node(n1) == "Node A"
        assert g_reconstructed.get_edge(EdgeKey(n1, n2, "is_a")) == "Edge A->B"

    def test_corrupted_checksum_raises(self) -> None:
        g: Graph[str, str] = Graph()
        n1 = SymbolicId.of("concept://a")
        g.add_node(n1, "Node A")

        serializer = GraphSerializer[str, str](
            graph_type="test",
            node_to_dict=node_to_dict,
            edge_to_dict=edge_to_dict,
        )
        loader = GraphLoader[str, str](
            expected_graph_type="test",
            dict_to_node=dict_to_node,
            dict_to_edge=dict_to_edge,
        )

        json_str = serializer.serialize(g)
        # Corrupt checksum
        json_str_corrupt = json_str.replace(
            '"checksum": "', '"checksum": "00000000000000000000000000000000'
        )

        with pytest.raises(ChecksumMismatchError):
            loader.deserialize(json_str_corrupt)
