"""Unit tests for InMemoryGraphRepository."""

from theo_core.symbolic._graph.graph import Graph
from theo_core.symbolic._graph.repository import InMemoryGraphRepository
from theo_core.symbolic._graph.serialization import GraphLoader, GraphSerializer
from theo_core.symbolic._primitives.identifiers import SymbolicId


def node_to_dict(n: str) -> dict:
    return {"label": n}


def dict_to_node(d: dict) -> str:
    return d["label"]


def edge_to_dict(e: str) -> dict:
    return {"name": e}


def dict_to_edge(d: dict) -> str:
    return d["name"]


class TestInMemoryGraphRepository:
    def test_save_and_load_integrity(self) -> None:
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
        repo = InMemoryGraphRepository[str, str](serializer, loader)

        g: Graph[str, str] = Graph()
        n1 = SymbolicId.of("concept://a")
        g.add_node(n1, "Node A")

        gid = SymbolicId.of("concept://my_graph")
        repo.save(gid, g)

        assert repo.exists(gid)

        g_loaded1 = repo.load(gid)
        g_loaded2 = repo.load(gid)

        assert g_loaded1 is not None
        assert g_loaded2 is not None
        assert g_loaded1.node_count == 1
        assert g_loaded1 is not g_loaded2  # Loaded instances are independent
