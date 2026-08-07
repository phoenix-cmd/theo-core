"""Unit tests for GraphTraversal algorithms (BFS, DFS, shortest path, ancestors, etc.)."""

from theo_core.symbolic._graph.graph import Graph
from theo_core.symbolic._graph.traversal import GraphTraversal
from theo_core.symbolic._graph.types import EdgeKey
from theo_core.symbolic._primitives.identifiers import SymbolicId


def _build_sample_graph() -> tuple[Graph[str, str], dict[str, SymbolicId]]:
    g: Graph[str, str] = Graph()
    nodes = {
        "a": SymbolicId.of("concept://a"),
        "b": SymbolicId.of("concept://b"),
        "c": SymbolicId.of("concept://c"),
        "d": SymbolicId.of("concept://d"),
        "e": SymbolicId.of("concept://e"),
    }
    for name, nid in nodes.items():
        g.add_node(nid, name.upper())

    # a -> b, a -> c, b -> d, c -> d, d -> e
    g.add_edge(EdgeKey(nodes["a"], nodes["b"], "rel"), "edge")
    g.add_edge(EdgeKey(nodes["a"], nodes["c"], "rel"), "edge")
    g.add_edge(EdgeKey(nodes["b"], nodes["d"], "rel"), "edge")
    g.add_edge(EdgeKey(nodes["c"], nodes["d"], "rel"), "edge")
    g.add_edge(EdgeKey(nodes["d"], nodes["e"], "rel"), "edge")

    return g, nodes


class TestGraphTraversal:
    def test_bfs_traversal(self) -> None:
        g, nodes = _build_sample_graph()
        visited = GraphTraversal.bfs(g, nodes["a"])
        assert [n.value for n in visited] == [
            "concept://a",
            "concept://b",
            "concept://c",
            "concept://d",
            "concept://e",
        ]

    def test_dfs_traversal(self) -> None:
        g, nodes = _build_sample_graph()
        visited = GraphTraversal.dfs(g, nodes["a"])
        assert visited[0] == nodes["a"]
        assert len(visited) == 5

    def test_depth_limited_search(self) -> None:
        g, nodes = _build_sample_graph()
        visited_d1 = GraphTraversal.depth_limited_search(g, nodes["a"], max_depth=1)
        assert [n.value for n in visited_d1] == [
            "concept://a",
            "concept://b",
            "concept://c",
        ]

    def test_ancestors_and_descendants(self) -> None:
        g, nodes = _build_sample_graph()
        anc = GraphTraversal.ancestors(g, nodes["d"])
        assert anc == {nodes["a"], nodes["b"], nodes["c"]}

        desc = GraphTraversal.descendants(g, nodes["a"])
        assert desc == {nodes["b"], nodes["c"], nodes["d"], nodes["e"]}

    def test_shortest_path(self) -> None:
        g, nodes = _build_sample_graph()
        path = GraphTraversal.shortest_path(g, nodes["a"], nodes["e"])
        assert path == [nodes["a"], nodes["b"], nodes["d"], nodes["e"]]
