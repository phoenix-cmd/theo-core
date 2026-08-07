"""ThoughtGraph — domain graph wrapper enforcing DAG semantics for thoughts."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from theo_core.symbolic._graph.graph import Graph
from theo_core.symbolic._graph.traversal import CycleDetector, GraphTraversal
from theo_core.symbolic._graph.types import EdgeKey
from theo_core.symbolic._primitives.errors import CycleDetectedError
from theo_core.symbolic.thoughts.models import Thought, ThoughtEdge, ThoughtId, ThoughtRelation

if TYPE_CHECKING:
    from theo_core.symbolic.beliefs.models import BeliefId


class ThoughtGraph:
    """Domain DAG graph wrapper managing Thought nodes and ThoughtEdge relationships.

    Enforces Directed Acyclic Graph (DAG) semantics. Attempts to add an edge that forms
    a cycle will raise CycleDetectedError.
    """

    def __init__(self) -> None:
        """Initialize empty ThoughtGraph instance."""
        self._graph: Graph[Thought, ThoughtEdge] = Graph()

    @classmethod
    def from_raw_graph(cls, raw: Graph[Thought, ThoughtEdge]) -> ThoughtGraph:
        """Wrap a reconstructed generic Graph into a ThoughtGraph.

        Args:
            raw: A deserialized Graph[Thought, ThoughtEdge].

        Returns:
            A ThoughtGraph backed by the given raw graph.

        """
        graph = cls()
        graph._graph = raw
        return graph

    @property
    def node_count(self) -> int:
        """Total number of thoughts in graph."""
        return self._graph.node_count

    @property
    def edge_count(self) -> int:
        """Total number of edges in graph."""
        return self._graph.edge_count

    @property
    def raw_graph(self) -> Graph[Thought, ThoughtEdge]:
        """Access underlying generic Graph instance."""
        return self._graph

    def add_thought(self, thought: Thought) -> None:
        """Add a Thought node to graph."""
        self._graph.add_node(thought.id.to_symbolic_id(), thought)

    def get_thought(self, thought_id: ThoughtId) -> Thought | None:
        """Look up a Thought by ThoughtId."""
        return self._graph.get_node(thought_id.to_symbolic_id())

    def has_thought(self, thought_id: ThoughtId) -> bool:
        """Check if thought exists in graph."""
        return self._graph.has_node(thought_id.to_symbolic_id())

    def remove_thought(self, thought_id: ThoughtId) -> None:
        """Remove a thought and connected edges."""
        self._graph.remove_node(thought_id.to_symbolic_id())

    def add_edge(self, edge: ThoughtEdge) -> None:
        """Add a ThoughtEdge to graph, verifying DAG invariant.

        Raises:
            CycleDetectedError: If adding the edge creates a cycle.

        """
        ek = EdgeKey(
            source=edge.source.to_symbolic_id(),
            target=edge.target.to_symbolic_id(),
            relation=edge.relation.value,
        )

        # Temporarily add edge to test for cycle
        self._graph.add_edge(ek, edge)
        if CycleDetector.has_cycle(self._graph):
            self._graph.remove_edge(ek)
            msg = f"Adding edge {ek} creates a cycle in the ThoughtGraph DAG."
            raise CycleDetectedError(msg)

    def get_edge(
        self, source: ThoughtId, target: ThoughtId, relation: ThoughtRelation
    ) -> ThoughtEdge | None:
        """Look up a ThoughtEdge."""
        ek = EdgeKey(
            source=source.to_symbolic_id(),
            target=target.to_symbolic_id(),
            relation=relation.value,
        )
        return self._graph.get_edge(ek)

    def get_thoughts(self) -> list[Thought]:
        """Return all thoughts in deterministic sorted order."""
        return [node for _, node in self._graph.get_nodes()]

    def get_edges(self) -> list[ThoughtEdge]:
        """Return all thought edges in deterministic sorted order."""
        return [edge for _, edge in self._graph.get_edges()]

    def topological_sort(self) -> list[Thought]:
        """Return thoughts in topological sorting order (Kahn's algorithm).

        Complexity Contract:
            Time: O(V + E)
            Memory: O(V)
            Deterministic: YES

        """
        in_degree: dict[ThoughtId, int] = {
            ThoughtId(value=nid.value): 0 for nid in self._graph.get_node_ids()
        }

        for ek, _ in self._graph.get_edges():
            tgt_id = ThoughtId(value=ek.target.value)
            in_degree[tgt_id] = in_degree.get(tgt_id, 0) + 1

        queue: deque[ThoughtId] = deque(
            sorted([tid for tid, deg in in_degree.items() if deg == 0], key=lambda x: x.value)
        )
        sorted_thoughts: list[Thought] = []

        while queue:
            current_id = queue.popleft()
            thought = self.get_thought(current_id)
            if thought is not None:
                sorted_thoughts.append(thought)

            edges = self._graph.get_edges_from(current_id.to_symbolic_id())
            for ek, _ in sorted(edges, key=lambda item: str(item[0])):
                target_id = ThoughtId(value=ek.target.value)
                in_degree[target_id] -= 1
                if in_degree[target_id] == 0:
                    queue.append(target_id)

        return sorted_thoughts

    def get_consumed_beliefs(self, thought_id: ThoughtId) -> set[BeliefId]:
        """Return all BeliefIds consumed by thought_id."""
        thought = self.get_thought(thought_id)
        if thought is None:
            return set()
        return set(thought.consumed_beliefs)

    def get_reasoning_chain(self, thought_id: ThoughtId) -> list[Thought]:
        """Return the sequence of thoughts in the DAG leading to thought_id."""
        ancestor_ids = GraphTraversal.ancestors(self._graph, thought_id.to_symbolic_id())
        all_chain_ids = ancestor_ids | {thought_id.to_symbolic_id()}

        # Filter topological sort by chain members
        full_topo = self.topological_sort()
        return [t for t in full_topo if t.id.to_symbolic_id() in all_chain_ids]

    def copy(self) -> ThoughtGraph:
        """Return an independent shallow copy of this ThoughtGraph."""
        new_graph = ThoughtGraph()
        new_graph._graph = self._graph.copy()
        return new_graph
