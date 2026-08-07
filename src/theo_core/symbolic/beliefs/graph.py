"""BeliefGraph — domain graph wrapper around generic Graph[Belief, BeliefEdge]."""

from __future__ import annotations

from decimal import Decimal

from theo_core.symbolic._graph.graph import Graph
from theo_core.symbolic._graph.types import EdgeKey
from theo_core.symbolic.beliefs.models import Belief, BeliefEdge, BeliefId, BeliefRelation


class BeliefGraph:
    """Domain graph wrapper managing Belief nodes and BeliefEdge relationships.

    Delegates structural storage to generic Graph[Belief, BeliefEdge].
    """

    def __init__(self) -> None:
        """Initialize empty BeliefGraph instance."""
        self._graph: Graph[Belief, BeliefEdge] = Graph()

    @property
    def node_count(self) -> int:
        """Total number of beliefs in graph."""
        return self._graph.node_count

    @property
    def edge_count(self) -> int:
        """Total number of edges in graph."""
        return self._graph.edge_count

    @property
    def raw_graph(self) -> Graph[Belief, BeliefEdge]:
        """Access underlying generic Graph instance."""
        return self._graph

    def add_belief(self, belief: Belief) -> None:
        """Add a Belief node to graph."""
        self._graph.add_node(belief.id.to_symbolic_id(), belief)

    def get_belief(self, belief_id: BeliefId) -> Belief | None:
        """Look up a Belief by BeliefId."""
        return self._graph.get_node(belief_id.to_symbolic_id())

    def has_belief(self, belief_id: BeliefId) -> bool:
        """Check if belief exists in graph."""
        return self._graph.has_node(belief_id.to_symbolic_id())

    def remove_belief(self, belief_id: BeliefId) -> None:
        """Remove a belief and connected edges."""
        self._graph.remove_node(belief_id.to_symbolic_id())

    def add_edge(self, edge: BeliefEdge) -> None:
        """Add a BeliefEdge to graph."""
        ek = EdgeKey(
            source=edge.source.to_symbolic_id(),
            target=edge.target.to_symbolic_id(),
            relation=edge.relation.value,
        )
        self._graph.add_edge(ek, edge)

    def get_edge(
        self, source: BeliefId, target: BeliefId, relation: BeliefRelation
    ) -> BeliefEdge | None:
        """Look up a BeliefEdge."""
        ek = EdgeKey(
            source=source.to_symbolic_id(),
            target=target.to_symbolic_id(),
            relation=relation.value,
        )
        return self._graph.get_edge(ek)

    def get_beliefs(self) -> list[Belief]:
        """Return all beliefs in deterministic sorted order."""
        return [node for _, node in self._graph.get_nodes()]

    def get_active_beliefs(self, min_confidence: Decimal = Decimal("0.1")) -> list[Belief]:
        """Return beliefs with confidence greater than or equal to threshold."""
        return [b for b in self.get_beliefs() if b.confidence >= min_confidence]

    def get_edges(self) -> list[BeliefEdge]:
        """Return all belief edges in deterministic sorted order."""
        return [edge for _, edge in self._graph.get_edges()]

    def get_supporting_beliefs(self, belief_id: BeliefId) -> set[BeliefId]:
        """Return beliefs that support belief_id (via incoming SUPPORTS edges)."""
        edges = self._graph.get_edges_to(belief_id.to_symbolic_id())
        return {
            BeliefId(value=ek.source.value)
            for ek, _ in edges
            if ek.relation == BeliefRelation.SUPPORTS.value
        }

    def get_contradicting_beliefs(self, belief_id: BeliefId) -> set[BeliefId]:
        """Return beliefs that contradict belief_id (via CONTRADICTS edges in either direction)."""
        incoming = self._graph.get_edges_to(belief_id.to_symbolic_id())
        outgoing = self._graph.get_edges_from(belief_id.to_symbolic_id())

        res: set[BeliefId] = set()
        for ek, _ in incoming:
            if ek.relation == BeliefRelation.CONTRADICTS.value:
                res.add(BeliefId(value=ek.source.value))
        for ek, _ in outgoing:
            if ek.relation == BeliefRelation.CONTRADICTS.value:
                res.add(BeliefId(value=ek.target.value))

        # Also include explicit contradictions recorded in Belief.contradictions field
        belief = self.get_belief(belief_id)
        if belief is not None:
            res.update(belief.contradictions)

        return res

    def get_dependent_beliefs(self, belief_id: BeliefId) -> set[BeliefId]:
        """Return beliefs that depend on belief_id (via incoming DEPENDS_ON edges)."""
        edges = self._graph.get_edges_to(belief_id.to_symbolic_id())
        return {
            BeliefId(value=ek.source.value)
            for ek, _ in edges
            if ek.relation == BeliefRelation.DEPENDS_ON.value
        }
