"""ConceptGraph — domain graph wrapper around generic Graph[Concept, ConceptEdge]."""

from __future__ import annotations

from theo_core.symbolic._graph.graph import Graph
from theo_core.symbolic._graph.traversal import GraphTraversal
from theo_core.symbolic._graph.types import EdgeKey
from theo_core.symbolic.concepts.models import Concept, ConceptEdge, ConceptId, RelationType


class ConceptGraph:
    """Domain graph wrapper managing Concept nodes and ConceptEdge relationships.

    Delegates structural graph storage and graph algorithms to generic Graph[Concept, ConceptEdge].
    """

    def __init__(self) -> None:
        """Initialize empty ConceptGraph instance."""
        self._graph: Graph[Concept, ConceptEdge] = Graph()

    @classmethod
    def from_raw_graph(cls, raw: Graph[Concept, ConceptEdge]) -> ConceptGraph:
        """Wrap a reconstructed generic Graph into a ConceptGraph.

        Args:
            raw: A deserialized Graph[Concept, ConceptEdge].

        Returns:
            A ConceptGraph backed by the given raw graph.

        """
        graph = cls()
        graph._graph = raw
        return graph

    @property
    def node_count(self) -> int:
        """Total number of concepts in graph."""
        return self._graph.node_count

    @property
    def edge_count(self) -> int:
        """Total number of edges in graph."""
        return self._graph.edge_count

    @property
    def raw_graph(self) -> Graph[Concept, ConceptEdge]:
        """Access underlying generic Graph instance."""
        return self._graph

    def add_concept(self, concept: Concept) -> None:
        """Add a Concept node to graph."""
        self._graph.add_node(concept.id.to_symbolic_id(), concept)

    def get_concept(self, concept_id: ConceptId) -> Concept | None:
        """Look up a Concept by ConceptId."""
        return self._graph.get_node(concept_id.to_symbolic_id())

    def has_concept(self, concept_id: ConceptId) -> bool:
        """Check if concept exists in graph."""
        return self._graph.has_node(concept_id.to_symbolic_id())

    def remove_concept(self, concept_id: ConceptId) -> None:
        """Remove a concept and connected edges."""
        self._graph.remove_node(concept_id.to_symbolic_id())

    def add_edge(self, edge: ConceptEdge) -> None:
        """Add a ConceptEdge to graph."""
        ek = EdgeKey(
            source=edge.source.to_symbolic_id(),
            target=edge.target.to_symbolic_id(),
            relation=edge.relation.value,
        )
        self._graph.add_edge(ek, edge)

    def get_edge(
        self, source: ConceptId, target: ConceptId, relation: RelationType
    ) -> ConceptEdge | None:
        """Look up a ConceptEdge."""
        ek = EdgeKey(
            source=source.to_symbolic_id(),
            target=target.to_symbolic_id(),
            relation=relation.value,
        )
        return self._graph.get_edge(ek)

    def get_concepts(self) -> list[Concept]:
        """Return all concepts in deterministic sorted order."""
        return [node for _, node in self._graph.get_nodes()]

    def get_edges(self) -> list[ConceptEdge]:
        """Return all concept edges in deterministic sorted order."""
        return [edge for _, edge in self._graph.get_edges()]

    def ancestors(self, concept_id: ConceptId) -> set[ConceptId]:
        """Find parent/ancestor concepts via outgoing IS_A hierarchy relationships."""
        symbolic_ancestors = GraphTraversal.descendants(
            self._graph,
            concept_id.to_symbolic_id(),
            edge_filter=lambda ek, _: ek.relation == RelationType.IS_A.value,
        )
        return {ConceptId(value=sid.value) for sid in symbolic_ancestors}

    def descendants(self, concept_id: ConceptId) -> set[ConceptId]:
        """Find child/descendant concepts via incoming IS_A hierarchy relationships."""
        symbolic_descendants = GraphTraversal.ancestors(
            self._graph,
            concept_id.to_symbolic_id(),
            edge_filter=lambda ek, _: ek.relation == RelationType.IS_A.value,
        )
        return {ConceptId(value=sid.value) for sid in symbolic_descendants}

    def copy(self) -> ConceptGraph:
        """Return an independent shallow copy of this ConceptGraph."""
        new_graph = ConceptGraph()
        new_graph._graph = self._graph.copy()
        return new_graph
