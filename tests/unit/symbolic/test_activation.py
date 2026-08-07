"""Unit tests for ActivationEngine spreading activation."""

from decimal import Decimal

from theo_core.symbolic.concepts.activation import ActivationEngine
from theo_core.symbolic.concepts.graph import ConceptGraph
from theo_core.symbolic.concepts.models import (
    Concept,
    ConceptEdge,
    ConceptId,
    RelationType,
)


class TestActivationEngine:
    def test_spreading_activation(self) -> None:
        cg = ConceptGraph()
        c1 = ConceptId.of("concept://a")
        c2 = ConceptId.of("concept://b")
        c3 = ConceptId.of("concept://c")

        cg.add_concept(Concept(id=c1, label="A"))
        cg.add_concept(Concept(id=c2, label="B"))
        cg.add_concept(Concept(id=c3, label="C"))

        # A -> B -> C
        cg.add_edge(
            ConceptEdge(
                source=c1,
                target=c2,
                relation=RelationType.RELATED_TO,
                weight=Decimal("1.0"),
            )
        )
        cg.add_edge(
            ConceptEdge(
                source=c2,
                target=c3,
                relation=RelationType.RELATED_TO,
                weight=Decimal("1.0"),
            )
        )

        # Activate A with seed = 1.0, decay = 0.5
        seeds = {c1: Decimal("1.0")}
        result = ActivationEngine.activate(cg, seeds, decay_factor=Decimal("0.5"), max_depth=3)

        assert result.activations[c1] == Decimal("1.0")
        assert result.activations[c2] == Decimal("0.5")  # 1.0 * 1.0 * 0.5
        assert result.activations[c3] == Decimal("0.25") # 0.5 * 1.0 * 0.5

    def test_activation_purity_graph_unmutated(self) -> None:
        cg = ConceptGraph()
        c1 = ConceptId.of("concept://a")
        cg.add_concept(Concept(id=c1, label="A"))

        initial_node_count = cg.node_count
        initial_edge_count = cg.edge_count

        ActivationEngine.activate(cg, {c1: Decimal("1.0")})

        assert cg.node_count == initial_node_count
        assert cg.edge_count == initial_edge_count
