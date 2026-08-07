"""Canon Edition C1 Conformance Tests — Determinism and Invariants for Concept System."""

from decimal import Decimal

from theo_core.symbolic.concepts.activation import ActivationEngine
from theo_core.symbolic.concepts.graph import ConceptGraph
from theo_core.symbolic.concepts.models import (
    Concept,
    ConceptEdge,
    ConceptId,
    RelationType,
)


class TestCanonConceptDeterminism:
    def test_100_run_determinism_invariant_8(self) -> None:
        """Canon Invariant 8: Every cognitive cycle MUST produce a deterministic state.

        Asserts 100 runs of spreading activation produce identical outputs.
        """
        cg = ConceptGraph()
        nodes = [ConceptId.of(f"concept://node/{i}") for i in range(20)]
        for nid in nodes:
            cg.add_concept(Concept(id=nid, label=f"Node {nid.value}"))

        for i in range(19):
            cg.add_edge(
                ConceptEdge(
                    source=nodes[i],
                    target=nodes[i + 1],
                    relation=RelationType.RELATED_TO,
                )
            )

        seed = {nodes[0]: Decimal("1.0")}

        first_run = ActivationEngine.activate(
            cg, seed, decay_factor=Decimal("0.8"), max_depth=10
        )

        for _ in range(99):
            subsequent_run = ActivationEngine.activate(
                cg, seed, decay_factor=Decimal("0.8"), max_depth=10
            )
            assert subsequent_run.activations == first_run.activations
