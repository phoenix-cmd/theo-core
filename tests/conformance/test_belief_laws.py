"""Canon Edition C1 Conformance Tests — Belief System Laws & Invariants."""


from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.models import (
    Belief,
    BeliefId,
    BeliefSource,
    EvidenceTrace,
)


class TestCanonBeliefLaws:
    def test_law_4_belief_source_validity(self) -> None:
        """Canon Law 4: Beliefs MUST be derived from valid sources."""
        valid_sources = {
            BeliefSource.MEMORY,
            BeliefSource.KNOWLEDGE,
            BeliefSource.INFERENCE,
            BeliefSource.PERCEPTION,
        }
        b = Belief(
            id=BeliefId.of("belief://canon_test"),
            proposition="Test proposition",
            source=BeliefSource.INFERENCE,
        )

        assert b.source in valid_sources

    def test_invariant_5_traceable_provenance(self) -> None:
        """Canon Invariant 5: Every Belief MUST have traceable provenance."""
        ev = EvidenceTrace(evidence_id=SymbolicId.of("memory://turn_1"), source_type="memory")
        b = Belief(
            id=BeliefId.of("belief://canon_provenance_test"),
            proposition="Test proposition",
            support=(ev,),
        )

        assert len(b.support) >= 1
        assert b.support[0].evidence_id == SymbolicId.of("memory://turn_1")
