"""Canon Edition C1 Conformance Tests — Conflict Resolution Laws."""

from decimal import Decimal

from theo_core.symbolic.beliefs.models import Belief, BeliefId
from theo_core.symbolic.conflict.models import ConflictPolicy
from theo_core.symbolic.conflict.resolver import ConflictResolver


class TestCanonConflictLaws:
    def test_law_7_contradiction_resolved_by_policy(self) -> None:
        """Canon Law 7: Contradictions MUST be resolved according to defined policies."""
        b1 = Belief(
            id=BeliefId.of("belief://canon_b1"),
            proposition="Contradictory claim 1",
            confidence=Decimal("0.9"),
        )
        b2 = Belief(
            id=BeliefId.of("belief://canon_b2"),
            proposition="Contradictory claim 2",
            confidence=Decimal("0.4"),
        )

        winner, deprecated_loser, record = ConflictResolver.resolve_belief_contradiction(
            b1, b2, policy=ConflictPolicy.HIGHER_CONFIDENCE
        )

        assert winner.id == b1.id
        assert deprecated_loser.confidence == Decimal("0.0")
        assert record.policy == ConflictPolicy.HIGHER_CONFIDENCE
        assert record.winning_id == b1.id.to_symbolic_id()
