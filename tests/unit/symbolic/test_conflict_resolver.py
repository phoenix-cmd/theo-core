"""Unit tests for ConflictResolver."""

from decimal import Decimal

from theo_core.symbolic.beliefs.models import Belief, BeliefId
from theo_core.symbolic.conflict.models import ConflictPolicy
from theo_core.symbolic.conflict.resolver import ConflictResolver
from theo_core.symbolic.hypotheses.models import Hypothesis, HypothesisId, HypothesisState


class TestConflictResolver:
    def test_resolve_belief_contradiction_higher_confidence(self) -> None:
        b1 = Belief(
            id=BeliefId.of("belief://b1"),
            proposition="Sky is blue",
            confidence=Decimal("0.9"),
        )
        b2 = Belief(
            id=BeliefId.of("belief://b2"),
            proposition="Sky is green",
            confidence=Decimal("0.2"),
        )

        winner, loser, record = ConflictResolver.resolve_belief_contradiction(
            b1, b2, policy=ConflictPolicy.HIGHER_CONFIDENCE
        )

        assert winner.id == b1.id
        assert loser.id == b2.id
        assert loser.confidence == Decimal("0.0")
        assert record.winning_id == b1.id.to_symbolic_id()

    def test_resolve_hypotheses_conflict(self) -> None:
        h1 = Hypothesis(
            id=HypothesisId.of("hypothesis://h1"),
            interpretation="Interp 1",
            score=Decimal("0.9"),
        )
        h2 = Hypothesis(
            id=HypothesisId.of("hypothesis://h2"),
            interpretation="Interp 2",
            score=Decimal("0.4"),
        )

        resolved, records = ConflictResolver.resolve_hypotheses_conflict([h1, h2])

        assert len(resolved) == 2
        assert resolved[0].state == HypothesisState.ACCEPTED
        assert resolved[1].state == HypothesisState.REJECTED
        assert len(records) == 1
