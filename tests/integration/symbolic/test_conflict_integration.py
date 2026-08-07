"""Integration test for Conflict Resolution workflow."""

from decimal import Decimal

from theo_core.symbolic.beliefs.models import Belief, BeliefId
from theo_core.symbolic.conflict.models import ConflictPolicy
from theo_core.symbolic.conflict.repository import InMemoryConflictRepository
from theo_core.symbolic.conflict.resolver import ConflictResolver


class TestConflictIntegration:
    def test_full_conflict_resolution_lifecycle(self) -> None:
        repo = InMemoryConflictRepository()

        b1 = Belief(
            id=BeliefId.of("belief://fact_a"),
            proposition="Fact A",
            confidence=Decimal("0.8"),
        )
        b2 = Belief(
            id=BeliefId.of("belief://fact_b"),
            proposition="Contradicting Fact B",
            confidence=Decimal("0.3"),
        )

        _winner, _loser, record = ConflictResolver.resolve_belief_contradiction(
            b1, b2, policy=ConflictPolicy.HIGHER_CONFIDENCE
        )

        repo.save(record)
        assert repo.load(record.conflict_id) == record
        assert len(repo.get_all()) == 1
