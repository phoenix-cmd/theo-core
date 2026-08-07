"""Unit tests for ConflictResolver."""

from decimal import Decimal

from theo_core.symbolic.beliefs.models import Belief, BeliefId, BeliefSource
from theo_core.symbolic.conflict.models import ConflictPolicy
from theo_core.symbolic.conflict.resolver import ConflictResolver
from theo_core.symbolic.hypotheses.models import Hypothesis, HypothesisId, HypothesisState
from theo_core.symbolic.thoughts.models import ThoughtId


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

    def test_explicit_authority_respects_source_rank(self) -> None:
        knowledge = Belief(
            id=BeliefId.of("belief://knowledge_b"),
            proposition="Known fact",
            confidence=Decimal("0.1"),
            source=BeliefSource.KNOWLEDGE,
        )
        memory = Belief(
            id=BeliefId.of("belief://memory_b"),
            proposition="Recalled fact",
            confidence=Decimal("0.9"),
            source=BeliefSource.MEMORY,
        )

        winner, loser, _record = ConflictResolver.resolve_belief_contradiction(
            memory, knowledge, policy=ConflictPolicy.EXPLICIT_AUTHORITY
        )

        assert winner.id == knowledge.id
        assert loser.id == memory.id
        assert loser.confidence == Decimal("0.0")

    def test_equal_confidence_tie_breaks_deterministically_by_id(self) -> None:
        b_a = Belief(id=BeliefId.of("belief://a"), proposition="A", confidence=Decimal("0.5"))
        b_b = Belief(id=BeliefId.of("belief://b"), proposition="B", confidence=Decimal("0.5"))

        winner, loser, _record = ConflictResolver.resolve_belief_contradiction(b_a, b_b)
        assert winner.id == b_a.id
        assert loser.id == b_b.id

        winner, loser, _record = ConflictResolver.resolve_belief_contradiction(b_b, b_a)
        assert winner.id == b_a.id
        assert loser.id == b_b.id

    def test_hypotheses_conflict_respects_evidence_count_policy(self) -> None:
        h1 = Hypothesis(
            id=HypothesisId.of("hypothesis://h1"),
            interpretation="Fewer supporting thoughts",
            score=Decimal("0.9"),
        )
        h2 = Hypothesis(
            id=HypothesisId.of("hypothesis://h2"),
            interpretation="More supporting thoughts",
            score=Decimal("0.4"),
            supporting_thoughts=(
                ThoughtId.of("thought://e1"),
                ThoughtId.of("thought://e2"),
                ThoughtId.of("thought://e3"),
            ),
        )

        resolved, _records = ConflictResolver.resolve_hypotheses_conflict(
            [h1, h2], policy=ConflictPolicy.EVIDENCE_COUNT
        )

        assert resolved[0].id == h2.id
        assert resolved[0].state == HypothesisState.ACCEPTED
