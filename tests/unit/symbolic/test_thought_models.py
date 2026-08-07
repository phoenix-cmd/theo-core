"""Unit tests for Thought Graph domain models."""

from decimal import Decimal

import pytest

from theo_core.symbolic.beliefs.models import BeliefId
from theo_core.symbolic.thoughts.models import (
    Thought,
    ThoughtEdge,
    ThoughtId,
    ThoughtRelation,
)


class TestThoughtModels:
    def test_thought_id_factory(self) -> None:
        tid = ThoughtId.of("thought://t1")
        assert tid.value == "thought://t1"
        assert str(tid) == "thought://t1"

    def test_invalid_thought_id_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="must start with 'thought://'"):
            ThoughtId.of("invalid://scheme")

    def test_thought_creation_and_immutability(self) -> None:
        tid = ThoughtId.of("thought://t1")
        b_id = BeliefId.of("belief://b1")
        t = Thought(
            id=tid,
            content="User prefers formal tone",
            confidence=Decimal("0.9"),
            consumed_beliefs=(b_id,),
        )

        assert t.id == tid
        assert t.content == "User prefers formal tone"
        assert t.consumed_beliefs == (b_id,)

        with pytest.raises((TypeError, Exception)):
            t.content = "User prefers informal tone"  # type: ignore[misc]

    def test_thought_edge_creation(self) -> None:
        t1 = ThoughtId.of("thought://t1")
        t2 = ThoughtId.of("thought://t2")
        edge = ThoughtEdge(
            source=t1,
            target=t2,
            relation=ThoughtRelation.DERIVED_FROM,
        )

        assert edge.source == t1
        assert edge.target == t2
        assert edge.relation == ThoughtRelation.DERIVED_FROM
