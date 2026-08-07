"""Canon Edition C1 Conformance Tests — Thought Graph Laws & Invariants."""

import pytest

from theo_core.symbolic.beliefs.models import BeliefId
from theo_core.symbolic.thoughts.models import Thought, ThoughtId


class TestCanonThoughtLaws:
    def test_invariant_1_thought_immutability(self) -> None:
        """Canon Invariant 1: Thoughts MUST be immutable after creation."""
        t = Thought(
            id=ThoughtId.of("thought://canon_immutability_test"),
            content="Original content",
        )

        with pytest.raises((TypeError, Exception)):
            t.content = "Modified content"  # type: ignore[misc]

    def test_law_3_thought_consumes_beliefs(self) -> None:
        """Canon Law 3: Every Thought MUST consume zero or more Beliefs."""
        b1 = BeliefId.of("belief://b1")
        t = Thought(
            id=ThoughtId.of("thought://canon_law_3_test"),
            content="Consuming belief b1",
            consumed_beliefs=(b1,),
        )

        assert len(t.consumed_beliefs) >= 0
        assert b1 in t.consumed_beliefs
