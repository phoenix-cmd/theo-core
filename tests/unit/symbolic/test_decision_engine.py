"""Unit tests for Decision Engine action selection."""

from decimal import Decimal

import pytest

from theo_core.symbolic.decisions.engine import DecisionEngine
from theo_core.symbolic.decisions.models import DecisionId, DecisionType
from theo_core.symbolic.hypotheses.models import Hypothesis, HypothesisId, HypothesisState
from theo_core.symbolic.thoughts.graph import ThoughtGraph
from theo_core.symbolic.thoughts.models import Thought, ThoughtId


class TestDecisionEngine:
    def test_decision_id_factory(self) -> None:
        did = DecisionId.of("decision://action_1")
        assert did.value == "decision://action_1"

    def test_invalid_decision_id_raises(self) -> None:
        with pytest.raises(ValueError, match="must start with 'decision://'"):
            DecisionId.of("invalid://scheme")

    def test_make_decision_selects_accepted_hypothesis(self) -> None:
        tg = ThoughtGraph()
        t1 = ThoughtId.of("thought://t1")
        tg.add_thought(Thought(id=t1, content="Evidence thought"))

        h1 = Hypothesis(
            id=HypothesisId.of("hypothesis://h1"),
            interpretation="Execute refactoring plan",
            score=Decimal("0.9"),
            state=HypothesisState.ACCEPTED,
            supporting_thoughts=(t1,),
        )

        decision = DecisionEngine.make_decision([h1], tg)

        assert decision.type == DecisionType.RESPONSE
        assert decision.action_text == "Execute refactoring plan"
        assert decision.confidence == Decimal("0.9")
        assert t1 in decision.referenced_thoughts
