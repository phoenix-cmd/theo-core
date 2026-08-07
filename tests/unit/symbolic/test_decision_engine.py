"""Unit tests for Decision Engine action selection."""

from decimal import Decimal

import pytest

from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.decisions.engine import DecisionEngine
from theo_core.symbolic.decisions.models import DecisionId, DecisionType, Intent
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

        decision = DecisionEngine.make_decision(
            [h1], tg, referenced_goal=SymbolicId.of("goal://execute_refactoring")
        )

        assert decision.type == DecisionType.RESPONSE
        assert decision.action_text == "Execute refactoring plan"
        assert decision.confidence == Decimal("0.9")
        assert t1 in decision.referenced_thoughts
        assert decision.action_spec.capability == "respond"
        assert decision.action_spec.parameters["content"] == "Execute refactoring plan"

    def test_intent_is_derived_from_referenced_goal(self) -> None:
        """The decision intent MUST derive from the referenced goal (Canon Law 6)."""
        tg = ThoughtGraph()
        h1 = Hypothesis(
            id=HypothesisId.of("hypothesis://h1"),
            interpretation="Winning interpretation",
            score=Decimal("0.8"),
            state=HypothesisState.ACCEPTED,
        )

        decision = DecisionEngine.make_decision(
            [h1], tg, referenced_goal=SymbolicId.of("goal://providerecommendation")
        )

        assert decision.intent == Intent.PROVIDE_RECOMMENDATION

        defer = DecisionEngine.make_decision(
            [], tg, referenced_goal=SymbolicId.of("goal://unknownslug")
        )

        assert defer.intent == Intent.MAINTAIN_CONVERSATION
        assert defer.action_spec.capability == "defer"

    def test_make_decision_is_pure_no_graph_mutation(self) -> None:
        tg = ThoughtGraph()
        goal = SymbolicId.of("goal://test")

        defer = DecisionEngine.make_decision([], tg, referenced_goal=goal)

        assert defer.type == DecisionType.DEFER
        assert tg.node_count == 0
        assert defer.referenced_thoughts == (ThoughtId.of("thought://sys/fallback_evaluation"),)

        h1 = Hypothesis(
            id=HypothesisId.of("hypothesis://h1"),
            interpretation="Winning interpretation",
            score=Decimal("0.8"),
            state=HypothesisState.ACCEPTED,
        )
        decision = DecisionEngine.make_decision([h1], tg, referenced_goal=goal)

        assert decision.type == DecisionType.RESPONSE
        assert tg.node_count == 0
        assert decision.referenced_thoughts == (ThoughtId.of("thought://sys/inference_rule"),)
