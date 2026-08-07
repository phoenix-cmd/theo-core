"""Canon Edition C1 Conformance Tests — Decision Engine Laws & Invariants."""

from decimal import Decimal

from theo_core.symbolic.decisions.engine import DecisionEngine
from theo_core.symbolic.hypotheses.models import Hypothesis, HypothesisId, HypothesisState
from theo_core.symbolic.thoughts.graph import ThoughtGraph
from theo_core.symbolic.thoughts.models import Thought, ThoughtId


class TestCanonDecisionLaws:
    def test_law_2_decision_references_thoughts(self) -> None:
        """Canon Law 2: Every Decision MUST reference one or more Thoughts."""
        tg = ThoughtGraph()
        t1_id = ThoughtId.of("thought://t1")
        tg.add_thought(Thought(id=t1_id, content="Reasoning thought"))

        h1 = Hypothesis(
            id=HypothesisId.of("hypothesis://h1"),
            interpretation="Proposed decision action",
            score=Decimal("0.9"),
            state=HypothesisState.ACCEPTED,
            supporting_thoughts=(t1_id,),
        )

        decision = DecisionEngine.make_decision([h1], tg)

        assert len(decision.referenced_thoughts) >= 1
        assert t1_id in decision.referenced_thoughts

    def test_invariant_2_decision_determinism(self) -> None:
        """Canon Invariant 2: Decisions MUST be fully deterministic given identical inputs."""
        tg = ThoughtGraph()
        t1_id = ThoughtId.of("thought://t1")
        tg.add_thought(Thought(id=t1_id, content="Reasoning thought"))

        h1 = Hypothesis(
            id=HypothesisId.of("hypothesis://h1"),
            interpretation="Proposed decision action",
            score=Decimal("0.85"),
            state=HypothesisState.ACCEPTED,
            supporting_thoughts=(t1_id,),
        )

        decisions = [DecisionEngine.make_decision([h1], tg) for _ in range(100)]

        # Verify all 100 decision outcomes are strictly identical
        first = (decisions[0].id.value, decisions[0].action_text, decisions[0].confidence)
        assert all(
            (d.id.value, d.action_text, d.confidence) == first for d in decisions
        )
