"""Unit tests for InferenceEngine forward and backward chaining."""

from decimal import Decimal

from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.beliefs.models import Belief, BeliefId, BeliefSource
from theo_core.symbolic.concepts.graph import ConceptGraph
from theo_core.symbolic.inference.engine import InferenceEngine
from theo_core.symbolic.inference.models import (
    InferenceRule,
    RuleCondition,
    RuleId,
)
from theo_core.symbolic.thoughts.graph import ThoughtGraph


class TestInferenceEngine:
    def test_forward_chaining_deduction(self) -> None:
        cg = ConceptGraph()
        bg = BeliefGraph()
        tg = ThoughtGraph()

        b1_id = BeliefId.of("belief://user_likes_astronomy")
        bg.add_belief(
            Belief(
                id=b1_id,
                proposition="User likes astronomy",
                confidence=Decimal("1.0"),
                source=BeliefSource.MEMORY,
            )
        )

        r1 = InferenceRule(
            id=RuleId.of("rule://recommend_astronomy"),
            name="Recommend Astronomy",
            conditions=(RuleCondition(premise_predicate="user likes astronomy"),),
            conclusion_template="Recommend stargazing guide",
            confidence_multiplier=Decimal("0.9"),
        )

        trace = InferenceEngine.forward_chain(cg, bg, tg, rules=[r1])

        assert len(trace.steps) == 1
        assert tg.node_count == 1
        assert bg.node_count == 2  # Original + derived belief

        # Check derived belief is tagged with INFERENCE source per Law 4
        derived_belief = bg.get_belief(trace.steps[0].produced_belief_id)  # type: ignore[arg-type]
        assert derived_belief is not None
        assert derived_belief.source == BeliefSource.INFERENCE
        assert derived_belief.confidence == Decimal("0.9")

    def test_backward_chaining_proof(self) -> None:
        bg = BeliefGraph()
        r1 = InferenceRule(
            id=RuleId.of("rule://r1"),
            name="R1",
            conditions=(RuleCondition(premise_predicate="sky_is_clear"),),
            conclusion_template="Good conditions for astronomy",
        )

        proof = InferenceEngine.backward_chain(
            goal_predicate="astronomy",
            beliefs=bg,
            rules=[r1],
        )

        assert len(proof) == 1
        assert proof[0].id == r1.id
