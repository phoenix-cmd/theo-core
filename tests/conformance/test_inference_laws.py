"""Canon Edition C1 Conformance Tests — Inference Engine Laws."""

from decimal import Decimal

from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.beliefs.models import Belief, BeliefId, BeliefSource
from theo_core.symbolic.concepts.graph import ConceptGraph
from theo_core.symbolic.inference.engine import InferenceEngine
from theo_core.symbolic.inference.models import InferenceRule, RuleCondition, RuleId
from theo_core.symbolic.thoughts.graph import ThoughtGraph


class TestCanonInferenceLaws:
    def test_law_5_inference_records_explicit_traceable_steps(self) -> None:
        """Canon Law 5: Inference MUST be explicit and record traceable steps."""
        cg = ConceptGraph()
        bg = BeliefGraph()
        tg = ThoughtGraph()

        b1_id = BeliefId.of("belief://b1")
        bg.add_belief(Belief(id=b1_id, proposition="Premise test", confidence=Decimal("1.0")))

        r1 = InferenceRule(
            id=RuleId.of("rule://r1"),
            name="R1",
            conditions=(RuleCondition(premise_predicate="premise test"),),
            conclusion_template="Derived conclusion",
        )

        trace = InferenceEngine.forward_chain(cg, bg, tg, rules=[r1])

        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.rule_id == r1.id
        assert step.matched_belief_ids == (b1_id,)
        assert step.produced_thought_id is not None
        assert step.produced_belief_id is not None

    def test_law_4_inference_derived_belief_source(self) -> None:
        """Canon Law 4: Beliefs produced by inference MUST have BeliefSource.INFERENCE."""
        cg = ConceptGraph()
        bg = BeliefGraph()
        tg = ThoughtGraph()

        bg.add_belief(Belief(id=BeliefId.of("belief://b1"), proposition="Premise test"))
        r1 = InferenceRule(
            id=RuleId.of("rule://r1"),
            name="R1",
            conditions=(RuleCondition(premise_predicate="premise test"),),
            conclusion_template="Derived conclusion",
        )

        trace = InferenceEngine.forward_chain(cg, bg, tg, rules=[r1])
        produced_belief = bg.get_belief(trace.steps[0].produced_belief_id)  # type: ignore[arg-type]

        assert produced_belief is not None
        assert produced_belief.source == BeliefSource.INFERENCE
