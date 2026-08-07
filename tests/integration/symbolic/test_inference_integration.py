"""Integration test for full Inference System workflow."""

from decimal import Decimal

from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.beliefs.models import Belief, BeliefId
from theo_core.symbolic.concepts.graph import ConceptGraph
from theo_core.symbolic.inference.engine import InferenceEngine
from theo_core.symbolic.inference.models import InferenceRule, RuleCondition, RuleId
from theo_core.symbolic.inference.repository import InMemoryRuleRepository
from theo_core.symbolic.thoughts.graph import ThoughtGraph


class TestInferenceIntegration:
    def test_full_inference_workflow(self) -> None:
        rule_repo = InMemoryRuleRepository()
        cg = ConceptGraph()
        bg = BeliefGraph()
        tg = ThoughtGraph()

        r1 = InferenceRule(
            id=RuleId.of("rule://r1"),
            name="R1",
            conditions=(RuleCondition(premise_predicate="weather_sunny"),),
            conclusion_template="Suggest outdoor activity",
            confidence_multiplier=Decimal("0.85"),
        )
        rule_repo.save(r1)

        bg.add_belief(
            Belief(
                id=BeliefId.of("belief://b1"),
                proposition="The weather_sunny forecast is high",
                confidence=Decimal("1.0"),
            )
        )

        trace = InferenceEngine.forward_chain(cg, bg, tg, rules=rule_repo.get_all())

        assert len(trace.steps) == 1
        assert tg.node_count == 1
        assert bg.node_count == 2

        # Verify ThoughtGraph topological order includes generated thought
        topo = tg.topological_sort()
        assert len(topo) == 1
