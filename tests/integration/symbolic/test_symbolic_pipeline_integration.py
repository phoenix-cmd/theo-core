"""Integration test for full end-to-end SymbolicCognitivePipeline execution."""

from decimal import Decimal

from theo_core.symbolic.inference.models import InferenceRule, RuleCondition, RuleId
from theo_core.symbolic.pipeline import SymbolicCognitivePipeline
from theo_core.symbolic.scheduler.models import ComputeBudget, CycleStage


class TestSymbolicPipelineIntegration:
    def test_full_pipeline_execution_cycle(self) -> None:
        rule = InferenceRule(
            id=RuleId.of("rule://r1"),
            name="Rule 1",
            conditions=(RuleCondition(premise_predicate="test"),),
            conclusion_template="Execute test action",
            confidence_multiplier=Decimal("0.95"),
        )

        pipeline = SymbolicCognitivePipeline(rules=[rule])
        budget = ComputeBudget(time_budget_ms=Decimal("2000.0"))

        decision, trace = pipeline.execute_cycle("test input percept", budget)

        # Verify decision selection
        assert decision.action_text is not None
        assert len(decision.referenced_thoughts) >= 1

        # Verify all 8 cycle stages were recorded
        assert len(trace.stages_executed) == 8
        assert CycleStage.PERCEPTION in trace.stages_executed
        assert CycleStage.DECISION in trace.stages_executed
        assert trace.budget_exhausted is False

    def test_pipeline_100_run_determinism(self) -> None:
        """Verify 100-run strict determinism across full pipeline execution."""
        pipeline = SymbolicCognitivePipeline()

        decisions = []
        for _ in range(100):
            d, _ = pipeline.execute_cycle("determinism check input")
            decisions.append((d.id.value, d.action_text, d.confidence))

        # All 100 decision outputs MUST be identical
        first = decisions[0]
        assert all(item == first for item in decisions)
