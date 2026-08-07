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

        decision, trace, _golden = pipeline.execute_cycle("test input percept", budget)

        # Verify decision selection
        assert decision.action_text is not None
        assert len(decision.referenced_thoughts) >= 1

        # Canon Invariant 7: every decision references the active goal
        assert decision.referenced_goal.value.startswith("goal://")
        assert decision.referenced_goal.value == "goal://maintainconversation"

        # Canon Law 6: decision carries a structured Intent + ActionSpec, not rendered text
        assert decision.intent.value == "maintain_conversation"
        assert decision.action_spec.capability in {"respond", "defer"}
        assert decision.action_spec.parameters is not None

        # Verify all 9 cycle stages were recorded
        assert len(trace.stages_executed) == 9
        assert CycleStage.PERCEPTION in trace.stages_executed
        assert CycleStage.DECISION in trace.stages_executed
        assert CycleStage.LEARNING in trace.stages_executed
        assert trace.budget_exhausted is False

    def test_pipeline_100_run_determinism(self) -> None:
        """Verify 100-run strict determinism across full pipeline execution."""
        pipeline = SymbolicCognitivePipeline()

        decisions = []
        for _ in range(100):
            d, _, _ = pipeline.execute_cycle("determinism check input")
            decisions.append((d.id.value, d.action_text, d.confidence))

        # All 100 decision outputs MUST be identical
        first = decisions[0]
        assert all(item == first for item in decisions)

    def test_cross_process_percept_id_determinism(self) -> None:
        """Verify percept belief ID generation is deterministic across processes (SHA-256)."""
        import hashlib
        input_text = "The sky is raining heavily outside"
        expected_hash = hashlib.sha256(input_text.encode("utf-8")).hexdigest()[:8]
        expected_id = f"belief://percept/{expected_hash}"

        p1 = SymbolicCognitivePipeline()
        p1.execute_cycle(input_text)
        b = p1.beliefs.get_active_beliefs()[0]

        assert b.id.value == expected_id

    def test_revision_stage_reconciles_contradictions(self) -> None:
        """Verify the REVISION stage deterministically reconciles contradictions."""
        from decimal import Decimal

        from theo_core.symbolic.beliefs.models import Belief, BeliefEdge, BeliefId, BeliefRelation

        pipeline = SymbolicCognitivePipeline()
        sky_blue = BeliefId.of("belief://sky_blue")
        sky_green = BeliefId.of("belief://sky_green")
        pipeline.beliefs.add_belief(
            Belief(id=sky_blue, proposition="Sky is blue", confidence=Decimal("0.9"))
        )
        pipeline.beliefs.add_belief(
            Belief(id=sky_green, proposition="Sky is green", confidence=Decimal("0.3"))
        )
        pipeline.beliefs.add_edge(
            BeliefEdge(source=sky_blue, target=sky_green, relation=BeliefRelation.CONTRADICTS)
        )

        pipeline.execute_cycle("sky observation")

        active_ids = {b.id for b in pipeline.beliefs.get_active_beliefs()}
        assert sky_blue in active_ids
        assert sky_green not in active_ids

    def test_percept_belief_is_evidence_derived(self) -> None:
        """Percepts enter as evidence; beliefs are derived (source=INFERENCE)."""
        from theo_core.symbolic.beliefs.models import BeliefSource

        pipeline = SymbolicCognitivePipeline()
        pipeline.execute_cycle("deterministic input")

        b = pipeline.beliefs.get_active_beliefs()[0]
        assert b.source == BeliefSource.INFERENCE
        assert len(b.support) == 1
        assert b.support[0].source_type == "perception"
        assert b.support[0].evidence_id.value.startswith("percept://")
        assert pipeline.state.percept is not None
        assert pipeline.state.percept.content == "deterministic input"
