"""CAUSAL-001 Benchmark Case — Causal Deduction Verification."""

from decimal import Decimal

from theo_core.evaluation.benchmark_schema import BenchmarkCase, BenchmarkId, GoldenTrace
from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.beliefs.models import Belief, BeliefId
from theo_core.symbolic.inference.models import InferenceRule, RuleCondition, RuleId
from theo_core.symbolic.pipeline import SymbolicCognitivePipeline


def test_causal_001_benchmark_execution() -> None:
    """Execute CAUSAL-001 benchmark case and verify against expected bounds and golden trace."""
    bm_case = BenchmarkCase(
        id=BenchmarkId.of("bm://causal_reasoning/001"),
        domain="causal_reasoning",
        name="CAUSAL-001: Rain and Wet Ground",
        description="Verify causal deduction that rain causes wet ground.",
        percept_input="The sky is raining heavily outside",
        expected_action_text="Interpretation based on belief 'The sky is raining heavily outside'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            fired_rule_ids=(SymbolicId.of("rule://causal/rain_umbrella"),),
            thought_dag_node_count=1,
        ),
    )

    # Setup pipeline with benchmark rule and knowledge
    r1 = InferenceRule(
        id=RuleId.of("rule://causal/rain_umbrella"),
        name="Rain umbrella suggestion rule",
        conditions=(RuleCondition(premise_predicate="raining"),),
        conclusion_template="Suggest carrying an umbrella",
        confidence_multiplier=Decimal("0.9"),
    )

    beliefs = BeliefGraph()
    beliefs.add_belief(
        Belief(
            id=BeliefId.of("belief://b_rain"),
            proposition="raining",
            confidence=Decimal("1.0"),
        )
    )

    pipeline = SymbolicCognitivePipeline(beliefs=beliefs, rules=[r1])
    decision, trace, golden_trace = pipeline.execute_cycle(bm_case.percept_input)

    # Assert correctness against benchmark case criteria and golden trace
    assert decision.action_text == bm_case.expected_action_text
    assert bm_case.min_confidence <= decision.confidence <= bm_case.max_confidence
    assert len(trace.stages_executed) == 8
    assert golden_trace.decision_id == decision.id.to_symbolic_id()
    assert golden_trace.response_text == decision.action_text
    assert golden_trace.thought_dag_node_count == 1
