"""CAUSAL-007 Benchmark Case — Adversarial Negative Control (Steam Is Not Smoke)."""

import hashlib

from theo_core.evaluation.benchmarks import causal_reasoning
from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.pipeline import SymbolicCognitivePipeline


def test_causal_007_steam_does_not_fire_smoke_rule() -> None:
    """A steam belief must not fire a smoke-gated causal rule."""
    case = causal_reasoning.CASES[6]
    assert case.id.value == "bm://causal_reasoning/007"

    beliefs = BeliefGraph()
    for belief in case.initial_beliefs:
        beliefs.add_belief(belief)

    pipeline = SymbolicCognitivePipeline(beliefs=beliefs, rules=list(case.rules))
    decision, trace, golden_trace = pipeline.execute_cycle(case.percept_input)

    assert len(trace.stages_executed) == 9
    assert golden_trace.fired_rule_ids == ()
    assert golden_trace.thought_dag_node_count == 0
    assert golden_trace.decision_id == decision.id.to_symbolic_id()

    percept_hash = hashlib.sha256(case.percept_input.encode("utf-8")).hexdigest()[:8]
    assert golden_trace.derived_belief_ids == (SymbolicId.of(f"belief://percept/{percept_hash}"),)

    active = [b.proposition for b in pipeline.beliefs.get_active_beliefs()]
    assert "steam is rising from the kettle" in active
    assert "Fire is likely nearby" not in active
