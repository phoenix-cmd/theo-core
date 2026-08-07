"""CAUSAL-008 Benchmark Case — Confidence-Gated Premise Negative Control."""

import hashlib

from theo_core.evaluation.benchmarks import causal_reasoning
from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.pipeline import SymbolicCognitivePipeline


def test_causal_008_low_confidence_premise_does_not_fire() -> None:
    """A premise below the rule's min_confidence must not fire the rule."""
    case = causal_reasoning.CASES[7]
    assert case.id.value == "bm://causal_reasoning/008"

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
    assert "the road is icy" in active
    assert "Driving is hazardous" not in active
