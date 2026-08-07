"""CAUSAL-005 Benchmark Case — Multi-Step Causal Chain Golden Trace."""

import hashlib

from theo_core.evaluation.benchmarks import causal_reasoning
from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.pipeline import SymbolicCognitivePipeline


def test_causal_005_chain_golden_trace() -> None:
    """Execute the two-rule causal chain and verify the complete golden trace."""
    case = causal_reasoning.CASES[4]
    assert case.id.value == "bm://causal_reasoning/005"

    beliefs = BeliefGraph()
    for belief in case.initial_beliefs:
        beliefs.add_belief(belief)

    pipeline = SymbolicCognitivePipeline(beliefs=beliefs, rules=list(case.rules))
    decision, trace, golden_trace = pipeline.execute_cycle(case.percept_input)

    percept_hash = hashlib.sha256(case.percept_input.encode("utf-8")).hexdigest()[:8]

    assert decision.action_text == case.expected_action_text
    assert len(trace.stages_executed) == 9

    assert golden_trace.retrieved_memory_ids == (SymbolicId.of("belief://c_rain"),)
    assert golden_trace.activated_concept_ids == ()
    assert golden_trace.generated_hypothesis_ids == (
        SymbolicId.of("hypothesis://cand/4"),
        SymbolicId.of("hypothesis://cand/1"),
        SymbolicId.of("hypothesis://cand/2"),
        SymbolicId.of("hypothesis://cand/3"),
    )
    assert golden_trace.fired_rule_ids == (
        SymbolicId.of("rule://causal/rain_wet_ground"),
        SymbolicId.of("rule://causal/wet_ground_slippery"),
    )
    assert golden_trace.derived_belief_ids == (
        SymbolicId.of("belief://inf/causal/rain_wet_ground/1"),
        SymbolicId.of("belief://inf/causal/wet_ground_slippery/2"),
        SymbolicId.of(f"belief://percept/{percept_hash}"),
    )
    assert golden_trace.resolved_conflict_ids == (
        SymbolicId.of("conflict://hyp/cand/4_cand/1"),
        SymbolicId.of("conflict://hyp/cand/4_cand/2"),
        SymbolicId.of("conflict://hyp/cand/4_cand/3"),
    )
    assert golden_trace.thought_dag_node_count == 2
    assert golden_trace.decision_id == decision.id.to_symbolic_id()
    assert golden_trace.response_text == decision.action_text

    for expected in case.expected_beliefs:
        assert expected in [b.proposition for b in pipeline.beliefs.get_active_beliefs()]

    assert case.min_confidence <= decision.confidence <= case.max_confidence
