"""CONT-006 Benchmark Case — Equal-Confidence Contradiction Tie-Break."""

from theo_core.evaluation.benchmarks import contradiction
from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.pipeline import SymbolicCognitivePipeline


def test_contradiction_006_equal_confidence_tie_break() -> None:
    """Equal-confidence contradictions must tie-break deterministically on id."""
    case = contradiction.CASES[5]
    assert case.id.value == "bm://contradiction/006"

    beliefs = BeliefGraph()
    for belief in case.initial_beliefs:
        beliefs.add_belief(belief)
    for edge in case.initial_belief_edges:
        beliefs.add_edge(edge)

    pipeline = SymbolicCognitivePipeline(beliefs=beliefs)
    decision, trace, golden_trace = pipeline.execute_cycle(case.percept_input)

    assert len(trace.stages_executed) == 9
    assert golden_trace.thought_dag_node_count == 0
    assert golden_trace.resolved_conflict_ids == (
        SymbolicId.of("conflict://hyp/cand/2_cand/1"),
    )
    assert golden_trace.decision_id == decision.id.to_symbolic_id()

    active = [b.proposition for b in pipeline.beliefs.get_active_beliefs()]
    assert "the light is off" in active
    assert "the light is on" not in active
