"""Ambiguity domain — Canon Law 6 golden-trace assertions.

Every ambiguity case must generate a distinct candidate hypothesis per matching
belief (no premature collapse) while leaving all seeded interpretations active.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from theo_core.evaluation.benchmarks import ambiguity
from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.pipeline import SymbolicCognitivePipeline

if TYPE_CHECKING:
    from theo_core.evaluation.benchmark_schema import BenchmarkCase

_CASE_IDS = tuple(case.id.value for case in ambiguity.CASES)


@pytest.mark.parametrize("case", ambiguity.CASES, ids=_CASE_IDS)
def test_ambiguity_case_golden_trace(case: BenchmarkCase) -> None:
    """Verify the complete golden trace and Law 6 semantics for each case."""
    beliefs = BeliefGraph()
    for belief in case.initial_beliefs:
        beliefs.add_belief(belief)

    pipeline = SymbolicCognitivePipeline(beliefs=beliefs, rules=list(case.rules))
    decision, trace, golden_trace = pipeline.execute_cycle(case.percept_input)

    assert decision.action_text == case.expected_action_text
    assert len(trace.stages_executed) == 9

    assert golden_trace.retrieved_memory_ids == tuple(
        SymbolicId.of(b.id.value) for b in case.initial_beliefs
    )
    assert golden_trace.fired_rule_ids == ()
    assert golden_trace.thought_dag_node_count == 0
    assert golden_trace.decision_id == decision.id.to_symbolic_id()
    assert golden_trace.response_text == decision.action_text

    for expected in case.expected_beliefs:
        assert expected in [b.proposition for b in pipeline.beliefs.get_active_beliefs()]

    assert case.min_confidence <= decision.confidence <= case.max_confidence


def test_ambiguity_generates_competing_hypotheses() -> None:
    """Canon Law 6: ambiguous input must not collapse to a single hypothesis.

    For each seeded interpretation plus the perceptual reading, the pipeline must
    record one candidate hypothesis — strictly more than one per case.
    """
    for case in ambiguity.CASES:
        assert len(case.golden_trace.generated_hypothesis_ids) > 1
        assert len(case.golden_trace.generated_hypothesis_ids) == (
            len(case.initial_beliefs) + 1
        )


def test_ambiguity_percept_belief_is_derived() -> None:
    """The perceptual reading is committed as an evidence-backed belief."""
    case = ambiguity.CASES[0]
    beliefs = BeliefGraph()
    for belief in case.initial_beliefs:
        beliefs.add_belief(belief)

    pipeline = SymbolicCognitivePipeline(beliefs=beliefs)
    _, _, golden_trace = pipeline.execute_cycle(case.percept_input)

    percept_hash = hashlib.sha256(case.percept_input.encode("utf-8")).hexdigest()[:8]
    assert SymbolicId.of(f"belief://percept/{percept_hash}") in golden_trace.derived_belief_ids
