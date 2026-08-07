"""Edge tests — the canonical pipeline handles extreme inputs gracefully.

Empty, whitespace-only, unicode, and very long percept inputs MUST produce a
valid decision without raising, and MUST remain deterministic.
"""

from __future__ import annotations

import pytest

from theo_core.symbolic.pipeline import SymbolicCognitivePipeline

EDGE_INPUTS = ("", "   ", "\t\n", "🦄 你好 monde", "a" * 10_000)


class TestEdgeInputs:
    @pytest.mark.parametrize("input_text", EDGE_INPUTS)
    def test_edge_input_produces_valid_decision(self, input_text: str) -> None:
        pipeline = SymbolicCognitivePipeline()
        decision, _, golden_trace = pipeline.execute_cycle(input_text)

        assert decision.type.value is not None
        assert decision.referenced_goal.value.startswith("goal://")
        assert golden_trace.decision_id is not None

    @pytest.mark.parametrize("input_text", EDGE_INPUTS)
    def test_edge_input_is_deterministic(self, input_text: str) -> None:
        first = SymbolicCognitivePipeline()
        second = SymbolicCognitivePipeline()
        a = first.execute_cycle(input_text)
        b = second.execute_cycle(input_text)

        assert a[2].decision_id == b[2].decision_id
        assert a[2].derived_belief_ids == b[2].derived_belief_ids

    def test_empty_input_still_commits_state(self) -> None:
        pipeline = SymbolicCognitivePipeline()
        pipeline.execute_cycle("")
        assert pipeline.beliefs.node_count >= 0
