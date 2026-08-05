"""Determinism benchmark suite for THEO v0.2.0-final."""

from __future__ import annotations

import os

from theo_core.composition.bootstrap import bootstrap


class TestDeterminismBenchmark:
    """Benchmark testing 100% replayable determinism across 1,000 runs."""

    def test_1000_run_determinism(self, tmp_path: object) -> None:
        """Run 1,000 cognitive cycles on identical input and verify 0 variance."""
        mem_file = str(tmp_path) + "/det_mem.json"
        know_file = str(tmp_path) + "/det_know.json"
        trace_dir = str(tmp_path) + "/traces"

        container = bootstrap(
            memory_file=mem_file,
            knowledge_file=know_file,
            trace_dir=trace_dir,
        )

        first_state = container.cognitive_engine.process("My name is Falcon")
        expected_text = first_state.response_text
        expected_goal = first_state.active_goal.description if first_state.active_goal else ""

        # Execute 1,000 iterations
        for _ in range(1000):
            s = container.cognitive_engine.process("My name is Falcon")
            assert s.response_text == expected_text
            assert s.active_goal is not None
            assert s.active_goal.description == expected_goal
            assert s.cognitive_depth == 12

        if os.path.exists(mem_file):
            os.remove(mem_file)
        if os.path.exists(know_file):
            os.remove(know_file)
