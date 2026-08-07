"""Tests for ExplainEngine, TraceRecorder, DecisionRecord, and ReplayEngine."""

from __future__ import annotations

import os

from theo_core.composition.bootstrap import bootstrap
from theo_core.domain.runtime.entities.decision_record import DecisionRecord
from theo_core.explanation.engine.explain_engine import ExplainEngine
from theo_core.explanation.replay.replay_engine import ReplayEngine
from theo_core.symbolic.pipeline import SymbolicCognitivePipeline
from theo_core.symbolic.runtime import SymbolicRuntime


class TestExplainEngine:
    """Tests for ExplainEngine markdown output."""

    def test_explain_record(self) -> None:
        """ExplainEngine should format DecisionRecord into text."""
        engine = ExplainEngine()
        rec = DecisionRecord(
            selected_option="Hello Falcon!",
            selection_reason="Greeting rule matched",
            confidence=1.0,
            used_memory_ids=("mem-000001",),
            used_rule_ids=("RULE-0001",),
            used_goal="AcknowledgeGreeting",
        )
        explanation = engine.explain_record(rec)
        assert "AcknowledgeGreeting" in explanation
        assert "RULE-0001" in explanation
        assert "mem-000001" in explanation


class TestTraceRecorderAndReplay:
    """Tests for TraceRecorder JSON persistence and ReplayEngine determinism."""

    def test_end_to_end_trace_and_replay(self, tmp_path: object) -> None:
        """Cognitive cycle should record trace file and replay with 0-variance match."""
        mem_file = str(tmp_path) + "/replay_mem.json"
        trace_dir = str(tmp_path) + "/traces"
        container = bootstrap(
            memory_file=mem_file,
            knowledge_file=str(tmp_path) + "/replay_know.json",
            trace_dir=trace_dir,
        )

        # Run cognitive cycle
        state = container.cognitive_engine.process("My name is Falcon")
        rec = container.cognitive_engine.last_record
        assert rec is not None
        assert rec.trace_id is not None

        # Verify trace file written to disk
        trace = container.trace_recorder.load_trace(rec.trace_id)
        assert trace is not None
        assert trace.response_text == state.response_text
        assert len(trace.spans) == 12

        # Replay trace using ReplayEngine
        replay_engine = ReplayEngine(
            recorder=container.trace_recorder,
            engine=container.cognitive_engine,
        )
        result = replay_engine.replay(str(rec.trace_id))

        assert result.matched
        assert result.variance == 0.0
        assert result.replayed_output == state.response_text

        # Test explanation output
        explanation = container.explain_engine.explain_state(state)
        assert "Falcon" in explanation

        if os.path.exists(mem_file):
            os.remove(mem_file)

    def test_container_replay_engine_drives_canonical_runtime(self, tmp_path: object) -> None:
        """The container's ReplayEngine must replay through the canonical runtime.

        No hidden legacy execution path: the container binds replay to a factory
        that builds a fresh canonical SymbolicRuntime (not the legacy
        CognitiveEngine), restores the recorded pre-cycle state, and regenerates
        the golden fingerprint.
        """
        container = bootstrap(
            memory_file=str(tmp_path) + "/replay_mem.json",
            knowledge_file=str(tmp_path) + "/replay_know.json",
            trace_dir=str(tmp_path) + "/traces",
            state_file=str(tmp_path) + "/symbolic_state.json",
        )

        result = container.symbolic_runtime.process("rain is falling")
        trace_id = container.symbolic_runtime.last_trace_id
        assert trace_id is not None

        replay = container.replay_engine.replay(trace_id)

        assert replay.matched
        assert replay.variance == 0.0
        assert replay.replayed_output == result.response_text

        # The container replay engine builds fresh canonical runtimes and is not
        # bound to the legacy CognitiveEngine.
        assert container.replay_engine._engine is None
        assert container.replay_engine._engine_factory is not None
        replay_runtime = container.replay_engine._engine_factory()
        assert isinstance(replay_runtime, SymbolicRuntime)
        assert isinstance(replay_runtime.pipeline, SymbolicCognitivePipeline)
