"""Replay — canonical pipeline determinism and trace-and-replay integration.

The symbolic pipeline is deterministic over its CognitiveTraceFingerprint:
re-running identical input on a fresh runtime MUST reproduce identical
response text and identical golden-trace identifiers.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from theo_core.explanation.replay.replay_engine import ReplayEngine
from theo_core.symbolic.runtime import SymbolicRuntime
from theo_core.telemetry.tracing.cognitive_trace import CognitiveTrace
from theo_core.telemetry.tracing.recorder import TraceRecorder

if TYPE_CHECKING:
    from pathlib import Path


class TestSymbolicDeterminism:
    def test_same_input_produces_identical_response(self) -> None:
        first = SymbolicRuntime()
        second = SymbolicRuntime()
        a = first.process("rain is falling")
        b = second.process("rain is falling")

        assert a.response_text == b.response_text
        assert a.golden_trace.decision_id == b.golden_trace.decision_id
        assert a.golden_trace.derived_belief_ids == b.golden_trace.derived_belief_ids


class TestTraceReplay:
    def test_trace_and_replay_matches(self, tmp_path: Path) -> None:
        recorder = TraceRecorder(str(tmp_path / "traces"))
        runtime = SymbolicRuntime()
        result = runtime.process("rain is falling")

        saved = recorder.close_trace(
            cycle_id=CognitiveTrace(raw_input="rain is falling").trace_id,
            raw_input="rain is falling",
            response_text=result.response_text,
        )

        fresh = SymbolicRuntime()
        replay_engine = ReplayEngine(recorder=recorder, engine=fresh)
        replay = replay_engine.replay(str(saved.trace_id))

        assert replay.matched
        assert replay.variance == 0.0
        assert replay.replayed_output == result.response_text

    def test_tampered_trace_is_detected(self, tmp_path: Path) -> None:
        recorder = TraceRecorder(str(tmp_path / "traces"))
        runtime = SymbolicRuntime()
        result = runtime.process("rain is falling")

        saved = recorder.close_trace(
            cycle_id=CognitiveTrace(raw_input="rain is falling").trace_id,
            raw_input="rain is falling",
            response_text=result.response_text,
        )

        trace_path = tmp_path / "traces" / f"{saved.trace_id}.json"
        data = json.loads(trace_path.read_text(encoding="utf-8"))
        data["response_text"] = "tampered response"
        trace_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

        replay_engine = ReplayEngine(recorder=recorder, engine=SymbolicRuntime())
        replay = replay_engine.replay(str(saved.trace_id))

        assert not replay.matched
        assert replay.variance == 1.0
        assert replay.original_output == "tampered response"
