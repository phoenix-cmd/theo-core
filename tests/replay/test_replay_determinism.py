"""Replay — canonical pipeline determinism and trace-and-replay integration.

The symbolic pipeline is deterministic over its CognitiveTraceFingerprint:
re-running identical input on a fresh runtime MUST reproduce identical
response text and identical golden-trace identifiers.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from theo_core.evaluation.benchmark_schema import FINGERPRINT_METADATA_KEY
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


class TestCanonicalRecording:
    def test_runtime_records_trace_with_golden_fingerprint(self, tmp_path: Path) -> None:
        recorder = TraceRecorder(str(tmp_path / "traces"))
        runtime = SymbolicRuntime(recorder=recorder)
        result = runtime.process("rain is falling")

        assert runtime.last_trace_id is not None
        trace = recorder.load_trace(runtime.last_trace_id)
        assert trace is not None
        assert trace.raw_input == "rain is falling"
        assert trace.response_text == result.response_text

        fingerprint = trace.metadata.get(FINGERPRINT_METADATA_KEY)
        assert fingerprint is not None
        assert fingerprint["decision_id"] == result.golden_trace.decision_id.value
        assert fingerprint["derived_belief_ids"] == [
            str(i) for i in result.golden_trace.derived_belief_ids
        ]


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

    def test_fingerprint_tamper_detected_even_when_text_matches(self, tmp_path: Path) -> None:
        recorder = TraceRecorder(str(tmp_path / "traces"))
        runtime = SymbolicRuntime(recorder=recorder)
        runtime.process("rain is falling")
        trace_path = tmp_path / "traces" / f"{runtime.last_trace_id}.json"

        data = json.loads(trace_path.read_text(encoding="utf-8"))
        recorded_fingerprint = data["metadata"][FINGERPRINT_METADATA_KEY]
        recorded_fingerprint["derived_belief_ids"] = ["belief://inf/forged/999"]
        trace_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        replay_engine = ReplayEngine(recorder=recorder, engine=SymbolicRuntime())
        replay = replay_engine.replay(str(runtime.last_trace_id))

        assert replay.matched is False
        assert replay.variance == 1.0
        assert replay.replayed_output == data["response_text"]


class TestMultiTurnReplayFidelity:
    def test_replay_restores_pre_cycle_state_for_later_turns(self, tmp_path: Path) -> None:
        recorder = TraceRecorder(str(tmp_path / "traces"))
        runtime = SymbolicRuntime(recorder=recorder)
        runtime.process("rain is falling")
        second = runtime.process("road is slippery")
        trace_id = runtime.last_trace_id
        assert trace_id is not None

        replay_engine = ReplayEngine(
            recorder=recorder,
            engine_factory=lambda: SymbolicRuntime(),
        )
        replay = replay_engine.replay(trace_id)

        assert replay.matched
        assert replay.variance == 0.0
        assert replay.replayed_output == second.response_text

    def test_replay_does_not_disturb_live_runtime_state(self, tmp_path: Path) -> None:
        recorder = TraceRecorder(str(tmp_path / "traces"))
        runtime = SymbolicRuntime(recorder=recorder)
        runtime.process("rain is falling")
        first_trace_id = runtime.last_trace_id
        runtime.process("road is slippery")
        second_trace_id = runtime.last_trace_id
        assert first_trace_id is not None
        assert second_trace_id is not None

        replay_engine = ReplayEngine(
            recorder=recorder,
            engine_factory=lambda: SymbolicRuntime(),
        )
        replay_engine.replay(second_trace_id)

        # Replay ran on a fresh engine: the live runtime's recording position
        # is untouched.
        assert runtime.last_trace_id == second_trace_id

        # An earlier turn replays faithfully even after later turns advanced
        # the live runtime, because its pre-cycle state was recorded.
        first_replay = replay_engine.replay(first_trace_id)
        assert first_replay.matched
        assert first_replay.variance == 0.0
