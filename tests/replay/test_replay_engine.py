"""Replay — ReplayEngine unit tests using stub recorder and engine."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from theo_core.explanation.replay.replay_engine import ReplayEngine
from theo_core.telemetry.tracing.cognitive_trace import CognitiveTrace


@dataclass(frozen=True)
class StubOutput:
    """Stub process() result carrying a response text."""

    response_text: str


class StubEngine:
    """Records inputs and returns a fixed response text."""

    def __init__(self, response_text: str = "") -> None:
        self._response = response_text
        self.inputs: list[str] = []

    def process(self, raw_input: str) -> StubOutput:
        self.inputs.append(raw_input)
        return StubOutput(self._response)


class StubRecorder:
    """Returns a preset trace (or None) on load."""

    def __init__(self, trace: CognitiveTrace | None = None) -> None:
        self._trace = trace

    def load_trace(self, trace_id: str) -> CognitiveTrace | None:
        return self._trace


def _trace(raw_input: str, response_text: str) -> CognitiveTrace:
    return CognitiveTrace(
        trace_id=uuid4(),
        raw_input=raw_input,
        response_text=response_text,
    )


class TestReplayEngine:
    def test_missing_trace_reports_unmatched(self) -> None:
        engine = ReplayEngine(StubRecorder(None), StubEngine())
        result = engine.replay("missing-trace")

        assert not result.matched
        assert result.variance == 1.0
        assert result.original_output == "Trace not found"
        assert result.replayed_output == ""

    def test_matching_output_reports_zero_variance(self) -> None:
        recorder = StubRecorder(_trace("hello theo", "Hello Theo"))
        engine = ReplayEngine(recorder, StubEngine("Hello Theo"))
        result = engine.replay("any-id")

        assert result.matched
        assert result.variance == 0.0
        assert result.original_output == "Hello Theo"
        assert result.replayed_output == "Hello Theo"

    def test_mismatched_output_reports_full_variance(self) -> None:
        recorder = StubRecorder(_trace("hello theo", "Hello Theo"))
        engine = ReplayEngine(recorder, StubEngine("Different response"))
        result = engine.replay("any-id")

        assert not result.matched
        assert result.variance == 1.0
        assert result.replayed_output == "Different response"

    def test_replay_passes_raw_input_to_engine(self) -> None:
        recorder = StubRecorder(_trace("my name is falcon", "Falcon"))
        stub_engine = StubEngine("Falcon")
        engine = ReplayEngine(recorder, stub_engine)
        engine.replay("any-id")

        assert stub_engine.inputs == ["my name is falcon"]
