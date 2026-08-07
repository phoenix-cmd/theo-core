"""ReplayEngine — replays historical traces and verifies 0-variance determinism.

The engine depends on a ``ReplayableEngine`` protocol so both the legacy
``CognitiveEngine`` and the canonical ``SymbolicRuntime`` can be driven.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    from theo_core.telemetry.tracing.cognitive_trace import CognitiveTrace
    from theo_core.telemetry.tracing.recorder import TraceRecorder


class ReplayProcessOutput(Protocol):
    """Structural protocol for any process() result that carries response text."""

    response_text: str


class ReplayableEngine(Protocol):
    """Structural protocol for an engine the ReplayEngine can drive.

    Both the legacy ``CognitiveEngine`` and the canonical ``SymbolicRuntime``
    satisfy this protocol via their ``process`` methods.
    """

    def process(self, raw_input: str) -> ReplayProcessOutput:
        """Execute one cognitive cycle over raw input."""
        ...


class ReplayResult(BaseModel):
    """Result of a cognitive replay comparison test.

    Attributes:
        trace_id: ID of the replayed trace.
        matched: True if replayed output matches original output 100%.
        variance: Output variance score (0.0 means 100% deterministic match).
        original_output: The response text from the original trace.
        replayed_output: The response text produced during replay.

    """

    trace_id: str
    matched: bool
    variance: float = 0.0
    original_output: str
    replayed_output: str


class ReplayEngine:
    """Loads recorded CognitiveTrace files and replays them to verify determinism."""

    def __init__(self, recorder: TraceRecorder, engine: ReplayableEngine) -> None:
        """Initialize ReplayEngine.

        Args:
            recorder: TraceRecorder instance to load trace files.
            engine: Replayable engine instance to execute replay.

        """
        self._recorder = recorder
        self._engine = engine

    def replay(self, trace_id: str) -> ReplayResult:
        """Replay a recorded trace and verify deterministic output match.

        Args:
            trace_id: ID string of the trace to load and replay.

        Returns:
            ReplayResult object showing match status and variance.

        """
        trace: CognitiveTrace | None = self._recorder.load_trace(trace_id)
        if trace is None:
            return ReplayResult(
                trace_id=trace_id,
                matched=False,
                variance=1.0,
                original_output="Trace not found",
                replayed_output="",
            )

        # Re-run cognitive cycle on original raw input
        replayed_state = self._engine.process(trace.raw_input)
        replayed_output = replayed_state.response_text
        matched = replayed_output == trace.response_text
        variance = 0.0 if matched else 1.0

        return ReplayResult(
            trace_id=str(trace.trace_id),
            matched=matched,
            variance=variance,
            original_output=trace.response_text,
            replayed_output=replayed_output,
        )
