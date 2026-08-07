"""ReplayEngine — replays historical traces and verifies 0-variance determinism.

The engine depends on a ``ReplayableEngine`` protocol so both the legacy
``CognitiveEngine`` and the canonical ``SymbolicRuntime`` can be driven.

For the canonical path, each recorded trace carries a structural golden
fingerprint (decision, fired rules, derived beliefs, hypotheses, thought DAG)
plus the pre-cycle committed state. Replay re-runs the raw input from that
recorded pre-cycle state on a fresh runtime and compares the regenerated
fingerprint field-for-field with the recorded one — a rendered-text match alone
is not sufficient, and replay must never re-run against a live runtime whose
state has already advanced. Traces without a recorded fingerprint (e.g. legacy
spans-based traces) fall back to the response-text comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from theo_core.evaluation.benchmark_schema import (
    FINGERPRINT_METADATA_KEY,
    PRE_CYCLE_STATE_METADATA_KEY,
    golden_fingerprint,
)
from theo_core.symbolic.persistence.store import deserialize_cycle_state

if TYPE_CHECKING:
    from collections.abc import Callable

    from theo_core.symbolic.scheduler.models import ComputeBudget
    from theo_core.telemetry.tracing.cognitive_trace import CognitiveTrace
    from theo_core.telemetry.tracing.recorder import TraceRecorder


class ReplayProcessOutput(Protocol):
    """Structural protocol for any process() result that carries response text.

    Declared read-only so frozen results (e.g. ``SymbolicRuntimeResult``)
    satisfy it; legacy results with plain writable attributes do too.
    """

    @property
    def response_text(self) -> str:
        """The rendered response text of a processed cycle."""
        ...


class ReplayableEngine(Protocol):
    """Structural protocol for an engine the ReplayEngine can drive.

    The canonical ``SymbolicRuntime`` satisfies this protocol directly. The
    legacy ``CognitiveEngine`` is duck-typed through the same method when used
    via the instance path (its process signature carries no budget).
    """

    def process(
        self,
        percept_input: str,
        budget: ComputeBudget | None = None,
    ) -> ReplayProcessOutput:
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

    def __init__(
        self,
        recorder: TraceRecorder,
        engine: ReplayableEngine | None = None,
        engine_factory: Callable[[], ReplayableEngine] | None = None,
    ) -> None:
        """Initialize ReplayEngine.

        Exactly one of ``engine`` or ``engine_factory`` must be provided.

        Args:
            recorder: TraceRecorder instance to load trace files.
            engine: Replayable engine instance to execute replay in place.
                Used for the legacy path and in tests; replays run against
                this instance's current state.
            engine_factory: Callable that builds a fresh replayable engine.
                Used for the canonical path so replay never re-runs against a
                live runtime's advanced state; the recorded pre-cycle state is
                restored into the fresh engine before the cycle is re-run.

        Raises:
            ValueError: If neither ``engine`` nor ``engine_factory`` is given.

        """
        if engine is None and engine_factory is None:
            msg = "ReplayEngine requires either an engine instance or an engine factory."
            raise ValueError(msg)
        self._recorder = recorder
        self._engine = engine
        self._engine_factory = engine_factory

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

        # Re-run cognitive cycle on original raw input from a faithful engine
        engine = self._build_engine(trace)
        replayed_state = engine.process(trace.raw_input)
        replayed_output = replayed_state.response_text
        matched = replayed_output == trace.response_text
        variance = 0.0 if matched else 1.0

        # Canonical path: compare the full golden fingerprint when recorded.
        recorded_fingerprint = trace.metadata.get(FINGERPRINT_METADATA_KEY)
        replayed_fingerprint = _fingerprint_of(replayed_state)
        if (
            recorded_fingerprint is not None
            and replayed_fingerprint is not None
            and recorded_fingerprint != replayed_fingerprint
        ):
            matched = False
            variance = 1.0

        return ReplayResult(
            trace_id=str(trace.trace_id),
            matched=matched,
            variance=variance,
            original_output=trace.response_text,
            replayed_output=replayed_output,
        )

    def _build_engine(self, trace: CognitiveTrace) -> ReplayableEngine:
        """Return the engine to drive this replay, restoring pre-cycle state.

        The factory path builds a fresh engine and restores the trace's recorded
        pre-cycle committed state into it before the cycle is re-run, so replay
        is faithful even for later turns of a multi-turn session. The instance
        path replays in place for the legacy path and in tests.
        """
        if self._engine_factory is not None:
            engine = self._engine_factory()
            self._restore_pre_cycle_state(engine, trace)
            return engine
        if self._engine is None:
            msg = "ReplayEngine has no engine configured."
            raise RuntimeError(msg)
        return self._engine

    def _restore_pre_cycle_state(self, engine: ReplayableEngine, trace: CognitiveTrace) -> None:
        """Restore the trace's pre-cycle committed state into a fresh engine.

        Raises:
            RuntimeError: If the trace carries pre-cycle state but the engine
                cannot restore it — replay would otherwise be unfaithful.

        """
        state_data = trace.metadata.get(PRE_CYCLE_STATE_METADATA_KEY)
        if state_data is None:
            return
        restore = getattr(engine, "restore_state", None)
        if restore is None:
            msg = (
                "Trace carries pre-cycle state but the replay engine cannot "
                "restore it; replay would be unfaithful."
            )
            raise RuntimeError(msg)
        restore(deserialize_cycle_state(state_data))


def _fingerprint_of(state: object) -> dict[str, object] | None:
    """Project a replayed result's golden trace into the canonical fingerprint.

    Returns None when the replayed result carries no golden trace (legacy path),
    in which case replay relies on the response-text comparison alone.
    """
    golden_trace = getattr(state, "golden_trace", None)
    if golden_trace is None:
        return None
    return golden_fingerprint(golden_trace, getattr(state, "response_text", ""))
