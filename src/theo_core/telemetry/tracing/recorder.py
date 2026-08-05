"""TraceRecorder — collects stage execution spans and computes execution statistics."""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

from theo_core.telemetry.tracing.cognitive_trace import CognitiveTrace, TraceSpan

if TYPE_CHECKING:
    from uuid import UUID


class TraceRecorder:
    """Collects stage spans, computes statistics, and writes CognitiveTrace JSON files."""

    def __init__(self, trace_dir: str = "data/traces") -> None:
        """Initialize TraceRecorder.

        Args:
            trace_dir: Path to directory where trace JSON files are saved.

        """
        self._trace_dir = trace_dir
        os.makedirs(self._trace_dir, exist_ok=True)
        self._spans: list[TraceSpan] = []
        self._stage_start_times: dict[str, float] = {}
        self._cycle_start_time: float = 0.0

    def start_cycle(self) -> None:
        """Start tracking a new cognitive cycle iteration."""
        self._spans.clear()
        self._stage_start_times.clear()
        self._cycle_start_time = time.perf_counter()

    def start_stage(self, stage_name: str) -> None:
        """Record the start time of a cognitive stage.

        Args:
            stage_name: Name of the cognitive stage.

        """
        self._stage_start_times[stage_name] = time.perf_counter()

    def end_stage(
        self,
        stage_name: str,
        input_summary: str = "",
        output_summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record the completion of a cognitive stage and append a TraceSpan.

        Args:
            stage_name: Name of the cognitive stage.
            input_summary: Brief summary of stage input.
            output_summary: Brief summary of stage output.
            metadata: Extensible stage details dictionary.

        """
        start_time = self._stage_start_times.get(stage_name, time.perf_counter())
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        span = TraceSpan(
            stage_name=stage_name,
            duration_ms=round(duration_ms, 3),
            input_summary=input_summary,
            output_summary=output_summary,
            metadata=metadata or {},
        )
        self._spans.append(span)

    def close_trace(
        self,
        cycle_id: UUID,
        raw_input: str,
        response_text: str,
        execution_stats: dict[str, Any] | None = None,
    ) -> CognitiveTrace:
        """Close the active trace, compute total latency, and write to JSON file.

        Args:
            cycle_id: UUID of the cognitive cycle.
            raw_input: Raw user input text string.
            response_text: Final response text string.
            execution_stats: Optional dictionary of quantified execution metrics.

        Returns:
            The saved CognitiveTrace object.

        """
        total_ms = (time.perf_counter() - self._cycle_start_time) * 1000.0
        stats = execution_stats or {}
        stats["pipeline_latency_ms"] = round(total_ms, 3)
        stats["stage_count"] = len(self._spans)

        trace = CognitiveTrace(
            cycle_id=cycle_id,
            raw_input=raw_input,
            response_text=response_text,
            spans=tuple(self._spans),
            total_duration_ms=round(total_ms, 3),
            execution_stats=stats,
        )

        trace_path = os.path.join(self._trace_dir, f"{trace.trace_id}.json")
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace.model_dump(mode="json"), f, indent=2)

        return trace

    def load_trace(self, trace_id: str | UUID) -> CognitiveTrace | None:
        """Load a CognitiveTrace object from disk by trace ID.

        Args:
            trace_id: UUID or string trace ID.

        Returns:
            CognitiveTrace object if found, or None.

        """
        trace_path = os.path.join(self._trace_dir, f"{trace_id}.json")
        if not os.path.exists(trace_path):
            return None

        try:
            with open(trace_path, encoding="utf-8") as f:
                data = json.load(f)
                return CognitiveTrace(**data)
        except Exception:
            return None
