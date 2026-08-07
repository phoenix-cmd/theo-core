"""CognitiveTrace and TraceSpan models for execution visibility and provenance."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from theo_core._version import __version__


class TraceSpan(BaseModel):
    """Represents an execution span for one stage in the 12-stage cognitive cycle.

    Attributes:
        span_id: Unique UUID identifier for this stage span.
        stage_name: Name of the cognitive stage (e.g. 'perception', 'inference').
        timestamp: UTC timestamp when the span executed.
        duration_ms: Execution duration in milliseconds.
        input_summary: Human-readable summary of stage inputs.
        output_summary: Human-readable summary of stage outputs.
        metadata: Stage-specific details dictionary.

    """

    span_id: UUID = Field(default_factory=uuid4)
    stage_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float = 0.0
    input_summary: str = ""
    output_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CognitiveTrace(BaseModel):
    """Full execution trace of one 12-stage cognitive cycle with provenance snapshots.

    Attributes:
        trace_id: Unique UUID identifier for the cognitive trace.
        cycle_id: UUID identifier of the cognitive cycle.
        raw_input: Raw user input text string.
        response_text: Final response text string generated.
        spans: Tuple of TraceSpan objects covering all 12 stages.
        total_duration_ms: Total cycle execution latency in milliseconds.
        config_snapshot: Version and hash metadata snapshot.
        execution_stats: Quantified metrics (retrievals, rules matched, latency).
        created_at: UTC creation timestamp.
        metadata: Extensible metadata dictionary.

    """

    trace_id: UUID = Field(default_factory=uuid4)
    cycle_id: UUID = Field(default_factory=uuid4)
    raw_input: str
    response_text: str = ""
    spans: tuple[TraceSpan, ...] = Field(default_factory=tuple)
    total_duration_ms: float = 0.0
    config_snapshot: dict[str, str] = Field(
        default_factory=lambda: {
            "theo_version": __version__,
            "rule_set_version": "v0.2.0",
            "memory_policy_version": "v0.2.0",
            "config_hash": "sha256-deterministic-v0.2",
        }
    )
    execution_stats: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
