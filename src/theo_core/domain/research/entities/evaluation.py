"""Benchmark and EvaluationResult — evaluation framework entities."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Metric(BaseModel, frozen=True):
    """A single metric measurement.

    Attributes:
        name: Metric name (e.g. "accuracy", "perplexity").
        value: The measured value.
        unit: Unit of measurement (e.g. "percent", "tokens/sec").
        higher_is_better: Whether a higher value indicates improvement.

    """

    name: str
    value: float
    unit: str = ""
    higher_is_better: bool = True


class EvaluationResult(BaseModel, frozen=True):
    """An immutable evaluation result for a single benchmark run.

    Attributes:
        id: Unique evaluation identifier.
        benchmark_name: Name of the benchmark evaluated against.
        model_version: Version of the model being evaluated.
        metrics: List of measured metrics.
        timestamp: UTC timestamp of evaluation.
        metadata: Extensible metadata dictionary.

    """

    id: UUID = Field(default_factory=uuid4)
    benchmark_name: str
    model_version: str = "0.1.0"
    metrics: tuple[Metric, ...] = Field(default_factory=tuple)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Benchmark(BaseModel):
    """A benchmark definition for evaluating cognitive capabilities.

    Attributes:
        id: Unique benchmark identifier.
        name: Human-readable benchmark name.
        description: What this benchmark measures.
        metric_names: The metrics this benchmark reports.
        metadata: Extensible metadata dictionary.

    """

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    metric_names: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
