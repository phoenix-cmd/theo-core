"""TrainingRun — a single training execution within an experiment."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TrainingRun(BaseModel):
    """A single training run with tracked metrics and checkpoints.

    Attributes:
        id: Unique run identifier.
        experiment_id: Parent experiment this run belongs to.
        config: Training configuration for this run.
        metrics: Dictionary of metric name to list of (step, value) tuples.
        started_at: UTC timestamp of run start.
        completed_at: UTC timestamp of run completion.
        metadata: Extensible metadata dictionary.

    """

    id: UUID = Field(default_factory=uuid4)
    experiment_id: UUID | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, list[tuple[int, float]]] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log a metric value at a given step.

        Args:
            name: The metric name.
            value: The metric value.
            step: The training step.

        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((step, value))
