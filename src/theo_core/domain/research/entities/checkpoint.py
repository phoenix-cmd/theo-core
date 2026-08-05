"""Checkpoint — a model state snapshot with lineage tracking."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Checkpoint(BaseModel, frozen=True):
    """An immutable snapshot of model state at a point in time.

    Attributes:
        id: Unique checkpoint identifier.
        run_id: The training run that produced this checkpoint.
        step: The training step at which this was saved.
        path: File system path to the checkpoint file.
        metrics_snapshot: Metrics at the time of checkpointing.
        created_at: UTC timestamp.
        metadata: Extensible metadata dictionary.

    """

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID | None = None
    step: int = 0
    path: str = ""
    metrics_snapshot: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
