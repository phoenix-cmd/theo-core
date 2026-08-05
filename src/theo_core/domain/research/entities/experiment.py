"""Experiment — a research experiment lifecycle aggregate.

An Experiment captures the full lifecycle from hypothesis to results,
including configuration snapshot, training runs, and evaluation outcomes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ExperimentStatus(StrEnum):
    """Lifecycle status of an experiment."""

    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Experiment(BaseModel):
    """A research experiment aggregate.

    Attributes:
        id: Unique experiment identifier.
        name: Human-readable experiment name.
        description: Description of the experiment hypothesis.
        config_snapshot: Frozen copy of the configuration used.
        status: Current lifecycle status.
        created_at: UTC timestamp of experiment creation.
        completed_at: UTC timestamp of experiment completion.
        tags: Tags for filtering and organization.
        metadata: Extensible metadata dictionary.

    """

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    status: ExperimentStatus = ExperimentStatus.PLANNED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
