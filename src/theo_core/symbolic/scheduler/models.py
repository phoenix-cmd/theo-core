"""Domain models for the Cognitive Scheduler."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field


class CycleStage(StrEnum):
    """Execution stages of a single symbolic cognitive cycle."""

    PERCEPTION = "perception"
    ACTIVATION = "activation"
    REVISION = "revision"
    INFERENCE = "inference"
    HYPOTHESIS = "hypothesis"
    CONFLICT_RESOLUTION = "conflict_resolution"
    DECISION = "decision"
    REALIZATION = "realization"
    LEARNING = "learning"


class ComputeBudget(BaseModel, frozen=True):
    """Resource budget constraints for cognitive cycle execution."""

    max_iterations: int = Field(default=5, ge=1)
    max_depth: int = Field(default=5, ge=1)
    time_budget_ms: Decimal = Field(default=Decimal("1000.0"), ge=Decimal("0.0"))


class SchedulerTrace(BaseModel, frozen=True):
    """Trace log of scheduler stage execution and budget utilization.

    When no ``ComputeBudget`` is supplied, timing is disabled entirely and
    ``total_time_ms`` is ``None`` so that the trace is fully deterministic.
    """

    stages_executed: tuple[CycleStage, ...] = Field(default_factory=tuple)
    total_time_ms: Decimal | None = Field(default=None, ge=Decimal("0.0"))
    budget_exhausted: bool = False


ComputeBudget.model_rebuild()
SchedulerTrace.model_rebuild()
