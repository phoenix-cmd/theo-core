"""Goal — a purposeful objective that drives Theo's planning.

Goals answer "Why?" while Plans answer "How?". Goals are managed
by the GoalManager and can be stacked by priority.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class GoalStatus(StrEnum):
    """Lifecycle status of a goal."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class GoalPriority(StrEnum):
    """Priority level for goal scheduling."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


class Goal(BaseModel):
    """A purposeful objective that drives cognitive behavior.

    Attributes:
        id: Unique goal identifier.
        description: Natural language description of the goal.
        priority: Scheduling priority for this goal.
        status: Current lifecycle status.
        parent_id: Optional parent goal for hierarchical decomposition.
        created_at: UTC timestamp of goal creation.
        metadata: Extensible metadata dictionary.

    """

    id: UUID = Field(default_factory=uuid4)
    description: str
    priority: GoalPriority = GoalPriority.MEDIUM
    status: GoalStatus = GoalStatus.ACTIVE
    parent_id: UUID | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
