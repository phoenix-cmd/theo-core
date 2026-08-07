"""Goal — a purposeful objective that drives Theo's planning.

Goals answer "Why?" while Plans answer "How?". Goals are managed
by the GoalManager and can be stacked by priority.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

_GOAL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*://[a-z0-9_./-]+$")


@dataclass(frozen=True, slots=True)
class GoalId:
    """Deterministic URI-style goal identifier (Canon Invariant 7).

    Derived from the goal description so identical goals share identical
    identifiers across runs and processes.
    """

    value: str

    @classmethod
    def of(cls, value: str) -> GoalId:
        """Validate the goal URI scheme and return a new GoalId."""
        if not value.startswith("goal://"):
            msg = f"GoalId URI must start with 'goal://', got {value!r}"
            raise ValueError(msg)
        if not _GOAL_ID_PATTERN.match(value):
            msg = (
                f"Invalid GoalId URI: {value!r}. "
                f"Expected format: goal://path (lowercase alphanumeric, _, ., -, /)"
            )
            raise ValueError(msg)
        return cls(value=value)

    def __str__(self) -> str:
        """Return the raw URI string."""
        return self.value


def _slugify(description: str) -> str:
    """Lowercase a description into a URI-safe path segment."""
    slug = re.sub(r"[^a-z0-9]+", "_", description.lower()).strip("_")
    return slug or "default"


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
    goal_id: GoalId | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _derive_goal_id(self) -> Goal:
        """Derive the deterministic GoalId from the description when unset."""
        if self.goal_id is None:
            self.goal_id = GoalId.of(f"goal://{_slugify(self.description)}")
        return self
