"""CognitiveWorkspace — intermediate shared workspace for cognitive execution cycles."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class CognitiveWorkspace(BaseModel):
    """Encapsulates the intermediate shared workspace during a cognitive cycle.

    All 12 stages read from and write to this shared workspace.

    Attributes:
        workspace_id: Unique identifier for this cycle's workspace instance.
        active_goal_id: ID of the active Goal.
        retrieved_memory_ids: Tuple of memory IDs retrieved during this cycle.
        used_memory_ids: Tuple of memory IDs actually consumed for decision-making.
        candidate_thoughts: List of intermediate thoughts generated.
        current_plan_id: ID of the generated Plan.
        reflection_notes: Dictionary of reflection outputs.

    """

    workspace_id: UUID = Field(default_factory=uuid4)
    active_goal_id: UUID | None = None
    retrieved_memory_ids: tuple[str, ...] = Field(default_factory=tuple)
    used_memory_ids: tuple[str, ...] = Field(default_factory=tuple)
    candidate_thoughts: list[dict[str, Any]] = Field(default_factory=list)
    current_plan_id: UUID | None = None
    reflection_notes: dict[str, Any] = Field(default_factory=dict)
