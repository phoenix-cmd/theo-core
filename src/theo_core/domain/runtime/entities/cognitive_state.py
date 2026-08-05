"""CognitiveState — snapshot of Theo's active cognitive state.

Captures the current working memory buffer, active identity traits,
and any active goals at a given moment.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class CognitiveState(BaseModel):
    """A snapshot of Theo's active cognitive state.

    Attributes:
        id: Unique state snapshot identifier.
        active_context: Summary of the current active context.
        working_memory_keys: Keys currently held in working memory.
        active_goals: IDs of currently active goals.
        identity_traits: Active identity trait values.
        timestamp: UTC timestamp of this state snapshot.
        metadata: Extensible metadata dictionary.

    """

    id: UUID = Field(default_factory=uuid4)
    active_context: str = ""
    working_memory_keys: list[str] = Field(default_factory=list)
    active_goals: list[UUID] = Field(default_factory=list)
    identity_traits: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
