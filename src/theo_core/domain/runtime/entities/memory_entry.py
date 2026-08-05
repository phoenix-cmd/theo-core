"""MemoryEntry — rich, immutable memory record with unique ID and provenance metadata."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class MemoryImportance(StrEnum):
    """Memory importance levels for retention and retrieval scoring."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PERMANENT = "permanent"


class MemoryEntry(BaseModel):
    """A rich memory entry with unique identifier and provenance metadata.

    Immutable design: memories are appended, never directly edited or deleted.
    When a fact changes, the old memory status becomes 'superseded' and a new
    memory entry is created pointing to its predecessor.

    Attributes:
        id: Unique memory entry identifier (e.g. 'mem-000001').
        memory_type: Memory classification ('identity', 'preference', 'experience', 'working').
        key: Key string identifying the memory (e.g. 'user.name').
        value: The memory payload value (e.g. 'Falcon').
        importance: Importance score level ('low', 'medium', 'high', 'permanent').
        confidence: Confidence score between 0.0 and 1.0.
        source: Provenance description (e.g. 'user_statement').
        status: Memory status ('active', 'superseded', 'expired').
        superseded_by: Optional ID of the memory entry that superseded this one.
        provenance: Detailed provenance chain mapping (message_id, percept_id, decision_id).
        created_at: UTC timestamp when created.
        last_verified: UTC timestamp when last verified.
        times_confirmed: Number of times confirmed in dialogue.
        metadata: Extensible metadata dictionary.

    """

    id: str
    memory_type: str = "identity"
    key: str
    value: Any
    importance: MemoryImportance = MemoryImportance.MEDIUM
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    source: str = "user_statement"
    status: str = "active"
    superseded_by: str | None = None
    provenance: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_verified: datetime = Field(default_factory=lambda: datetime.now(UTC))
    times_confirmed: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)

    def mark_superseded(self, new_memory_id: str) -> None:
        """Mark this memory entry as superseded by a newer entry.

        Args:
            new_memory_id: The ID of the memory entry superseding this one.

        """
        self.status = "superseded"
        self.superseded_by = new_memory_id

    def confirm(self) -> None:
        """Increment confirmation count and update last_verified timestamp."""
        self.times_confirmed += 1
        self.last_verified = datetime.now(UTC)
