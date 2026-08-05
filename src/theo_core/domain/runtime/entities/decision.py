"""Decision — rich decision entity produced by the Decision Engine."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Decision(BaseModel):
    """The formulated decision output of a cognitive cycle.

    Attributes:
        id: Unique UUID identifier for the decision.
        response: Candidate response text payload.
        confidence: Confidence score between 0.0 and 1.0.
        goal: The goal description this decision addresses.
        actions_taken: List of capabilities or action names executed.
        used_memory_ids: Tuple of memory IDs referenced in forming this decision.
        memory_updates: Dict of memory entries queued for creation.
        knowledge_updates: Dict of knowledge graph updates queued.
        reasoning_summary: Human-readable summary of the cognitive policy evaluation.
        metadata: Extensible metadata dictionary.

    """

    id: UUID = Field(default_factory=uuid4)
    response: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    goal: str = "MaintainConversation"
    actions_taken: tuple[str, ...] = Field(default_factory=tuple)
    used_memory_ids: tuple[str, ...] = Field(default_factory=tuple)
    memory_updates: dict[str, Any] = Field(default_factory=dict)
    knowledge_updates: dict[str, Any] = Field(default_factory=dict)
    reasoning_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
