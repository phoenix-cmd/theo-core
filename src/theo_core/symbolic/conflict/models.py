"""Domain models for Conflict Resolution."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from theo_core.symbolic._primitives.identifiers import SymbolicId  # noqa: TC001


class ConflictPolicy(StrEnum):
    """Strategies for resolving conflicting beliefs or hypotheses."""

    HIGHER_CONFIDENCE = "higher_confidence"
    RECENT_SOURCE = "recent_source"
    EVIDENCE_COUNT = "evidence_count"
    EXPLICIT_AUTHORITY = "explicit_authority"


class ConflictRecord(BaseModel, frozen=True):
    """Traceable record of a resolved contradiction enforcing Canon Law 7."""

    conflict_id: SymbolicId
    conflicting_ids: tuple[SymbolicId, ...]
    policy: ConflictPolicy
    winning_id: SymbolicId
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)


ConflictRecord.model_rebuild()
