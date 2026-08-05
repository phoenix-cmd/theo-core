"""DecisionRecord — rich, formal decision provenance record."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DecisionRecord(BaseModel):
    """Rich formal decision record capturing all selection details and provenance.

    Attributes:
        decision_id: Unique UUID identifier for the decision.
        selected_option: The candidate response text selected.
        candidate_options: Tuple of candidate response texts evaluated.
        selection_reason: Human-readable explanation of why this option was chosen.
        confidence: Confidence score between 0.0 and 1.0.
        rejected_candidates: Tuple of candidate response texts that were rejected.
        used_memory_ids: Tuple of memory IDs referenced in forming this decision.
        used_rule_ids: Tuple of stable rule IDs (e.g. 'RULE-0001') evaluated.
        used_goal: The goal description this decision addresses.
        trace_id: Optional trace ID associated with this decision cycle.
        metadata: Extensible metadata dictionary.

    """

    decision_id: UUID = Field(default_factory=uuid4)
    selected_option: str
    candidate_options: tuple[str, ...] = Field(default_factory=tuple)
    selection_reason: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    rejected_candidates: tuple[str, ...] = Field(default_factory=tuple)
    used_memory_ids: tuple[str, ...] = Field(default_factory=tuple)
    used_rule_ids: tuple[str, ...] = Field(default_factory=tuple)
    used_goal: str = "MaintainConversation"
    trace_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
