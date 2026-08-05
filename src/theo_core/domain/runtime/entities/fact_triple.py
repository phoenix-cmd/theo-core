"""FactTriple — Entity-Predicate-Entity graph relationship triple."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class FactTriple(BaseModel):
    """Represents a structured relationship triple in the Knowledge Graph.

    Entity -> Predicate -> Entity (e.g. Falcon --likes--> Astronomy).

    Attributes:
        subject: Subject entity string (e.g. 'Falcon').
        predicate: Predicate relationship string (e.g. 'likes', 'related_to', 'requires').
        object: Object entity string (e.g. 'Astronomy').
        confidence: Confidence score between 0.0 and 1.0.
        source: Provenance description (e.g. 'memory_inference').
        created_at: UTC creation timestamp.
        metadata: Extensible metadata dictionary.

    """

    subject: str
    predicate: str
    object: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    source: str = "memory_inference"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
