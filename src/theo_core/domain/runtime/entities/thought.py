"""Thought — an internal cognitive unit produced during reasoning.

A Thought is an intermediate cognitive artifact that Theo generates
as part of its reasoning process. Thoughts have a confidence score
and a traceable source subsystem.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Thought(BaseModel, frozen=True):
    """An immutable internal cognitive unit.

    Attributes:
        id: Unique thought identifier.
        content: The textual content of this thought.
        confidence: Confidence score between 0.0 and 1.0.
        source: The subsystem or process that generated this thought.
        timestamp: UTC timestamp of thought generation.
        metadata: Extensible metadata dictionary.

    """

    id: UUID = Field(default_factory=uuid4)
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
