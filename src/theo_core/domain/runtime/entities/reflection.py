"""Reflection — output of Theo's self-assessment process.

A Reflection captures insights produced when Theo evaluates its own
thoughts, reasoning quality, or behavioral patterns.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from theo_core.domain.runtime.entities.thought import Thought  # noqa: TC001


class Reflection(BaseModel, frozen=True):
    """An immutable self-reflection result.

    Attributes:
        id: Unique reflection identifier.
        source_thoughts: The thoughts that triggered this reflection.
        insights: List of insight strings produced by self-assessment.
        confidence: Overall confidence in the reflection quality.
        timestamp: UTC timestamp of reflection completion.
        metadata: Extensible metadata dictionary.

    """

    id: UUID = Field(default_factory=uuid4)
    source_thoughts: tuple[Thought, ...] = Field(default_factory=tuple)
    insights: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
