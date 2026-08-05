"""Percept — normalized internal representation of raw input.

All external inputs (text, images, audio, documents) are converted into
Percept objects before entering the cognitive pipeline. This ensures
the cognitive system never directly processes raw input formats.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class PerceptModality(StrEnum):
    """The sensory modality of a percept."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    MULTIMODAL = "multimodal"


class Percept(BaseModel, frozen=True):
    """A normalized internal representation of raw input.

    Attributes:
        id: Unique percept identifier.
        modality: The sensory modality of this input.
        content: The normalized content representation.
        raw_source: Optional reference to the original raw source.
        confidence: Confidence in the perceptual processing quality.
        timestamp: UTC timestamp of percept creation.
        metadata: Extensible metadata dictionary.

    """

    id: UUID = Field(default_factory=uuid4)
    modality: PerceptModality
    content: str
    raw_source: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
