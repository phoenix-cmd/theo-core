"""DatasetManifest and DatasetSample — versioned dataset metadata and samples."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DatasetManifest(BaseModel, frozen=True):
    """Immutable metadata describing a versioned dataset.

    Attributes:
        id: Unique dataset identifier.
        name: Human-readable dataset name.
        version: Semantic version string.
        checksum: Content hash for integrity verification.
        source: URI or description of the data source.
        license: License governing this dataset.
        sample_count: Total number of samples.
        quality_score: Optional quality assessment score (0.0 to 1.0).
        created_at: UTC timestamp.
        metadata: Extensible metadata dictionary.

    """

    id: UUID = Field(default_factory=uuid4)
    name: str
    version: str = "0.1.0"
    checksum: str = ""
    source: str = ""
    license: str = "unknown"
    sample_count: int = 0
    quality_score: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetSample(BaseModel, frozen=True):
    """An immutable individual sample from a dataset.

    Attributes:
        id: Unique sample identifier.
        features: The input features.
        labels: The expected labels.
        split: The dataset split (train, val, test).
        metadata: Extensible metadata dictionary.

    """

    id: UUID = Field(default_factory=uuid4)
    features: dict[str, Any] = Field(default_factory=dict)
    labels: dict[str, Any] = Field(default_factory=dict)
    split: str = "train"
    metadata: dict[str, Any] = Field(default_factory=dict)
