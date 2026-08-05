"""EmbeddingVector — an immutable embedding with dimensionality and source."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EmbeddingVector(BaseModel, frozen=True):
    """An immutable embedding vector with provenance.

    Attributes:
        values: The raw float values of the embedding.
        model_name: Name of the model that produced this embedding.

    """

    values: tuple[float, ...] = Field(default_factory=tuple)
    model_name: str = "unknown"

    @property
    def dimension(self) -> int:
        """Return the dimensionality of this embedding."""
        return len(self.values)
