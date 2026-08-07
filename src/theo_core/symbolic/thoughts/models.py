"""Domain models for the Thought Graph reasoning DAG."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.models import BeliefId  # noqa: TC001


@dataclass(frozen=True, slots=True)
class ThoughtId:
    """URI-style thought identifier value object.

    Wraps SymbolicId with thought-specific URI validation (must start with 'thought://').
    """

    value: str

    @classmethod
    def of(cls, value: str) -> ThoughtId:
        """Validate thought URI scheme and return a new ThoughtId."""
        if not value.startswith("thought://"):
            msg = f"ThoughtId URI must start with 'thought://', got {value!r}"
            raise ValueError(msg)
        symbolic_id = SymbolicId.of(value)
        return cls(value=symbolic_id.value)

    def to_symbolic_id(self) -> SymbolicId:
        """Convert to primitive SymbolicId."""
        return SymbolicId(value=self.value)

    def __str__(self) -> str:
        """Return raw string value."""
        return self.value


class ThoughtRelation(StrEnum):
    """Relationship between thoughts in the Thought Graph DAG."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"
    DERIVED_FROM = "derived_from"
    REPLACES = "replaces"
    INVALIDATES = "invalidates"


class ThoughtEdge(BaseModel, frozen=True):
    """A typed, weighted, immutable edge connecting two thoughts."""

    source: ThoughtId
    target: ThoughtId
    relation: ThoughtRelation = ThoughtRelation.DERIVED_FROM
    weight: Decimal = Field(default=Decimal("1.0"), ge=Decimal("0.0"))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Thought(BaseModel, frozen=True):
    """An immutable internal reasoning unit, enforcing Canon Invariant 1.

    Consumes Beliefs per Law 3 and provides evidence links for Decisions per Law 2.
    """

    id: ThoughtId
    content: str
    confidence: Decimal = Field(default=Decimal("1.0"), ge=Decimal("0.0"), le=Decimal("1.0"))
    consumed_beliefs: tuple[BeliefId, ...] = Field(default_factory=tuple)
    evidence_links: tuple[SymbolicId, ...] = Field(default_factory=tuple)
    source_subsystem: str = "inference"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_valid: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


Thought.model_rebuild()
