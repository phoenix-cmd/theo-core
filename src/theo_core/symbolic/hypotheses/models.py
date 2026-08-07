"""Domain models for the Hypothesis Engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.models import BeliefId  # noqa: TC001
from theo_core.symbolic.thoughts.models import ThoughtId  # noqa: TC001


@dataclass(frozen=True, slots=True)
class HypothesisId:
    """URI-style hypothesis identifier value object.

    Wraps SymbolicId with hypothesis-specific URI validation (must start with 'hypothesis://').
    """

    value: str

    @classmethod
    def of(cls, value: str) -> HypothesisId:
        """Validate hypothesis URI scheme and return a new HypothesisId."""
        if not value.startswith("hypothesis://"):
            msg = f"HypothesisId URI must start with 'hypothesis://', got {value!r}"
            raise ValueError(msg)
        symbolic_id = SymbolicId.of(value)
        return cls(value=symbolic_id.value)

    def to_symbolic_id(self) -> SymbolicId:
        """Convert to primitive SymbolicId."""
        return SymbolicId(value=self.value)

    def __str__(self) -> str:
        """Return raw string value."""
        return self.value


class HypothesisState(StrEnum):
    """Lifecycle state of a candidate Hypothesis."""

    CANDIDATE = "candidate"
    EVALUATED = "evaluated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PRUNED = "pruned"


class Hypothesis(BaseModel, frozen=True):
    """An immutable candidate interpretation enforcing Canon Law 6."""

    id: HypothesisId
    interpretation: str
    score: Decimal = Field(default=Decimal("0.5"), ge=Decimal("0.0"), le=Decimal("1.0"))
    state: HypothesisState = HypothesisState.CANDIDATE
    supporting_thoughts: tuple[ThoughtId, ...] = Field(default_factory=tuple)
    supporting_beliefs: tuple[BeliefId, ...] = Field(default_factory=tuple)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


Hypothesis.model_rebuild()
