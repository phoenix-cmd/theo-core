"""Domain models for the Decision Engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.hypotheses.models import HypothesisId  # noqa: TC001
from theo_core.symbolic.thoughts.models import ThoughtId  # noqa: TC001


@dataclass(frozen=True, slots=True)
class DecisionId:
    """URI-style decision identifier value object.

    Wraps SymbolicId with decision-specific URI validation (must start with 'decision://').
    """

    value: str

    @classmethod
    def of(cls, value: str) -> DecisionId:
        """Validate decision URI scheme and return a new DecisionId."""
        if not value.startswith("decision://"):
            msg = f"DecisionId URI must start with 'decision://', got {value!r}"
            raise ValueError(msg)
        symbolic_id = SymbolicId.of(value)
        return cls(value=symbolic_id.value)

    def to_symbolic_id(self) -> SymbolicId:
        """Convert to primitive SymbolicId."""
        return SymbolicId(value=self.value)

    def __str__(self) -> str:
        """Return raw string value."""
        return self.value


class DecisionType(StrEnum):
    """Category of decision selected by the Cognitive Engine."""

    ACTION = "action"
    RESPONSE = "response"
    DEFER = "defer"
    NO_OP = "no_op"


class DecisionRecord(BaseModel, frozen=True):
    """An immutable decision selection record enforcing Canon Law 2 and Invariant 2."""

    id: DecisionId
    type: DecisionType = DecisionType.RESPONSE
    action_text: str
    confidence: Decimal = Field(default=Decimal("1.0"), ge=Decimal("0.0"), le=Decimal("1.0"))
    referenced_thoughts: tuple[ThoughtId, ...] = Field(default_factory=tuple)
    accepted_hypothesis_id: HypothesisId | None = None
    active_goal_id: SymbolicId | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


DecisionRecord.model_rebuild()
