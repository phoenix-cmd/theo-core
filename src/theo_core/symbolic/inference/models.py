"""Domain models for the Inference Engine."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.models import BeliefId  # noqa: TC001
from theo_core.symbolic.concepts.models import ConceptId  # noqa: TC001
from theo_core.symbolic.thoughts.models import ThoughtId  # noqa: TC001


@dataclass(frozen=True, slots=True)
class RuleId:
    """URI-style rule identifier value object.

    Wraps SymbolicId with rule-specific URI validation (must start with 'rule://').
    """

    value: str

    @classmethod
    def of(cls, value: str) -> RuleId:
        """Validate rule URI scheme and return a new RuleId."""
        if not value.startswith("rule://"):
            msg = f"RuleId URI must start with 'rule://', got {value!r}"
            raise ValueError(msg)
        symbolic_id = SymbolicId.of(value)
        return cls(value=symbolic_id.value)

    def to_symbolic_id(self) -> SymbolicId:
        """Convert to primitive SymbolicId."""
        return SymbolicId(value=self.value)

    def __str__(self) -> str:
        """Return raw string value."""
        return self.value


class InferenceMode(StrEnum):
    """Execution mode of the Inference Engine."""

    FORWARD_CHAINING = "forward_chaining"
    BACKWARD_CHAINING = "backward_chaining"


class RuleCondition(BaseModel, frozen=True):
    """Premise condition that must be satisfied for an InferenceRule to fire."""

    premise_predicate: str
    required_concept_ids: tuple[ConceptId, ...] = Field(default_factory=tuple)
    min_confidence: Decimal = Field(default=Decimal("0.5"), ge=Decimal("0.0"), le=Decimal("1.0"))


class InferenceRule(BaseModel, frozen=True):
    """An explicit symbolic deduction rule enforcing Canon Law 5."""

    id: RuleId
    name: str
    conditions: tuple[RuleCondition, ...]
    conclusion_template: str
    confidence_multiplier: Decimal = Field(
        default=Decimal("0.9"), ge=Decimal("0.0"), le=Decimal("1.0")
    )
    salience: Decimal = Field(default=Decimal("1.0"), ge=Decimal("0.0"))
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceStep(BaseModel, frozen=True):
    """Traceable deduction step record enforcing Canon Law 5."""

    step_id: SymbolicId
    rule_id: RuleId
    matched_belief_ids: tuple[BeliefId, ...]
    produced_thought_id: ThoughtId
    produced_belief_id: BeliefId | None = None
    confidence: Decimal = Field(default=Decimal("1.0"), ge=Decimal("0.0"), le=Decimal("1.0"))


class InferenceTrace(BaseModel, frozen=True):
    """Full execution trace of an inference cycle."""

    steps: tuple[InferenceStep, ...] = Field(default_factory=tuple)
    mode: InferenceMode = InferenceMode.FORWARD_CHAINING
    execution_time_ms: Decimal = Field(default=Decimal("0.0"), ge=Decimal("0.0"))


RuleCondition.model_rebuild()
InferenceRule.model_rebuild()
InferenceStep.model_rebuild()
InferenceTrace.model_rebuild()
