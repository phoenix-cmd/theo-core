"""Domain models for the Constraint Engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from theo_core.symbolic._primitives.identifiers import SymbolicId


@dataclass(frozen=True, slots=True)
class ConstraintId:
    """URI-style constraint identifier value object.

    Wraps SymbolicId with constraint-specific URI validation (must start with 'constraint://').
    """

    value: str

    @classmethod
    def of(cls, value: str) -> ConstraintId:
        """Validate constraint URI scheme and return a new ConstraintId."""
        if not value.startswith("constraint://"):
            msg = f"ConstraintId URI must start with 'constraint://', got {value!r}"
            raise ValueError(msg)
        symbolic_id = SymbolicId.of(value)
        return cls(value=symbolic_id.value)

    def to_symbolic_id(self) -> SymbolicId:
        """Convert to primitive SymbolicId."""
        return SymbolicId(value=self.value)

    def __str__(self) -> str:
        """Return raw string value."""
        return self.value


class ConstraintSeverity(StrEnum):
    """Severity classification of a constraint violation."""

    FATAL = "fatal"
    WARNING = "warning"
    ADVISORY = "advisory"


class ConstraintRule(BaseModel, frozen=True):
    """An immutable system constraint rule."""

    id: ConstraintId
    name: str
    description: str
    severity: ConstraintSeverity = ConstraintSeverity.FATAL
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConstraintViolation(BaseModel, frozen=True):
    """Record of a constraint violation."""

    constraint_id: ConstraintId
    target_id: SymbolicId
    reason: str
    severity: ConstraintSeverity = ConstraintSeverity.FATAL
    metadata: dict[str, Any] = Field(default_factory=dict)


ConstraintRule.model_rebuild()
ConstraintViolation.model_rebuild()
