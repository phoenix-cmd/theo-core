"""Domain models for the Probabilistic Epistemic Belief System."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from theo_core.symbolic._primitives.identifiers import SymbolicId


@dataclass(frozen=True, slots=True)
class BeliefId:
    """URI-style belief identifier value object.

    Wraps SymbolicId with belief-specific URI validation (must start with 'belief://').
    """

    value: str

    @classmethod
    def of(cls, value: str) -> BeliefId:
        """Validate belief URI scheme and return a new BeliefId."""
        if not value.startswith("belief://"):
            msg = f"BeliefId URI must start with 'belief://', got {value!r}"
            raise ValueError(msg)
        symbolic_id = SymbolicId.of(value)
        return cls(value=symbolic_id.value)

    def to_symbolic_id(self) -> SymbolicId:
        """Convert to primitive SymbolicId."""
        return SymbolicId(value=self.value)

    def __str__(self) -> str:
        """Return raw string value."""
        return self.value


class BeliefSource(StrEnum):
    """Origin source of a belief, enforcing Canon Law 4.

    Perception is intentionally absent: percepts enter cognition as *evidence*,
    and beliefs are mechanically derived by Inference (see ADR-0026).
    """

    MEMORY = "memory"
    KNOWLEDGE = "knowledge"
    INFERENCE = "inference"


class EvidenceTrace(BaseModel, frozen=True):
    """Traceable evidence supporting a belief, enforcing Canon Invariant 5."""

    evidence_id: SymbolicId
    source_type: str = "general"
    weight: Decimal = Field(default=Decimal("1.0"), ge=Decimal("0.0"), le=Decimal("1.0"))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Belief(BaseModel, frozen=True):
    """An immutable epistemic belief proposition.

    Maintains full provenance, evidence traces, confidence, uncertainty,
    and revision tracking.
    """

    id: BeliefId
    proposition: str
    confidence: Decimal = Field(default=Decimal("1.0"), ge=Decimal("0.0"), le=Decimal("1.0"))
    uncertainty: Decimal = Field(default=Decimal("0.0"), ge=Decimal("0.0"), le=Decimal("1.0"))
    support: tuple[EvidenceTrace, ...] = Field(default_factory=tuple)
    contradictions: tuple[BeliefId, ...] = Field(default_factory=tuple)
    source: BeliefSource = BeliefSource.INFERENCE
    last_verified: datetime = Field(default_factory=lambda: datetime.now(UTC))
    evidence_count: int = Field(default=1, ge=0)
    reasoning_depth: int = Field(default=0, ge=0)
    revision_id: int = Field(default=1, ge=1)
    previous_version_id: BeliefId | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BeliefRelation(StrEnum):
    """Relationship between beliefs in the BeliefGraph."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REPLACES = "replaces"
    DEPENDS_ON = "depends_on"


class BeliefEdge(BaseModel, frozen=True):
    """A typed, weighted, immutable edge connecting two beliefs."""

    source: BeliefId
    target: BeliefId
    relation: BeliefRelation = BeliefRelation.DEPENDS_ON
    weight: Decimal = Field(default=Decimal("1.0"), ge=Decimal("0.0"))
    metadata: dict[str, Any] = Field(default_factory=dict)
