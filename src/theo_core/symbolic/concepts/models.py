"""Domain models for the Concept System."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from theo_core.symbolic._primitives.identifiers import SymbolicId


@dataclass(frozen=True, slots=True)
class ConceptId:
    """URI-style concept identifier value object.

    Wraps SymbolicId with concept-specific URI validation (must start with 'concept://').
    """

    value: str

    @classmethod
    def of(cls, value: str) -> ConceptId:
        """Validate concept URI scheme and return a new ConceptId."""
        if not value.startswith("concept://"):
            msg = f"ConceptId URI must start with 'concept://', got {value!r}"
            raise ValueError(msg)
        symbolic_id = SymbolicId.of(value)
        return cls(value=symbolic_id.value)

    def to_symbolic_id(self) -> SymbolicId:
        """Convert to primitive SymbolicId."""
        return SymbolicId(value=self.value)

    def __str__(self) -> str:
        """Return raw string value."""
        return self.value


class ConceptType(StrEnum):
    """Semantic category of a Concept node."""

    ENTITY = "entity"
    ATTRIBUTE = "attribute"
    ACTION = "action"
    STATE = "state"
    RELATION = "relation"
    ABSTRACT = "abstract"


class RelationType(StrEnum):
    """Taxonomic relationship type between concepts."""

    IS_A = "is_a"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    DEPENDS_ON = "depends_on"
    OPPOSITE_OF = "opposite_of"
    SIMILAR_TO = "similar_to"
    CUSTOM = "custom"


class Concept(BaseModel, frozen=True):
    """An immutable semantic concept node.

    Contains knowledge attributes only — no transient runtime state (e.g. activation).
    """

    id: ConceptId
    label: str
    concept_type: ConceptType = ConceptType.ENTITY
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConceptEdge(BaseModel, frozen=True):
    """A typed, weighted, immutable edge connecting two concepts."""

    source: ConceptId
    target: ConceptId
    relation: RelationType = RelationType.RELATED_TO
    weight: Decimal = Field(default=Decimal("1.0"), ge=Decimal("0.0"))
    metadata: dict[str, Any] = Field(default_factory=dict)
