"""Percept models for the symbolic perception stage.

Perception produces an immutable ``Percept`` value (the structured abstraction of
raw sensory input). Percepts are NOT beliefs: they enter cognition as evidence,
and beliefs are derived from them during inference (Canon Law 4, Law 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from theo_core.symbolic._primitives.identifiers import SymbolicId


@dataclass(frozen=True, slots=True)
class PerceptId:
    """URI-style percept identifier value object.

    Wraps SymbolicId with percept-specific URI validation (must start with 'percept://').
    """

    value: str

    @classmethod
    def of(cls, value: str) -> PerceptId:
        """Validate percept URI scheme and return a new PerceptId."""
        if not value.startswith("percept://"):
            msg = f"PerceptId URI must start with 'percept://', got {value!r}"
            raise ValueError(msg)
        symbolic_id = SymbolicId.of(value)
        return cls(value=symbolic_id.value)

    def to_symbolic_id(self) -> SymbolicId:
        """Convert to primitive SymbolicId."""
        return SymbolicId(value=self.value)

    def __str__(self) -> str:
        """Return raw string value."""
        return self.value


@dataclass(frozen=True, slots=True)
class Percept:
    """An immutable structured abstraction of raw sensory input.

    Attributes:
        id: Stable URI identifier (deterministic from content).
        content: Normalized content representation.
        modality: Sensory modality label.
        metadata: Extensible metadata dictionary.

    """

    id: PerceptId
    content: str
    modality: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)
