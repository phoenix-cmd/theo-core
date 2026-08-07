"""Constraint rule persistence protocols and in-memory repository implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from theo_core.symbolic.constraints.models import ConstraintId, ConstraintRule


@runtime_checkable
class ConstraintRepository(Protocol):
    """Abstract persistence protocol for ConstraintRule collections.

    Behavioral guarantees:
    - save() preserves rule integrity.
    - load() reconstructs identical rule data.
    - Operations MUST NOT mutate the supplied rule.
    """

    def save(self, rule: ConstraintRule) -> None:
        """Persist a ConstraintRule."""
        ...

    def load(self, constraint_id: ConstraintId) -> ConstraintRule | None:
        """Load a ConstraintRule by constraint_id."""
        ...

    def get_all(self) -> list[ConstraintRule]:
        """Return all stored constraint rules in deterministic sorted order."""
        ...

    def delete(self, constraint_id: ConstraintId) -> None:
        """Delete a ConstraintRule by constraint_id."""
        ...


class InMemoryConstraintRepository:
    """In-memory implementation of ConstraintRepository."""

    def __init__(self) -> None:
        """Initialize empty constraint store."""
        self._store: dict[str, ConstraintRule] = {}

    def save(self, rule: ConstraintRule) -> None:
        """Store a ConstraintRule."""
        self._store[rule.id.value] = rule

    def load(self, constraint_id: ConstraintId) -> ConstraintRule | None:
        """Retrieve stored ConstraintRule by constraint_id."""
        return self._store.get(constraint_id.value)

    def get_all(self) -> list[ConstraintRule]:
        """Return all constraint rules sorted by ID."""
        sorted_keys = sorted(self._store.keys())
        return [self._store[k] for k in sorted_keys]

    def delete(self, constraint_id: ConstraintId) -> None:
        """Delete constraint rule from store."""
        self._store.pop(constraint_id.value, None)
