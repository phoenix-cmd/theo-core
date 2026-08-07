"""ConflictRecord persistence protocols and in-memory repository implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from theo_core.symbolic._primitives.identifiers import SymbolicId
    from theo_core.symbolic.conflict.models import ConflictRecord


@runtime_checkable
class ConflictRepository(Protocol):
    """Abstract persistence protocol for ConflictRecord collections.

    Behavioral guarantees:
    - save() preserves record integrity.
    - load() reconstructs identical conflict records.
    - Operations MUST NOT mutate the supplied record.
    """

    def save(self, record: ConflictRecord) -> None:
        """Persist a ConflictRecord."""
        ...

    def load(self, conflict_id: SymbolicId) -> ConflictRecord | None:
        """Load a ConflictRecord by conflict_id."""
        ...

    def get_all(self) -> list[ConflictRecord]:
        """Return all stored conflict records in deterministic sorted order."""
        ...

    def delete(self, conflict_id: SymbolicId) -> None:
        """Delete a ConflictRecord by conflict_id."""
        ...


class InMemoryConflictRepository:
    """In-memory implementation of ConflictRepository."""

    def __init__(self) -> None:
        """Initialize empty conflict store."""
        self._store: dict[str, ConflictRecord] = {}

    def save(self, record: ConflictRecord) -> None:
        """Store a ConflictRecord."""
        self._store[record.conflict_id.value] = record

    def load(self, conflict_id: SymbolicId) -> ConflictRecord | None:
        """Retrieve stored ConflictRecord by conflict_id."""
        return self._store.get(conflict_id.value)

    def get_all(self) -> list[ConflictRecord]:
        """Return all conflict records sorted by ID."""
        sorted_keys = sorted(self._store.keys())
        return [self._store[k] for k in sorted_keys]

    def delete(self, conflict_id: SymbolicId) -> None:
        """Delete conflict record from store."""
        self._store.pop(conflict_id.value, None)
