"""Decision persistence protocols and in-memory repository implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from theo_core.symbolic.decisions.models import DecisionId, DecisionRecord


@runtime_checkable
class DecisionRepository(Protocol):
    """Abstract persistence protocol for DecisionRecord collections.

    Behavioral guarantees:
    - save() preserves decision record integrity.
    - load() reconstructs identical decision record data.
    - Operations MUST NOT mutate the supplied decision record.
    """

    def save(self, record: DecisionRecord) -> None:
        """Persist a DecisionRecord."""
        ...

    def load(self, decision_id: DecisionId) -> DecisionRecord | None:
        """Load a DecisionRecord by decision_id."""
        ...

    def get_all(self) -> list[DecisionRecord]:
        """Return all stored decision records in deterministic sorted order."""
        ...

    def delete(self, decision_id: DecisionId) -> None:
        """Delete a DecisionRecord by decision_id."""
        ...


class InMemoryDecisionRepository:
    """In-memory implementation of DecisionRepository."""

    def __init__(self) -> None:
        """Initialize empty decision store."""
        self._store: dict[str, DecisionRecord] = {}

    def save(self, record: DecisionRecord) -> None:
        """Store a DecisionRecord."""
        self._store[record.id.value] = record

    def load(self, decision_id: DecisionId) -> DecisionRecord | None:
        """Retrieve stored DecisionRecord by decision_id."""
        return self._store.get(decision_id.value)

    def get_all(self) -> list[DecisionRecord]:
        """Return all decision records sorted by ID."""
        sorted_keys = sorted(self._store.keys())
        return [self._store[k] for k in sorted_keys]

    def delete(self, decision_id: DecisionId) -> None:
        """Delete decision record from store."""
        self._store.pop(decision_id.value, None)
