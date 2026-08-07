"""Hypothesis persistence protocols and in-memory repository implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from theo_core.symbolic.hypotheses.models import Hypothesis, HypothesisId


@runtime_checkable
class HypothesisRepository(Protocol):
    """Abstract persistence protocol for Hypothesis collections.

    Behavioral guarantees:
    - save() preserves hypothesis integrity.
    - load() reconstructs identical hypothesis data.
    - Operations MUST NOT mutate the supplied hypothesis.
    """

    def save(self, hypothesis: Hypothesis) -> None:
        """Persist a Hypothesis."""
        ...

    def load(self, hypothesis_id: HypothesisId) -> Hypothesis | None:
        """Load a Hypothesis by hypothesis_id."""
        ...

    def get_all(self) -> list[Hypothesis]:
        """Return all stored hypotheses in deterministic sorted order."""
        ...

    def delete(self, hypothesis_id: HypothesisId) -> None:
        """Delete a Hypothesis by hypothesis_id."""
        ...


class InMemoryHypothesisRepository:
    """In-memory implementation of HypothesisRepository."""

    def __init__(self) -> None:
        """Initialize empty hypothesis store."""
        self._store: dict[str, Hypothesis] = {}

    def save(self, hypothesis: Hypothesis) -> None:
        """Store a Hypothesis."""
        self._store[hypothesis.id.value] = hypothesis

    def load(self, hypothesis_id: HypothesisId) -> Hypothesis | None:
        """Retrieve stored Hypothesis by hypothesis_id."""
        return self._store.get(hypothesis_id.value)

    def get_all(self) -> list[Hypothesis]:
        """Return all hypotheses sorted by ID."""
        sorted_keys = sorted(self._store.keys())
        return [self._store[k] for k in sorted_keys]

    def delete(self, hypothesis_id: HypothesisId) -> None:
        """Delete hypothesis from store."""
        self._store.pop(hypothesis_id.value, None)
