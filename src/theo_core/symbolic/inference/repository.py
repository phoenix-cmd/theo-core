"""Rule persistence protocols and in-memory repository implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from theo_core.symbolic.inference.models import InferenceRule, RuleId


@runtime_checkable
class RuleRepository(Protocol):
    """Abstract persistence protocol for InferenceRule collections.

    Behavioral guarantees:
    - save() preserves rule integrity.
    - load() reconstructs identical rules.
    - Operations MUST NOT mutate the supplied rule.
    """

    def save(self, rule: InferenceRule) -> None:
        """Persist an InferenceRule."""
        ...

    def load(self, rule_id: RuleId) -> InferenceRule | None:
        """Load an InferenceRule by rule_id."""
        ...

    def get_all(self) -> list[InferenceRule]:
        """Return all stored rules in deterministic sorted order."""
        ...

    def delete(self, rule_id: RuleId) -> None:
        """Delete an InferenceRule by rule_id."""
        ...


class InMemoryRuleRepository:
    """In-memory implementation of RuleRepository."""

    def __init__(self) -> None:
        """Initialize empty rule store."""
        self._store: dict[str, InferenceRule] = {}

    def save(self, rule: InferenceRule) -> None:
        """Store an InferenceRule."""
        self._store[rule.id.value] = rule

    def load(self, rule_id: RuleId) -> InferenceRule | None:
        """Retrieve stored InferenceRule by rule_id."""
        return self._store.get(rule_id.value)

    def get_all(self) -> list[InferenceRule]:
        """Return all rules sorted by rule ID."""
        sorted_keys = sorted(self._store.keys())
        return [self._store[k] for k in sorted_keys]

    def delete(self, rule_id: RuleId) -> None:
        """Delete rule from store."""
        self._store.pop(rule_id.value, None)
