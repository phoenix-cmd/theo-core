"""ProviderResolver — deterministic capability-to-provider resolution.

ADR-0028 amendment (v0.5.0 Phase 1): the runtime resolves exactly one provider
per capability, deterministically, never depending on set or hash iteration
order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from theo_core.models.ports.snapshots import ProviderCapabilities
from theo_core.runtime.providers.models import ProviderEntry

if TYPE_CHECKING:
    from collections.abc import Sequence

_CAPABILITY_METHODS: dict[ProviderCapabilities, tuple[str, ...]] = {
    ProviderCapabilities.HYPOTHESIS_PROPOSAL: ("propose_hypotheses",),
    ProviderCapabilities.CALIBRATION: ("score_hypotheses", "score_confidence"),
    ProviderCapabilities.SALIENCE: ("rank_goals", "rank_rules"),
    ProviderCapabilities.RULE_DISCOVERY: ("discover_rules",),
}


@dataclass(frozen=True, slots=True)
class ResolvedProvider:
    """The provider selected for a capability plus resolution metadata.

    ``order`` is the explicit configuration index, used only to break true
    ties deterministically.
    """

    provider: object
    name: str
    priority: int
    order: int


class ProviderResolver:
    """Resolves exactly one provider per capability, deterministically.

    Resolution rule (ADR-0028 amendment):

        capability
            → eligible providers (those advertising the capability)
            → sort by (-priority, provider name, configuration order)
            → exactly one selected provider, or none

    Selection is driven purely by the explicit configuration sequence and the
    sort key; it never iterates a ``frozenset`` of providers or capabilities to
    pick a winner.
    """

    def __init__(self, providers: Sequence[ProviderEntry | object] = ()) -> None:
        """Initialize the resolver.

        Args:
            providers: Providers in explicit configuration order. Plain objects
                are wrapped with priority 0. Every provider MUST expose
                ``capabilities()``; anything else is a configuration error.

        Raises:
            ValueError: If a provider lacks ``capabilities()`` or its
                capability declaration is invalid.

        """
        entries = [
            item if isinstance(item, ProviderEntry) else ProviderEntry(provider=item)
            for item in providers
        ]
        self._entries = tuple(entries)
        self._by_capability = self._build_index()

    @property
    def provider_count(self) -> int:
        """Return the number of registered providers."""
        return len(self._entries)

    def resolve(self, capability: ProviderCapabilities) -> ResolvedProvider | None:
        """Resolve the single provider for a capability, or None.

        Args:
            capability: The capability a hook requires.

        Returns:
            The deterministically selected provider, or None when no eligible
            provider is registered.

        """
        candidates = self._by_capability.get(capability, ())
        return candidates[0] if candidates else None

    def advertised(self) -> frozenset[ProviderCapabilities]:
        """Return the union of capabilities advertised by all providers."""
        return frozenset(self._by_capability)

    def _build_index(self) -> dict[ProviderCapabilities, tuple[ResolvedProvider, ...]]:
        """Register eligible providers per capability, sorted deterministically."""
        index: dict[ProviderCapabilities, list[ResolvedProvider]] = {}
        for order, entry in enumerate(self._entries):
            provider = entry.provider
            capabilities = getattr(provider, "capabilities", None)
            if not callable(capabilities):
                msg = (
                    f"Provider {type(provider).__name__!r} lacks a callable "
                    "capabilities() and cannot be resolved."
                )
                raise ValueError(msg)
            try:
                declared = frozenset(capabilities())
            except Exception as exc:
                msg = (
                    f"Provider {type(provider).__name__!r} capabilities() raised "
                    f"during registration: {exc}"
                )
                raise ValueError(msg) from exc
            for capability in declared:
                if not isinstance(capability, ProviderCapabilities):
                    msg = (
                        f"Provider {type(provider).__name__!r} declared an invalid "
                        f"capability: {capability!r}"
                    )
                    raise ValueError(msg)
                missing = [
                    method
                    for method in _CAPABILITY_METHODS.get(capability, ())
                    if not callable(getattr(provider, method, None))
                ]
                if missing:
                    msg = (
                        f"Provider {type(provider).__name__!r} advertises "
                        f"{capability.value!r} but lacks {' and '.join(missing)}."
                    )
                    raise ValueError(msg)
                resolved = ResolvedProvider(
                    provider=provider,
                    name=type(provider).__name__,
                    priority=entry.priority,
                    order=order,
                )
                index.setdefault(capability, []).append(resolved)

        return {
            capability: tuple(
                sorted(
                    candidates,
                    key=lambda rp: (-rp.priority, rp.name, rp.order),
                )
            )
            for capability, candidates in index.items()
        }
