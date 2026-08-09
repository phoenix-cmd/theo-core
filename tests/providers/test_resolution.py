"""Provider resolution determinism (ADR-0028 amendment).

Resolution MUST select exactly one provider per capability deterministically:
capability -> eligible providers -> sort by (-priority, provider name,
configuration order) -> one or none. Never set or hash iteration order.
"""

from __future__ import annotations

import pytest
from stubs import (
    InvalidCapabilityProvider,
    RecordingProvider,
    UnimplementedCapabilityProvider,
)

from theo_core.models.ports.snapshots import ProviderCapabilities
from theo_core.runtime.providers.models import ProviderEntry
from theo_core.runtime.providers.resolution import ProviderResolver


class AlphaProvider(RecordingProvider):
    """Distinct class name for name-tiebreak tests."""


class BetaProvider(RecordingProvider):
    """Distinct class name for name-tiebreak tests."""


class TestResolutionDeterminism:
    def test_no_providers_resolves_nothing(self) -> None:
        resolver = ProviderResolver()
        assert resolver.provider_count == 0
        for capability in ProviderCapabilities:
            assert resolver.resolve(capability) is None
        assert resolver.advertised() == frozenset()

    def test_single_provider_resolves_for_every_advertised_capability(self) -> None:
        provider = RecordingProvider()
        resolver = ProviderResolver([provider])
        for capability in ProviderCapabilities:
            resolved = resolver.resolve(capability)
            assert resolved is not None
            assert resolved.provider is provider
            assert resolved.name == "RecordingProvider"
            assert resolved.priority == 0
            assert resolved.order == 0

    def test_higher_priority_wins(self) -> None:
        low = RecordingProvider()
        high = RecordingProvider()
        resolver = ProviderResolver(
            [
                ProviderEntry(low, priority=1),
                ProviderEntry(high, priority=10),
            ]
        )
        for capability in ProviderCapabilities:
            resolved = resolver.resolve(capability)
            assert resolved is not None
            assert resolved.provider is high

    def test_name_tiebreak_beats_configuration_order(self) -> None:
        beta = BetaProvider()
        alpha = AlphaProvider()
        resolver = ProviderResolver([ProviderEntry(beta), ProviderEntry(alpha)])
        for capability in ProviderCapabilities:
            resolved = resolver.resolve(capability)
            assert resolved is not None
            assert resolved.name == "AlphaProvider"
            assert resolved.provider is alpha

    def test_configuration_order_breaks_identical_name_and_priority_ties(self) -> None:
        provider_a = RecordingProvider()
        provider_b = RecordingProvider()
        resolver = ProviderResolver([provider_a, provider_b])
        for capability in ProviderCapabilities:
            resolved = resolver.resolve(capability)
            assert resolved is not None
            assert resolved.order == 0
            assert resolved.provider is provider_a

    def test_resolution_is_stable_across_repeated_calls(self) -> None:
        resolver = ProviderResolver([RecordingProvider()])
        first = resolver.resolve(ProviderCapabilities.SALIENCE)
        for _ in range(5):
            assert resolver.resolve(ProviderCapabilities.SALIENCE) == first


class TestResolutionValidation:
    def test_capabilities_that_raise_are_rejected_at_registration(self) -> None:
        class ExplodingProvider:
            def capabilities(self) -> frozenset[ProviderCapabilities]:
                msg = "no capabilities here"
                raise RuntimeError(msg)

        with pytest.raises(ValueError, match="capabilities"):
            ProviderResolver([ExplodingProvider()])

    def test_provider_without_callable_capabilities_is_rejected(self) -> None:
        class NoCapabilitiesProvider:
            capabilities = None

        with pytest.raises(ValueError, match="lacks a callable capabilities"):
            ProviderResolver([NoCapabilitiesProvider()])

    def test_advertised_but_unimplemented_capability_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="advertises"):
            ProviderResolver([UnimplementedCapabilityProvider()])

    def test_invalid_capability_value_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid capability"):
            ProviderResolver([InvalidCapabilityProvider()])

    def test_advertised_union_is_computed(self) -> None:
        partial = RecordingProvider(capabilities=frozenset({ProviderCapabilities.SALIENCE}))
        resolver = ProviderResolver([partial])
        assert resolver.advertised() == frozenset({ProviderCapabilities.SALIENCE})
