"""Provider hook orchestration for the runtime boundary (ADR-0028).

Theo-core-owned coordination of the neural symbolic provider surface. The
``ProviderCoordinator`` is the only theo-core code that calls providers;
``ProviderResolver`` performs deterministic capability resolution; the models
module carries the provenance and failure contracts.
"""

from theo_core.runtime.providers.coordinator import ProviderCoordinator
from theo_core.runtime.providers.models import (
    CalibrationHookResult,
    GoalRankHookResult,
    ProposalHookResult,
    ProviderEntry,
    ProviderFailure,
    ProviderInvocation,
    ProviderStatus,
    RuleRankHookResult,
)
from theo_core.runtime.providers.resolution import ProviderResolver

__all__ = [
    "CalibrationHookResult",
    "GoalRankHookResult",
    "ProposalHookResult",
    "ProviderCoordinator",
    "ProviderEntry",
    "ProviderFailure",
    "ProviderInvocation",
    "ProviderResolver",
    "ProviderStatus",
    "RuleRankHookResult",
]
