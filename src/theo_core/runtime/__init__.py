"""Runtime provider orchestration (ADR-0028).

The runtime boundary owns provider resolution, hook consultation, and
provenance. Symbolic engines never see providers: the pipeline consults a
``ProviderCoordinator`` and theo-providers implementations live outside
theo-core.
"""

from theo_core.runtime.providers import (
    CalibrationHookResult,
    GoalRankHookResult,
    ProposalHookResult,
    ProviderCoordinator,
    ProviderEntry,
    ProviderFailure,
    ProviderInvocation,
    ProviderResolver,
    ProviderStatus,
    RuleRankHookResult,
)

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
