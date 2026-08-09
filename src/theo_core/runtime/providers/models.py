"""Provider hook orchestration models (ADR-0028).

These types are theo-core-owned and live inside the runtime boundary. They are
never part of the provider-visible ``theo_core.models.ports`` surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theo_core.models.ports.snapshots import (
        GoalSnapshotCollection,
        HypothesisProposal,
        JSONValue,
        ProviderCapabilities,
        RuleSnapshotCollection,
        ScoredHypothesis,
    )


class ProviderStatus(StrEnum):
    """Outcome of a provider invocation recorded in trace provenance."""

    EXECUTED = "executed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ProviderInvocation:
    """Deterministic provenance record for one provider call (ADR-0028).

    Contains no timestamps or wall-clock data: it is reproducible and
    JSON-serializable so it can live in trace metadata without breaking the
    fingerprint/replay contract.
    """

    capability: ProviderCapabilities
    status: ProviderStatus
    provider_name: str
    provider_version: str
    model_name: str
    model_hash: str
    summary: JSONValue = None

    def to_json(self) -> dict[str, JSONValue]:
        """Project the invocation into JSON-safe trace metadata."""
        return {
            "capability": self.capability.value,
            "status": self.status.value,
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "model_name": self.model_name,
            "model_hash": self.model_hash,
            "summary": self.summary,
        }


class ProviderFailure(RuntimeError):  # noqa: N818 - ADR-0028 names the type ProviderFailure
    """Raised when a resolved provider errors during a hook (ADR-0028).

    There is no silent fallback: a provider error fails the cycle unless an
    explicit fallback policy is configured (none exists in v0.5.0).
    """

    def __init__(
        self,
        capability: ProviderCapabilities,
        provider_name: str,
        message: str,
    ) -> None:
        """Initialize a provider failure with attribution context.

        Args:
            capability: The capability whose hook the provider was serving.
            provider_name: The resolved provider that raised.
            message: Human-readable failure detail.

        """
        super().__init__(message)
        self.capability = capability
        self.provider_name = provider_name


@dataclass(frozen=True, slots=True)
class ProviderEntry:
    """A provider plus its explicit configuration priority (default 0)."""

    provider: object
    priority: int = 0


@dataclass(frozen=True, slots=True)
class ProposalHookResult:
    """Provider-proposed hypotheses for the hypothesis hook (ADR-0028).

    ``invocation`` is None when no provider was resolved; the symbolic path
    executes unchanged. Phase 1 records the outcome; consumption of proposals
    is defined in Phase 5.
    """

    proposals: tuple[HypothesisProposal, ...]
    invocation: ProviderInvocation | None = None


@dataclass(frozen=True, slots=True)
class CalibrationHookResult:
    """Provider-assigned scores for the calibration hooks (ADR-0028).

    Phase 1 records the outcome; applying scores to confidence is Phase 4.
    """

    scored: tuple[ScoredHypothesis, ...]
    invocation: ProviderInvocation | None = None


@dataclass(frozen=True, slots=True)
class RuleRankHookResult:
    """Provider-reordered rule snapshots for the rule-salience hook.

    Phase 1 records the outcome; applying the reorder is Phase 3.
    """

    rules: RuleSnapshotCollection
    invocation: ProviderInvocation | None = None


@dataclass(frozen=True, slots=True)
class GoalRankHookResult:
    """Provider-reordered goal snapshots for the goal-salience hook.

    Phase 1 records the outcome; applying the reorder is Phase 3.
    """

    goals: GoalSnapshotCollection
    invocation: ProviderInvocation | None = None
