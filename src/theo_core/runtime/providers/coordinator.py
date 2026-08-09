"""ProviderCoordinator — the single invocation point for provider hooks.

ADR-0028: the coordinator is the ONLY theo-core code that calls providers. It
resolves exactly one provider per capability, invokes it with snapshot-only
inputs, and records deterministic provenance. Symbolic engines never see
providers.

Failure semantics (v0.5.0): any provider error raises ``ProviderFailure`` and
fails the cycle. There is no silent fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from theo_core.models.ports.snapshots import (
    ProviderCapabilities,
    ProviderExecution,
)
from theo_core.runtime.providers.models import (
    CalibrationHookResult,
    GoalRankHookResult,
    ProposalHookResult,
    ProviderFailure,
    ProviderInvocation,
    ProviderStatus,
    RuleRankHookResult,
)

if TYPE_CHECKING:
    from theo_core.models.ports.snapshots import (
        BeliefSnapshotCollection,
        ConceptSnapshotCollection,
        DecisionSnapshot,
        GoalSnapshotCollection,
        GroundingSnapshot,
        HypothesisSnapshotCollection,
        RuleSnapshotCollection,
    )
    from theo_core.runtime.providers.resolution import (
        ProviderResolver,
        ResolvedProvider,
    )


class ProviderCoordinator:
    """Coordinates provider invocations against the resolved configuration."""

    def __init__(self, resolver: ProviderResolver) -> None:
        """Initialize the coordinator.

        Args:
            resolver: Resolver carrying the explicitly ordered provider
                configuration.

        """
        self._resolver = resolver

    def propose_hypotheses(
        self,
        *,
        percept: str,
        concepts: ConceptSnapshotCollection,
        beliefs: BeliefSnapshotCollection,
        rules: RuleSnapshotCollection,
        grounding: GroundingSnapshot,
    ) -> ProposalHookResult:
        """Run the hypothesis proposal hook (capability HYPOTHESIS_PROPOSAL)."""
        capability = ProviderCapabilities.HYPOTHESIS_PROPOSAL
        resolved = self._resolver.resolve(capability)
        if resolved is None:
            return ProposalHookResult(proposals=())
        execution = self._execute(
            resolved,
            capability,
            "propose_hypotheses",
            (percept, concepts, beliefs, rules, grounding),
        )
        return ProposalHookResult(
            proposals=execution.output,
            invocation=self._invocation(capability, execution),
        )

    def score_hypotheses(
        self,
        *,
        hypotheses: HypothesisSnapshotCollection,
        beliefs: BeliefSnapshotCollection,
        percept: str,
        grounding: GroundingSnapshot,
    ) -> CalibrationHookResult:
        """Run the hypothesis scoring hook (capability CALIBRATION)."""
        capability = ProviderCapabilities.CALIBRATION
        resolved = self._resolver.resolve(capability)
        if resolved is None:
            return CalibrationHookResult(scored=())
        execution = self._execute(
            resolved,
            capability,
            "score_hypotheses",
            (hypotheses, beliefs, percept, grounding),
        )
        return CalibrationHookResult(
            scored=execution.output,
            invocation=self._invocation(capability, execution),
        )

    def score_confidence(
        self,
        *,
        decision: DecisionSnapshot,
        hypotheses: HypothesisSnapshotCollection,
        beliefs: BeliefSnapshotCollection,
        grounding: GroundingSnapshot,
    ) -> CalibrationHookResult:
        """Run the confidence calibration hook (capability CALIBRATION)."""
        capability = ProviderCapabilities.CALIBRATION
        resolved = self._resolver.resolve(capability)
        if resolved is None:
            return CalibrationHookResult(scored=())
        execution = self._execute(
            resolved,
            capability,
            "score_confidence",
            (decision, hypotheses, beliefs, grounding),
        )
        return CalibrationHookResult(
            scored=execution.output,
            invocation=self._invocation(capability, execution),
        )

    def rank_goals(
        self,
        *,
        goals: GoalSnapshotCollection,
        percept: str,
        beliefs: BeliefSnapshotCollection,
        grounding: GroundingSnapshot,
    ) -> GoalRankHookResult:
        """Run the goal salience hook (capability SALIENCE)."""
        capability = ProviderCapabilities.SALIENCE
        resolved = self._resolver.resolve(capability)
        if resolved is None:
            return GoalRankHookResult(goals=())
        execution = self._execute(
            resolved,
            capability,
            "rank_goals",
            (goals, percept, beliefs, grounding),
        )
        return GoalRankHookResult(
            goals=execution.output,
            invocation=self._invocation(capability, execution),
        )

    def rank_rules(
        self,
        *,
        rules: RuleSnapshotCollection,
        concepts: ConceptSnapshotCollection,
        beliefs: BeliefSnapshotCollection,
        grounding: GroundingSnapshot,
    ) -> RuleRankHookResult:
        """Run the rule salience hook (capability SALIENCE)."""
        capability = ProviderCapabilities.SALIENCE
        resolved = self._resolver.resolve(capability)
        if resolved is None:
            return RuleRankHookResult(rules=())
        execution = self._execute(
            resolved,
            capability,
            "rank_rules",
            (rules, concepts, beliefs, grounding),
        )
        return RuleRankHookResult(
            rules=execution.output,
            invocation=self._invocation(capability, execution),
        )

    def _execute(
        self,
        resolved: ResolvedProvider,
        capability: ProviderCapabilities,
        method_name: str,
        args: tuple[Any, ...],
    ) -> ProviderExecution[Any]:
        """Invoke the resolved provider method, wrapping failures fail-fast."""
        provider = resolved.provider
        method = getattr(provider, method_name)
        try:
            execution = method(*args)
        except ProviderFailure:
            raise
        except Exception as exc:
            msg = (
                f"Provider {resolved.name!r} raised {type(exc).__name__} during "
                f"{method_name}: {exc}"
            )
            raise ProviderFailure(capability, resolved.name, msg) from exc
        if not isinstance(execution, ProviderExecution) or not isinstance(
            execution.output, tuple
        ):
            msg = (
                f"Provider {resolved.name!r} returned an invalid execution from "
                f"{method_name}; expected ProviderExecution wrapping a tuple."
            )
            raise ProviderFailure(capability, resolved.name, msg)
        return execution

    @staticmethod
    def _invocation(
        capability: ProviderCapabilities,
        execution: ProviderExecution[Any],
    ) -> ProviderInvocation:
        """Record deterministic provenance for an executed provider."""
        return ProviderInvocation(
            capability=capability,
            status=ProviderStatus.EXECUTED,
            provider_name=execution.provider_name,
            provider_version=execution.provider_version,
            model_name=execution.model_name,
            model_hash=execution.model_hash,
            summary={"count": len(execution.output)},
        )
