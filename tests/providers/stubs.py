"""Stub providers for theo-core provider-boundary tests (ADR-0028 Phase 1).

Theo-core MUST NOT import ``theo_providers`` (firewall); these in-repo stubs
satisfy the same provider protocols so the runtime boundary is exercised
without crossing the firewall.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from theo_core.models.ports.snapshots import (
    BeliefSnapshotCollection,
    ConceptSnapshotCollection,
    DecisionSnapshot,
    GoalSnapshotCollection,
    GroundingSnapshot,
    HypothesisSnapshotCollection,
    ProviderCapabilities,
    ProviderExecution,
    RuleSnapshotCollection,
)

if TYPE_CHECKING:
    from theo_core.models.ports.snapshots import KnowledgeGapReport


class RecordingProvider:
    """Declares every capability and records hook calls for assertions.

    Returns ``output`` (empty by default) so cognition is unchanged; Phase 1
    records the invocations and does not consume the outputs.
    """

    def __init__(
        self,
        *,
        capabilities: frozenset[ProviderCapabilities] = frozenset(ProviderCapabilities),
        raise_on: str | None = None,
        output: tuple[object, ...] = (),
    ) -> None:
        """Initialize the recording provider.

        Args:
            capabilities: Capabilities to advertise.
            raise_on: Method name to raise from (fail-fast tests).
            output: Execution output returned by every hook.

        """
        self._capabilities = capabilities
        self._raise_on = raise_on
        self._output = output
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def capabilities(self) -> frozenset[ProviderCapabilities]:
        """Declare the advertised capabilities."""
        return self._capabilities

    def propose_hypotheses(
        self,
        percept: str,
        concepts: ConceptSnapshotCollection,
        beliefs: BeliefSnapshotCollection,
        rules: RuleSnapshotCollection,
        grounding: GroundingSnapshot,
    ) -> ProviderExecution[tuple[object, ...]]:
        """Propose candidate hypotheses; records the call."""
        return self._run("propose_hypotheses", (percept, concepts, beliefs, rules, grounding))

    def score_hypotheses(
        self,
        hypotheses: HypothesisSnapshotCollection,
        beliefs: BeliefSnapshotCollection,
        percept: str,
        grounding: GroundingSnapshot,
    ) -> ProviderExecution[tuple[object, ...]]:
        """Score hypothesis support; records the call."""
        return self._run("score_hypotheses", (hypotheses, beliefs, percept, grounding))

    def score_confidence(
        self,
        decision: DecisionSnapshot,
        hypotheses: HypothesisSnapshotCollection,
        beliefs: BeliefSnapshotCollection,
        grounding: GroundingSnapshot,
    ) -> ProviderExecution[tuple[object, ...]]:
        """Recalibrate confidence; records the call."""
        return self._run("score_confidence", (decision, hypotheses, beliefs, grounding))

    def rank_goals(
        self,
        goals: GoalSnapshotCollection,
        percept: str,
        beliefs: BeliefSnapshotCollection,
        grounding: GroundingSnapshot,
    ) -> ProviderExecution[tuple[object, ...]]:
        """Reorder goals by salience; records the call."""
        return self._run("rank_goals", (goals, percept, beliefs, grounding))

    def rank_rules(
        self,
        rules: RuleSnapshotCollection,
        concepts: ConceptSnapshotCollection,
        beliefs: BeliefSnapshotCollection,
        grounding: GroundingSnapshot,
    ) -> ProviderExecution[tuple[object, ...]]:
        """Reorder rules by salience; records the call."""
        return self._run("rank_rules", (rules, concepts, beliefs, grounding))

    def discover_rules(
        self,
        knowledge_gap: KnowledgeGapReport,
        concepts: ConceptSnapshotCollection,
        beliefs: BeliefSnapshotCollection,
        grounding: GroundingSnapshot,
    ) -> ProviderExecution[tuple[object, ...]]:
        """Discover candidate rules; records the call."""
        return self._run("discover_rules", (knowledge_gap, concepts, beliefs, grounding))

    def _run(self, method: str, args: tuple[object, ...]) -> ProviderExecution[tuple[object, ...]]:
        """Record a call and return the configured execution."""
        self.calls.append((method, args))
        if self._raise_on == method:
            msg = f"{method} exploded"
            raise RuntimeError(msg)
        return ProviderExecution(
            provider_name="recording",
            provider_version="0.1.0",
            model_name="none",
            model_hash="",
            seed=0,
            temperature=0.0,
            output=self._output,
        )


class GoalSalienceOnlyProvider(RecordingProvider):
    """Advertises only SALIENCE; other hooks must never be called."""

    def capabilities(self) -> frozenset[ProviderCapabilities]:
        """Declare only the SALIENCE capability."""
        return frozenset({ProviderCapabilities.SALIENCE})


class UnimplementedCapabilityProvider:
    """Advertises HYPOTHESIS_PROPOSAL but lacks ``propose_hypotheses``."""

    def capabilities(self) -> frozenset[ProviderCapabilities]:
        """Declare the HYPOTHESIS_PROPOSAL capability."""
        return frozenset({ProviderCapabilities.HYPOTHESIS_PROPOSAL})


class InvalidCapabilityProvider:
    """Declares a non-``ProviderCapabilities`` value."""

    def capabilities(self) -> frozenset[str]:
        """Declare an invalid capability value."""
        return frozenset({"not_a_capability"})
