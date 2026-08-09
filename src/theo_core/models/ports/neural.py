"""Neural symbolic provider protocols (ADR-0028).

Exactly four provider protocols. Providers are replaceable implementations, not
architectural components. Every signature is snapshot-only: providers MUST NOT
import or receive internal runtime objects.

Architectural invariant (ADR-0028):

> If deleting ``theo_core.symbolic`` internals would break a provider
> implementation, the boundary is wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from theo_core.models.ports.snapshots import (
        BeliefSnapshotCollection,
        ConceptSnapshotCollection,
        DecisionSnapshot,
        GoalSnapshotCollection,
        GroundingSnapshot,
        HypothesisProposal,
        HypothesisSnapshotCollection,
        KnowledgeGapReport,
        ProviderCapabilities,
        ProviderExecution,
        RuleSnapshot,
        RuleSnapshotCollection,
        ScoredHypothesis,
    )


class HypothesisProposalProvider(Protocol):
    """Proposes candidate hypotheses for the Hypothesis Engine (ADR-0028).

    Proposals are evidence, never decisions. The symbolic runtime verifies
    strict grounding and decides.
    """

    def capabilities(self) -> frozenset[ProviderCapabilities]:
        """Declare the capabilities this provider offers."""
        ...

    def propose_hypotheses(
        self,
        percept: str,
        concepts: ConceptSnapshotCollection,
        beliefs: BeliefSnapshotCollection,
        rules: RuleSnapshotCollection,
        grounding: GroundingSnapshot,
    ) -> ProviderExecution[tuple[HypothesisProposal, ...]]:
        """Propose candidate hypotheses for the given cognitive context."""
        ...


class CalibrationProvider(Protocol):
    """Scores hypotheses and confidence for the Decision Engine (ADR-0028).

    Contains exactly two operations: ``score_hypotheses`` and
    ``score_confidence``.
    """

    def capabilities(self) -> frozenset[ProviderCapabilities]:
        """Declare the capabilities this provider offers."""
        ...

    def score_hypotheses(
        self,
        hypotheses: HypothesisSnapshotCollection,
        beliefs: BeliefSnapshotCollection,
        percept: str,
        grounding: GroundingSnapshot,
    ) -> ProviderExecution[tuple[ScoredHypothesis, ...]]:
        """Score the support each hypothesis has in the belief context."""
        ...

    def score_confidence(
        self,
        decision: DecisionSnapshot,
        hypotheses: HypothesisSnapshotCollection,
        beliefs: BeliefSnapshotCollection,
        grounding: GroundingSnapshot,
    ) -> ProviderExecution[tuple[ScoredHypothesis, ...]]:
        """Recalibrate confidence for the candidate hypotheses of a decision."""
        ...


class SalienceProvider(Protocol):
    """Ranks goals and rules (ADR-0028).

    Ranking reorders candidates only; it can neither create nor delete
    candidates.
    """

    def capabilities(self) -> frozenset[ProviderCapabilities]:
        """Declare the capabilities this provider offers."""
        ...

    def rank_goals(
        self,
        goals: GoalSnapshotCollection,
        percept: str,
        beliefs: BeliefSnapshotCollection,
        grounding: GroundingSnapshot,
    ) -> ProviderExecution[GoalSnapshotCollection]:
        """Reorder goals by salience for the given percept."""
        ...

    def rank_rules(
        self,
        rules: RuleSnapshotCollection,
        concepts: ConceptSnapshotCollection,
        beliefs: BeliefSnapshotCollection,
        grounding: GroundingSnapshot,
    ) -> ProviderExecution[RuleSnapshotCollection]:
        """Reorder rules by salience for the given cognitive context."""
        ...


class RuleDiscoveryProvider(Protocol):
    """Discovers candidate rules offline (ADR-0028).

    Offline only: driven by benchmark failures and ``KnowledgeGapReport`` during
    knowledge engineering. It has no runtime hook and its proposed rules are
    never auto-committed to the knowledge base.
    """

    def capabilities(self) -> frozenset[ProviderCapabilities]:
        """Declare the capabilities this provider offers."""
        ...

    def discover_rules(
        self,
        knowledge_gap: KnowledgeGapReport,
        concepts: ConceptSnapshotCollection,
        beliefs: BeliefSnapshotCollection,
        grounding: GroundingSnapshot,
    ) -> ProviderExecution[tuple[RuleSnapshot, ...]]:
        """Discover candidate rules that address the reported knowledge gaps."""
        ...
