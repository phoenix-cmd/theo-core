"""theo_core.models.ports — provider boundary contracts (ADR-0028).

Exposes the snapshot DTOs and the four neural symbolic provider protocols.
Providers (in theo-providers) MAY import from ``theo_core.models.ports`` and
MUST NOT import ``theo_core.symbolic``.
"""

from theo_core.models.ports.neural import (
    CalibrationProvider,
    HypothesisProposalProvider,
    RuleDiscoveryProvider,
    SalienceProvider,
)
from theo_core.models.ports.snapshots import (
    BeliefSnapshot,
    BeliefSnapshotCollection,
    BenchmarkFailureSnapshot,
    ConceptSnapshot,
    ConceptSnapshotCollection,
    DecisionSnapshot,
    GapItem,
    GoalSnapshot,
    GoalSnapshotCollection,
    GroundingSnapshot,
    HypothesisProposal,
    HypothesisSnapshot,
    HypothesisSnapshotCollection,
    JSONValue,
    KnowledgeGapReport,
    ProviderCapabilities,
    ProviderExecution,
    ProviderTelemetry,
    RuleSnapshot,
    RuleSnapshotCollection,
    ScoredHypothesis,
    verify_grounding,
)

__all__ = [
    "BeliefSnapshot",
    "BeliefSnapshotCollection",
    "BenchmarkFailureSnapshot",
    "CalibrationProvider",
    "ConceptSnapshot",
    "ConceptSnapshotCollection",
    "DecisionSnapshot",
    "GapItem",
    "GoalSnapshot",
    "GoalSnapshotCollection",
    "GroundingSnapshot",
    "HypothesisProposal",
    "HypothesisProposalProvider",
    "HypothesisSnapshot",
    "HypothesisSnapshotCollection",
    "JSONValue",
    "KnowledgeGapReport",
    "ProviderCapabilities",
    "ProviderExecution",
    "ProviderTelemetry",
    "RuleDiscoveryProvider",
    "RuleSnapshot",
    "RuleSnapshotCollection",
    "SalienceProvider",
    "ScoredHypothesis",
    "verify_grounding",
]
