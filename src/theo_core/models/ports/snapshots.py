"""Provider boundary snapshot DTOs (ADR-0028).

All inputs and outputs that cross the neural symbolic boundary are immutable
snapshot dataclasses. Providers MUST NOT see internal runtime objects
(``theo_core.symbolic``); every entity is represented here by its snapshot
counterpart carrying only semantic, canonicalizable fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from decimal import Decimal


type JSONValue = (
    str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None
)


class ProviderCapabilities(StrEnum):
    """Capability flags a provider may declare (ADR-0028 capability discovery).

    This is an enum used for discovery; it is not itself a provider protocol.
    """

    HYPOTHESIS_PROPOSAL = "hypothesis_proposal"
    CALIBRATION = "calibration"
    SALIENCE = "salience"
    RULE_DISCOVERY = "rule_discovery"


@dataclass(frozen=True, slots=True)
class ConceptSnapshot:
    """Semantic snapshot of a concept, safe to expose to providers."""

    concept_id: str
    name: str
    definition: str


@dataclass(frozen=True, slots=True)
class BeliefSnapshot:
    """Semantic snapshot of a belief, safe to expose to providers.

    ``source`` carries the canonical BeliefSource value (e.g. ``memory``,
    ``knowledge``, ``inference``). No timestamps or lifecycle state.
    """

    belief_id: str
    proposition: str
    source: str
    confidence: Decimal


@dataclass(frozen=True, slots=True)
class RuleSnapshot:
    """Semantic snapshot of an inference rule, safe to expose to providers."""

    rule_id: str
    name: str
    premise_text: str
    conclusion_text: str
    salience: Decimal


@dataclass(frozen=True, slots=True)
class HypothesisSnapshot:
    """Semantic snapshot of a hypothesis, safe to expose to providers."""

    hypothesis_id: str
    content: str
    confidence: Decimal


@dataclass(frozen=True, slots=True)
class GoalSnapshot:
    """Semantic snapshot of a goal, safe to expose to providers."""

    goal_id: str
    description: str


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    """Semantic snapshot of a decision, safe to expose to providers.

    Carries no timestamps or mutable lifecycle state; those belong in telemetry.
    """

    decision_id: str
    action_text: str
    intent: str
    confidence: Decimal
    referenced_goal: str | None = None


@dataclass(frozen=True, slots=True)
class HypothesisProposal:
    """A provider-proposed hypothesis (ADR-0028).

    A proposal is evidence, never a decision. ``referenced_ids`` names existing
    symbolic entities (concept/belief/rule/evidence URIs) and is subject to
    strict grounding verification.
    """

    proposal_id: str
    content: str
    referenced_ids: frozenset[str] = field(default_factory=frozenset)
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class ScoredHypothesis:
    """A hypothesis paired with a provider-assigned score and evidence text."""

    hypothesis_id: str
    score: Decimal
    evidence: str = ""


@dataclass(frozen=True, slots=True)
class ProviderExecution[T]:
    """The replay contract for a single provider execution (ADR-0028).

    Contains only what is required to reproduce the cognitive result. There is
    no free-form metadata field; provider-specific detail belongs in
    ``ProviderTelemetry``.
    """

    provider_name: str
    provider_version: str
    model_name: str
    model_hash: str
    seed: int
    temperature: float
    output: T


@dataclass(frozen=True, slots=True)
class ProviderTelemetry:
    """Non-replay provider observation stream (ADR-0028).

    Deliberately separate from ``ProviderExecution``: timings and provider
    detail are not part of the replay contract.
    """

    provider_name: str
    provider_version: str
    model_name: str
    model_hash: str
    duration_ms: Decimal
    details: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GroundingSnapshot:
    """Read-only lookup boundary for strict grounding (ADR-0028).

    Holds the identifiers of every existing symbolic entity a provider may
    reference. Proposals referencing identifiers absent from all four sets are
    REJECTED; there is no ``grounded=False`` fallback.
    """

    belief_ids: frozenset[str] = field(default_factory=frozenset)
    concept_ids: frozenset[str] = field(default_factory=frozenset)
    rule_ids: frozenset[str] = field(default_factory=frozenset)
    evidence_ids: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def empty(cls) -> GroundingSnapshot:
        """Return an empty grounding snapshot (rejects all references)."""
        return cls()


@dataclass(frozen=True, slots=True)
class GapItem:
    """A single structured remediation finding (ADR-0028 gap analysis).

    ``references`` are semantic identifiers of related entities (no raw runtime
    state). ``severity`` is a coarse triage level (``high``/``medium``/``low``).
    """

    description: str
    domain: str | None = None
    severity: str = "low"
    references: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class BenchmarkFailureSnapshot:
    """Canonical failure record for one benchmark case (ADR-0028 gap analysis)."""

    case_id: str
    domain: str
    failure_type: str
    expected_decision: str
    actual_decision: str
    expected_trace_hash: str | None = None
    actual_trace_hash: str | None = None
    expected_state_hash: str | None = None
    actual_state_hash: str | None = None
    missing_rules: tuple[str, ...] = field(default_factory=tuple)
    missing_concepts: tuple[str, ...] = field(default_factory=tuple)
    missing_premises: tuple[str, ...] = field(default_factory=tuple)
    confidence_delta: Decimal | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class KnowledgeGapReport:
    """Symbolic-owned gap report structured by remediation action (ADR-0028).

    Produced by ``GapAnalyzer`` (theo-core) for offline rule discovery. Sections
    map to remediations rather than raw runtime state.
    """

    missing_premises: tuple[GapItem, ...] = field(default_factory=tuple)
    weak_rule_coverage: tuple[GapItem, ...] = field(default_factory=tuple)
    unresolved_ambiguities: tuple[GapItem, ...] = field(default_factory=tuple)
    low_confidence_regions: tuple[GapItem, ...] = field(default_factory=tuple)
    retrieval_failures: tuple[GapItem, ...] = field(default_factory=tuple)
    contradiction_patterns: tuple[GapItem, ...] = field(default_factory=tuple)
    benchmark_failures: tuple[BenchmarkFailureSnapshot, ...] = field(default_factory=tuple)


type ConceptSnapshotCollection = tuple[ConceptSnapshot, ...]
type BeliefSnapshotCollection = tuple[BeliefSnapshot, ...]
type RuleSnapshotCollection = tuple[RuleSnapshot, ...]
type HypothesisSnapshotCollection = tuple[HypothesisSnapshot, ...]
type GoalSnapshotCollection = tuple[GoalSnapshot, ...]


def verify_grounding(
    proposal: HypothesisProposal,
    grounding: GroundingSnapshot,
) -> bool:
    """Verify strict grounding of a proposal (ADR-0028).

    A proposal MUST reference at least one existing entity. Proposals with no
    references, or with references to nonexistent entities, are rejected.

    """
    known = (
        grounding.belief_ids
        | grounding.concept_ids
        | grounding.rule_ids
        | grounding.evidence_ids
    )
    return bool(set(proposal.referenced_ids) & known)
