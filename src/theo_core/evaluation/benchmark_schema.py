"""Structured Schema for THEO Cognitive Benchmark Corpus (v0.4.1).

Defines formal benchmark case structures and golden execution trace schemas to enable
longitudinal cognitive evaluation across symbolic (v0.4), neural (v0.5), and hybrid (v0.6) runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.models import Belief, BeliefEdge  # noqa: TC001
from theo_core.symbolic.concepts.models import Concept, ConceptEdge  # noqa: TC001
from theo_core.symbolic.decisions.models import Intent  # noqa: TC001
from theo_core.symbolic.inference.models import InferenceRule  # noqa: TC001


@dataclass(frozen=True, slots=True)
class BenchmarkId:
    """URI-style benchmark identifier value object."""

    value: str

    @classmethod
    def of(cls, value: str) -> BenchmarkId:
        """Validate benchmark URI scheme."""
        if not value.startswith("bm://"):
            msg = f"BenchmarkId URI must start with 'bm://', got {value!r}"
            raise ValueError(msg)
        symbolic_id = SymbolicId.of(value)
        return cls(value=symbolic_id.value)

    def __str__(self) -> str:
        """Return raw string value."""
        return self.value


class GoldenTrace(BaseModel, frozen=True):
    """Step-by-step golden execution trace snapshot for cognitive diagnostic tracing."""

    activated_concept_ids: tuple[SymbolicId, ...] = Field(default_factory=tuple)
    retrieved_memory_ids: tuple[SymbolicId, ...] = Field(default_factory=tuple)
    generated_hypothesis_ids: tuple[SymbolicId, ...] = Field(default_factory=tuple)
    fired_rule_ids: tuple[SymbolicId, ...] = Field(default_factory=tuple)
    derived_belief_ids: tuple[SymbolicId, ...] = Field(default_factory=tuple)
    resolved_conflict_ids: tuple[SymbolicId, ...] = Field(default_factory=tuple)
    thought_dag_node_count: int = 0
    decision_id: SymbolicId | None = None
    response_text: str = ""


class FailureMode(StrEnum):
    """Declared weakness a case is designed to probe (architecture-neutral).

    Phase 2: every new case declares exactly one failure mode. The mode names
    the measured limitation (e.g. confidence compression, lexical distractors)
    without prescribing how any provider should address it. The declared mode
    is corpus metadata, never a provider contract.
    """

    MULTI_HOP = "multi_hop"
    SPARSE_KNOWLEDGE = "sparse_knowledge"
    DISTRACTOR_EVIDENCE = "distractor_evidence"
    SYNONYM_AMBIGUITY = "synonym_ambiguity"
    FALSE_ASSOCIATION = "false_association"
    CALIBRATION = "calibration"


class BenchmarkCase(BaseModel, frozen=True):
    """Formal structured benchmark case definition.

    Fields added in Phase 2 (all additive with defaults, so no pre-v0.5 case
    is affected): ``expected_intent``, ``failure_mode``, ``initial_goals``, and
    ``baseline`` (the frozen v0.4.1 measurement captured at case creation).
    """

    id: BenchmarkId
    domain: str  # e.g., "commonsense", "causal_reasoning", "contradiction", "taxonomy"
    name: str
    description: str
    initial_concepts: tuple[Concept, ...] = Field(default_factory=tuple)
    initial_concept_edges: tuple[ConceptEdge, ...] = Field(default_factory=tuple)
    initial_beliefs: tuple[Belief, ...] = Field(default_factory=tuple)
    initial_belief_edges: tuple[BeliefEdge, ...] = Field(default_factory=tuple)
    rules: tuple[InferenceRule, ...] = Field(default_factory=tuple)
    initial_knowledge_base: list[dict[str, Any]] = Field(default_factory=list)
    percept_input: str
    expected_beliefs: tuple[str, ...] = Field(default_factory=tuple)
    excluded_beliefs: tuple[str, ...] = Field(default_factory=tuple)
    expected_decision_type: str = "response"
    expected_action_text: str
    min_confidence: Decimal = Field(default=Decimal("0.5"), ge=Decimal("0.0"), le=Decimal("1.0"))
    max_confidence: Decimal = Field(default=Decimal("1.0"), ge=Decimal("0.0"), le=Decimal("1.0"))
    expected_intent: Intent | None = None
    failure_mode: FailureMode | None = None
    initial_goals: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Goal descriptions seeded into the GoalManager before the cycle; "
            "the deterministic goal slug drives the decision intent "
            "(Canon Invariant 7 / audit F5)."
        ),
    )
    baseline: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Frozen v0.4.1 baseline measurement captured when the case was "
            "created, before any provider was evaluated. Diagnostic metadata "
            "only; the harness never asserts on it."
        ),
    )
    golden_trace: GoldenTrace | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


FINGERPRINT_METADATA_KEY = "theo_golden_fingerprint"
PRE_CYCLE_STATE_METADATA_KEY = "theo_pre_cycle_state"
STATE_HASH_BEFORE_METADATA_KEY = "theo_state_hash_before"
STATE_HASH_AFTER_METADATA_KEY = "theo_state_hash_after"
PROVIDER_PROVENANCE_METADATA_KEY = "theo_provider_provenance"


def golden_fingerprint(golden_trace: GoldenTrace, response_text: str) -> dict[str, object]:
    """Project a GoldenTrace into a canonical, JSON-serializable fingerprint.

    Recording, replay, and cross-process determinism all use this projection so
    a replayed cycle is compared field-for-field — decision, fired rules, derived
    beliefs, hypotheses, activated concepts, and thought-DAG size — and never
    just by rendered response text.
    """
    return {
        "decision_id": golden_trace.decision_id.value if golden_trace.decision_id else None,
        "response_text": response_text,
        "activated_concept_ids": [str(i) for i in golden_trace.activated_concept_ids],
        "retrieved_memory_ids": [str(i) for i in golden_trace.retrieved_memory_ids],
        "generated_hypothesis_ids": [str(i) for i in golden_trace.generated_hypothesis_ids],
        "fired_rule_ids": [str(i) for i in golden_trace.fired_rule_ids],
        "derived_belief_ids": [str(i) for i in golden_trace.derived_belief_ids],
        "resolved_conflict_ids": [str(i) for i in golden_trace.resolved_conflict_ids],
        "thought_dag_node_count": golden_trace.thought_dag_node_count,
    }


GoldenTrace.model_rebuild()
BenchmarkCase.model_rebuild()
