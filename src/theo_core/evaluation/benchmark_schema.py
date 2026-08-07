"""Structured Schema for THEO Cognitive Benchmark Corpus (v0.4.1).

Defines formal benchmark case structures and golden execution trace schemas to enable
longitudinal cognitive evaluation across symbolic (v0.4), neural (v0.5), and hybrid (v0.6) runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.models import Belief, BeliefEdge  # noqa: TC001
from theo_core.symbolic.concepts.models import Concept, ConceptEdge  # noqa: TC001
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


class BenchmarkCase(BaseModel, frozen=True):
    """Formal structured benchmark case definition."""

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
    golden_trace: GoldenTrace | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


GoldenTrace.model_rebuild()
BenchmarkCase.model_rebuild()
