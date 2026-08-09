"""commonsense domain — everyday reasoning and state-tracking benchmark cases."""

from __future__ import annotations

from decimal import Decimal

from theo_core.evaluation.benchmark_schema import (
    BenchmarkCase,
    BenchmarkId,
    FailureMode,
    GoldenTrace,
)
from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.models import Belief, BeliefEdge, BeliefId, BeliefRelation
from theo_core.symbolic.decisions.models import Intent


def _belief(uri: str, proposition: str, confidence: Decimal = Decimal("1.0")) -> Belief:
    """Build a Belief with the given id, proposition and confidence."""
    return Belief(id=BeliefId.of(uri), proposition=proposition, confidence=confidence)


CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        id=BenchmarkId.of("bm://commonsense/001"),
        domain="commonsense",
        name="COMMON-001: Water Freezing Point",
        description="Retrieve the everyday fact that water freezes below zero.",
        initial_beliefs=(
            _belief("belief://c_water_freezing", "water freezes below zero degrees Celsius"),
        ),
        percept_input="Water freezes in the winter",
        expected_beliefs=("water freezes below zero degrees Celsius",),
        expected_action_text=(
            "Interpretation based on belief 'water freezes below zero degrees Celsius'"
        ),
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(SymbolicId.of("belief://c_water_freezing"),),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://commonsense/002"),
        domain="commonsense",
        name="COMMON-002: Birds Migrate South",
        description="Retrieve the everyday fact that birds fly south in autumn.",
        initial_beliefs=(_belief("belief://c_birds", "birds fly south in autumn"),),
        percept_input="Birds fly south",
        expected_beliefs=("birds fly south in autumn",),
        expected_action_text="Interpretation based on belief 'birds fly south in autumn'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(SymbolicId.of("belief://c_birds"),),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://commonsense/003"),
        domain="commonsense",
        name="COMMON-003: Fire Needs Oxygen",
        description="Retrieve the everyday fact that fire requires oxygen.",
        initial_beliefs=(_belief("belief://c_fire_oxygen", "fire needs oxygen to burn"),),
        percept_input="Fire burns in the air",
        expected_beliefs=("fire needs oxygen to burn",),
        expected_action_text="Interpretation based on belief 'fire needs oxygen to burn'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(SymbolicId.of("belief://c_fire_oxygen"),),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://commonsense/004"),
        domain="commonsense",
        name="COMMON-004: Plants Seek Sunlight",
        description="Retrieve the everyday fact that plants grow towards sunlight.",
        initial_beliefs=(_belief("belief://c_plants", "plants grow towards sunlight"),),
        percept_input="Plants grow in the garden",
        expected_beliefs=("plants grow towards sunlight",),
        expected_action_text="Interpretation based on belief 'plants grow towards sunlight'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(SymbolicId.of("belief://c_plants"),),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://commonsense/005"),
        domain="commonsense",
        name="COMMON-005: Metal Expansion",
        description="Retrieve the everyday fact that metal expands when heated.",
        initial_beliefs=(_belief("belief://c_metal", "metal expands when heated"),),
        percept_input="Metal expands",
        expected_beliefs=("metal expands when heated",),
        expected_action_text="Interpretation based on belief 'metal expands when heated'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(SymbolicId.of("belief://c_metal"),),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://commonsense/006"),
        domain="commonsense",
        name="COMMON-006: Apple Distractor Evidence",
        description="Probe DISTRACTOR_EVIDENCE: a high-confidence distractor "
        "belief sharing the word 'apple' must not displace the correct fruit "
        "reading; both stay active with declared ordering.",
        initial_beliefs=(
            _belief("belief://c_apple_fruit", "an apple is a fruit", Decimal("0.9")),
            _belief("belief://c_apple_tech", "apple makes computers", Decimal("0.85")),
        ),
        percept_input="is an apple healthy",
        expected_beliefs=("an apple is a fruit", "apple makes computers"),
        expected_action_text="Interpretation based on belief 'is an apple healthy'",
        min_confidence=Decimal("0.9"),
        max_confidence=Decimal("1.0"),
        expected_intent=Intent.PROVIDE_RECOMMENDATION,
        failure_mode=FailureMode.DISTRACTOR_EVIDENCE,
        initial_goals=("ProvideRecommendation",),
        metadata={
            "ground_truth_ordering": {
                "an apple is a fruit": "0.9",
                "apple makes computers": "0.85",
            }
        },
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://c_apple_fruit"),
                SymbolicId.of("belief://c_apple_tech"),
            ),
            generated_hypothesis_ids=(
                SymbolicId.of("hypothesis://cand/3"),
                SymbolicId.of("hypothesis://cand/1"),
                SymbolicId.of("hypothesis://cand/2"),
            ),
            resolved_conflict_ids=(
                SymbolicId.of("conflict://hyp/cand/3_cand/1"),
                SymbolicId.of("conflict://hyp/cand/3_cand/2"),
            ),
            thought_dag_node_count=0,
        ),
        baseline={
            "accepted_hypothesis_id": "hypothesis://cand/3",
            "action_text": "Interpretation based on belief 'is an apple healthy'",
            "activated_concepts": [],
            "captured_at": "v0.4.1 (ADR-0028 Phase 2, pre-provider)",
            "confidence": "1.0000",
            "dag_node_count": 0,
            "decision_id": "decision://select/cand/3",
            "decision_type": "response",
            "derived_beliefs": ["belief://percept/db2c9754"],
            "fingerprint": {
                "activated_concept_ids": [],
                "decision_id": "decision://select/cand/3",
                "derived_belief_ids": ["belief://percept/db2c9754"],
                "fired_rule_ids": [],
                "generated_hypothesis_ids": [
                    "hypothesis://cand/3",
                    "hypothesis://cand/1",
                    "hypothesis://cand/2",
                ],
                "resolved_conflict_ids": [
                    "conflict://hyp/cand/3_cand/1",
                    "conflict://hyp/cand/3_cand/2",
                ],
                "response_text": "Interpretation based on belief 'is an apple healthy'",
                "retrieved_memory_ids": ["belief://c_apple_fruit", "belief://c_apple_tech"],
                "thought_dag_node_count": 0,
            },
            "fired_rules": [],
            "generated_hypotheses": [
                "hypothesis://cand/3",
                "hypothesis://cand/1",
                "hypothesis://cand/2",
            ],
            "intent": "provide_recommendation",
            "referenced_goal": "goal://providerecommendation",
            "resolved_conflicts": ["conflict://hyp/cand/3_cand/1", "conflict://hyp/cand/3_cand/2"],
            "retrieved_memories": ["belief://c_apple_fruit", "belief://c_apple_tech"],
            "stages": [
                "perception",
                "activation",
                "revision",
                "inference",
                "hypothesis",
                "conflict_resolution",
                "decision",
                "realization",
                "learning",
            ],
            "state_checksum": "399758530c04a47810ce274a478584b5ec9317be2ab5d8d45693eae7a7823e2e",
        },
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://commonsense/007"),
        domain="commonsense",
        name="COMMON-007: Bats False Association",
        description="Probe FALSE_ASSOCIATION: the false association 'bats are "
        "birds' contradicts the correct 'bats are mammals' and must be "
        "deprecated by revision while the correct fact stays active.",
        initial_beliefs=(
            _belief("belief://c_bats_birds", "bats are birds", Decimal("0.55")),
            _belief("belief://c_bats_mammals", "bats are mammals", Decimal("0.9")),
        ),
        initial_belief_edges=(
            BeliefEdge(
                source=BeliefId.of("belief://c_bats_birds"),
                target=BeliefId.of("belief://c_bats_mammals"),
                relation=BeliefRelation.CONTRADICTS,
            ),
        ),
        percept_input="what do bats eat",
        expected_beliefs=("bats are mammals",),
        excluded_beliefs=("bats are birds",),
        expected_action_text="Interpretation based on belief 'what do bats eat'",
        min_confidence=Decimal("0.9"),
        max_confidence=Decimal("1.0"),
        expected_intent=Intent.REMEMBER_FACT,
        failure_mode=FailureMode.FALSE_ASSOCIATION,
        initial_goals=("RememberFact",),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://c_bats_birds"),
                SymbolicId.of("belief://c_bats_mammals"),
            ),
            generated_hypothesis_ids=(
                SymbolicId.of("hypothesis://cand/2"),
                SymbolicId.of("hypothesis://cand/1"),
            ),
            resolved_conflict_ids=(SymbolicId.of("conflict://hyp/cand/2_cand/1"),),
            thought_dag_node_count=0,
        ),
        baseline={
            "accepted_hypothesis_id": "hypothesis://cand/2",
            "action_text": "Interpretation based on belief 'what do bats eat'",
            "activated_concepts": [],
            "captured_at": "v0.4.1 (ADR-0028 Phase 2, pre-provider)",
            "confidence": "1.0000",
            "dag_node_count": 0,
            "decision_id": "decision://select/cand/2",
            "decision_type": "response",
            "derived_beliefs": ["belief://percept/f614a16f"],
            "fingerprint": {
                "activated_concept_ids": [],
                "decision_id": "decision://select/cand/2",
                "derived_belief_ids": ["belief://percept/f614a16f"],
                "fired_rule_ids": [],
                "generated_hypothesis_ids": ["hypothesis://cand/2", "hypothesis://cand/1"],
                "resolved_conflict_ids": ["conflict://hyp/cand/2_cand/1"],
                "response_text": "Interpretation based on belief 'what do bats eat'",
                "retrieved_memory_ids": ["belief://c_bats_birds", "belief://c_bats_mammals"],
                "thought_dag_node_count": 0,
            },
            "fired_rules": [],
            "generated_hypotheses": ["hypothesis://cand/2", "hypothesis://cand/1"],
            "intent": "remember_fact",
            "referenced_goal": "goal://rememberfact",
            "resolved_conflicts": ["conflict://hyp/cand/2_cand/1"],
            "retrieved_memories": ["belief://c_bats_birds", "belief://c_bats_mammals"],
            "stages": [
                "perception",
                "activation",
                "revision",
                "inference",
                "hypothesis",
                "conflict_resolution",
                "decision",
                "realization",
                "learning",
            ],
            "state_checksum": "faf61fc5ed5b90ad92c15b43c27579305316d1b58bb3c888eed5d6a5510596d4",
        },
    ),
)
