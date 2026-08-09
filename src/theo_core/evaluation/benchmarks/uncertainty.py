"""uncertainty domain — probabilistic confidence and propagation benchmark cases."""

from __future__ import annotations

from decimal import Decimal

from theo_core.evaluation.benchmark_schema import (
    BenchmarkCase,
    BenchmarkId,
    FailureMode,
    GoldenTrace,
)
from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.models import Belief, BeliefId
from theo_core.symbolic.decisions.models import Intent
from theo_core.symbolic.inference.models import InferenceRule, RuleCondition, RuleId


def _belief(uri: str, proposition: str, confidence: Decimal = Decimal("1.0")) -> Belief:
    """Build a Belief with the given id, proposition and confidence."""
    return Belief(id=BeliefId.of(uri), proposition=proposition, confidence=confidence)


def _rule(uri: str, premise: str, conclusion: str, multiplier: Decimal) -> InferenceRule:
    """Build a single-condition InferenceRule."""
    return InferenceRule(
        id=RuleId.of(uri),
        name=conclusion,
        conditions=(RuleCondition(premise_predicate=premise),),
        conclusion_template=conclusion,
        confidence_multiplier=multiplier,
    )


CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        id=BenchmarkId.of("bm://uncertainty/001"),
        domain="uncertainty",
        name="UNC-001: Rain Confidence at Half",
        description="Verify confidence 0.5 propagation drags the decision to 0.75.",
        rules=(
            _rule(
                "rule://uncertainty/rain_wet",
                "rain",
                "The ground is wet",
                Decimal("0.5"),
            ),
        ),
        percept_input="rain is falling",
        expected_beliefs=("The ground is wet",),
        expected_action_text="Interpretation based on belief 'rain is falling'",
        min_confidence=Decimal("0.6"),
        max_confidence=Decimal("0.85"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(),
            fired_rule_ids=(SymbolicId.of("rule://uncertainty/rain_wet"),),
            thought_dag_node_count=1,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://uncertainty/002"),
        domain="uncertainty",
        name="UNC-002: Cloud Forecast Propagation",
        description="Verify sub-threshold derived belief while the percept "
        "decision remains confident.",
        rules=(
            _rule(
                "rule://uncertainty/clouds_gather",
                "cloud",
                "Sky is cloudy",
                Decimal("0.4"),
            ),
        ),
        percept_input="clouds gather",
        expected_beliefs=("Sky is cloudy",),
        expected_action_text="Interpretation based on belief 'clouds gather'",
        min_confidence=Decimal("0.6"),
        max_confidence=Decimal("0.8"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(),
            fired_rule_ids=(SymbolicId.of("rule://uncertainty/clouds_gather"),),
            thought_dag_node_count=1,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://uncertainty/003"),
        domain="uncertainty",
        name="UNC-003: Rain Confidence Propagation",
        description="Verify confidence multiplication through a rule fired on the percept.",
        rules=(
            _rule(
                "rule://uncertainty/weather_rain",
                "rain",
                "The ground is wet",
                Decimal("0.6"),
            ),
        ),
        percept_input="rain is falling",
        expected_beliefs=("The ground is wet",),
        expected_action_text="Interpretation based on belief 'rain is falling'",
        min_confidence=Decimal("0.6"),
        max_confidence=Decimal("0.9"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(),
            fired_rule_ids=(SymbolicId.of("rule://uncertainty/weather_rain"),),
            thought_dag_node_count=1,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://uncertainty/004"),
        domain="uncertainty",
        name="UNC-004: Weak Signal Propagation",
        description="Verify a low-multiplier rule still propagates a derived belief.",
        rules=(
            _rule(
                "rule://uncertainty/signal_weak",
                "signal",
                "Signal is weak",
                Decimal("0.3"),
            ),
        ),
        percept_input="signal detected",
        expected_beliefs=("Signal is weak",),
        expected_action_text="Interpretation based on belief 'signal detected'",
        min_confidence=Decimal("0.55"),
        max_confidence=Decimal("0.75"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(),
            fired_rule_ids=(SymbolicId.of("rule://uncertainty/signal_weak"),),
            thought_dag_node_count=1,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://uncertainty/005"),
        domain="uncertainty",
        name="UNC-005: Low-Confidence Premise",
        description="Verify a low-confidence premise still yields a derived belief.",
        initial_beliefs=(_belief("belief://c_market", "market volatile", Decimal("0.5")),),
        rules=(
            _rule(
                "rule://uncertainty/volatile_diversify",
                "volatile",
                "Diversify your portfolio",
                Decimal("0.9"),
            ),
        ),
        percept_input="market is volatile today",
        expected_beliefs=("Diversify your portfolio",),
        expected_action_text="Interpretation based on belief 'market is volatile today'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(SymbolicId.of("belief://c_market"),),
            fired_rule_ids=(SymbolicId.of("rule://uncertainty/volatile_diversify"),),
            thought_dag_node_count=1,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://uncertainty/006"),
        domain="uncertainty",
        name="UNC-006: Strong Rule Calibration",
        description="Probe CALIBRATION (strong rung): a strong 0.9 multiplier "
        "rule must yield a high-confidence decision; the confidence ladder "
        "across UNC-006/007/008 is monotone in rule strength.",
        rules=(
            _rule(
                "rule://uncertainty/storm_weather",
                "storm",
                "severe weather is approaching",
                Decimal("0.9"),
            ),
        ),
        percept_input="a storm is approaching",
        expected_beliefs=("severe weather is approaching",),
        expected_action_text="Interpretation based on belief 'a storm is approaching'",
        min_confidence=Decimal("0.9"),
        max_confidence=Decimal("1.0"),
        expected_intent=Intent.MAINTAIN_CONVERSATION,
        failure_mode=FailureMode.CALIBRATION,
        metadata={
            "calibration_rung": "strong",
            "rule_multiplier": "0.9",
            "ground_truth_ordering": {
                "strong_rule_0.9": "decision confidence should be the highest in the ladder"
            },
        },
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(),
            fired_rule_ids=(SymbolicId.of("rule://uncertainty/storm_weather"),),
            generated_hypothesis_ids=(
                SymbolicId.of("hypothesis://cand/2"),
                SymbolicId.of("hypothesis://cand/1"),
            ),
            resolved_conflict_ids=(SymbolicId.of("conflict://hyp/cand/2_cand/1"),),
            thought_dag_node_count=1,
        ),
        baseline={
            "accepted_hypothesis_id": "hypothesis://cand/2",
            "action_text": "Interpretation based on belief 'a storm is approaching'",
            "activated_concepts": [],
            "captured_at": "v0.4.1 (ADR-0028 Phase 2, pre-provider)",
            "confidence": "0.9500",
            "dag_node_count": 1,
            "decision_id": "decision://select/cand/2",
            "decision_type": "response",
            "derived_beliefs": [
                "belief://inf/uncertainty/storm_weather/1",
                "belief://percept/b2432293",
            ],
            "fingerprint": {
                "activated_concept_ids": [],
                "decision_id": "decision://select/cand/2",
                "derived_belief_ids": [
                    "belief://inf/uncertainty/storm_weather/1",
                    "belief://percept/b2432293",
                ],
                "fired_rule_ids": ["rule://uncertainty/storm_weather"],
                "generated_hypothesis_ids": ["hypothesis://cand/2", "hypothesis://cand/1"],
                "resolved_conflict_ids": ["conflict://hyp/cand/2_cand/1"],
                "response_text": "Interpretation based on belief 'a storm is approaching'",
                "retrieved_memory_ids": [],
                "thought_dag_node_count": 1,
            },
            "fired_rules": ["rule://uncertainty/storm_weather"],
            "generated_hypotheses": ["hypothesis://cand/2", "hypothesis://cand/1"],
            "intent": "maintain_conversation",
            "referenced_goal": "goal://maintainconversation",
            "resolved_conflicts": ["conflict://hyp/cand/2_cand/1"],
            "retrieved_memories": [],
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
            "state_checksum": "64875249952ede07bcfaea29a50cf39acc29ca93bb1ac4c023979bb6fae2456f",
        },
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://uncertainty/007"),
        domain="uncertainty",
        name="UNC-007: Sparse-Evidence Moderate Rule",
        description="Probe SPARSE_KNOWLEDGE/CALIBRATION (middle rung): with a "
        "single moderate 0.4 rule as the only evidence, the decision confidence "
        "must land below the strong rung; the weak support must not be masked.",
        rules=(
            _rule(
                "rule://uncertainty/fog_visibility",
                "fog",
                "visibility is reduced",
                Decimal("0.4"),
            ),
        ),
        percept_input="the fog rolls in",
        expected_beliefs=("visibility is reduced",),
        expected_action_text="Interpretation based on belief 'the fog rolls in'",
        min_confidence=Decimal("0.6"),
        max_confidence=Decimal("0.8"),
        expected_intent=Intent.MAINTAIN_CONVERSATION,
        failure_mode=FailureMode.SPARSE_KNOWLEDGE,
        metadata={
            "calibration_rung": "moderate",
            "rule_multiplier": "0.4",
            "ground_truth_ordering": {
                "moderate_rule_0.4": "decision confidence should sit between strong and weak rungs"
            },
        },
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(),
            fired_rule_ids=(SymbolicId.of("rule://uncertainty/fog_visibility"),),
            generated_hypothesis_ids=(SymbolicId.of("hypothesis://cand/1"),),
            thought_dag_node_count=1,
        ),
        baseline={
            "accepted_hypothesis_id": "hypothesis://cand/1",
            "action_text": "Interpretation based on belief 'the fog rolls in'",
            "activated_concepts": [],
            "captured_at": "v0.4.1 (ADR-0028 Phase 2, pre-provider)",
            "confidence": "0.7000",
            "dag_node_count": 1,
            "decision_id": "decision://select/cand/1",
            "decision_type": "response",
            "derived_beliefs": [
                "belief://inf/uncertainty/fog_visibility/1",
                "belief://percept/a7529822",
            ],
            "fingerprint": {
                "activated_concept_ids": [],
                "decision_id": "decision://select/cand/1",
                "derived_belief_ids": [
                    "belief://inf/uncertainty/fog_visibility/1",
                    "belief://percept/a7529822",
                ],
                "fired_rule_ids": ["rule://uncertainty/fog_visibility"],
                "generated_hypothesis_ids": ["hypothesis://cand/1"],
                "resolved_conflict_ids": [],
                "response_text": "Interpretation based on belief 'the fog rolls in'",
                "retrieved_memory_ids": [],
                "thought_dag_node_count": 1,
            },
            "fired_rules": ["rule://uncertainty/fog_visibility"],
            "generated_hypotheses": ["hypothesis://cand/1"],
            "intent": "maintain_conversation",
            "referenced_goal": "goal://maintainconversation",
            "resolved_conflicts": [],
            "retrieved_memories": [],
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
            "state_checksum": "8754d66fbf9b9f12b951f15d3cf267b5c887ab877ed9e43487a8f5fa4eab4e0e",
        },
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://uncertainty/008"),
        domain="uncertainty",
        name="UNC-008: Weak Rule Calibration",
        description="Probe CALIBRATION (weak rung): a weak 0.2 multiplier rule "
        "must yield the lowest decision confidence in the ladder while the "
        "derived belief still persists.",
        rules=(
            _rule(
                "rule://uncertainty/tremor_earth",
                "tremor",
                "minor earth movement detected",
                Decimal("0.2"),
            ),
        ),
        percept_input="a tremor is felt",
        expected_beliefs=("minor earth movement detected",),
        expected_action_text="Interpretation based on belief 'a tremor is felt'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("0.7"),
        expected_intent=Intent.MAINTAIN_CONVERSATION,
        failure_mode=FailureMode.CALIBRATION,
        metadata={
            "calibration_rung": "weak",
            "rule_multiplier": "0.2",
            "ground_truth_ordering": {
                "weak_rule_0.2": "decision confidence should be the lowest in the ladder"
            },
        },
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(),
            fired_rule_ids=(SymbolicId.of("rule://uncertainty/tremor_earth"),),
            generated_hypothesis_ids=(
                SymbolicId.of("hypothesis://cand/2"),
                SymbolicId.of("hypothesis://cand/1"),
            ),
            resolved_conflict_ids=(SymbolicId.of("conflict://hyp/cand/2_cand/1"),),
            thought_dag_node_count=1,
        ),
        baseline={
            "accepted_hypothesis_id": "hypothesis://cand/2",
            "action_text": "Interpretation based on belief 'a tremor is felt'",
            "activated_concepts": [],
            "captured_at": "v0.4.1 (ADR-0028 Phase 2, pre-provider)",
            "confidence": "0.6000",
            "dag_node_count": 1,
            "decision_id": "decision://select/cand/2",
            "decision_type": "response",
            "derived_beliefs": [
                "belief://inf/uncertainty/tremor_earth/1",
                "belief://percept/dbdfa0a8",
            ],
            "fingerprint": {
                "activated_concept_ids": [],
                "decision_id": "decision://select/cand/2",
                "derived_belief_ids": [
                    "belief://inf/uncertainty/tremor_earth/1",
                    "belief://percept/dbdfa0a8",
                ],
                "fired_rule_ids": ["rule://uncertainty/tremor_earth"],
                "generated_hypothesis_ids": ["hypothesis://cand/2", "hypothesis://cand/1"],
                "resolved_conflict_ids": ["conflict://hyp/cand/2_cand/1"],
                "response_text": "Interpretation based on belief 'a tremor is felt'",
                "retrieved_memory_ids": [],
                "thought_dag_node_count": 1,
            },
            "fired_rules": ["rule://uncertainty/tremor_earth"],
            "generated_hypotheses": ["hypothesis://cand/2", "hypothesis://cand/1"],
            "intent": "maintain_conversation",
            "referenced_goal": "goal://maintainconversation",
            "resolved_conflicts": ["conflict://hyp/cand/2_cand/1"],
            "retrieved_memories": [],
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
            "state_checksum": "1491e3d3303dcdfe18026422ae7ac4b53294d0353af4b050c0b4c85b2fcb866d",
        },
    ),
)
