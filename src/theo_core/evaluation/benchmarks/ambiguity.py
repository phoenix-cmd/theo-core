"""ambiguity domain — competing-interpretation benchmark cases.

Each case seeds multiple beliefs that plausibly interpret the same percept.
Canon Law 6 requires the system to generate competing hypotheses rather than
collapsing to a single interpretation prematurely: the seeded beliefs must all
remain active post-cycle, and the golden trace must record a distinct candidate
hypothesis per matching belief.
"""

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


def _belief(uri: str, proposition: str, confidence: Decimal) -> Belief:
    """Build a Belief with the given id, proposition and confidence."""
    return Belief(id=BeliefId.of(uri), proposition=proposition, confidence=confidence)


CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        id=BenchmarkId.of("bm://ambiguity/001"),
        domain="ambiguity",
        name="AMB-001: Sky Colour Ambiguity",
        description="Verify competing sky-colour interpretations generate "
        "multiple hypotheses and both beliefs stay active.",
        initial_beliefs=(
            _belief("belief://b_sky_blue", "the sky is blue", Decimal("0.8")),
            _belief("belief://b_sky_grey", "the sky is grey", Decimal("0.7")),
        ),
        percept_input="what color is the sky",
        expected_beliefs=("the sky is blue", "the sky is grey"),
        expected_action_text="Interpretation based on belief 'what color is the sky'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://b_sky_blue"),
                SymbolicId.of("belief://b_sky_grey"),
            ),
            generated_hypothesis_ids=(
                SymbolicId.of("hypothesis://cand/3"),
                SymbolicId.of("hypothesis://cand/1"),
                SymbolicId.of("hypothesis://cand/2"),
            ),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://ambiguity/002"),
        domain="ambiguity",
        name="AMB-002: Door State Equal-Confidence Ambiguity",
        description="Verify equal-confidence door interpretations both stay "
        "active without premature collapse.",
        initial_beliefs=(
            _belief("belief://b_door_closed", "the door is closed", Decimal("0.8")),
            _belief("belief://b_door_open", "the door is open", Decimal("0.8")),
        ),
        percept_input="check the door",
        expected_beliefs=("the door is closed", "the door is open"),
        expected_action_text="Interpretation based on belief 'check the door'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://b_door_closed"),
                SymbolicId.of("belief://b_door_open"),
            ),
            generated_hypothesis_ids=(
                SymbolicId.of("hypothesis://cand/3"),
                SymbolicId.of("hypothesis://cand/1"),
                SymbolicId.of("hypothesis://cand/2"),
            ),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://ambiguity/003"),
        domain="ambiguity",
        name="AMB-003: Three-Way Weather Ambiguity",
        description="Verify a three-way weather ambiguity generates a competing "
        "hypothesis per matching belief.",
        initial_beliefs=(
            _belief("belief://b_weather_rain", "it will rain today", Decimal("0.8")),
            _belief("belief://b_weather_snow", "it will snow today", Decimal("0.7")),
            _belief("belief://b_weather_sunny", "it will be sunny today", Decimal("0.6")),
        ),
        percept_input="what is the weather today",
        expected_beliefs=(
            "it will rain today",
            "it will snow today",
            "it will be sunny today",
        ),
        expected_action_text="Interpretation based on belief 'what is the weather today'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://b_weather_rain"),
                SymbolicId.of("belief://b_weather_snow"),
                SymbolicId.of("belief://b_weather_sunny"),
            ),
            generated_hypothesis_ids=(
                SymbolicId.of("hypothesis://cand/4"),
                SymbolicId.of("hypothesis://cand/1"),
                SymbolicId.of("hypothesis://cand/2"),
                SymbolicId.of("hypothesis://cand/3"),
            ),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://ambiguity/004"),
        domain="ambiguity",
        name="AMB-004: Meeting Time Ambiguity",
        description="Verify meeting-related beliefs with different confidences "
        "still generate competing hypotheses.",
        initial_beliefs=(
            _belief("belief://b_meeting_cancel", "the meeting is cancelled", Decimal("0.5")),
            _belief("belief://b_meeting_noon", "the meeting is at noon", Decimal("0.9")),
        ),
        percept_input="what time is the meeting",
        expected_beliefs=("the meeting is cancelled", "the meeting is at noon"),
        expected_action_text="Interpretation based on belief 'what time is the meeting'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://b_meeting_cancel"),
                SymbolicId.of("belief://b_meeting_noon"),
            ),
            generated_hypothesis_ids=(
                SymbolicId.of("hypothesis://cand/3"),
                SymbolicId.of("hypothesis://cand/2"),
                SymbolicId.of("hypothesis://cand/1"),
            ),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://ambiguity/005"),
        domain="ambiguity",
        name="AMB-005: Lexical Bank Ambiguity",
        description="Verify lexical ambiguity (river bank vs financial bank) "
        "generates competing hypotheses.",
        initial_beliefs=(
            _belief("belief://b_bank_open", "the bank opens at nine", Decimal("0.8")),
            _belief("belief://b_bank_river", "the bank is near the river", Decimal("0.8")),
        ),
        percept_input="which bank closes today",
        expected_beliefs=("the bank opens at nine", "the bank is near the river"),
        expected_action_text="Interpretation based on belief 'which bank closes today'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://b_bank_open"),
                SymbolicId.of("belief://b_bank_river"),
            ),
            generated_hypothesis_ids=(
                SymbolicId.of("hypothesis://cand/3"),
                SymbolicId.of("hypothesis://cand/1"),
                SymbolicId.of("hypothesis://cand/2"),
            ),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://ambiguity/006"),
        domain="ambiguity",
        name="AMB-006: Graded Sky Interpretation",
        description="Probe confidence calibration (CALIBRATION): three graded "
        "sky readings (clear 0.9, overcast 0.7, hazy 0.5) with a known correct "
        "ordering must all remain active; the decision confidence is measured "
        "against the strongest interpretation.",
        initial_beliefs=(
            _belief("belief://b_sky_clear", "the sky is clear", Decimal("0.9")),
            _belief("belief://b_sky_hazy", "the sky is hazy", Decimal("0.5")),
            _belief("belief://b_sky_overcast", "the sky is overcast", Decimal("0.7")),
        ),
        percept_input="how is the sky looking",
        expected_beliefs=("the sky is clear", "the sky is hazy", "the sky is overcast"),
        expected_action_text="Interpretation based on belief 'how is the sky looking'",
        min_confidence=Decimal("0.9"),
        max_confidence=Decimal("1.0"),
        expected_intent=Intent.MAINTAIN_CONVERSATION,
        failure_mode=FailureMode.CALIBRATION,
        metadata={
            "ground_truth_ordering": {
                "the sky is clear": "0.9",
                "the sky is overcast": "0.7",
                "the sky is hazy": "0.5",
            }
        },
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://b_sky_clear"),
                SymbolicId.of("belief://b_sky_hazy"),
                SymbolicId.of("belief://b_sky_overcast"),
            ),
            generated_hypothesis_ids=(
                SymbolicId.of("hypothesis://cand/4"),
                SymbolicId.of("hypothesis://cand/1"),
                SymbolicId.of("hypothesis://cand/3"),
                SymbolicId.of("hypothesis://cand/2"),
            ),
            resolved_conflict_ids=(
                SymbolicId.of("conflict://hyp/cand/4_cand/1"),
                SymbolicId.of("conflict://hyp/cand/4_cand/3"),
                SymbolicId.of("conflict://hyp/cand/4_cand/2"),
            ),
            thought_dag_node_count=0,
        ),
        baseline={
            "accepted_hypothesis_id": "hypothesis://cand/4",
            "action_text": "Interpretation based on belief 'how is the sky looking'",
            "activated_concepts": [],
            "captured_at": "v0.4.1 (ADR-0028 Phase 2, pre-provider)",
            "confidence": "1.0000",
            "dag_node_count": 0,
            "decision_id": "decision://select/cand/4",
            "decision_type": "response",
            "derived_beliefs": ["belief://percept/3867c31d"],
            "fingerprint": {
                "activated_concept_ids": [],
                "decision_id": "decision://select/cand/4",
                "derived_belief_ids": ["belief://percept/3867c31d"],
                "fired_rule_ids": [],
                "generated_hypothesis_ids": [
                    "hypothesis://cand/4",
                    "hypothesis://cand/1",
                    "hypothesis://cand/3",
                    "hypothesis://cand/2",
                ],
                "resolved_conflict_ids": [
                    "conflict://hyp/cand/4_cand/1",
                    "conflict://hyp/cand/4_cand/3",
                    "conflict://hyp/cand/4_cand/2",
                ],
                "response_text": "Interpretation based on belief 'how is the sky looking'",
                "retrieved_memory_ids": [
                    "belief://b_sky_clear",
                    "belief://b_sky_hazy",
                    "belief://b_sky_overcast",
                ],
                "thought_dag_node_count": 0,
            },
            "fired_rules": [],
            "generated_hypotheses": [
                "hypothesis://cand/4",
                "hypothesis://cand/1",
                "hypothesis://cand/3",
                "hypothesis://cand/2",
            ],
            "intent": "maintain_conversation",
            "referenced_goal": "goal://maintainconversation",
            "resolved_conflicts": [
                "conflict://hyp/cand/4_cand/1",
                "conflict://hyp/cand/4_cand/3",
                "conflict://hyp/cand/4_cand/2",
            ],
            "retrieved_memories": [
                "belief://b_sky_clear",
                "belief://b_sky_hazy",
                "belief://b_sky_overcast",
            ],
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
            "state_checksum": "c7d05e3aa777ea21466504b8aa3cb8674e76e5fb0eb251739e14a5643eb741ea",
        },
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://ambiguity/007"),
        domain="ambiguity",
        name="AMB-007: Date Synonym Ambiguity",
        description="Probe lexical ambiguity (SYNONYM_AMBIGUITY): the percept "
        "'date' maps to a fruit reading and a meeting reading of differing "
        "confidence; both must remain active per Canon Law 6.",
        initial_beliefs=(
            _belief("belief://b_date_fruit", "the date is a sweet fruit", Decimal("0.7")),
            _belief("belief://b_date_meeting", "the date is set for the meeting", Decimal("0.9")),
        ),
        percept_input="what is the date",
        expected_beliefs=("the date is a sweet fruit", "the date is set for the meeting"),
        expected_action_text="Interpretation based on belief 'what is the date'",
        min_confidence=Decimal("0.9"),
        max_confidence=Decimal("1.0"),
        expected_intent=Intent.MAINTAIN_CONVERSATION,
        failure_mode=FailureMode.SYNONYM_AMBIGUITY,
        metadata={
            "ground_truth_ordering": {
                "the date is set for the meeting": "0.9",
                "the date is a sweet fruit": "0.7",
            }
        },
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://b_date_fruit"),
                SymbolicId.of("belief://b_date_meeting"),
            ),
            generated_hypothesis_ids=(
                SymbolicId.of("hypothesis://cand/3"),
                SymbolicId.of("hypothesis://cand/2"),
                SymbolicId.of("hypothesis://cand/1"),
            ),
            resolved_conflict_ids=(
                SymbolicId.of("conflict://hyp/cand/3_cand/2"),
                SymbolicId.of("conflict://hyp/cand/3_cand/1"),
            ),
            thought_dag_node_count=0,
        ),
        baseline={
            "accepted_hypothesis_id": "hypothesis://cand/3",
            "action_text": "Interpretation based on belief 'what is the date'",
            "activated_concepts": [],
            "captured_at": "v0.4.1 (ADR-0028 Phase 2, pre-provider)",
            "confidence": "1.0000",
            "dag_node_count": 0,
            "decision_id": "decision://select/cand/3",
            "decision_type": "response",
            "derived_beliefs": ["belief://percept/f621c9cc"],
            "fingerprint": {
                "activated_concept_ids": [],
                "decision_id": "decision://select/cand/3",
                "derived_belief_ids": ["belief://percept/f621c9cc"],
                "fired_rule_ids": [],
                "generated_hypothesis_ids": [
                    "hypothesis://cand/3",
                    "hypothesis://cand/2",
                    "hypothesis://cand/1",
                ],
                "resolved_conflict_ids": [
                    "conflict://hyp/cand/3_cand/2",
                    "conflict://hyp/cand/3_cand/1",
                ],
                "response_text": "Interpretation based on belief 'what is the date'",
                "retrieved_memory_ids": ["belief://b_date_fruit", "belief://b_date_meeting"],
                "thought_dag_node_count": 0,
            },
            "fired_rules": [],
            "generated_hypotheses": [
                "hypothesis://cand/3",
                "hypothesis://cand/2",
                "hypothesis://cand/1",
            ],
            "intent": "maintain_conversation",
            "referenced_goal": "goal://maintainconversation",
            "resolved_conflicts": ["conflict://hyp/cand/3_cand/2", "conflict://hyp/cand/3_cand/1"],
            "retrieved_memories": ["belief://b_date_fruit", "belief://b_date_meeting"],
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
            "state_checksum": "760d0291619752487ab176dbdd09369788fc5bbc7690925930664246b27df9a5",
        },
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://ambiguity/008"),
        domain="ambiguity",
        name="AMB-008: False Association Light",
        description="Probe FALSE_ASSOCIATION: percept word 'light' triggers a "
        "low-confidence illumination reading that is a false association with "
        "the weight reading; both stay active and the ordering is declared.",
        initial_beliefs=(
            _belief("belief://b_light_bright", "the light is bright", Decimal("0.6")),
            _belief(
                "belief://b_light_weight",
                "the light package is easy to carry",
                Decimal("0.9"),
            ),
        ),
        percept_input="how heavy is the light",
        expected_beliefs=("the light is bright", "the light package is easy to carry"),
        expected_action_text="Interpretation based on belief 'how heavy is the light'",
        min_confidence=Decimal("0.9"),
        max_confidence=Decimal("1.0"),
        expected_intent=Intent.MAINTAIN_CONVERSATION,
        failure_mode=FailureMode.FALSE_ASSOCIATION,
        metadata={
            "ground_truth_ordering": {
                "the light package is easy to carry": "0.9",
                "the light is bright": "0.6",
            }
        },
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://b_light_bright"),
                SymbolicId.of("belief://b_light_weight"),
            ),
            generated_hypothesis_ids=(
                SymbolicId.of("hypothesis://cand/3"),
                SymbolicId.of("hypothesis://cand/2"),
                SymbolicId.of("hypothesis://cand/1"),
            ),
            resolved_conflict_ids=(
                SymbolicId.of("conflict://hyp/cand/3_cand/2"),
                SymbolicId.of("conflict://hyp/cand/3_cand/1"),
            ),
            thought_dag_node_count=0,
        ),
        baseline={
            "accepted_hypothesis_id": "hypothesis://cand/3",
            "action_text": "Interpretation based on belief 'how heavy is the light'",
            "activated_concepts": [],
            "captured_at": "v0.4.1 (ADR-0028 Phase 2, pre-provider)",
            "confidence": "1.0000",
            "dag_node_count": 0,
            "decision_id": "decision://select/cand/3",
            "decision_type": "response",
            "derived_beliefs": ["belief://percept/c2f97590"],
            "fingerprint": {
                "activated_concept_ids": [],
                "decision_id": "decision://select/cand/3",
                "derived_belief_ids": ["belief://percept/c2f97590"],
                "fired_rule_ids": [],
                "generated_hypothesis_ids": [
                    "hypothesis://cand/3",
                    "hypothesis://cand/2",
                    "hypothesis://cand/1",
                ],
                "resolved_conflict_ids": [
                    "conflict://hyp/cand/3_cand/2",
                    "conflict://hyp/cand/3_cand/1",
                ],
                "response_text": "Interpretation based on belief 'how heavy is the light'",
                "retrieved_memory_ids": ["belief://b_light_bright", "belief://b_light_weight"],
                "thought_dag_node_count": 0,
            },
            "fired_rules": [],
            "generated_hypotheses": [
                "hypothesis://cand/3",
                "hypothesis://cand/2",
                "hypothesis://cand/1",
            ],
            "intent": "maintain_conversation",
            "referenced_goal": "goal://maintainconversation",
            "resolved_conflicts": ["conflict://hyp/cand/3_cand/2", "conflict://hyp/cand/3_cand/1"],
            "retrieved_memories": ["belief://b_light_bright", "belief://b_light_weight"],
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
            "state_checksum": "85d557f45c36f144ab107acd4843fd0596dbfc6fc4623d99ae16a77cab70124c",
        },
    ),
)
