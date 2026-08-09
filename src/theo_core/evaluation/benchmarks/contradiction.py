"""contradiction domain — epistemic conflict resolution benchmark cases.

Each case seeds two mutually-contradicting beliefs (linked by a CONTRADICTS
edge); the pipeline's Revision stage deprecates the lower-confidence loser.
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
from theo_core.symbolic.beliefs.models import Belief, BeliefEdge, BeliefId, BeliefRelation
from theo_core.symbolic.decisions.models import Intent


def _belief(uri: str, proposition: str, confidence: Decimal) -> Belief:
    """Build a Belief with the given id, proposition and confidence."""
    return Belief(id=BeliefId.of(uri), proposition=proposition, confidence=confidence)


def _contradicts(source_uri: str, target_uri: str) -> BeliefEdge:
    """Build a CONTRADICTS edge between two belief ids."""
    return BeliefEdge(
        source=BeliefId.of(source_uri),
        target=BeliefId.of(target_uri),
        relation=BeliefRelation.CONTRADICTS,
    )


CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        id=BenchmarkId.of("bm://contradiction/001"),
        domain="contradiction",
        name="CONT-001: Sky Colour Conflict",
        description="Verify that the higher-confidence sky belief survives a contradiction.",
        initial_beliefs=(
            _belief("belief://b_sky_blue", "the sky is blue", Decimal("0.9")),
            _belief("belief://b_sky_grey", "the sky is grey", Decimal("0.6")),
        ),
        initial_belief_edges=(_contradicts("belief://b_sky_blue", "belief://b_sky_grey"),),
        percept_input="Look at the sky",
        expected_beliefs=("the sky is blue",),
        excluded_beliefs=("the sky is grey",),
        expected_action_text="Interpretation based on belief 'Look at the sky'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://b_sky_blue"),
                SymbolicId.of("belief://b_sky_grey"),
            ),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://contradiction/002"),
        domain="contradiction",
        name="CONT-002: Door State Conflict",
        description="Verify that the higher-confidence door belief survives a contradiction.",
        initial_beliefs=(
            _belief("belief://b_door_open", "the door is open", Decimal("0.8")),
            _belief("belief://b_door_closed", "the door is closed", Decimal("0.5")),
        ),
        initial_belief_edges=(_contradicts("belief://b_door_open", "belief://b_door_closed"),),
        percept_input="Check the door",
        expected_beliefs=("the door is open",),
        excluded_beliefs=("the door is closed",),
        expected_action_text="Interpretation based on belief 'Check the door'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://b_door_closed"),
                SymbolicId.of("belief://b_door_open"),
            ),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://contradiction/003"),
        domain="contradiction",
        name="CONT-003: Day and Night Conflict",
        description="Verify that the higher-confidence time-of-day belief "
        "survives a contradiction.",
        initial_beliefs=(
            _belief("belief://b_day", "it is day", Decimal("0.95")),
            _belief("belief://b_night", "it is night", Decimal("0.4")),
        ),
        initial_belief_edges=(_contradicts("belief://b_day", "belief://b_night"),),
        percept_input="What time is it",
        expected_beliefs=("it is day",),
        excluded_beliefs=("it is night",),
        expected_action_text="Interpretation based on belief 'What time is it'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://b_day"),
                SymbolicId.of("belief://b_night"),
            ),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://contradiction/004"),
        domain="contradiction",
        name="CONT-004: Market Trend Conflict",
        description="Verify that the higher-confidence market trend belief "
        "survives a contradiction.",
        initial_beliefs=(
            _belief("belief://b_stock_rising", "the stock is rising", Decimal("0.7")),
            _belief("belief://b_stock_falling", "the stock is falling", Decimal("0.6")),
        ),
        initial_belief_edges=(_contradicts("belief://b_stock_rising", "belief://b_stock_falling"),),
        percept_input="Check the market",
        expected_beliefs=("the stock is rising",),
        excluded_beliefs=("the stock is falling",),
        expected_action_text="Interpretation based on belief 'Check the market'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://b_stock_falling"),
                SymbolicId.of("belief://b_stock_rising"),
            ),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://contradiction/005"),
        domain="contradiction",
        name="CONT-005: Engine Temperature Conflict",
        description="Verify that the higher-confidence engine belief survives a contradiction.",
        initial_beliefs=(
            _belief("belief://b_engine_hot", "the engine is hot", Decimal("0.9")),
            _belief("belief://b_engine_cool", "the engine is cool", Decimal("0.3")),
        ),
        initial_belief_edges=(_contradicts("belief://b_engine_hot", "belief://b_engine_cool"),),
        percept_input="Inspect the engine",
        expected_beliefs=("the engine is hot",),
        excluded_beliefs=("the engine is cool",),
        expected_action_text="Interpretation based on belief 'Inspect the engine'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://b_engine_cool"),
                SymbolicId.of("belief://b_engine_hot"),
            ),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://contradiction/006"),
        domain="contradiction",
        name="CONT-006: Light State Equal-Confidence Tie",
        description="Verify that equal-confidence contradictions tie-break "
        "deterministically on belief id (Canon Invariant 8).",
        initial_beliefs=(
            _belief("belief://b_light_off", "the light is off", Decimal("0.7")),
            _belief("belief://b_light_on", "the light is on", Decimal("0.7")),
        ),
        initial_belief_edges=(_contradicts("belief://b_light_off", "belief://b_light_on"),),
        percept_input="check the light",
        expected_beliefs=("the light is off",),
        excluded_beliefs=("the light is on",),
        expected_action_text="Interpretation based on belief 'check the light'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://b_light_off"),
                SymbolicId.of("belief://b_light_on"),
            ),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://contradiction/007"),
        domain="contradiction",
        name="CONT-007: Low-Confidence Signal Conflict",
        description="Verify that even low-confidence conflicts are resolved "
        "deterministically: the stronger signal belief survives.",
        initial_beliefs=(
            _belief("belief://b_signal_strong", "the signal is strong", Decimal("0.4")),
            _belief("belief://b_signal_weak", "the signal is weak", Decimal("0.3")),
        ),
        initial_belief_edges=(_contradicts("belief://b_signal_strong", "belief://b_signal_weak"),),
        percept_input="check the signal",
        expected_beliefs=("the signal is strong",),
        excluded_beliefs=("the signal is weak",),
        expected_action_text="Interpretation based on belief 'check the signal'",
        min_confidence=Decimal("0.0"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://b_signal_strong"),
                SymbolicId.of("belief://b_signal_weak"),
            ),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://contradiction/008"),
        domain="contradiction",
        name="CONT-008: Graded Market Conflict",
        description="Probe CALIBRATION: a wide confidence gap (rising 0.9 vs "
        "falling 0.3) is resolved to the stronger reading; the case records the "
        "declared correct ordering and residual confidence.",
        initial_beliefs=(
            _belief("belief://b_market_falling", "the market is falling", Decimal("0.3")),
            _belief("belief://b_market_rising", "the market is rising", Decimal("0.9")),
        ),
        initial_belief_edges=(
            _contradicts("belief://b_market_rising", "belief://b_market_falling"),
        ),
        percept_input="check the market",
        expected_beliefs=("the market is rising",),
        excluded_beliefs=("the market is falling",),
        expected_action_text="Interpretation based on belief 'check the market'",
        min_confidence=Decimal("0.9"),
        max_confidence=Decimal("1.0"),
        expected_intent=Intent.MAINTAIN_CONVERSATION,
        failure_mode=FailureMode.CALIBRATION,
        metadata={
            "ground_truth_ordering": {
                "the market is rising": "0.9",
                "the market is falling": "0.3",
            }
        },
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://b_market_falling"),
                SymbolicId.of("belief://b_market_rising"),
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
            "action_text": "Interpretation based on belief 'check the market'",
            "activated_concepts": [],
            "captured_at": "v0.4.1 (ADR-0028 Phase 2, pre-provider)",
            "confidence": "1.0000",
            "dag_node_count": 0,
            "decision_id": "decision://select/cand/2",
            "decision_type": "response",
            "derived_beliefs": ["belief://percept/3321b075"],
            "fingerprint": {
                "activated_concept_ids": [],
                "decision_id": "decision://select/cand/2",
                "derived_belief_ids": ["belief://percept/3321b075"],
                "fired_rule_ids": [],
                "generated_hypothesis_ids": ["hypothesis://cand/2", "hypothesis://cand/1"],
                "resolved_conflict_ids": ["conflict://hyp/cand/2_cand/1"],
                "response_text": "Interpretation based on belief 'check the market'",
                "retrieved_memory_ids": ["belief://b_market_falling", "belief://b_market_rising"],
                "thought_dag_node_count": 0,
            },
            "fired_rules": [],
            "generated_hypotheses": ["hypothesis://cand/2", "hypothesis://cand/1"],
            "intent": "maintain_conversation",
            "referenced_goal": "goal://maintainconversation",
            "resolved_conflicts": ["conflict://hyp/cand/2_cand/1"],
            "retrieved_memories": ["belief://b_market_falling", "belief://b_market_rising"],
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
            "state_checksum": "3e828ab966ec98609a1a64b86e1e39500c6d21e38fa60c23ecdfa595b817c1e6",
        },
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://contradiction/009"),
        domain="contradiction",
        name="CONT-009: Low-Absolute-Confidence Conflict",
        description="Probe CALIBRATION (weakest rung): even after resolution "
        "the winner carries only 0.45 absolute confidence, documenting residual "
        "uncertainty the percept cannot mask.",
        initial_beliefs=(
            _belief("belief://b_signal_gate_low", "the gate signal is low", Decimal("0.35")),
            _belief("belief://b_signal_gate_high", "the gate signal is high", Decimal("0.45")),
        ),
        initial_belief_edges=(
            _contradicts("belief://b_signal_gate_high", "belief://b_signal_gate_low"),
        ),
        percept_input="read the gate signal",
        expected_beliefs=("the gate signal is high",),
        excluded_beliefs=("the gate signal is low",),
        expected_action_text="Interpretation based on belief 'read the gate signal'",
        min_confidence=Decimal("0.9"),
        max_confidence=Decimal("1.0"),
        expected_intent=Intent.MAINTAIN_CONVERSATION,
        failure_mode=FailureMode.CALIBRATION,
        metadata={
            "ground_truth_ordering": {
                "the gate signal is high": "0.45",
                "the gate signal is low": "0.35",
            },
            "residual_uncertainty": (
                "winner confidence 0.45 remains below the percept self-confidence"
            ),
        },
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(
                SymbolicId.of("belief://b_signal_gate_high"),
                SymbolicId.of("belief://b_signal_gate_low"),
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
            "action_text": "Interpretation based on belief 'read the gate signal'",
            "activated_concepts": [],
            "captured_at": "v0.4.1 (ADR-0028 Phase 2, pre-provider)",
            "confidence": "1.0000",
            "dag_node_count": 0,
            "decision_id": "decision://select/cand/2",
            "decision_type": "response",
            "derived_beliefs": ["belief://percept/cd3281fd"],
            "fingerprint": {
                "activated_concept_ids": [],
                "decision_id": "decision://select/cand/2",
                "derived_belief_ids": ["belief://percept/cd3281fd"],
                "fired_rule_ids": [],
                "generated_hypothesis_ids": ["hypothesis://cand/2", "hypothesis://cand/1"],
                "resolved_conflict_ids": ["conflict://hyp/cand/2_cand/1"],
                "response_text": "Interpretation based on belief 'read the gate signal'",
                "retrieved_memory_ids": [
                    "belief://b_signal_gate_high",
                    "belief://b_signal_gate_low",
                ],
                "thought_dag_node_count": 0,
            },
            "fired_rules": [],
            "generated_hypotheses": ["hypothesis://cand/2", "hypothesis://cand/1"],
            "intent": "maintain_conversation",
            "referenced_goal": "goal://maintainconversation",
            "resolved_conflicts": ["conflict://hyp/cand/2_cand/1"],
            "retrieved_memories": ["belief://b_signal_gate_high", "belief://b_signal_gate_low"],
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
            "state_checksum": "cb857f4cf528e449d5b57e7df0dcde9aec626ec7dcc408199f40aafd751dbc37",
        },
    ),
)
