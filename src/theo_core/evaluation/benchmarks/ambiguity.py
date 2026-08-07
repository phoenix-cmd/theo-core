"""ambiguity domain — competing-interpretation benchmark cases.

Each case seeds multiple beliefs that plausibly interpret the same percept.
Canon Law 6 requires the system to generate competing hypotheses rather than
collapsing to a single interpretation prematurely: the seeded beliefs must all
remain active post-cycle, and the golden trace must record a distinct candidate
hypothesis per matching belief.
"""

from __future__ import annotations

from decimal import Decimal

from theo_core.evaluation.benchmark_schema import BenchmarkCase, BenchmarkId, GoldenTrace
from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.models import Belief, BeliefId


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
)
