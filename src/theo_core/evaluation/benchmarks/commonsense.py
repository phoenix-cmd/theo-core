"""commonsense domain — everyday reasoning and state-tracking benchmark cases."""

from __future__ import annotations

from decimal import Decimal

from theo_core.evaluation.benchmark_schema import BenchmarkCase, BenchmarkId, GoldenTrace
from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.models import Belief, BeliefId


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
)
