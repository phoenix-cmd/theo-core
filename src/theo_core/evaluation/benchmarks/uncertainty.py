"""uncertainty domain — probabilistic confidence and propagation benchmark cases."""

from __future__ import annotations

from decimal import Decimal

from theo_core.evaluation.benchmark_schema import BenchmarkCase, BenchmarkId, GoldenTrace
from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.models import Belief, BeliefId
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
)
