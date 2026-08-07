"""causal_reasoning domain — premise-to-conclusion deduction benchmark cases.

Each case seeds a BeliefGraph (and rules) then verifies the derived belief,
fired rule ids, and golden execution trace produced by the canonical pipeline.
"""

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
        id=BenchmarkId.of("bm://causal_reasoning/001"),
        domain="causal_reasoning",
        name="CAUSAL-001: Rain and Wet Ground",
        description="Verify causal deduction that rain causes wet ground.",
        initial_beliefs=(_belief("belief://c_rain", "raining"),),
        rules=(
            _rule(
                "rule://causal/rain_umbrella",
                "raining",
                "Suggest carrying an umbrella",
                Decimal("0.9"),
            ),
        ),
        percept_input="The sky is raining heavily outside",
        expected_beliefs=("Suggest carrying an umbrella",),
        expected_action_text="Interpretation based on belief 'The sky is raining heavily outside'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(SymbolicId.of("belief://c_rain"),),
            fired_rule_ids=(SymbolicId.of("rule://causal/rain_umbrella"),),
            thought_dag_node_count=1,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://causal_reasoning/002"),
        domain="causal_reasoning",
        name="CAUSAL-002: Smoke Indicates Fire",
        description="Verify causal deduction that smoke implies fire.",
        initial_beliefs=(_belief("belief://c_smoke", "smoke"),),
        rules=(
            _rule(
                "rule://causal/smoke_fire",
                "smoke",
                "Fire is likely nearby",
                Decimal("0.9"),
            ),
        ),
        percept_input="Smoke is rising from the chimney",
        expected_beliefs=("Fire is likely nearby",),
        expected_action_text="Interpretation based on belief 'Smoke is rising from the chimney'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(SymbolicId.of("belief://c_smoke"),),
            fired_rule_ids=(SymbolicId.of("rule://causal/smoke_fire"),),
            thought_dag_node_count=1,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://causal_reasoning/003"),
        domain="causal_reasoning",
        name="CAUSAL-003: Drought and Crop Failure",
        description="Verify causal deduction that drought threatens crops.",
        initial_beliefs=(_belief("belief://c_drought", "drought"),),
        rules=(
            _rule(
                "rule://causal/drought_crop_failure",
                "drought",
                "Crops may fail",
                Decimal("0.85"),
            ),
        ),
        percept_input="There is a severe drought this season",
        expected_beliefs=("Crops may fail",),
        expected_action_text=(
            "Interpretation based on belief 'There is a severe drought this season'"
        ),
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(SymbolicId.of("belief://c_drought"),),
            fired_rule_ids=(SymbolicId.of("rule://causal/drought_crop_failure"),),
            thought_dag_node_count=1,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://causal_reasoning/004"),
        domain="causal_reasoning",
        name="CAUSAL-004: Studying Builds Knowledge",
        description="Verify causal deduction that studying produces knowledge.",
        initial_beliefs=(_belief("belief://c_study", "studying"),),
        rules=(
            _rule(
                "rule://causal/study_knowledge",
                "studying",
                "Knowledge improves with studying",
                Decimal("0.9"),
            ),
        ),
        percept_input="I am studying every evening",
        expected_beliefs=("Knowledge improves with studying",),
        expected_action_text="Interpretation based on belief 'I am studying every evening'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(SymbolicId.of("belief://c_study"),),
            fired_rule_ids=(SymbolicId.of("rule://causal/study_knowledge"),),
            thought_dag_node_count=1,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://causal_reasoning/005"),
        domain="causal_reasoning",
        name="CAUSAL-005: Rain-to-Slippery Causal Chain",
        description="Verify multi-step causal chaining across two sequential rules.",
        initial_beliefs=(_belief("belief://c_rain", "raining"),),
        rules=(
            _rule(
                "rule://causal/rain_wet_ground",
                "rain",
                "the ground is wet",
                Decimal("0.8"),
            ),
            _rule(
                "rule://causal/wet_ground_slippery",
                "ground is wet",
                "the ground is slippery",
                Decimal("0.9"),
            ),
        ),
        percept_input="rain is falling",
        expected_beliefs=("the ground is wet", "the ground is slippery"),
        expected_action_text="Interpretation based on belief 'rain is falling'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(SymbolicId.of("belief://c_rain"),),
            fired_rule_ids=(
                SymbolicId.of("rule://causal/rain_wet_ground"),
                SymbolicId.of("rule://causal/wet_ground_slippery"),
            ),
            thought_dag_node_count=2,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://causal_reasoning/006"),
        domain="causal_reasoning",
        name="CAUSAL-006: Sunrise and Daylight",
        description="Verify causal deduction that sunrise produces daylight.",
        initial_beliefs=(_belief("belief://c_sunrise", "sunrise"),),
        rules=(
            _rule(
                "rule://causal/sunrise_daylight",
                "sunrise",
                "Daylight follows sunrise",
                Decimal("0.9"),
            ),
        ),
        percept_input="The sun rises in the east",
        expected_beliefs=("Daylight follows sunrise",),
        expected_action_text="Interpretation based on belief 'The sun rises in the east'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(SymbolicId.of("belief://c_sunrise"),),
            fired_rule_ids=(SymbolicId.of("rule://causal/sunrise_daylight"),),
            thought_dag_node_count=1,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://causal_reasoning/007"),
        domain="causal_reasoning",
        name="CAUSAL-007: Steam Is Not Smoke",
        description="Adversarial negative control: a steam belief must not fire "
        "a smoke-based causal rule.",
        initial_beliefs=(_belief("belief://c_steam", "steam is rising from the kettle"),),
        rules=(
            _rule(
                "rule://causal/steam_smoke_fire",
                "smoke",
                "Fire is likely nearby",
                Decimal("0.9"),
            ),
        ),
        percept_input="I can see steam in the kitchen",
        expected_beliefs=("steam is rising from the kettle",),
        excluded_beliefs=("Fire is likely nearby",),
        expected_action_text=(
            "Interpretation based on belief 'steam is rising from the kettle'"
        ),
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(SymbolicId.of("belief://c_steam"),),
            fired_rule_ids=(),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://causal_reasoning/008"),
        domain="causal_reasoning",
        name="CAUSAL-008: Low-Confidence Premise Fails to Fire",
        description="Negative control: a premise below the rule's min_confidence "
        "must not fire the causal rule.",
        initial_beliefs=(_belief("belief://c_road", "the road is icy", Decimal("0.4")),),
        rules=(
            InferenceRule(
                id=RuleId.of("rule://causal/icy_road_hazard"),
                name="Driving is hazardous",
                conditions=(
                    RuleCondition(
                        premise_predicate="the road is icy",
                        min_confidence=Decimal("0.8"),
                    ),
                ),
                conclusion_template="Driving is hazardous",
                confidence_multiplier=Decimal("0.9"),
            ),
        ),
        percept_input="check the road conditions",
        expected_beliefs=("the road is icy",),
        excluded_beliefs=("Driving is hazardous",),
        expected_action_text="Interpretation based on belief 'check the road conditions'",
        min_confidence=Decimal("0.0"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(SymbolicId.of("belief://c_road"),),
            fired_rule_ids=(),
            thought_dag_node_count=0,
        ),
    ),
)
