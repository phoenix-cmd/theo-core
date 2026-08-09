"""causal_reasoning domain — premise-to-conclusion deduction benchmark cases.

Each case seeds a BeliefGraph (and rules) then verifies the derived belief,
fired rule ids, and golden execution trace produced by the canonical pipeline.
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
        expected_action_text=("Interpretation based on belief 'steam is rising from the kettle'"),
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
    BenchmarkCase(
        id=BenchmarkId.of("bm://causal_reasoning/009"),
        domain="causal_reasoning",
        name="CAUSAL-009: Three-Hop Forecast Chain",
        description="Probe MULTI_HOP: a three-rule causal chain "
        "(rain -> wet ground -> slippery path -> warn about slipping) must "
        "fire in sequence and commit all intermediate derivations.",
        initial_beliefs=(_belief("belief://c_rain", "raining"),),
        rules=(
            _rule(
                "rule://causal/rain_wet",
                "rain",
                "the ground is wet",
                Decimal("0.9"),
            ),
            _rule(
                "rule://causal/wet_slippery",
                "wet",
                "the path is slippery",
                Decimal("0.85"),
            ),
            _rule(
                "rule://causal/slippery_fall",
                "slippery",
                "warn about slipping",
                Decimal("0.8"),
            ),
        ),
        percept_input="what does the forecast mean",
        expected_beliefs=(
            "the ground is wet",
            "the path is slippery",
            "warn about slipping",
        ),
        expected_action_text="Interpretation based on belief 'what does the forecast mean'",
        min_confidence=Decimal("0.9"),
        max_confidence=Decimal("1.0"),
        expected_intent=Intent.MAINTAIN_CONVERSATION,
        failure_mode=FailureMode.MULTI_HOP,
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(SymbolicId.of("belief://c_rain"),),
            fired_rule_ids=(
                SymbolicId.of("rule://causal/rain_wet"),
                SymbolicId.of("rule://causal/slippery_fall"),
                SymbolicId.of("rule://causal/wet_slippery"),
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
            thought_dag_node_count=3,
        ),
        baseline={
            "accepted_hypothesis_id": "hypothesis://cand/3",
            "action_text": "Interpretation based on belief 'what does the forecast mean'",
            "activated_concepts": [],
            "captured_at": "v0.4.1 (ADR-0028 Phase 2, pre-provider)",
            "confidence": "1.0000",
            "dag_node_count": 3,
            "decision_id": "decision://select/cand/3",
            "decision_type": "response",
            "derived_beliefs": [
                "belief://inf/causal/rain_wet/1",
                "belief://inf/causal/slippery_fall/3",
                "belief://inf/causal/wet_slippery/2",
                "belief://percept/a40bbddb",
            ],
            "fingerprint": {
                "activated_concept_ids": [],
                "decision_id": "decision://select/cand/3",
                "derived_belief_ids": [
                    "belief://inf/causal/rain_wet/1",
                    "belief://inf/causal/slippery_fall/3",
                    "belief://inf/causal/wet_slippery/2",
                    "belief://percept/a40bbddb",
                ],
                "fired_rule_ids": [
                    "rule://causal/rain_wet",
                    "rule://causal/slippery_fall",
                    "rule://causal/wet_slippery",
                ],
                "generated_hypothesis_ids": [
                    "hypothesis://cand/3",
                    "hypothesis://cand/1",
                    "hypothesis://cand/2",
                ],
                "resolved_conflict_ids": [
                    "conflict://hyp/cand/3_cand/1",
                    "conflict://hyp/cand/3_cand/2",
                ],
                "response_text": "Interpretation based on belief 'what does the forecast mean'",
                "retrieved_memory_ids": ["belief://c_rain"],
                "thought_dag_node_count": 3,
            },
            "fired_rules": [
                "rule://causal/rain_wet",
                "rule://causal/slippery_fall",
                "rule://causal/wet_slippery",
            ],
            "generated_hypotheses": [
                "hypothesis://cand/3",
                "hypothesis://cand/1",
                "hypothesis://cand/2",
            ],
            "intent": "maintain_conversation",
            "referenced_goal": "goal://maintainconversation",
            "resolved_conflicts": ["conflict://hyp/cand/3_cand/1", "conflict://hyp/cand/3_cand/2"],
            "retrieved_memories": ["belief://c_rain"],
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
            "state_checksum": "a4d682c56b038b251daa5a6574b344f67d5335623005ed3d5aca04dc18d848c4",
        },
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://causal_reasoning/010"),
        domain="causal_reasoning",
        name="CAUSAL-010: Sleet Distractor Rule",
        description="Probe DISTRACTOR_EVIDENCE: a near-synonym distractor rule "
        "(sleet -> travel cancelled) must not fire when only snow is observed; "
        "only the correct causal rule fires.",
        initial_beliefs=(_belief("belief://c_snow", "snow is falling"),),
        rules=(
            _rule(
                "rule://causal/snow_slippery",
                "snow",
                "the roads are slippery",
                Decimal("0.85"),
            ),
            _rule(
                "rule://causal/sleet_cancel",
                "sleet",
                "travel is cancelled",
                Decimal("0.9"),
            ),
        ),
        percept_input="snow is falling",
        expected_beliefs=("the roads are slippery",),
        excluded_beliefs=("travel is cancelled",),
        expected_action_text="Interpretation based on belief 'snow is falling'",
        min_confidence=Decimal("0.9"),
        max_confidence=Decimal("1.0"),
        expected_intent=Intent.MAINTAIN_CONVERSATION,
        failure_mode=FailureMode.DISTRACTOR_EVIDENCE,
        golden_trace=GoldenTrace(
            retrieved_memory_ids=(SymbolicId.of("belief://c_snow"),),
            fired_rule_ids=(SymbolicId.of("rule://causal/snow_slippery"),),
            generated_hypothesis_ids=(
                SymbolicId.of("hypothesis://cand/2"),
                SymbolicId.of("hypothesis://cand/1"),
            ),
            resolved_conflict_ids=(SymbolicId.of("conflict://hyp/cand/2_cand/1"),),
            thought_dag_node_count=1,
        ),
        baseline={
            "accepted_hypothesis_id": "hypothesis://cand/2",
            "action_text": "Interpretation based on belief 'snow is falling'",
            "activated_concepts": [],
            "captured_at": "v0.4.1 (ADR-0028 Phase 2, pre-provider)",
            "confidence": "1.0000",
            "dag_node_count": 1,
            "decision_id": "decision://select/cand/2",
            "decision_type": "response",
            "derived_beliefs": ["belief://inf/causal/snow_slippery/1", "belief://percept/34d58b81"],
            "fingerprint": {
                "activated_concept_ids": [],
                "decision_id": "decision://select/cand/2",
                "derived_belief_ids": [
                    "belief://inf/causal/snow_slippery/1",
                    "belief://percept/34d58b81",
                ],
                "fired_rule_ids": ["rule://causal/snow_slippery"],
                "generated_hypothesis_ids": ["hypothesis://cand/2", "hypothesis://cand/1"],
                "resolved_conflict_ids": ["conflict://hyp/cand/2_cand/1"],
                "response_text": "Interpretation based on belief 'snow is falling'",
                "retrieved_memory_ids": ["belief://c_snow"],
                "thought_dag_node_count": 1,
            },
            "fired_rules": ["rule://causal/snow_slippery"],
            "generated_hypotheses": ["hypothesis://cand/2", "hypothesis://cand/1"],
            "intent": "maintain_conversation",
            "referenced_goal": "goal://maintainconversation",
            "resolved_conflicts": ["conflict://hyp/cand/2_cand/1"],
            "retrieved_memories": ["belief://c_snow"],
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
            "state_checksum": "daec4053b1236364be99012a6306f8da3a46dcc6e977633a97345b5afb42447d",
        },
    ),
)
