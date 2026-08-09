"""taxonomy domain — is-a hierarchy and concept-activation benchmark cases."""

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
from theo_core.symbolic.concepts.models import (
    Concept,
    ConceptEdge,
    ConceptId,
    ConceptType,
    RelationType,
)
from theo_core.symbolic.decisions.models import Intent
from theo_core.symbolic.inference.models import InferenceRule, RuleCondition, RuleId


def _concept(uri: str, label: str) -> Concept:
    """Build a Concept node with the given id and label."""
    return Concept(id=ConceptId.of(uri), label=label, concept_type=ConceptType.ENTITY)


def _is_a(child_uri: str, parent_uri: str) -> ConceptEdge:
    """Build an IS_A hierarchy edge from child to parent."""
    return ConceptEdge(
        source=ConceptId.of(child_uri),
        target=ConceptId.of(parent_uri),
        relation=RelationType.IS_A,
    )


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
        id=BenchmarkId.of("bm://taxonomy/001"),
        domain="taxonomy",
        name="TAXO-001: Mammal Is A Animal",
        description="Verify concept activation over a mammal/animal hierarchy.",
        initial_concepts=(
            _concept("concept://animal", "Animal"),
            _concept("concept://mammal", "Mammal"),
        ),
        initial_concept_edges=(_is_a("concept://mammal", "concept://animal"),),
        percept_input="What is a mammal",
        expected_action_text="Interpretation based on belief 'What is a mammal'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            activated_concept_ids=(
                SymbolicId.of("concept://animal"),
                SymbolicId.of("concept://mammal"),
            ),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://taxonomy/002"),
        domain="taxonomy",
        name="TAXO-002: Car Is A Vehicle",
        description="Verify concept activation over a car/vehicle hierarchy.",
        initial_concepts=(
            _concept("concept://vehicle", "Vehicle"),
            _concept("concept://car", "Car"),
        ),
        initial_concept_edges=(_is_a("concept://car", "concept://vehicle"),),
        percept_input="Tell me about cars",
        expected_action_text="Interpretation based on belief 'Tell me about cars'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            activated_concept_ids=(
                SymbolicId.of("concept://car"),
                SymbolicId.of("concept://vehicle"),
            ),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://taxonomy/003"),
        domain="taxonomy",
        name="TAXO-003: Plant Is A Organism",
        description="Verify concept activation over a plant/organism hierarchy.",
        initial_concepts=(
            _concept("concept://organism", "Organism"),
            _concept("concept://plant", "Plant"),
        ),
        initial_concept_edges=(_is_a("concept://plant", "concept://organism"),),
        percept_input="Describe plants",
        expected_action_text="Interpretation based on belief 'Describe plants'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            activated_concept_ids=(
                SymbolicId.of("concept://organism"),
                SymbolicId.of("concept://plant"),
            ),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://taxonomy/004"),
        domain="taxonomy",
        name="TAXO-004: Fruit Is A Food",
        description="Verify concept activation over a fruit/food hierarchy.",
        initial_concepts=(
            _concept("concept://food", "Food"),
            _concept("concept://fruit", "Fruit"),
        ),
        initial_concept_edges=(_is_a("concept://fruit", "concept://food"),),
        percept_input="Talk about fruit",
        expected_action_text="Interpretation based on belief 'Talk about fruit'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            activated_concept_ids=(
                SymbolicId.of("concept://food"),
                SymbolicId.of("concept://fruit"),
            ),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://taxonomy/005"),
        domain="taxonomy",
        name="TAXO-005: City Is A Place",
        description="Verify concept activation over a city/place hierarchy.",
        initial_concepts=(
            _concept("concept://place", "Place"),
            _concept("concept://city", "City"),
        ),
        initial_concept_edges=(_is_a("concept://city", "concept://place"),),
        percept_input="Explain cities",
        expected_action_text="Interpretation based on belief 'Explain cities'",
        min_confidence=Decimal("0.5"),
        max_confidence=Decimal("1.0"),
        golden_trace=GoldenTrace(
            activated_concept_ids=(
                SymbolicId.of("concept://city"),
                SymbolicId.of("concept://place"),
            ),
            thought_dag_node_count=0,
        ),
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://taxonomy/006"),
        domain="taxonomy",
        name="TAXO-006: Inheritance Property Chain",
        description="Probe MULTI_HOP: a two-step property chain across the "
        "taxonomy (mammal -> warm-blooded -> animals need oxygen) must fire in "
        "sequence from a single seeded fact.",
        initial_concepts=(
            _concept("concept://animal", "Animal"),
            _concept("concept://dog", "Dog"),
            _concept("concept://mammal", "Mammal"),
        ),
        initial_concept_edges=(
            _is_a("concept://dog", "concept://mammal"),
            _is_a("concept://mammal", "concept://animal"),
        ),
        initial_beliefs=(_belief("belief://c_dog", "a dog is a mammal"),),
        rules=(
            _rule(
                "rule://taxonomy/mammal_inherit",
                "mammal",
                "warm-blooded animals regulate heat",
                Decimal("0.95"),
            ),
            _rule(
                "rule://taxonomy/animal_breathe",
                "animal",
                "oxygen is needed to survive",
                Decimal("0.9"),
            ),
        ),
        percept_input="tell me about dogs",
        expected_beliefs=(
            "warm-blooded animals regulate heat",
            "oxygen is needed to survive",
        ),
        expected_action_text="Interpretation based on belief 'tell me about dogs'",
        min_confidence=Decimal("0.9"),
        max_confidence=Decimal("1.0"),
        expected_intent=Intent.ANSWER_QUESTION,
        failure_mode=FailureMode.MULTI_HOP,
        initial_goals=("AnswerQuestion",),
        golden_trace=GoldenTrace(
            activated_concept_ids=(
                SymbolicId.of("concept://animal"),
                SymbolicId.of("concept://dog"),
                SymbolicId.of("concept://mammal"),
            ),
            retrieved_memory_ids=(SymbolicId.of("belief://c_dog"),),
            fired_rule_ids=(
                SymbolicId.of("rule://taxonomy/animal_breathe"),
                SymbolicId.of("rule://taxonomy/mammal_inherit"),
            ),
            generated_hypothesis_ids=(SymbolicId.of("hypothesis://cand/1"),),
            thought_dag_node_count=2,
        ),
        baseline={
            "accepted_hypothesis_id": "hypothesis://cand/1",
            "action_text": "Interpretation based on belief 'tell me about dogs'",
            "activated_concepts": ["concept://animal", "concept://dog", "concept://mammal"],
            "captured_at": "v0.4.1 (ADR-0028 Phase 2, pre-provider)",
            "confidence": "1.0000",
            "dag_node_count": 2,
            "decision_id": "decision://select/cand/1",
            "decision_type": "response",
            "derived_beliefs": [
                "belief://inf/taxonomy/animal_breathe/2",
                "belief://inf/taxonomy/mammal_inherit/1",
                "belief://percept/ff171ec6",
            ],
            "fingerprint": {
                "activated_concept_ids": ["concept://animal", "concept://dog", "concept://mammal"],
                "decision_id": "decision://select/cand/1",
                "derived_belief_ids": [
                    "belief://inf/taxonomy/animal_breathe/2",
                    "belief://inf/taxonomy/mammal_inherit/1",
                    "belief://percept/ff171ec6",
                ],
                "fired_rule_ids": [
                    "rule://taxonomy/animal_breathe",
                    "rule://taxonomy/mammal_inherit",
                ],
                "generated_hypothesis_ids": ["hypothesis://cand/1"],
                "resolved_conflict_ids": [],
                "response_text": "Interpretation based on belief 'tell me about dogs'",
                "retrieved_memory_ids": ["belief://c_dog"],
                "thought_dag_node_count": 2,
            },
            "fired_rules": ["rule://taxonomy/animal_breathe", "rule://taxonomy/mammal_inherit"],
            "generated_hypotheses": ["hypothesis://cand/1"],
            "intent": "answer_question",
            "referenced_goal": "goal://answerquestion",
            "resolved_conflicts": [],
            "retrieved_memories": ["belief://c_dog"],
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
            "state_checksum": "ed6fb8aa8c991ddd35097eaa42a6a0e2260e568a9946939caa883f7e236248e0",
        },
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://taxonomy/007"),
        domain="taxonomy",
        name="TAXO-007: Canine Synonym Ambiguity",
        description="Probe SYNONYM_AMBIGUITY: percept word 'canine' must keep "
        "both the near-synonym and the canonical fact active; the higher-"
        "confidence known fact wins the reading.",
        initial_concepts=(
            _concept("concept://animal", "Animal"),
            _concept("concept://dog", "Dog"),
        ),
        initial_concept_edges=(_is_a("concept://dog", "concept://animal"),),
        initial_beliefs=(
            _belief("belief://c_canine", "a canine is a dog", Decimal("0.9")),
            _belief("belief://c_dog", "a dog is a mammal", Decimal("1.0")),
        ),
        percept_input="is a canine a dog",
        expected_beliefs=("a canine is a dog", "a dog is a mammal"),
        expected_action_text="Interpretation based on belief 'a dog is a mammal'",
        min_confidence=Decimal("0.9"),
        max_confidence=Decimal("1.0"),
        expected_intent=Intent.MAINTAIN_CONVERSATION,
        failure_mode=FailureMode.SYNONYM_AMBIGUITY,
        metadata={
            "ground_truth_ordering": {
                "a dog is a mammal": "1.0",
                "a canine is a dog": "0.9",
            }
        },
        golden_trace=GoldenTrace(
            activated_concept_ids=(
                SymbolicId.of("concept://animal"),
                SymbolicId.of("concept://dog"),
            ),
            retrieved_memory_ids=(
                SymbolicId.of("belief://c_canine"),
                SymbolicId.of("belief://c_dog"),
            ),
            generated_hypothesis_ids=(
                SymbolicId.of("hypothesis://cand/2"),
                SymbolicId.of("hypothesis://cand/3"),
                SymbolicId.of("hypothesis://cand/1"),
            ),
            resolved_conflict_ids=(
                SymbolicId.of("conflict://hyp/cand/2_cand/3"),
                SymbolicId.of("conflict://hyp/cand/2_cand/1"),
            ),
            thought_dag_node_count=0,
        ),
        baseline={
            "accepted_hypothesis_id": "hypothesis://cand/2",
            "action_text": "Interpretation based on belief 'a dog is a mammal'",
            "activated_concepts": ["concept://animal", "concept://dog"],
            "captured_at": "v0.4.1 (ADR-0028 Phase 2, pre-provider)",
            "confidence": "1.0000",
            "dag_node_count": 0,
            "decision_id": "decision://select/cand/2",
            "decision_type": "response",
            "derived_beliefs": ["belief://percept/c88451cd"],
            "fingerprint": {
                "activated_concept_ids": ["concept://animal", "concept://dog"],
                "decision_id": "decision://select/cand/2",
                "derived_belief_ids": ["belief://percept/c88451cd"],
                "fired_rule_ids": [],
                "generated_hypothesis_ids": [
                    "hypothesis://cand/2",
                    "hypothesis://cand/3",
                    "hypothesis://cand/1",
                ],
                "resolved_conflict_ids": [
                    "conflict://hyp/cand/2_cand/3",
                    "conflict://hyp/cand/2_cand/1",
                ],
                "response_text": "Interpretation based on belief 'a dog is a mammal'",
                "retrieved_memory_ids": ["belief://c_canine", "belief://c_dog"],
                "thought_dag_node_count": 0,
            },
            "fired_rules": [],
            "generated_hypotheses": [
                "hypothesis://cand/2",
                "hypothesis://cand/3",
                "hypothesis://cand/1",
            ],
            "intent": "maintain_conversation",
            "referenced_goal": "goal://maintainconversation",
            "resolved_conflicts": ["conflict://hyp/cand/2_cand/3", "conflict://hyp/cand/2_cand/1"],
            "retrieved_memories": ["belief://c_canine", "belief://c_dog"],
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
            "state_checksum": "d89e4787f09ca92a3626e6652ed1d8d107e2dd98fcf830d65c2ea5a42ded5db2",
        },
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://taxonomy/008"),
        domain="taxonomy",
        name="TAXO-008: Three-Level Property Spread",
        description="Probe MULTI_HOP: properties spread across a three-level "
        "taxonomy (dog -> bark, mammal -> nurse, animal -> oxygen) firing in "
        "sequence and committing all three derivations.",
        initial_concepts=(
            _concept("concept://animal", "Animal"),
            _concept("concept://dog", "Dog"),
            _concept("concept://mammal", "Mammal"),
        ),
        initial_concept_edges=(
            _is_a("concept://dog", "concept://mammal"),
            _is_a("concept://mammal", "concept://animal"),
        ),
        initial_beliefs=(_belief("belief://c_dog", "a dog is a mammal"),),
        rules=(
            _rule(
                "rule://taxonomy/dog_property",
                "dog",
                "canines bark loudly",
                Decimal("0.9"),
            ),
            _rule(
                "rule://taxonomy/mammal_property",
                "mammal",
                "warm-blooded animals nurse offspring",
                Decimal("0.85"),
            ),
            _rule(
                "rule://taxonomy/animal_property",
                "animal",
                "living creatures need oxygen",
                Decimal("0.8"),
            ),
        ),
        percept_input="the family pet is a dog",
        expected_beliefs=(
            "canines bark loudly",
            "warm-blooded animals nurse offspring",
            "living creatures need oxygen",
        ),
        expected_action_text="Interpretation based on belief 'the family pet is a dog'",
        min_confidence=Decimal("0.9"),
        max_confidence=Decimal("1.0"),
        expected_intent=Intent.MAINTAIN_CONVERSATION,
        failure_mode=FailureMode.MULTI_HOP,
        golden_trace=GoldenTrace(
            activated_concept_ids=(
                SymbolicId.of("concept://animal"),
                SymbolicId.of("concept://dog"),
                SymbolicId.of("concept://mammal"),
            ),
            retrieved_memory_ids=(SymbolicId.of("belief://c_dog"),),
            fired_rule_ids=(
                SymbolicId.of("rule://taxonomy/animal_property"),
                SymbolicId.of("rule://taxonomy/dog_property"),
                SymbolicId.of("rule://taxonomy/mammal_property"),
            ),
            generated_hypothesis_ids=(
                SymbolicId.of("hypothesis://cand/5"),
                SymbolicId.of("hypothesis://cand/1"),
                SymbolicId.of("hypothesis://cand/3"),
                SymbolicId.of("hypothesis://cand/4"),
                SymbolicId.of("hypothesis://cand/2"),
            ),
            resolved_conflict_ids=(
                SymbolicId.of("conflict://hyp/cand/5_cand/1"),
                SymbolicId.of("conflict://hyp/cand/5_cand/3"),
                SymbolicId.of("conflict://hyp/cand/5_cand/4"),
                SymbolicId.of("conflict://hyp/cand/5_cand/2"),
            ),
            thought_dag_node_count=3,
        ),
        baseline={
            "accepted_hypothesis_id": "hypothesis://cand/5",
            "action_text": "Interpretation based on belief 'the family pet is a dog'",
            "activated_concepts": ["concept://animal", "concept://dog", "concept://mammal"],
            "captured_at": "v0.4.1 (ADR-0028 Phase 2, pre-provider)",
            "confidence": "1.0000",
            "dag_node_count": 3,
            "decision_id": "decision://select/cand/5",
            "decision_type": "response",
            "derived_beliefs": [
                "belief://inf/taxonomy/animal_property/3",
                "belief://inf/taxonomy/dog_property/1",
                "belief://inf/taxonomy/mammal_property/2",
                "belief://percept/31146037",
            ],
            "fingerprint": {
                "activated_concept_ids": ["concept://animal", "concept://dog", "concept://mammal"],
                "decision_id": "decision://select/cand/5",
                "derived_belief_ids": [
                    "belief://inf/taxonomy/animal_property/3",
                    "belief://inf/taxonomy/dog_property/1",
                    "belief://inf/taxonomy/mammal_property/2",
                    "belief://percept/31146037",
                ],
                "fired_rule_ids": [
                    "rule://taxonomy/animal_property",
                    "rule://taxonomy/dog_property",
                    "rule://taxonomy/mammal_property",
                ],
                "generated_hypothesis_ids": [
                    "hypothesis://cand/5",
                    "hypothesis://cand/1",
                    "hypothesis://cand/3",
                    "hypothesis://cand/4",
                    "hypothesis://cand/2",
                ],
                "resolved_conflict_ids": [
                    "conflict://hyp/cand/5_cand/1",
                    "conflict://hyp/cand/5_cand/3",
                    "conflict://hyp/cand/5_cand/4",
                    "conflict://hyp/cand/5_cand/2",
                ],
                "response_text": "Interpretation based on belief 'the family pet is a dog'",
                "retrieved_memory_ids": ["belief://c_dog"],
                "thought_dag_node_count": 3,
            },
            "fired_rules": [
                "rule://taxonomy/animal_property",
                "rule://taxonomy/dog_property",
                "rule://taxonomy/mammal_property",
            ],
            "generated_hypotheses": [
                "hypothesis://cand/5",
                "hypothesis://cand/1",
                "hypothesis://cand/3",
                "hypothesis://cand/4",
                "hypothesis://cand/2",
            ],
            "intent": "maintain_conversation",
            "referenced_goal": "goal://maintainconversation",
            "resolved_conflicts": [
                "conflict://hyp/cand/5_cand/1",
                "conflict://hyp/cand/5_cand/3",
                "conflict://hyp/cand/5_cand/4",
                "conflict://hyp/cand/5_cand/2",
            ],
            "retrieved_memories": ["belief://c_dog"],
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
            "state_checksum": "0206f327c2e8144de833694f38192e3e86f4bdcb56180679316b4a5b57a0ae38",
        },
    ),
    BenchmarkCase(
        id=BenchmarkId.of("bm://taxonomy/009"),
        domain="taxonomy",
        name="TAXO-009: Sparse Knowledge Abstention",
        description="Probe SPARSE_KNOWLEDGE: a percept referencing an entity "
        "with zero supporting knowledge must not fabricate inference; the "
        "absence of an abstention signal is the measured weakness.",
        initial_concepts=(
            _concept("concept://mammal", "Mammal"),
            _concept("concept://platypus", "Platypus"),
        ),
        initial_concept_edges=(_is_a("concept://platypus", "concept://mammal"),),
        percept_input="what is a platypus",
        expected_action_text="Interpretation based on belief 'what is a platypus'",
        min_confidence=Decimal("0.9"),
        max_confidence=Decimal("1.0"),
        expected_intent=Intent.MAINTAIN_CONVERSATION,
        failure_mode=FailureMode.SPARSE_KNOWLEDGE,
        metadata={
            "expected_property": (
                "no rules fired, no derived beliefs, maximal confidence with zero knowledge"
            ),
        },
        golden_trace=GoldenTrace(
            activated_concept_ids=(
                SymbolicId.of("concept://mammal"),
                SymbolicId.of("concept://platypus"),
            ),
            retrieved_memory_ids=(),
            fired_rule_ids=(),
            generated_hypothesis_ids=(SymbolicId.of("hypothesis://cand/1"),),
            thought_dag_node_count=0,
        ),
        baseline={
            "accepted_hypothesis_id": "hypothesis://cand/1",
            "action_text": "Interpretation based on belief 'what is a platypus'",
            "activated_concepts": ["concept://mammal", "concept://platypus"],
            "captured_at": "v0.4.1 (ADR-0028 Phase 2, pre-provider)",
            "confidence": "1.0000",
            "dag_node_count": 0,
            "decision_id": "decision://select/cand/1",
            "decision_type": "response",
            "derived_beliefs": ["belief://percept/90dbf604"],
            "fingerprint": {
                "activated_concept_ids": ["concept://mammal", "concept://platypus"],
                "decision_id": "decision://select/cand/1",
                "derived_belief_ids": ["belief://percept/90dbf604"],
                "fired_rule_ids": [],
                "generated_hypothesis_ids": ["hypothesis://cand/1"],
                "resolved_conflict_ids": [],
                "response_text": "Interpretation based on belief 'what is a platypus'",
                "retrieved_memory_ids": [],
                "thought_dag_node_count": 0,
            },
            "fired_rules": [],
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
            "state_checksum": "eda10e80e2c043297222f261452f4e82cfc2add5bd950db282cd57f8e6d463de",
        },
    ),
)
