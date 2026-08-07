"""taxonomy domain — is-a hierarchy and concept-activation benchmark cases."""

from __future__ import annotations

from decimal import Decimal

from theo_core.evaluation.benchmark_schema import BenchmarkCase, BenchmarkId, GoldenTrace
from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.concepts.models import (
    Concept,
    ConceptEdge,
    ConceptId,
    ConceptType,
    RelationType,
)


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
)
