"""Unit tests for Concept System domain models."""

from decimal import Decimal

import pytest

from theo_core.symbolic.concepts.models import (
    Concept,
    ConceptEdge,
    ConceptId,
    ConceptType,
    RelationType,
)


class TestConceptModels:
    def test_concept_id_factory(self) -> None:
        cid = ConceptId.of("concept://animal/dog")
        assert cid.value == "concept://animal/dog"
        assert str(cid) == "concept://animal/dog"

    def test_invalid_concept_id_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="must start with 'concept://'"):
            ConceptId.of("invalid://scheme")

    def test_concept_creation_and_immutability(self) -> None:
        cid = ConceptId.of("concept://animal/dog")
        c = Concept(id=cid, label="Dog", concept_type=ConceptType.ENTITY)

        assert c.id == cid
        assert c.label == "Dog"
        assert c.concept_type == ConceptType.ENTITY

        with pytest.raises((TypeError, Exception)):
            c.label = "Cat"  # type: ignore[misc]

    def test_concept_edge_creation(self) -> None:
        c1 = ConceptId.of("concept://animal/dog")
        c2 = ConceptId.of("concept://animal/canine")
        edge = ConceptEdge(
            source=c1,
            target=c2,
            relation=RelationType.IS_A,
            weight=Decimal("1.0"),
        )

        assert edge.source == c1
        assert edge.target == c2
        assert edge.relation == RelationType.IS_A
        assert edge.weight == Decimal("1.0")
