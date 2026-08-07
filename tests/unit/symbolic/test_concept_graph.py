"""Unit tests for ConceptGraph domain graph wrapper."""

from theo_core.symbolic.concepts.graph import ConceptGraph
from theo_core.symbolic.concepts.models import (
    Concept,
    ConceptEdge,
    ConceptId,
    ConceptType,
    RelationType,
)


class TestConceptGraph:
    def test_add_and_retrieve_concept(self) -> None:
        cg = ConceptGraph()
        cid = ConceptId.of("concept://animal/dog")
        c = Concept(id=cid, label="Dog", concept_type=ConceptType.ENTITY)

        cg.add_concept(c)
        assert cg.node_count == 1
        assert cg.has_concept(cid)
        assert cg.get_concept(cid) == c

    def test_add_edge_and_taxonomy_lookup(self) -> None:
        cg = ConceptGraph()
        dog_id = ConceptId.of("concept://animal/dog")
        canine_id = ConceptId.of("concept://animal/canine")
        mammal_id = ConceptId.of("concept://animal/mammal")

        cg.add_concept(Concept(id=dog_id, label="Dog"))
        cg.add_concept(Concept(id=canine_id, label="Canine"))
        cg.add_concept(Concept(id=mammal_id, label="Mammal"))

        # Dog -> IS_A -> Canine -> IS_A -> Mammal
        cg.add_edge(ConceptEdge(source=dog_id, target=canine_id, relation=RelationType.IS_A))
        cg.add_edge(ConceptEdge(source=canine_id, target=mammal_id, relation=RelationType.IS_A))

        ancestors = cg.ancestors(dog_id)
        assert ancestors == {canine_id, mammal_id}

        descendants = cg.descendants(mammal_id)
        assert descendants == {dog_id, canine_id}
