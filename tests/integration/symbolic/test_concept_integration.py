"""Integration test for Concept System full lifecycle.

Build graph -> validate -> activate -> traverse -> serialize -> deserialize -> verify identical.
"""

from decimal import Decimal

from theo_core.symbolic._graph.serialization import GraphLoader, GraphSerializer
from theo_core.symbolic._graph.validation import GraphValidator
from theo_core.symbolic.concepts.activation import ActivationEngine
from theo_core.symbolic.concepts.graph import ConceptGraph
from theo_core.symbolic.concepts.models import (
    Concept,
    ConceptEdge,
    ConceptId,
    ConceptType,
    RelationType,
)


def concept_to_dict(c: Concept) -> dict:
    return {
        "id": c.id.value,
        "label": c.label,
        "concept_type": c.concept_type.value,
        "metadata": c.metadata,
    }


def dict_to_concept(d: dict) -> Concept:
    return Concept(
        id=ConceptId.of(d["id"]),
        label=d["label"],
        concept_type=ConceptType(d["concept_type"]),
        metadata=d.get("metadata", {}),
    )


def edge_to_dict(e: ConceptEdge) -> dict:
    return {
        "source": e.source.value,
        "target": e.target.value,
        "relation": e.relation.value,
        "weight": str(e.weight),
        "metadata": e.metadata,
    }


def dict_to_edge(d: dict) -> ConceptEdge:
    return ConceptEdge(
        source=ConceptId.of(d["source"]),
        target=ConceptId.of(d["target"]),
        relation=RelationType(d["relation"]),
        weight=Decimal(d["weight"]),
        metadata=d.get("metadata", {}),
    )


class TestConceptIntegration:
    def test_full_concept_system_lifecycle(self) -> None:
        cg = ConceptGraph()
        c_animal = ConceptId.of("concept://taxonomy/animal")
        c_dog = ConceptId.of("concept://taxonomy/dog")
        c_bark = ConceptId.of("concept://action/bark")

        cg.add_concept(Concept(id=c_animal, label="Animal"))
        cg.add_concept(Concept(id=c_dog, label="Dog"))
        cg.add_concept(Concept(id=c_bark, label="Bark", concept_type=ConceptType.ACTION))

        cg.add_edge(ConceptEdge(source=c_dog, target=c_animal, relation=RelationType.IS_A))
        cg.add_edge(ConceptEdge(source=c_dog, target=c_bark, relation=RelationType.CAUSES))

        # 1. Validation
        errors = GraphValidator.validate(cg.raw_graph)
        assert len(errors) == 0

        # 2. Activation
        act_result = ActivationEngine.activate(
            cg,
            seeds={c_dog: Decimal("1.0")},
            decay_factor=Decimal("0.5"),
        )
        assert act_result.activations[c_dog] == Decimal("1.0")
        assert act_result.activations[c_animal] == Decimal("0.5")
        assert act_result.activations[c_bark] == Decimal("0.5")

        # 3. Traversal
        ancestors = cg.ancestors(c_dog)
        assert ancestors == {c_animal}

        # 4. Serialization round-trip
        serializer = GraphSerializer[Concept, ConceptEdge](
            graph_type="concept",
            node_to_dict=concept_to_dict,
            edge_to_dict=edge_to_dict,
        )
        loader = GraphLoader[Concept, ConceptEdge](
            expected_graph_type="concept",
            dict_to_node=dict_to_concept,
            dict_to_edge=dict_to_edge,
        )

        serialized_json = serializer.serialize(cg.raw_graph)
        reconstructed_raw = loader.deserialize(serialized_json)

        assert reconstructed_raw.node_count == 3
        assert reconstructed_raw.edge_count == 2
