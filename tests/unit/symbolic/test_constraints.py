"""Unit tests for Constraint Engine models and validation."""

import pytest

from theo_core.symbolic._primitives.errors import ConstraintViolationError
from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.beliefs.models import Belief, BeliefId
from theo_core.symbolic.concepts.graph import ConceptGraph
from theo_core.symbolic.constraints.engine import ConstraintEngine
from theo_core.symbolic.constraints.models import (
    ConstraintId,
    ConstraintRule,
    ConstraintSeverity,
)
from theo_core.symbolic.thoughts.graph import ThoughtGraph
from theo_core.symbolic.thoughts.models import Thought, ThoughtId


class TestConstraints:
    def test_constraint_id_factory(self) -> None:
        cid = ConstraintId.of("constraint://c1")
        assert cid.value == "constraint://c1"

    def test_invalid_constraint_id_raises(self) -> None:
        with pytest.raises(ValueError, match="must start with 'constraint://'"):
            ConstraintId.of("invalid://scheme")

    def test_constraint_engine_validates_system_graphs(self) -> None:
        cg = ConceptGraph()
        bg = BeliefGraph()
        tg = ThoughtGraph()

        # Add valid belief with source
        bg.add_belief(Belief(id=BeliefId.of("belief://b1"), proposition="Valid belief"))

        violations = ConstraintEngine.validate_all(cg, bg, tg)
        assert len(violations) == 0

    def test_unknown_custom_predicate_fails_loudly_as_advisory(self) -> None:
        rule = ConstraintRule(
            id=ConstraintId.of("constraint://custom/typo"),
            name="Typo rule",
            description="A rule with an unrecognized predicate.",
            metadata={"predicate": "bogus_predicate"},
        )
        violations = ConstraintEngine.validate_all(
            ConceptGraph(), BeliefGraph(), ThoughtGraph(), custom_rules=[rule]
        )

        assert len(violations) == 1
        assert violations[0].severity == ConstraintSeverity.ADVISORY
        assert "Unknown custom predicate 'bogus_predicate'" in violations[0].reason

    def test_custom_no_cycle_rule_detects_cycle(self) -> None:
        tg = ThoughtGraph()
        tg.add_thought(Thought(id=ThoughtId.of("thought://n1"), content="Node 1"))
        tg.add_thought(Thought(id=ThoughtId.of("thought://n2"), content="Node 2"))
        # Inject a cycle through the raw graph to bypass the DAG guard.
        from theo_core.symbolic._graph.types import EdgeKey
        from theo_core.symbolic.thoughts.models import ThoughtEdge, ThoughtRelation

        n1 = ThoughtId.of("thought://n1").to_symbolic_id()
        n2 = ThoughtId.of("thought://n2").to_symbolic_id()
        tg.raw_graph.add_edge(
            EdgeKey(source=n1, target=n2, relation=ThoughtRelation.DERIVED_FROM.value),
            ThoughtEdge(source=ThoughtId.of("thought://n1"), target=ThoughtId.of("thought://n2")),
        )
        tg.raw_graph.add_edge(
            EdgeKey(source=n2, target=n1, relation=ThoughtRelation.DERIVED_FROM.value),
            ThoughtEdge(source=ThoughtId.of("thought://n2"), target=ThoughtId.of("thought://n1")),
        )

        rule = ConstraintRule(
            id=ConstraintId.of("constraint://custom/no_cycles"),
            name="No cycles",
            description="ThoughtGraph must remain acyclic.",
            metadata={"predicate": "no_thought_graph_cycles"},
        )
        violations = ConstraintEngine.validate_all(
            ConceptGraph(), BeliefGraph(), tg, custom_rules=[rule]
        )

        assert any(v.constraint_id == rule.id for v in violations)

    def test_assert_valid_raises_on_fatal_violation(self) -> None:
        tg = ThoughtGraph()
        tg.add_thought(Thought(id=ThoughtId.of("thought://n1"), content="Node 1"))
        tg.add_thought(Thought(id=ThoughtId.of("thought://n2"), content="Node 2"))
        from theo_core.symbolic._graph.types import EdgeKey
        from theo_core.symbolic.thoughts.models import ThoughtEdge, ThoughtRelation

        n1 = ThoughtId.of("thought://n1").to_symbolic_id()
        n2 = ThoughtId.of("thought://n2").to_symbolic_id()
        tg.raw_graph.add_edge(
            EdgeKey(source=n1, target=n2, relation=ThoughtRelation.DERIVED_FROM.value),
            ThoughtEdge(source=ThoughtId.of("thought://n1"), target=ThoughtId.of("thought://n2")),
        )
        tg.raw_graph.add_edge(
            EdgeKey(source=n2, target=n1, relation=ThoughtRelation.DERIVED_FROM.value),
            ThoughtEdge(source=ThoughtId.of("thought://n2"), target=ThoughtId.of("thought://n1")),
        )

        with pytest.raises(ConstraintViolationError):
            ConstraintEngine.assert_valid(ConceptGraph(), BeliefGraph(), tg)
