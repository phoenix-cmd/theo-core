"""Unit tests for Constraint Engine models and validation."""

import pytest

from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.beliefs.models import Belief, BeliefId
from theo_core.symbolic.concepts.graph import ConceptGraph
from theo_core.symbolic.constraints.engine import ConstraintEngine
from theo_core.symbolic.constraints.models import ConstraintId
from theo_core.symbolic.thoughts.graph import ThoughtGraph


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
