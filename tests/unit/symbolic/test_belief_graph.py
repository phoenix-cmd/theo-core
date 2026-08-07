"""Unit tests for BeliefGraph domain graph wrapper."""

from decimal import Decimal

from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.beliefs.models import (
    Belief,
    BeliefEdge,
    BeliefId,
    BeliefRelation,
)


class TestBeliefGraph:
    def test_add_and_retrieve_belief(self) -> None:
        bg = BeliefGraph()
        bid = BeliefId.of("belief://b1")
        b = Belief(id=bid, proposition="Test proposition")

        bg.add_belief(b)
        assert bg.node_count == 1
        assert bg.has_belief(bid)
        assert bg.get_belief(bid) == b

    def test_active_beliefs_filtering(self) -> None:
        bg = BeliefGraph()
        b1 = Belief(
            id=BeliefId.of("belief://active"),
            proposition="Active",
            confidence=Decimal("0.8"),
        )
        b2 = Belief(
            id=BeliefId.of("belief://deprecated"),
            proposition="Deprecated",
            confidence=Decimal("0.0"),
        )

        bg.add_belief(b1)
        bg.add_belief(b2)

        active = bg.get_active_beliefs(min_confidence=Decimal("0.1"))
        assert len(active) == 1
        assert active[0].id == b1.id

    def test_supporting_and_contradicting_lookups(self) -> None:
        bg = BeliefGraph()
        b1 = BeliefId.of("belief://b1")
        b2 = BeliefId.of("belief://b2")
        b3 = BeliefId.of("belief://b3")

        bg.add_belief(Belief(id=b1, proposition="P1"))
        bg.add_belief(Belief(id=b2, proposition="P2"))
        bg.add_belief(Belief(id=b3, proposition="P3"))

        # b1 supports b2
        bg.add_edge(BeliefEdge(source=b1, target=b2, relation=BeliefRelation.SUPPORTS))
        # b3 contradicts b2
        bg.add_edge(BeliefEdge(source=b3, target=b2, relation=BeliefRelation.CONTRADICTS))

        assert bg.get_supporting_beliefs(b2) == {b1}
        assert bg.get_contradicting_beliefs(b2) == {b3}
