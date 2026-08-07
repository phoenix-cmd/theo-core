"""Unit tests for Belief System domain models."""

from decimal import Decimal

import pytest

from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.models import (
    Belief,
    BeliefEdge,
    BeliefId,
    BeliefRelation,
    BeliefSource,
    EvidenceTrace,
)


class TestBeliefModels:
    def test_belief_id_factory(self) -> None:
        bid = BeliefId.of("belief://sun_rises_east")
        assert bid.value == "belief://sun_rises_east"
        assert str(bid) == "belief://sun_rises_east"

    def test_invalid_belief_id_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="must start with 'belief://'"):
            BeliefId.of("invalid://scheme")

    def test_belief_creation_and_immutability(self) -> None:
        bid = BeliefId.of("belief://sun_rises_east")
        ev = EvidenceTrace(evidence_id=SymbolicId.of("memory://obs_1"))
        b = Belief(
            id=bid,
            proposition="The sun rises in the east",
            confidence=Decimal("0.99"),
            support=(ev,),
            source=BeliefSource.MEMORY,
        )

        assert b.id == bid
        assert b.confidence == Decimal("0.99")
        assert b.source == BeliefSource.MEMORY
        assert len(b.support) == 1

        with pytest.raises((TypeError, Exception)):
            b.confidence = Decimal("0.5")  # type: ignore[misc]

    def test_belief_edge_creation(self) -> None:
        b1 = BeliefId.of("belief://premise_1")
        b2 = BeliefId.of("belief://conclusion_1")
        edge = BeliefEdge(
            source=b1,
            target=b2,
            relation=BeliefRelation.SUPPORTS,
            weight=Decimal("1.0"),
        )

        assert edge.source == b1
        assert edge.target == b2
        assert edge.relation == BeliefRelation.SUPPORTS
