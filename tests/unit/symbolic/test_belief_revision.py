"""Unit tests for BeliefRevision engine."""

from decimal import Decimal

from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.beliefs.models import (
    Belief,
    BeliefEdge,
    BeliefId,
    BeliefRelation,
    EvidenceTrace,
)
from theo_core.symbolic.beliefs.revision import BeliefRevision


class TestBeliefRevision:
    def test_revise_belief_creates_new_version(self) -> None:
        bg = BeliefGraph()
        b1_id = BeliefId.of("belief://earth_flat")
        b1 = Belief(id=b1_id, proposition="Earth is flat", confidence=Decimal("0.5"))
        bg.add_belief(b1)

        new_ev = EvidenceTrace(evidence_id=SymbolicId.of("memory://satellite_img"))
        revised = BeliefRevision.revise_belief(
            bg,
            existing_id=b1_id,
            new_confidence=Decimal("0.0"),
            new_evidence=(new_ev,),
        )

        assert bg.node_count == 2
        assert revised.previous_version_id == b1_id
        assert revised.revision_id == 2
        assert revised.confidence == Decimal("0.0")
        assert len(revised.support) == 1

        # Check REPLACES edge
        edge = bg.get_edge(revised.id, b1_id, BeliefRelation.REPLACES)
        assert edge is not None

    def test_reconcile_contradiction(self) -> None:
        bg = BeliefGraph()
        b1_id = BeliefId.of("belief://rain_today")
        b2_id = BeliefId.of("belief://no_rain_today")

        bg.add_belief(Belief(id=b1_id, proposition="It rains today", confidence=Decimal("0.9")))
        bg.add_belief(Belief(id=b2_id, proposition="No rain today", confidence=Decimal("0.3")))

        bg.add_edge(BeliefEdge(source=b1_id, target=b2_id, relation=BeliefRelation.CONTRADICTS))

        winner_id = BeliefRevision.reconcile_contradiction(bg, b1_id, b2_id)
        assert winner_id == b1_id

        # Winning belief confidence unchanged; loser confidence reduced to 0.0 in new version
        active_beliefs = bg.get_active_beliefs(min_confidence=Decimal("0.1"))
        active_ids = {b.id for b in active_beliefs}
        assert b1_id in active_ids
