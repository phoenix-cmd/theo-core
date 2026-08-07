"""Integration test for Belief System lifecycle."""

from decimal import Decimal

from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.beliefs.models import (
    Belief,
    BeliefId,
    BeliefSource,
    EvidenceTrace,
)
from theo_core.symbolic.beliefs.repository import InMemoryBeliefRepository
from theo_core.symbolic.beliefs.revision import BeliefRevision


class TestBeliefIntegration:
    def test_full_belief_lifecycle(self) -> None:
        repo = InMemoryBeliefRepository()
        bg = BeliefGraph()
        gid = SymbolicId.of("concept://workspace/beliefs")

        # 1. Create beliefs with evidence traces
        ev1 = EvidenceTrace(evidence_id=SymbolicId.of("memory://turn_1"), source_type="memory")
        b1_id = BeliefId.of("belief://user_name_is_alex")
        b1 = Belief(
            id=b1_id,
            proposition="User's name is Alex",
            confidence=Decimal("0.95"),
            support=(ev1,),
            source=BeliefSource.MEMORY,
        )
        bg.add_belief(b1)

        # 2. Revise belief
        ev2 = EvidenceTrace(evidence_id=SymbolicId.of("memory://turn_5"), source_type="memory")
        revised = BeliefRevision.revise_belief(
            bg,
            existing_id=b1_id,
            new_confidence=Decimal("0.99"),
            new_evidence=(ev2,),
        )

        # 3. Save to repository
        repo.save(gid, bg)
        assert repo.exists(gid)

        # 4. Load from repository
        loaded_bg = repo.load(gid)
        assert loaded_bg is not None
        assert loaded_bg.node_count == 2
        assert loaded_bg.has_belief(revised.id)
