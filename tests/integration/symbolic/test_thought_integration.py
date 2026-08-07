"""Integration test for Thought Graph lifecycle."""

from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.models import BeliefId
from theo_core.symbolic.thoughts.graph import ThoughtGraph
from theo_core.symbolic.thoughts.models import (
    Thought,
    ThoughtEdge,
    ThoughtId,
    ThoughtRelation,
)
from theo_core.symbolic.thoughts.repository import InMemoryThoughtRepository


class TestThoughtIntegration:
    def test_full_thought_graph_lifecycle(self) -> None:
        repo = InMemoryThoughtRepository()
        tg = ThoughtGraph()
        gid = SymbolicId.of("concept://workspace/thoughts")

        b1 = BeliefId.of("belief://user_wants_code")
        t1_id = ThoughtId.of("thought://parse_user_intent")
        t2_id = ThoughtId.of("thought://plan_code_generation")

        t1 = Thought(
            id=t1_id,
            content="User intent is code generation",
            consumed_beliefs=(b1,),
        )
        t2 = Thought(
            id=t2_id,
            content="Plan 3 steps for implementation",
        )

        tg.add_thought(t1)
        tg.add_thought(t2)
        tg.add_edge(ThoughtEdge(source=t1_id, target=t2_id, relation=ThoughtRelation.DERIVED_FROM))

        # 1. Topological sort check
        topo = tg.topological_sort()
        assert [t.id for t in topo] == [t1_id, t2_id]

        # 2. Save & Load repository check
        repo.save(gid, tg)
        assert repo.exists(gid)

        loaded_tg = repo.load(gid)
        assert loaded_tg is not None
        assert loaded_tg.node_count == 2
        assert loaded_tg.edge_count == 1
