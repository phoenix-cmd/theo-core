"""Unit tests for ThoughtGraph DAG operations."""

import pytest

from theo_core.symbolic._primitives.errors import CycleDetectedError
from theo_core.symbolic.beliefs.models import BeliefId
from theo_core.symbolic.thoughts.graph import ThoughtGraph
from theo_core.symbolic.thoughts.models import (
    Thought,
    ThoughtEdge,
    ThoughtId,
    ThoughtRelation,
)


class TestThoughtGraph:
    def test_add_and_retrieve_thought(self) -> None:
        tg = ThoughtGraph()
        tid = ThoughtId.of("thought://t1")
        b_id = BeliefId.of("belief://b1")
        t = Thought(id=tid, content="Premise thought", consumed_beliefs=(b_id,))

        tg.add_thought(t)
        assert tg.node_count == 1
        assert tg.has_thought(tid)
        assert tg.get_thought(tid) == t
        assert tg.get_consumed_beliefs(tid) == {b_id}

    def test_cycle_detection_rejects_cyclic_edge(self) -> None:
        tg = ThoughtGraph()
        t1 = ThoughtId.of("thought://t1")
        t2 = ThoughtId.of("thought://t2")

        tg.add_thought(Thought(id=t1, content="T1"))
        tg.add_thought(Thought(id=t2, content="T2"))

        # t1 -> t2
        tg.add_edge(ThoughtEdge(source=t1, target=t2, relation=ThoughtRelation.DERIVED_FROM))
        assert tg.edge_count == 1

        # t2 -> t1 (would create cycle)
        with pytest.raises(CycleDetectedError):
            tg.add_edge(ThoughtEdge(source=t2, target=t1, relation=ThoughtRelation.DERIVED_FROM))

        # Edge count should remain 1 because cyclic edge was rejected and cleaned up
        assert tg.edge_count == 1

    def test_topological_sort(self) -> None:
        tg = ThoughtGraph()
        t1 = ThoughtId.of("thought://t1")
        t2 = ThoughtId.of("thought://t2")
        t3 = ThoughtId.of("thought://t3")

        tg.add_thought(Thought(id=t1, content="T1"))
        tg.add_thought(Thought(id=t2, content="T2"))
        tg.add_thought(Thought(id=t3, content="T3"))

        # t1 -> t2 -> t3
        tg.add_edge(ThoughtEdge(source=t1, target=t2))
        tg.add_edge(ThoughtEdge(source=t2, target=t3))

        topo = tg.topological_sort()
        assert [t.id for t in topo] == [t1, t2, t3]

    def test_reasoning_chain_retrieval(self) -> None:
        tg = ThoughtGraph()
        t1 = ThoughtId.of("thought://t1")
        t2 = ThoughtId.of("thought://t2")
        t3 = ThoughtId.of("thought://t3")

        tg.add_thought(Thought(id=t1, content="T1"))
        tg.add_thought(Thought(id=t2, content="T2"))
        tg.add_thought(Thought(id=t3, content="T3"))

        tg.add_edge(ThoughtEdge(source=t1, target=t2))
        tg.add_edge(ThoughtEdge(source=t2, target=t3))

        chain = tg.get_reasoning_chain(t3)
        assert [t.id for t in chain] == [t1, t2, t3]
