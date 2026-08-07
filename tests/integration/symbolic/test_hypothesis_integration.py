"""Integration test for Hypothesis System lifecycle."""

from decimal import Decimal

from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.beliefs.models import Belief, BeliefId
from theo_core.symbolic.concepts.graph import ConceptGraph
from theo_core.symbolic.hypotheses.engine import HypothesisEngine
from theo_core.symbolic.hypotheses.repository import InMemoryHypothesisRepository
from theo_core.symbolic.thoughts.graph import ThoughtGraph


class TestHypothesisIntegration:
    def test_full_hypothesis_workflow(self) -> None:
        repo = InMemoryHypothesisRepository()
        cg = ConceptGraph()
        bg = BeliefGraph()
        tg = ThoughtGraph()

        bg.add_belief(
            Belief(
                id=BeliefId.of("belief://opt_a"),
                proposition="Option A",
                confidence=Decimal("0.95"),
            )
        )
        bg.add_belief(
            Belief(
                id=BeliefId.of("belief://opt_b"),
                proposition="Option B",
                confidence=Decimal("0.1"),
            )
        )

        # 1. Generate candidate hypotheses
        cands = HypothesisEngine.generate_hypotheses("Option", cg, bg, tg)
        assert len(cands) == 2

        # 2. Evaluate candidates
        evaluated = HypothesisEngine.evaluate_hypotheses(cands, tg, bg)

        # 3. Prune low-confidence candidates
        kept, _pruned = HypothesisEngine.prune_candidates(
            evaluated, min_score=Decimal("0.3"), max_keep=1
        )

        # 4. Save to repository
        for h in kept:
            repo.save(h)

        assert len(repo.get_all()) == 1
        assert repo.get_all()[0].id == kept[0].id
