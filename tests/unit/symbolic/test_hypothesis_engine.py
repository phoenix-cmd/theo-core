"""Unit tests for HypothesisEngine generation, evaluation, and pruning."""

from decimal import Decimal

from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.beliefs.models import Belief, BeliefId
from theo_core.symbolic.concepts.graph import ConceptGraph
from theo_core.symbolic.hypotheses.engine import HypothesisEngine
from theo_core.symbolic.hypotheses.models import HypothesisState
from theo_core.symbolic.thoughts.graph import ThoughtGraph


class TestHypothesisEngine:
    def test_multi_hypothesis_generation(self) -> None:
        cg = ConceptGraph()
        bg = BeliefGraph()
        tg = ThoughtGraph()

        b1 = Belief(
            id=BeliefId.of("belief://user_wants_python"),
            proposition="User wants Python implementation",
        )
        b2 = Belief(
            id=BeliefId.of("belief://user_wants_rust"),
            proposition="User wants Rust implementation",
        )

        bg.add_belief(b1)
        bg.add_belief(b2)

        # Percept text contains ambiguous keywords matching both beliefs
        hypotheses = HypothesisEngine.generate_hypotheses("implementation details", cg, bg, tg)

        # Generates competing hypotheses (Canon Law 6)
        assert len(hypotheses) == 2

    def test_hypothesis_evaluation_and_pruning(self) -> None:
        cg = ConceptGraph()
        bg = BeliefGraph()
        tg = ThoughtGraph()

        b1 = Belief(
            id=BeliefId.of("belief://python"),
            proposition="python code",
            confidence=Decimal("0.9"),
        )
        b2 = Belief(
            id=BeliefId.of("belief://rust"),
            proposition="rust code",
            confidence=Decimal("0.2"),
        )

        bg.add_belief(b1)
        bg.add_belief(b2)

        hypotheses = HypothesisEngine.generate_hypotheses("code", cg, bg, tg)
        evaluated = HypothesisEngine.evaluate_hypotheses(hypotheses, tg, bg)

        assert len(evaluated) == 2
        # Highest score should be accepted
        accepted = [h for h in evaluated if h.state == HypothesisState.ACCEPTED]
        assert len(accepted) == 1

        kept, pruned = HypothesisEngine.prune_candidates(
            evaluated, min_score=Decimal("0.3"), max_keep=1
        )
        assert len(kept) == 1
        assert len(pruned) == 1
        assert pruned[0].state == HypothesisState.PRUNED
