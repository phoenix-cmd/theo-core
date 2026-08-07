"""Canon Edition C1 Conformance Tests — Hypothesis Engine Laws."""

from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.beliefs.models import Belief, BeliefId
from theo_core.symbolic.concepts.graph import ConceptGraph
from theo_core.symbolic.hypotheses.engine import HypothesisEngine
from theo_core.symbolic.thoughts.graph import ThoughtGraph


class TestCanonHypothesisLaws:
    def test_law_6_ambiguous_input_generates_competing_hypotheses(self) -> None:
        """Canon Law 6: When input is ambiguous, the system MUST generate competing Hypotheses."""
        cg = ConceptGraph()
        bg = BeliefGraph()
        tg = ThoughtGraph()

        bg.add_belief(
            Belief(
                id=BeliefId.of("belief://intent_refactor"),
                proposition="User intent refactor",
            )
        )
        bg.add_belief(
            Belief(
                id=BeliefId.of("belief://intent_feature"),
                proposition="User intent feature",
            )
        )

        hypotheses = HypothesisEngine.generate_hypotheses("intent", cg, bg, tg)

        # Must generate multiple competing hypotheses rather than collapsing prematurely
        assert len(hypotheses) >= 2
