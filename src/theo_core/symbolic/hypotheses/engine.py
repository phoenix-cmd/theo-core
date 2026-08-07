"""HypothesisEngine — multi-hypothesis generation, scoring, and candidate pruning."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from theo_core.symbolic.hypotheses.models import (
    Hypothesis,
    HypothesisId,
    HypothesisState,
)

if TYPE_CHECKING:
    from theo_core.symbolic.beliefs.graph import BeliefGraph
    from theo_core.symbolic.concepts.graph import ConceptGraph
    from theo_core.symbolic.thoughts.graph import ThoughtGraph


class HypothesisEngine:
    """Deterministic multi-hypothesis generation and evaluation engine.

    Enforces Canon Law 6: When input is ambiguous, the system MUST generate
    competing Hypotheses rather than collapsing prematurely.

    Complexity Contract:
        Time: O(H * (V + E)) where H is hypothesis count
        Memory: O(H)
        Deterministic: YES
    """

    @staticmethod
    def generate_hypotheses(
        percept_text: str,
        _concepts: ConceptGraph,
        beliefs: BeliefGraph,
        thoughts: ThoughtGraph,
    ) -> list[Hypothesis]:
        """Generate competing candidate Hypotheses for a percept input.

        Args:
            percept_text: Raw input text.
            _concepts: ConceptGraph for taxonomy expansion.
            beliefs: Active BeliefGraph.
            thoughts: Active ThoughtGraph.

        Returns:
            List of competing Hypothesis instances (minimum 1, multiple if ambiguous).

        """
        hypotheses: list[Hypothesis] = []
        active_beliefs = beliefs.get_active_beliefs()
        all_thoughts = thoughts.get_thoughts()

        words = percept_text.lower().split()
        matching_beliefs = [
            b for b in active_beliefs if any(w in b.proposition.lower() for w in words)
        ]

        if not matching_beliefs:
            # Default single hypothesis for simple inputs
            h_id = HypothesisId.of("hypothesis://cand/1")
            hypotheses.append(
                Hypothesis(
                    id=h_id,
                    interpretation=f"Direct interpretation of '{percept_text}'",
                    score=Decimal("0.5"),
                    state=HypothesisState.CANDIDATE,
                )
            )
        else:
            # Generate competing hypotheses for each matching belief (Law 6)
            for idx, belief in enumerate(matching_beliefs, start=1):
                h_id = HypothesisId.of(f"hypothesis://cand/{idx}")
                supp_thoughts = tuple(t.id for t in all_thoughts if belief.id in t.consumed_beliefs)
                hypotheses.append(
                    Hypothesis(
                        id=h_id,
                        interpretation=f"Interpretation based on belief '{belief.proposition}'",
                        score=belief.confidence,
                        state=HypothesisState.CANDIDATE,
                        supporting_beliefs=(belief.id,),
                        supporting_thoughts=supp_thoughts,
                    )
                )

        return hypotheses

    @staticmethod
    def evaluate_hypotheses(
        hypotheses: list[Hypothesis],
        thoughts: ThoughtGraph,
        beliefs: BeliefGraph,
    ) -> list[Hypothesis]:
        """Evaluate and score competing hypotheses against evidence.

        Args:
            hypotheses: List of candidate Hypothesis instances.
            thoughts: Active ThoughtGraph.
            beliefs: Active BeliefGraph.

        Returns:
            New list of updated Hypothesis instances with scores and states.

        """
        evaluated: list[Hypothesis] = []
        if not hypotheses:
            return []

        # Calculate scores for each candidate
        scored_candidates: list[tuple[Hypothesis, Decimal]] = []
        for h in hypotheses:
            # Base score from existing score
            score = h.score

            # Add weight for supporting beliefs
            for b_id in h.supporting_beliefs:
                b = beliefs.get_belief(b_id)
                if b is not None:
                    score = (score + b.confidence) / Decimal("2.0")

            # Add weight for supporting thoughts
            for t_id in h.supporting_thoughts:
                t = thoughts.get_thought(t_id)
                if t is not None:
                    score = (score + t.confidence) / Decimal("2.0")

            scored_candidates.append((h, round(score, 4)))

        # Sort candidates deterministically by score descending, then ID string
        scored_candidates.sort(key=lambda item: (-item[1], item[0].id.value))
        max_score = scored_candidates[0][1]

        for h, score in scored_candidates:
            if score == max_score and score >= Decimal("0.5"):
                new_state = HypothesisState.ACCEPTED
            elif score < Decimal("0.3"):
                new_state = HypothesisState.REJECTED
            else:
                new_state = HypothesisState.EVALUATED

            evaluated.append(
                Hypothesis(
                    id=h.id,
                    interpretation=h.interpretation,
                    score=score,
                    state=new_state,
                    supporting_thoughts=h.supporting_thoughts,
                    supporting_beliefs=h.supporting_beliefs,
                    created_at=h.created_at,
                    metadata=h.metadata,
                )
            )

        return evaluated

    @staticmethod
    def prune_candidates(
        hypotheses: list[Hypothesis],
        min_score: Decimal = Decimal("0.3"),
        max_keep: int = 3,
    ) -> tuple[list[Hypothesis], list[Hypothesis]]:
        """Prune low-scoring candidate hypotheses.

        Args:
            hypotheses: List of candidate hypotheses.
            min_score: Minimum required score threshold.
            max_keep: Maximum number of hypotheses to retain.

        Returns:
            Tuple of (kept_hypotheses, pruned_hypotheses).

        """
        sorted_hyp = sorted(hypotheses, key=lambda h: (-h.score, h.id.value))
        kept: list[Hypothesis] = []
        pruned: list[Hypothesis] = []

        for idx, h in enumerate(sorted_hyp):
            if idx < max_keep and h.score >= min_score and h.state != HypothesisState.REJECTED:
                kept.append(h)
            else:
                pruned_h = Hypothesis(
                    id=h.id,
                    interpretation=h.interpretation,
                    score=h.score,
                    state=HypothesisState.PRUNED,
                    supporting_thoughts=h.supporting_thoughts,
                    supporting_beliefs=h.supporting_beliefs,
                    created_at=h.created_at,
                    metadata=h.metadata,
                )
                pruned.append(pruned_h)

        return kept, pruned
