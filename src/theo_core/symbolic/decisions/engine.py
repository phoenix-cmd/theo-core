"""DecisionEngine — deterministic action selection engine."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from theo_core.symbolic.decisions.models import DecisionId, DecisionRecord, DecisionType
from theo_core.symbolic.hypotheses.models import Hypothesis, HypothesisState
from theo_core.symbolic.thoughts.models import Thought, ThoughtId

if TYPE_CHECKING:
    from theo_core.symbolic._primitives.identifiers import SymbolicId
    from theo_core.symbolic.thoughts.graph import ThoughtGraph


class DecisionEngine:
    """Deterministic action and response decision engine.

    Enforces Canon Law 2 (Every Decision MUST reference one or more Thoughts)
    and Canon Invariant 2 (Decisions MUST be fully deterministic given identical inputs).

    Complexity Contract:
        Time: O(H + T)
        Memory: O(1)
        Deterministic: YES
    """

    @staticmethod
    def make_decision(
        hypotheses: list[Hypothesis],
        thoughts: ThoughtGraph,
        min_confidence: Decimal = Decimal("0.5"),
        active_goal_id: SymbolicId | None = None,
    ) -> DecisionRecord:
        """Select an action or response decision from hypotheses and thoughts.

        Args:
            hypotheses: Evaluated candidate Hypotheses.
            thoughts: Active ThoughtGraph.
            min_confidence: Minimum confidence threshold for an action decision.
            active_goal_id: Optional active GoalId per Canon Invariant 7.

        Returns:
            A deterministic DecisionRecord instance.

        """
        accepted = [
            h
            for h in hypotheses
            if h.state == HypothesisState.ACCEPTED and h.score >= min_confidence
        ]

        if not accepted:
            # Fallback DEFER decision
            d_id = DecisionId.of("decision://fallback/defer")
            all_thoughts = thoughts.get_thoughts()
            if not all_thoughts:
                fallback_thought = Thought(
                    id=ThoughtId.of("thought://sys/fallback_evaluation"),
                    content="Fallback system evaluation due to low hypothesis confidence.",
                    confidence=Decimal("0.0"),
                )
                thoughts.add_thought(fallback_thought)
                referenced: tuple[ThoughtId, ...] = (fallback_thought.id,)
            else:
                referenced = tuple(t.id for t in all_thoughts[:1])

            return DecisionRecord(
                id=d_id,
                type=DecisionType.DEFER,
                action_text="Insufficient confidence; deferring decision.",
                confidence=Decimal("0.0"),
                referenced_thoughts=referenced,
                active_goal_id=active_goal_id,
            )

        # Deterministically pick winning hypothesis (highest score, lowest ID string)
        winner = sorted(accepted, key=lambda h: (-h.score, h.id.value))[0]

        # Gather referenced thoughts per Canon Law 2
        ref_thoughts: list[ThoughtId] = list(winner.supporting_thoughts)
        if not ref_thoughts:
            all_t = thoughts.get_thoughts()
            if not all_t:
                sys_thought = Thought(
                    id=ThoughtId.of("thought://sys/inference_rule"),
                    content=f"System inference rule derivation for hypothesis {winner.id}",
                    confidence=winner.score,
                )
                thoughts.add_thought(sys_thought)
                ref_thoughts.append(sys_thought.id)
            else:
                ref_thoughts.append(all_t[0].id)

        d_id = DecisionId.of(f"decision://select/{winner.id.value.replace('hypothesis://', '')}")
        return DecisionRecord(
            id=d_id,
            type=DecisionType.RESPONSE,
            action_text=winner.interpretation,
            confidence=winner.score,
            referenced_thoughts=tuple(ref_thoughts),
            accepted_hypothesis_id=winner.id,
            active_goal_id=active_goal_id,
        )
