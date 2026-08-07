"""DecisionEngine — deterministic action selection engine."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from theo_core.symbolic.decisions.models import (
    ActionSpec,
    DecisionId,
    DecisionRecord,
    DecisionType,
    Intent,
)
from theo_core.symbolic.hypotheses.models import Hypothesis, HypothesisState
from theo_core.symbolic.thoughts.models import Thought, ThoughtId

if TYPE_CHECKING:
    from theo_core.symbolic._primitives.identifiers import SymbolicId
    from theo_core.symbolic.thoughts.graph import ThoughtGraph


_INTENT_BY_GOAL_SLUG: dict[str, Intent] = {
    "acknowledgegreeting": Intent.ACKNOWLEDGE_GREETING,
    "rememberfact": Intent.REMEMBER_FACT,
    "providerecommendation": Intent.PROVIDE_RECOMMENDATION,
    "answerquestion": Intent.ANSWER_QUESTION,
    "maintainconversation": Intent.MAINTAIN_CONVERSATION,
}


def _intent_for_goal(goal_id: SymbolicId) -> Intent:
    """Derive the Intent from the referenced goal's deterministic slug."""
    slug = goal_id.value.rsplit("/", 1)[-1]
    return _INTENT_BY_GOAL_SLUG.get(slug, Intent.MAINTAIN_CONVERSATION)


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
        referenced_goal: SymbolicId,
        min_confidence: Decimal = Decimal("0.5"),
    ) -> DecisionRecord:
        """Select an action or response decision from hypotheses and thoughts.

        This method is pure: it never mutates the ThoughtGraph. When no
        existing thought is available to reference (Canon Law 2), a synthetic
        Thought value is constructed inline and referenced, leaving persistence
        to the caller (Canon Invariant 2 / §6).

        Args:
            hypotheses: Evaluated candidate Hypotheses.
            thoughts: Active ThoughtGraph.
            referenced_goal: The active goal referenced by the decision
                (Canon Invariant 7).
            min_confidence: Minimum confidence threshold for an action decision.

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
                referenced: tuple[ThoughtId, ...] = (fallback_thought.id,)
            else:
                referenced = tuple(t.id for t in all_thoughts[:1])

            return DecisionRecord(
                id=d_id,
                type=DecisionType.DEFER,
                action_text="Insufficient confidence; deferring decision.",
                confidence=Decimal("0.0"),
                referenced_thoughts=referenced,
                referenced_goal=referenced_goal,
                intent=_intent_for_goal(referenced_goal),
                action_spec=ActionSpec(capability="defer"),
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
            referenced_goal=referenced_goal,
            intent=_intent_for_goal(referenced_goal),
            action_spec=ActionSpec(
                capability="respond",
                parameters={"content": winner.interpretation},
            ),
        )
