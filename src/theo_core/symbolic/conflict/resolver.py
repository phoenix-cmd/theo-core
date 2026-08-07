"""ConflictResolver — multi-policy contradiction resolution engine."""

from __future__ import annotations

from decimal import Decimal

from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.models import Belief, BeliefSource
from theo_core.symbolic.conflict.models import ConflictPolicy, ConflictRecord
from theo_core.symbolic.hypotheses.models import Hypothesis, HypothesisState


class ConflictResolver:
    """Deterministic contradiction resolution engine enforcing Canon Law 7.

    Complexity Contract:
        Time: O(1) per belief pair, O(N log N) per hypothesis set
        Memory: O(1)
        Deterministic: YES
    """

    @staticmethod
    def resolve_belief_contradiction(
        b1: Belief,
        b2: Belief,
        policy: ConflictPolicy = ConflictPolicy.HIGHER_CONFIDENCE,
    ) -> tuple[Belief, Belief, ConflictRecord]:
        """Resolve contradiction between two beliefs per specified policy.

        Args:
            b1: First conflicting Belief.
            b2: Second conflicting Belief.
            policy: Resolution strategy.

        Returns:
            Tuple of (winning_belief, deprecated_losing_belief, conflict_record).

        """
        # Determine winner based on policy
        if policy == ConflictPolicy.HIGHER_CONFIDENCE:
            winner, loser = (b1, b2) if b1.confidence >= b2.confidence else (b2, b1)
            reason = (
                f"Selected {winner.id} with higher confidence "
                f"{winner.confidence} >= {loser.confidence}"
            )
        elif policy == ConflictPolicy.EVIDENCE_COUNT:
            winner, loser = (b1, b2) if b1.evidence_count >= b2.evidence_count else (b2, b1)
            reason = (
                f"Selected {winner.id} with higher evidence count "
                f"{winner.evidence_count} >= {loser.evidence_count}"
            )
        elif policy == ConflictPolicy.RECENT_SOURCE:
            winner, loser = (b1, b2) if b1.last_verified >= b2.last_verified else (b2, b1)
            reason = f"Selected {winner.id} verified more recently at {winner.last_verified}"
        elif policy == ConflictPolicy.EXPLICIT_AUTHORITY:
            source_rank = {
                BeliefSource.KNOWLEDGE: 4,
                BeliefSource.INFERENCE: 3,
                BeliefSource.MEMORY: 2,
                BeliefSource.PERCEPTION: 1,
            }
            r1, r2 = source_rank.get(b1.source, 0), source_rank.get(b2.source, 0)
            if r1 != r2:
                winner, loser = (b1, b2) if r1 > r2 else (b2, b1)
                reason = f"Explicit authority source rank {winner.source} > {loser.source}"
            else:
                winner, loser = (b1, b2) if b1.confidence >= b2.confidence else (b2, b1)
                reason = f"Equal rank; fallback {winner.confidence} >= {loser.confidence}"
        else:
            winner, loser = (b1, b2) if b1.confidence >= b2.confidence else (b2, b1)
            reason = f"Fallback confidence selected {winner.id}"

        # Create updated deprecated loser belief with zero confidence
        deprecated_loser = Belief(
            id=loser.id,
            proposition=loser.proposition,
            confidence=Decimal("0.0"),
            uncertainty=Decimal("1.0"),
            support=loser.support,
            contradictions=loser.contradictions,
            source=loser.source,
            last_verified=loser.last_verified,
            evidence_count=loser.evidence_count,
            metadata=loser.metadata,
        )

        b1_tag = b1.id.value.replace("belief://", "")
        b2_tag = b2.id.value.replace("belief://", "")
        conflict_id = SymbolicId.of(f"conflict://belief/{b1_tag}_{b2_tag}")
        record = ConflictRecord(
            conflict_id=conflict_id,
            conflicting_ids=(b1.id.to_symbolic_id(), b2.id.to_symbolic_id()),
            policy=policy,
            winning_id=winner.id.to_symbolic_id(),
            reason=reason,
        )

        return winner, deprecated_loser, record

    @staticmethod
    def resolve_hypotheses_conflict(
        hypotheses: list[Hypothesis],
        policy: ConflictPolicy = ConflictPolicy.HIGHER_CONFIDENCE,
    ) -> tuple[list[Hypothesis], list[ConflictRecord]]:
        """Resolve conflicts across a list of competing candidate hypotheses.

        Args:
            hypotheses: List of candidate hypotheses.
            policy: Resolution policy.

        Returns:
            Tuple of (resolved_hypotheses, conflict_records).

        """
        if not hypotheses:
            return [], []

        if policy == ConflictPolicy.EVIDENCE_COUNT:
            sorted_hyp = sorted(
                hypotheses,
                key=lambda h: (-len(h.supporting_thoughts), -h.score, h.id.value),
            )
        else:
            sorted_hyp = sorted(hypotheses, key=lambda h: (-h.score, h.id.value))

        winner = sorted_hyp[0]

        resolved: list[Hypothesis] = []
        records: list[ConflictRecord] = []

        # Winner is accepted
        resolved.append(
            Hypothesis(
                id=winner.id,
                interpretation=winner.interpretation,
                score=winner.score,
                state=HypothesisState.ACCEPTED,
                supporting_thoughts=winner.supporting_thoughts,
                supporting_beliefs=winner.supporting_beliefs,
                created_at=winner.created_at,
                metadata=winner.metadata,
            )
        )

        # Losers are rejected
        for loser in sorted_hyp[1:]:
            resolved.append(
                Hypothesis(
                    id=loser.id,
                    interpretation=loser.interpretation,
                    score=loser.score,
                    state=HypothesisState.REJECTED,
                    supporting_thoughts=loser.supporting_thoughts,
                    supporting_beliefs=loser.supporting_beliefs,
                    created_at=loser.created_at,
                    metadata=loser.metadata,
                )
            )

            w_tag = winner.id.value.replace("hypothesis://", "")
            l_tag = loser.id.value.replace("hypothesis://", "")
            c_id = SymbolicId.of(f"conflict://hyp/{w_tag}_{l_tag}")
            records.append(
                ConflictRecord(
                    conflict_id=c_id,
                    conflicting_ids=(winner.id.to_symbolic_id(), loser.id.to_symbolic_id()),
                    policy=policy,
                    winning_id=winner.id.to_symbolic_id(),
                    reason=f"Hypothesis {winner.id} scored {winner.score} > {loser.score}",
                )
            )

        return resolved, records
