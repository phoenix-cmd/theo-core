"""Pure deterministic belief revision engine.

Handles belief updates, confidence recalculation, superseding old versions with
new revision objects, and resolving contradictions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from theo_core.symbolic.beliefs.models import (
    Belief,
    BeliefEdge,
    BeliefId,
    BeliefRelation,
    BeliefSource,
    EvidenceTrace,
)

if TYPE_CHECKING:
    from theo_core.symbolic.beliefs.graph import BeliefGraph


class BeliefRevision:
    """Deterministic belief revision engine.

    Complexity Contract:
        Time: O(V + E)
        Memory: O(V)
        Deterministic: YES
    """

    @staticmethod
    def revise_belief(
        graph: BeliefGraph,
        existing_id: BeliefId,
        new_confidence: Decimal,
        new_evidence: tuple[EvidenceTrace, ...] = (),
        source: BeliefSource | None = None,
    ) -> Belief:
        """Revise an existing belief by creating a new versioned Belief object.

        Performs append-only superseding: the old belief remains in the graph,
        and a new belief is added with incremented ``revision_id``, updated confidence,
        merged support evidence, and a ``REPLACES`` edge pointing from the new version
        to the old version.

        Args:
            graph: The BeliefGraph to update.
            existing_id: ID of the belief being revised.
            new_confidence: Updated confidence value.
            new_evidence: New evidence traces to append to support.
            source: Optional new BeliefSource (defaults to previous source).

        Returns:
            The newly created revised Belief object.

        Raises:
            KeyError: If existing_id is not found in graph.

        """
        old_belief = graph.get_belief(existing_id)
        if old_belief is None:
            msg = f"Belief {existing_id.value!r} not found in BeliefGraph."
            raise KeyError(msg)

        new_version_id = BeliefId.of(f"{existing_id.value}/v{old_belief.revision_id + 1}")
        merged_support = old_belief.support + new_evidence
        new_source = source if source is not None else old_belief.source

        revised_belief = Belief(
            id=new_version_id,
            proposition=old_belief.proposition,
            confidence=new_confidence,
            uncertainty=Decimal("1.0") - new_confidence,
            support=merged_support,
            contradictions=old_belief.contradictions,
            source=new_source,
            last_verified=datetime.now(UTC),
            evidence_count=old_belief.evidence_count + len(new_evidence),
            reasoning_depth=old_belief.reasoning_depth,
            revision_id=old_belief.revision_id + 1,
            previous_version_id=existing_id,
            metadata=old_belief.metadata,
        )

        # Deactivate old version by setting confidence to 0.0 per Invariant 3
        deactivated_old = Belief(
            id=old_belief.id,
            proposition=old_belief.proposition,
            confidence=Decimal("0.0"),
            uncertainty=Decimal("1.0"),
            support=old_belief.support,
            contradictions=old_belief.contradictions,
            source=old_belief.source,
            last_verified=old_belief.last_verified,
            evidence_count=old_belief.evidence_count,
            reasoning_depth=old_belief.reasoning_depth,
            revision_id=old_belief.revision_id,
            previous_version_id=old_belief.previous_version_id,
            metadata=old_belief.metadata,
        )
        graph.add_belief(deactivated_old, overwrite=True)

        graph.add_belief(revised_belief)
        graph.add_edge(
            BeliefEdge(
                source=new_version_id,
                target=existing_id,
                relation=BeliefRelation.REPLACES,
            )
        )

        return revised_belief

    @staticmethod
    def detect_contradictions(graph: BeliefGraph) -> list[tuple[BeliefId, BeliefId]]:
        """Identify all pairs of active beliefs that contradict each other.

        Returns:
            List of (BeliefId, BeliefId) tuples representing active contradictions.

        """
        contradictions: list[tuple[BeliefId, BeliefId]] = []
        visited_pairs: set[tuple[str, str]] = set()

        for b1 in graph.get_active_beliefs():
            contradicting_ids = graph.get_contradicting_beliefs(b1.id)
            for c_id in sorted(contradicting_ids, key=lambda x: x.value):
                pair_key: tuple[str, str] = (
                    min(b1.id.value, c_id.value),
                    max(b1.id.value, c_id.value),
                )
                if pair_key not in visited_pairs:
                    visited_pairs.add(pair_key)
                    contradictions.append((b1.id, c_id))

        return contradictions

    @staticmethod
    def reconcile_contradiction(
        graph: BeliefGraph,
        belief_id_1: BeliefId,
        belief_id_2: BeliefId,
    ) -> BeliefId:
        """Reconcile a contradiction between two beliefs by keeping the higher confidence belief.

        Marks the lower confidence belief's confidence to Decimal("0.0") via revision.

        Args:
            graph: The BeliefGraph.
            belief_id_1: First contradicting BeliefId.
            belief_id_2: Second contradicting BeliefId.

        Returns:
            The BeliefId of the winning belief.

        """
        b1 = graph.get_belief(belief_id_1)
        b2 = graph.get_belief(belief_id_2)

        if b1 is None or b2 is None:
            msg = "One or both beliefs for reconciliation do not exist in graph."
            raise KeyError(msg)

        if b1.confidence >= b2.confidence:
            winner, loser = b1, b2
        else:
            winner, loser = b2, b1

        # Deprecate losing belief by setting confidence to 0.0
        BeliefRevision.revise_belief(
            graph,
            existing_id=loser.id,
            new_confidence=Decimal("0.0"),
        )

        return winner.id
