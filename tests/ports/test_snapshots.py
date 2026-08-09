"""Port contract tests — snapshot DTOs and grounding (ADR-0028)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from theo_core.models.ports.snapshots import (
    GroundingSnapshot,
    HypothesisProposal,
    KnowledgeGapReport,
    ProviderCapabilities,
    ProviderExecution,
    verify_grounding,
)


class TestSnapshotContracts:
    def test_snapshots_are_frozen(self) -> None:
        """All provider-boundary DTOs must be immutable."""
        proposal = HypothesisProposal(
            proposal_id="hyp://p1", content="candidate interpretation"
        )
        with pytest.raises(FrozenInstanceError):
            proposal.content = "overwrite"  # type: ignore[misc]

    def test_provider_execution_is_frozen_and_carries_replay_fields(self) -> None:
        """ProviderExecution carries exactly the replay-contract fields."""
        execution = ProviderExecution(
            provider_name="null",
            provider_version="0.1.0",
            model_name="none",
            model_hash="",
            seed=0,
            temperature=0.0,
            output=(),
        )
        assert execution.provider_name == "null"
        assert execution.seed == 0
        assert execution.output == ()
        with pytest.raises(FrozenInstanceError):
            execution.temperature = 0.5  # type: ignore[misc]

    def test_capability_enum_has_four_values(self) -> None:
        """Capability discovery exposes exactly four capability flags."""
        values = {c.value for c in ProviderCapabilities}
        assert values == {
            "hypothesis_proposal",
            "calibration",
            "salience",
            "rule_discovery",
        }


class TestGrounding:
    def test_referenced_known_id_is_grounded(self) -> None:
        proposal = HypothesisProposal(
            proposal_id="hyp://p1",
            content="uses belief",
            referenced_ids=frozenset({"belief://b1"}),
        )
        grounding = GroundingSnapshot(belief_ids=frozenset({"belief://b1"}))
        assert verify_grounding(proposal, grounding)

    def test_referenced_unknown_id_is_rejected(self) -> None:
        proposal = HypothesisProposal(
            proposal_id="hyp://p1",
            content="uses missing belief",
            referenced_ids=frozenset({"belief://ghost"}),
        )
        grounding = GroundingSnapshot(belief_ids=frozenset({"belief://b1"}))
        assert not verify_grounding(proposal, grounding)

    def test_unreferenced_proposal_is_rejected(self) -> None:
        proposal = HypothesisProposal(
            proposal_id="hyp://p1", content="no references"
        )
        assert not verify_grounding(proposal, GroundingSnapshot.empty())

    def test_grounding_covers_all_entity_kinds(self) -> None:
        """Referencing any known entity kind (concept/rule/evidence) grounds."""
        proposal = HypothesisProposal(
            proposal_id="hyp://p1",
            content="uses evidence",
            referenced_ids=frozenset({"evidence://e1"}),
        )
        grounding = GroundingSnapshot(
            evidence_ids=frozenset({"evidence://e1"})
        )
        assert verify_grounding(proposal, grounding)

    def test_grounding_requires_no_raw_runtime_state(self) -> None:
        assert not hasattr(GroundingSnapshot, "store")
        assert not hasattr(GroundingSnapshot, "graph")


class TestKnowledgeGapReport:
    def test_defaults_are_empty(self) -> None:
        report = KnowledgeGapReport()
        assert report.missing_premises == ()
        assert report.weak_rule_coverage == ()
        assert report.benchmark_failures == ()

    def test_scores_preserve_decimal_precision(self) -> None:
        from theo_core.models.ports.snapshots import ScoredHypothesis

        scored = ScoredHypothesis(
            hypothesis_id="hyp://h1", score=Decimal("0.123456")
        )
        assert scored.score == Decimal("0.123456")
