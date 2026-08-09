"""ProviderCoordinator hook behavior (ADR-0028).

The coordinator is the only theo-core code that calls providers: it resolves
one provider per capability, records deterministic provenance, and fails fast
with ``ProviderFailure`` on any provider error. No silent fallback.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from stubs import RecordingProvider

from theo_core.models.ports.snapshots import (
    DecisionSnapshot,
    GroundingSnapshot,
    HypothesisProposal,
    ProviderCapabilities,
    ProviderExecution,
)
from theo_core.runtime.providers.coordinator import ProviderCoordinator
from theo_core.runtime.providers.models import ProviderFailure, ProviderStatus
from theo_core.runtime.providers.resolution import ProviderResolver


class BadExecutionProvider(RecordingProvider):
    """Returns a non-``ProviderExecution`` value from every hook."""

    def _run(self, method: str, args: tuple[object, ...]) -> ProviderExecution[tuple[object, ...]]:
        """Record a call and return an invalid execution."""
        self.calls.append((method, args))
        return "not an execution"  # type: ignore[return-value]


class NonTupleOutputProvider(RecordingProvider):
    """Returns a ``ProviderExecution`` whose output is not a tuple."""

    def _run(self, method: str, args: tuple[object, ...]) -> ProviderExecution[tuple[object, ...]]:
        """Record a call and return a non-tuple output."""
        self.calls.append((method, args))
        return ProviderExecution(
            provider_name="recording",
            provider_version="0.1.0",
            model_name="none",
            model_hash="",
            seed=0,
            temperature=0.0,
            output="not a tuple",  # type: ignore[arg-type]
        )


class TestUnconfiguredCoordinator:
    def test_every_hook_returns_empty_without_invocation(self) -> None:
        coordinator = ProviderCoordinator(ProviderResolver())
        grounding = GroundingSnapshot.empty()
        proposal = coordinator.propose_hypotheses(
            percept="hello", concepts=(), beliefs=(), rules=(), grounding=grounding
        )
        assert proposal.proposals == ()
        assert proposal.invocation is None

        scored = coordinator.score_hypotheses(
            hypotheses=(), beliefs=(), percept="hello", grounding=grounding
        )
        assert scored.scored == ()
        assert scored.invocation is None

        calibrated = coordinator.score_confidence(
            decision=DecisionSnapshot(
                decision_id="decision://d1",
                action_text="respond",
                intent="maintain_conversation",
                confidence=Decimal("1.0"),
            ),
            hypotheses=(),
            beliefs=(),
            grounding=grounding,
        )
        assert calibrated.scored == ()
        assert calibrated.invocation is None

        goals = coordinator.rank_goals(
            goals=(), percept="hello", beliefs=(), grounding=grounding
        )
        assert goals.goals == ()
        assert goals.invocation is None

        rules = coordinator.rank_rules(
            rules=(), concepts=(), beliefs=(), grounding=grounding
        )
        assert rules.rules == ()
        assert rules.invocation is None


class TestConfiguredCoordinator:
    def test_invocation_records_deterministic_provenance(self) -> None:
        provider = RecordingProvider()
        coordinator = ProviderCoordinator(ProviderResolver([provider]))
        result = coordinator.rank_rules(
            rules=(), concepts=(), beliefs=(), grounding=GroundingSnapshot.empty()
        )
        assert result.invocation is not None
        assert result.invocation.status == ProviderStatus.EXECUTED
        assert result.invocation.provider_name == "recording"
        assert result.invocation.provider_version == "0.1.0"
        assert result.invocation.model_name == "none"
        assert result.invocation.capability == ProviderCapabilities.SALIENCE
        assert result.invocation.summary == {"count": 0}
        assert provider.calls == [("rank_rules", ((), (), (), GroundingSnapshot.empty()))]

    def test_summary_reports_nonempty_output_count(self) -> None:
        output = (
            HypothesisProposal(
                proposal_id="proposal://p1",
                content="proposed",
                referenced_ids=frozenset(),
            ),
        )
        provider = RecordingProvider(output=output)
        coordinator = ProviderCoordinator(ProviderResolver([provider]))
        result = coordinator.propose_hypotheses(
            percept="hello", concepts=(), beliefs=(), rules=(), grounding=GroundingSnapshot.empty()
        )
        assert result.proposals == output
        assert result.invocation is not None
        assert result.invocation.summary == {"count": 1}

    def test_unresolved_capability_never_invokes_provider(self) -> None:
        provider = RecordingProvider(
            capabilities=frozenset({ProviderCapabilities.SALIENCE})
        )
        coordinator = ProviderCoordinator(ProviderResolver([provider]))
        result = coordinator.propose_hypotheses(
            percept="hello", concepts=(), beliefs=(), rules=(), grounding=GroundingSnapshot.empty()
        )
        assert result.invocation is None
        assert provider.calls == []


class TestFailFastSemantics:
    def test_provider_error_raises_provider_failure(self) -> None:
        provider = RecordingProvider(raise_on="rank_goals")
        coordinator = ProviderCoordinator(ProviderResolver([provider]))
        with pytest.raises(ProviderFailure) as excinfo:
            coordinator.rank_goals(
                goals=(), percept="hello", beliefs=(), grounding=GroundingSnapshot.empty()
            )
        assert excinfo.value.capability == ProviderCapabilities.SALIENCE
        assert excinfo.value.provider_name == "RecordingProvider"

    def test_non_execution_return_raises_provider_failure(self) -> None:
        coordinator = ProviderCoordinator(ProviderResolver([BadExecutionProvider()]))
        with pytest.raises(ProviderFailure) as excinfo:
            coordinator.propose_hypotheses(
                percept="hello",
                concepts=(),
                beliefs=(),
                rules=(),
                grounding=GroundingSnapshot.empty(),
            )
        assert excinfo.value.capability == ProviderCapabilities.HYPOTHESIS_PROPOSAL

    def test_non_tuple_output_raises_provider_failure(self) -> None:
        coordinator = ProviderCoordinator(ProviderResolver([NonTupleOutputProvider()]))
        with pytest.raises(ProviderFailure) as excinfo:
            coordinator.score_hypotheses(
                hypotheses=(), beliefs=(), percept="hello", grounding=GroundingSnapshot.empty()
            )
        assert excinfo.value.capability == ProviderCapabilities.CALIBRATION
