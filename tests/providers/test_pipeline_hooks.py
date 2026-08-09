"""Provider hook integration with the symbolic pipeline and runtime (ADR-0028).

Phase 1 inserts the provider hook orchestration at the runtime boundary without
changing cognition: hooks are consulted and recorded for provenance, but their
outputs are never consumed. These tests pin the equivalence and fail-fast
contracts inside theo-core.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pytest
from stubs import GoalSalienceOnlyProvider, RecordingProvider

from theo_core.evaluation.benchmark_schema import (
    PROVIDER_PROVENANCE_METADATA_KEY,
    golden_fingerprint,
)
from theo_core.models.ports.snapshots import (
    BeliefSnapshotCollection,
    DecisionSnapshot,
    GroundingSnapshot,
    HypothesisProposal,
    HypothesisSnapshotCollection,
    ProviderCapabilities,
    ProviderExecution,
    ScoredHypothesis,
)
from theo_core.runtime.providers.coordinator import ProviderCoordinator
from theo_core.runtime.providers.models import ProviderFailure, ProviderStatus
from theo_core.runtime.providers.resolution import ProviderResolver
from theo_core.symbolic.persistence.store import checksum_cycle_state
from theo_core.symbolic.pipeline import SymbolicCognitivePipeline
from theo_core.symbolic.runtime import SymbolicRuntime
from theo_core.telemetry.tracing.recorder import TraceRecorder

if TYPE_CHECKING:
    from pathlib import Path

_HOOK_ORDER = (
    "rank_rules",
    "propose_hypotheses",
    "score_hypotheses",
    "rank_goals",
    "score_confidence",
)


def _run_fingerprint(pipeline: SymbolicCognitivePipeline) -> dict[str, object]:
    """Execute one cycle and return its canonical golden fingerprint."""
    decision, _trace, golden_trace = pipeline.execute_cycle("rain is falling")
    return golden_fingerprint(golden_trace, decision.action_text)


def _state_checksum(pipeline: SymbolicCognitivePipeline) -> str:
    """Return the committed state checksum after the last cycle."""
    return checksum_cycle_state(pipeline.state)


def _coordinator_with(provider: object) -> ProviderCoordinator:
    return ProviderCoordinator(ProviderResolver([provider]))


class TestPipelineProvenance:
    def test_default_pipeline_records_no_provenance(self) -> None:
        pipeline = SymbolicCognitivePipeline()
        pipeline.execute_cycle("rain is falling")
        assert pipeline.provider_provenance == ()

    def test_hooks_invoked_in_deterministic_stage_order(self) -> None:
        provider = RecordingProvider()
        pipeline = SymbolicCognitivePipeline(coordinator=_coordinator_with(provider))
        pipeline.execute_cycle("rain is falling")

        methods = [method for method, _ in provider.calls]
        assert methods == list(_HOOK_ORDER)
        assert [i.capability for i in pipeline.provider_provenance] == [
            ProviderCapabilities.SALIENCE,
            ProviderCapabilities.HYPOTHESIS_PROPOSAL,
            ProviderCapabilities.CALIBRATION,
            ProviderCapabilities.SALIENCE,
            ProviderCapabilities.CALIBRATION,
        ]
        for invocation in pipeline.provider_provenance:
            assert invocation.status == ProviderStatus.EXECUTED
            assert invocation.provider_name == "recording"
            assert invocation.summary == {"count": 0}


class TestNullEquivalence:
    def test_null_provider_is_cognitively_equivalent_to_no_provider(self) -> None:
        baseline = SymbolicCognitivePipeline()
        null_configured = SymbolicCognitivePipeline(
            coordinator=_coordinator_with(RecordingProvider())
        )

        assert _run_fingerprint(baseline) == _run_fingerprint(null_configured)
        assert _state_checksum(baseline) == _state_checksum(null_configured)
        assert baseline.provider_provenance == ()
        assert len(null_configured.provider_provenance) == 5

    def test_nonempty_hook_outputs_are_never_consumed(self) -> None:
        proposal = HypothesisProposal(
            proposal_id="proposal://p1",
            content="ignored in Phase 1",
            referenced_ids=frozenset({"belief://percept/rain"}),
        )
        provider = RecordingProvider(output=(proposal,))
        configured = SymbolicCognitivePipeline(coordinator=_coordinator_with(provider))
        baseline = SymbolicCognitivePipeline()
        configured_cycle = configured.execute_cycle("rain is falling")
        baseline_cycle = baseline.execute_cycle("rain is falling")

        configured_fingerprint = golden_fingerprint(
            configured_cycle[2], configured_cycle[0].action_text
        )
        baseline_fingerprint = golden_fingerprint(baseline_cycle[2], baseline_cycle[0].action_text)
        assert configured_fingerprint == baseline_fingerprint
        assert configured_cycle[1] == baseline_cycle[1]
        assert configured_cycle[2] == baseline_cycle[2]
        assert _state_checksum(configured) == _state_checksum(baseline)

    def test_wellformed_calibration_scores_are_consumed(self) -> None:
        """Phase 4: well-formed ScoredHypothesis output recalibrates confidence.

        The accepted (first) hypothesis receives the provider score while the
        decision identity and action text are preserved.
        """

        class RecalibrateProvider(RecordingProvider):
            def score_confidence(
                self,
                decision: DecisionSnapshot,
                hypotheses: HypothesisSnapshotCollection,
                beliefs: BeliefSnapshotCollection,
                grounding: GroundingSnapshot,
            ) -> ProviderExecution[tuple[object, ...]]:
                return ProviderExecution(
                    provider_name="recalibrate",
                    provider_version="0.1.0",
                    model_name="test",
                    model_hash="",
                    seed=0,
                    temperature=0.0,
                    output=tuple(
                        ScoredHypothesis(
                            hypothesis_id=h.hypothesis_id,
                            score=Decimal("0.42") if i == 0 else h.confidence,
                            evidence="test",
                        )
                        for i, h in enumerate(hypotheses)
                    ),
                )

        baseline = SymbolicCognitivePipeline()
        configured = SymbolicCognitivePipeline(coordinator=_coordinator_with(RecalibrateProvider()))
        baseline_decision, _, _ = baseline.execute_cycle("rain is falling")
        configured_decision, _, _ = configured.execute_cycle("rain is falling")

        assert configured_decision.id == baseline_decision.id
        assert configured_decision.action_text == baseline_decision.action_text
        assert configured_decision.confidence == Decimal("0.42")
        assert baseline_decision.confidence != Decimal("0.42")

    def test_unsupported_capability_is_never_called(self) -> None:
        provider = GoalSalienceOnlyProvider()
        pipeline = SymbolicCognitivePipeline(coordinator=_coordinator_with(provider))
        baseline = SymbolicCognitivePipeline()

        assert _run_fingerprint(pipeline) == _run_fingerprint(baseline)
        methods = [method for method, _ in provider.calls]
        assert methods == ["rank_rules", "rank_goals"]
        assert len(pipeline.provider_provenance) == 2
        assert all(
            i.capability == ProviderCapabilities.SALIENCE for i in pipeline.provider_provenance
        )


class TestFailFast:
    def test_provider_failure_fails_the_cycle(self) -> None:
        provider = RecordingProvider(raise_on="score_confidence")
        pipeline = SymbolicCognitivePipeline(coordinator=_coordinator_with(provider))
        with pytest.raises(ProviderFailure) as excinfo:
            pipeline.execute_cycle("rain is falling")
        assert excinfo.value.capability == ProviderCapabilities.CALIBRATION
        assert excinfo.value.provider_name == "RecordingProvider"

    def test_provider_failure_propagates_through_runtime(self) -> None:
        provider = RecordingProvider(raise_on="rank_goals")
        runtime = SymbolicRuntime(coordinator=_coordinator_with(provider))
        with pytest.raises(ProviderFailure) as excinfo:
            runtime.process("rain is falling")
        assert excinfo.value.capability == ProviderCapabilities.SALIENCE


class TestRuntimeProvenance:
    def test_runtime_surfaces_provenance_on_result(self, tmp_path: Path) -> None:
        runtime = SymbolicRuntime(
            coordinator=_coordinator_with(RecordingProvider()),
            recorder=TraceRecorder(str(tmp_path / "traces")),
        )
        result = runtime.process("rain is falling")
        assert len(result.provider_provenance) == 5
        assert result.provider_provenance[0].capability == ProviderCapabilities.SALIENCE

    def test_runtime_records_provenance_as_trace_metadata(self, tmp_path: Path) -> None:
        recorder = TraceRecorder(str(tmp_path / "traces"))
        runtime = SymbolicRuntime(
            coordinator=_coordinator_with(RecordingProvider()),
            recorder=recorder,
        )
        runtime.process("rain is falling")
        trace = recorder.load_trace(str(runtime.last_trace_id))
        assert trace is not None
        recorded = trace.metadata[PROVIDER_PROVENANCE_METADATA_KEY]
        assert isinstance(recorded, list)
        assert len(recorded) == 5
        assert all(entry["status"] == "executed" for entry in recorded)
        assert all(entry["summary"] == {"count": 0} for entry in recorded)

    def test_metadata_distinguishes_not_configured_from_configured_empty(
        self, tmp_path: Path
    ) -> None:
        plain_recorder = TraceRecorder(str(tmp_path / "plain"))
        plain = SymbolicRuntime(recorder=plain_recorder)
        plain.process("rain is falling")
        plain_trace = plain_recorder.load_trace(str(plain.last_trace_id))
        assert plain_trace is not None
        assert PROVIDER_PROVENANCE_METADATA_KEY not in plain_trace.metadata

        configured_recorder = TraceRecorder(str(tmp_path / "configured"))
        configured = SymbolicRuntime(
            coordinator=_coordinator_with(RecordingProvider()),
            recorder=configured_recorder,
        )
        configured.process("rain is falling")
        configured_trace = configured_recorder.load_trace(str(configured.last_trace_id))
        assert configured_trace is not None
        assert PROVIDER_PROVENANCE_METADATA_KEY in configured_trace.metadata
        assert len(configured_trace.metadata[PROVIDER_PROVENANCE_METADATA_KEY]) == 5

    def test_runtime_with_null_provider_matches_baseline_fingerprint(self) -> None:
        baseline = SymbolicRuntime()
        null_configured = SymbolicRuntime(coordinator=_coordinator_with(RecordingProvider()))
        baseline_result = baseline.process("rain is falling")
        null_result = null_configured.process("rain is falling")

        assert golden_fingerprint(
            null_result.golden_trace, null_result.response_text
        ) == golden_fingerprint(baseline_result.golden_trace, baseline_result.response_text)
        assert null_result.provider_provenance != ()
        assert baseline_result.provider_provenance == ()
