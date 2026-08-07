"""Unit tests for CognitiveScheduler."""

from decimal import Decimal

from theo_core.symbolic.scheduler.models import ComputeBudget, CycleStage
from theo_core.symbolic.scheduler.scheduler import CognitiveScheduler


class TestCognitiveScheduler:
    def test_scheduler_stage_tracking_and_budget(self) -> None:
        budget = ComputeBudget(time_budget_ms=Decimal("500.0"))
        scheduler = CognitiveScheduler(budget)
        scheduler.start_cycle()

        allowed = scheduler.record_stage(CycleStage.PERCEPTION)
        assert allowed is True

        allowed = scheduler.record_stage(CycleStage.DECISION)
        assert allowed is True

        trace = scheduler.finalize_trace()
        assert len(trace.stages_executed) == 2
        assert CycleStage.PERCEPTION in trace.stages_executed
        assert CycleStage.DECISION in trace.stages_executed
        assert trace.budget_exhausted is False

    def test_unbudgeted_scheduler_is_timing_free_and_deterministic(self) -> None:
        scheduler = CognitiveScheduler()  # budget=None
        scheduler.start_cycle()

        assert scheduler.record_stage(CycleStage.PERCEPTION) is True
        assert scheduler.record_stage(CycleStage.DECISION) is True

        trace = scheduler.finalize_trace()
        assert trace.total_time_ms is None
        assert trace.budget_exhausted is False
        assert tuple(trace.stages_executed) == (
            CycleStage.PERCEPTION,
            CycleStage.DECISION,
        )

        second_trace = CognitiveScheduler().finalize_trace()
        assert second_trace.total_time_ms is None
        assert second_trace.budget_exhausted is False
