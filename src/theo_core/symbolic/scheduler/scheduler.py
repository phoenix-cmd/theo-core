"""CognitiveScheduler — budget-constrained stage execution manager."""

from __future__ import annotations

import time
from decimal import Decimal

from theo_core.symbolic.scheduler.models import ComputeBudget, CycleStage, SchedulerTrace


class CognitiveScheduler:
    """Manages cognitive cycle stage sequencing and compute budget limits."""

    def __init__(self, budget: ComputeBudget | None = None) -> None:
        """Initialize scheduler with an optional ComputeBudget."""
        self._budget = budget or ComputeBudget()
        self._executed_stages: list[CycleStage] = []
        self._start_time: float = 0.0

    def start_cycle(self) -> None:
        """Begin a cognitive cycle timer."""
        self._start_time = time.perf_counter()
        self._executed_stages.clear()

    def record_stage(self, stage: CycleStage) -> bool:
        """Record stage execution and check if budget remains.

        Returns:
            True if budget allows continuing, False if budget is exhausted.

        """
        self._executed_stages.append(stage)
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        return Decimal(str(round(elapsed_ms, 3))) <= self._budget.time_budget_ms

    def finalize_trace(self) -> SchedulerTrace:
        """Generate final SchedulerTrace summary."""
        elapsed_ms = Decimal(str(round((time.perf_counter() - self._start_time) * 1000, 3)))
        exhausted = elapsed_ms > self._budget.time_budget_ms
        return SchedulerTrace(
            stages_executed=tuple(self._executed_stages),
            total_time_ms=elapsed_ms,
            budget_exhausted=exhausted,
        )
