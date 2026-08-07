"""Integration tests — end-to-end boot and event routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from theo_core.composition.bootstrap import bootstrap
from theo_core.composition.container import TheoContainer
from theo_core.events.events import DomainEvent, SystemReadyV1
from theo_core.infrastructure.config import TheoSettings

if TYPE_CHECKING:
    from pathlib import Path


def _bootstrap(tmp_path: Path, settings: TheoSettings | None = None) -> TheoContainer:
    """Bootstrap a container with all state paths under tmp_path."""
    return bootstrap(
        settings,
        memory_file=str(tmp_path / "memory_store.json"),
        knowledge_file=str(tmp_path / "knowledge_graph.json"),
        trace_dir=str(tmp_path / "traces"),
        state_file=str(tmp_path / "symbolic_state.json"),
    )


@pytest.mark.integration
class TestBootSequence:
    """Integration tests for the full boot sequence."""

    def test_bootstrap_creates_container(self, tmp_path: Path) -> None:
        """Bootstrap should create a fully wired container."""
        container = _bootstrap(tmp_path)
        assert isinstance(container, TheoContainer)
        assert container.settings is not None
        assert container.event_bus is not None
        assert container.kernel is not None

    def test_kernel_boots_successfully(self, tmp_path: Path) -> None:
        """The kernel should boot and emit SystemReadyV1."""
        settings = TheoSettings(
            logging={"level": "WARNING", "format": "console"},  # type: ignore[arg-type]
        )
        container = _bootstrap(tmp_path, settings)

        ready_events: list[DomainEvent] = []
        container.event_bus.subscribe(SystemReadyV1, lambda e: ready_events.append(e))

        container.kernel.boot()

        assert container.kernel.is_booted
        assert len(ready_events) == 1

    def test_experiment_tracker_lifecycle(self, tmp_path: Path) -> None:
        """The experiment tracker should complete a full lifecycle."""
        container = _bootstrap(tmp_path)
        tracker = container.experiment_tracker
        run_id = tracker.start_run("integration-test", {"epochs": 1})
        tracker.log_metric("loss", 0.42, step=0)
        tracker.log_param("lr", 0.001)
        tracker.end_run()
        assert run_id is not None

    def test_kernel_shutdown(self, tmp_path: Path) -> None:
        """The kernel should shut down cleanly."""
        container = _bootstrap(tmp_path)
        container.kernel.boot()
        container.kernel.shutdown()
        assert not container.kernel.is_booted

    def test_symbolic_runtime_boots_and_processes(self, tmp_path: Path) -> None:
        """The symbolic runtime should boot, process, and persist on shutdown."""
        container = _bootstrap(tmp_path)
        container.kernel.boot()

        result = container.symbolic_runtime.process("Hello Theo")

        assert result.response_text
        assert result.golden_trace.response_text == result.response_text
        assert result.decision.referenced_goal.value.startswith("goal://")

        container.kernel.shutdown()
        assert (tmp_path / "symbolic_state.json").exists()
