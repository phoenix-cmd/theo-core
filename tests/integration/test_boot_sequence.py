"""Integration tests — end-to-end boot and event routing."""

from __future__ import annotations

import pytest

from theo_core.composition.bootstrap import bootstrap
from theo_core.composition.container import TheoContainer
from theo_core.events.events import DomainEvent, SystemReadyV1
from theo_core.infrastructure.config import TheoSettings


@pytest.mark.integration
class TestBootSequence:
    """Integration tests for the full boot sequence."""

    def test_bootstrap_creates_container(self) -> None:
        """Bootstrap should create a fully wired container."""
        container = bootstrap()
        assert isinstance(container, TheoContainer)
        assert container.settings is not None
        assert container.event_bus is not None
        assert container.kernel is not None

    def test_kernel_boots_successfully(self) -> None:
        """The kernel should boot and emit SystemReadyV1."""
        settings = TheoSettings(
            logging={"level": "WARNING", "format": "console"},  # type: ignore[arg-type]
        )
        container = bootstrap(settings)

        ready_events: list[DomainEvent] = []
        container.event_bus.subscribe(SystemReadyV1, lambda e: ready_events.append(e))

        container.kernel.boot()

        assert container.kernel.is_booted
        assert len(ready_events) == 1

    def test_experiment_tracker_lifecycle(self) -> None:
        """The experiment tracker should complete a full lifecycle."""
        container = bootstrap()
        tracker = container.experiment_tracker
        run_id = tracker.start_run("integration-test", {"epochs": 1})
        tracker.log_metric("loss", 0.42, step=0)
        tracker.log_param("lr", 0.001)
        tracker.end_run()
        assert run_id is not None

    def test_kernel_shutdown(self) -> None:
        """The kernel should shut down cleanly."""
        container = bootstrap()
        container.kernel.boot()
        container.kernel.shutdown()
        assert not container.kernel.is_booted
