"""State machine — kernel boot/shutdown lifecycle behavior.

Boot drives subsystems REGISTERED -> STARTING -> RUNNING; shutdown drives
RUNNING -> STOPPING -> STOPPED. Boot and shutdown are idempotent, and start
failures MUST surface a FAILED state.
"""

from __future__ import annotations

from theo_core.events.bus import EventBus
from theo_core.events.events import DomainEvent, SubsystemStartedV1, SystemReadyV1
from theo_core.kernel.boot import Kernel
from theo_core.kernel.lifecycle import LifecycleManager
from theo_core.kernel.registry import SubsystemRegistry, SubsystemState


class FlakySubsystem:
    """Subsystem that can be configured to fail on start or stop."""

    def __init__(self, fail_start: bool = False, fail_stop: bool = False) -> None:
        """Configure failure modes."""
        self.started = 0
        self.stopped = 0
        self._fail_start = fail_start
        self._fail_stop = fail_stop

    def start(self) -> None:
        """Count starts, raising when configured to fail."""
        self.started += 1
        if self._fail_start:
            raise RuntimeError("start failed")

    def stop(self) -> None:
        """Count stops, raising when configured to fail."""
        self.stopped += 1
        if self._fail_stop:
            raise RuntimeError("stop failed")


def _kernel_with_sub(
    name: str, instance: FlakySubsystem
) -> tuple[Kernel, SubsystemRegistry, EventBus]:
    registry = SubsystemRegistry()
    registry.register(name, instance)
    event_bus = EventBus()
    kernel = Kernel(registry, event_bus, LifecycleManager(), start_order=[name])
    return kernel, registry, event_bus


class TestKernelBootTransitions:
    def test_boot_drives_starting_then_running(self) -> None:
        kernel, registry, _ = _kernel_with_sub("sub", FlakySubsystem())
        kernel.boot()
        assert registry.all_entries()[0].state == SubsystemState.RUNNING
        assert kernel.is_booted

    def test_double_boot_is_noop(self) -> None:
        kernel, registry, event_bus = _kernel_with_sub("sub", FlakySubsystem())
        ready_events: list[DomainEvent] = []
        event_bus.subscribe(SystemReadyV1, lambda e: ready_events.append(e))
        sub = registry.get("sub")

        kernel.boot()
        kernel.boot()

        assert sub is not None
        assert sub.started == 1
        assert len(ready_events) == 1

    def test_shutdown_drives_stopped(self) -> None:
        kernel, registry, _ = _kernel_with_sub("sub", FlakySubsystem())
        kernel.boot()
        kernel.shutdown()
        assert registry.all_entries()[0].state == SubsystemState.STOPPED
        assert not kernel.is_booted

    def test_shutdown_before_boot_is_noop(self) -> None:
        kernel, registry, _ = _kernel_with_sub("sub", FlakySubsystem())
        sub = registry.get("sub")
        kernel.shutdown()
        assert sub is not None
        assert sub.stopped == 0
        assert not kernel.is_booted

    def test_double_shutdown_is_noop(self) -> None:
        kernel, registry, _ = _kernel_with_sub("sub", FlakySubsystem())
        sub = registry.get("sub")
        kernel.boot()
        kernel.shutdown()
        kernel.shutdown()
        assert sub is not None
        assert sub.stopped == 1


class TestKernelFailurePaths:
    def test_start_failure_marks_failed(self) -> None:
        kernel, registry, event_bus = _kernel_with_sub(
            "sub", FlakySubsystem(fail_start=True)
        )
        started_events: list[DomainEvent] = []
        event_bus.subscribe(
            SubsystemStartedV1, lambda e: started_events.append(e)
        )

        kernel.boot()

        assert registry.all_entries()[0].state == SubsystemState.FAILED
        assert not started_events

    def test_stop_failure_marks_failed(self) -> None:
        kernel, registry, _ = _kernel_with_sub(
            "sub", FlakySubsystem(fail_stop=True)
        )
        kernel.boot()
        kernel.shutdown()
        assert registry.all_entries()[0].state == SubsystemState.FAILED
        assert not kernel.is_booted

    def test_failed_subsystem_skipped_on_shutdown(self) -> None:
        registry = SubsystemRegistry()
        registry.register("good", FlakySubsystem())
        registry.register("bad", FlakySubsystem(fail_start=True))
        event_bus = EventBus()
        kernel = Kernel(
            registry,
            event_bus,
            LifecycleManager(),
            start_order=["good", "bad"],
        )

        kernel.boot()
        kernel.shutdown()

        states = {e.name: e.state for e in registry.all_entries()}
        assert states["good"] == SubsystemState.STOPPED
        assert states["bad"] == SubsystemState.FAILED
