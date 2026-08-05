"""Tests for the Kernel boot sequence."""

from __future__ import annotations

from theo_core.events.bus import EventBus
from theo_core.events.events import DomainEvent, SubsystemStartedV1, SystemReadyV1
from theo_core.kernel.boot import Kernel
from theo_core.kernel.lifecycle import LifecycleManager
from theo_core.kernel.registry import SubsystemRegistry, SubsystemState


class StubSubsystem:
    """A test subsystem that tracks start/stop calls."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        """Mark as started."""
        self.started = True

    def stop(self) -> None:
        """Mark as stopped."""
        self.stopped = True


class TestKernel:
    """Tests for the Kernel boot and shutdown sequence."""

    def test_boot_starts_subsystems(self) -> None:
        """Boot should start registered subsystems and emit events."""
        registry = SubsystemRegistry()
        event_bus = EventBus()
        lifecycle = LifecycleManager()

        sub = StubSubsystem()
        registry.register("test_sub", sub)

        events_received: list[DomainEvent] = []
        event_bus.subscribe(SubsystemStartedV1, lambda e: events_received.append(e))
        event_bus.subscribe(SystemReadyV1, lambda e: events_received.append(e))

        kernel = Kernel(registry, event_bus, lifecycle, start_order=["test_sub"])
        kernel.boot()

        assert sub.started
        assert kernel.is_booted
        assert len(events_received) == 2  # SubsystemStartedV1 + SystemReadyV1

    def test_shutdown_stops_subsystems(self) -> None:
        """Shutdown should stop all running subsystems."""
        registry = SubsystemRegistry()
        event_bus = EventBus()
        lifecycle = LifecycleManager()

        sub = StubSubsystem()
        registry.register("test_sub", sub)

        kernel = Kernel(registry, event_bus, lifecycle, start_order=["test_sub"])
        kernel.boot()
        kernel.shutdown()

        assert sub.stopped
        assert not kernel.is_booted

    def test_missing_subsystem_in_start_order(self) -> None:
        """Subsystems in start_order but not registered should be skipped."""
        registry = SubsystemRegistry()
        event_bus = EventBus()
        lifecycle = LifecycleManager()

        kernel = Kernel(registry, event_bus, lifecycle, start_order=["nonexistent"])
        kernel.boot()  # Should not raise
        assert kernel.is_booted


class TestSubsystemRegistry:
    """Tests for the SubsystemRegistry."""

    def test_register_and_get(self) -> None:
        """A registered subsystem should be retrievable by name."""
        registry = SubsystemRegistry()
        sub = StubSubsystem()
        registry.register("my_sub", sub)
        assert registry.get("my_sub") is sub
        assert registry.count == 1

    def test_get_nonexistent(self) -> None:
        """Getting a nonexistent subsystem should return None."""
        registry = SubsystemRegistry()
        assert registry.get("nope") is None

    def test_state_management(self) -> None:
        """State should be updatable for registered subsystems."""
        registry = SubsystemRegistry()
        registry.register("sub", StubSubsystem())
        registry.set_state("sub", SubsystemState.RUNNING)
        entries = registry.all_entries()
        assert entries[0].state == SubsystemState.RUNNING
