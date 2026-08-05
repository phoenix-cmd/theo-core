"""Boot — the kernel boot sequence orchestrator.

Responsible for initializing all subsystems in deterministic order,
firing lifecycle hooks, and emitting the SystemReady event.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from theo_core.events.events import SubsystemStarted, SystemReady
from theo_core.kernel.registry import SubsystemRegistry, SubsystemState

if TYPE_CHECKING:
    from theo_core.events.bus import EventBus
    from theo_core.kernel.lifecycle import LifecycleManager

logger = logging.getLogger(__name__)


class Kernel:
    """The THEO cognitive kernel — orchestrates boot and lifecycle.

    The kernel is the entry point for the entire system. It manages
    the deterministic boot sequence, subsystem registration, and
    event routing.
    """

    def __init__(
        self,
        registry: SubsystemRegistry,
        event_bus: EventBus,
        lifecycle: LifecycleManager,
        start_order: list[str] | None = None,
    ) -> None:
        """Initialize the kernel.

        Args:
            registry: The subsystem registry.
            event_bus: The central event bus.
            lifecycle: The lifecycle manager.
            start_order: Optional ordered list of subsystem names to start.

        """
        self._registry = registry
        self._event_bus = event_bus
        self._lifecycle = lifecycle
        self._start_order = start_order or []
        self._booted = False

    def boot(self) -> None:
        """Execute the full boot sequence.

        Starts all registered subsystems in the configured order,
        fires SubsystemStarted events for each, and emits SystemReady
        when complete.
        """
        logger.info("=== THEO Kernel Boot Sequence ===")

        # Start subsystems in configured order
        started_count = 0
        for name in self._start_order:
            entry_instance = self._registry.get(name)
            if entry_instance is None:
                logger.warning("Subsystem '%s' in start_order but not registered, skipping.", name)
                continue

            self._registry.set_state(name, SubsystemState.STARTING)
            success = self._lifecycle.start_subsystem(name, entry_instance)

            if success:
                self._registry.set_state(name, SubsystemState.RUNNING)
                self._event_bus.publish(SubsystemStarted(source="kernel", subsystem_name=name))
                started_count += 1
            else:
                self._registry.set_state(name, SubsystemState.FAILED)
                logger.error("Subsystem '%s' failed to start.", name)

        # Start any remaining subsystems not in the explicit start_order
        for entry in self._registry.all_entries():
            if entry.state == SubsystemState.REGISTERED:
                self._registry.set_state(entry.name, SubsystemState.STARTING)
                success = self._lifecycle.start_subsystem(entry.name, entry.instance)
                if success:
                    self._registry.set_state(entry.name, SubsystemState.RUNNING)
                    self._event_bus.publish(
                        SubsystemStarted(source="kernel", subsystem_name=entry.name)
                    )
                    started_count += 1
                else:
                    self._registry.set_state(entry.name, SubsystemState.FAILED)

        self._booted = True
        self._event_bus.publish(SystemReady(source="kernel", subsystem_count=started_count))
        logger.info(
            "=== THEO Kernel Ready (%d subsystems started) ===",
            started_count,
        )

    def shutdown(self) -> None:
        """Shut down all subsystems in reverse order."""
        logger.info("=== THEO Kernel Shutdown ===")
        for entry in reversed(self._registry.all_entries()):
            if entry.state == SubsystemState.RUNNING:
                self._registry.set_state(entry.name, SubsystemState.STOPPING)
                self._lifecycle.stop_subsystem(entry.name, entry.instance)
                self._registry.set_state(entry.name, SubsystemState.STOPPED)
        self._booted = False
        logger.info("=== THEO Kernel Stopped ===")

    @property
    def is_booted(self) -> bool:
        """Return True if the kernel has completed booting."""
        return self._booted

    @property
    def registry(self) -> SubsystemRegistry:
        """Return the subsystem registry."""
        return self._registry

    @property
    def event_bus(self) -> EventBus:
        """Return the event bus."""
        return self._event_bus
