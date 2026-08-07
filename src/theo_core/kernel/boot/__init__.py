"""Kernel — orchestrates boot sequence and subsystem lifecycle.

The Kernel is the central operating system engine. It executes the boot sequence,
manages subsystem registries, runs health checks, and emits SystemReady.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from theo_core._version import __version__
from theo_core.events.events import SubsystemStartedV1, SystemReadyV1
from theo_core.kernel.registry import SubsystemState

if TYPE_CHECKING:
    from theo_core.events.bus import EventBus
    from theo_core.kernel.lifecycle import LifecycleManager
    from theo_core.kernel.registry import SubsystemRegistry

logger = logging.getLogger(__name__)


class Kernel:
    """THEO Kernel orchestrator.

    Manages subsystem startup sequence, monitors health, and emits boot completion events.
    """

    def __init__(
        self,
        registry: SubsystemRegistry,
        event_bus: EventBus,
        lifecycle: LifecycleManager,
        start_order: list[str] | None = None,
    ) -> None:
        """Initialize the Kernel.

        Args:
            registry: The SubsystemRegistry instance.
            event_bus: The EventBus instance.
            lifecycle: The LifecycleManager instance.
            start_order: Ordered list of subsystem names for boot.

        """
        self._registry = registry
        self._event_bus = event_bus
        self._lifecycle = lifecycle
        self._start_order = start_order or []
        self._is_booted = False

    def boot(self) -> None:
        """Execute the deterministic kernel boot sequence.

        Boots subsystems in start_order sequence, drives each through
        REGISTERED -> STARTING -> RUNNING, and publishes SystemReady when
        complete. Idempotent: calling boot while already booted is a no-op.
        """
        if self._is_booted:
            logger.info("Kernel already booted, ignoring repeated boot() call.")
            return

        logger.info("=== THEO Kernel Boot Sequence ===")

        started_count = 0
        for name in self._start_order:
            subsystem = self._registry.get(name)
            if subsystem is None:
                logger.warning(
                    "Subsystem '%s' in start_order but not registered, skipping.",
                    name,
                )
                continue

            self._registry.transition(name, SubsystemState.STARTING)
            try:
                started = self._lifecycle.start_subsystem(name, subsystem)
            except Exception:
                logger.exception("Failed to start subsystem '%s'", name)
                self._registry.transition(name, SubsystemState.FAILED)
                raise
            if not started:
                self._registry.transition(name, SubsystemState.FAILED)
                logger.error("Subsystem '%s' failed to start.", name)
                continue

            self._registry.transition(name, SubsystemState.RUNNING)
            started_count += 1
            self._event_bus.publish(
                SubsystemStartedV1(
                    source="kernel",
                    subsystem_name=name,
                    version=__version__,
                )
            )

        self._is_booted = True
        logger.info("=== THEO Kernel Ready (%d subsystems started) ===", started_count)
        self._event_bus.publish(
            SystemReadyV1(
                source="kernel",
                subsystem_count=started_count,
            )
        )

    def shutdown(self) -> None:
        """Shutdown all registered subsystems in reverse boot order.

        Drives each active subsystem through RUNNING/STARTING -> STOPPING ->
        STOPPED. Idempotent: calling shutdown while not booted is a no-op.
        """
        if not self._is_booted:
            logger.info("Kernel not booted, ignoring shutdown() call.")
            return

        logger.info("=== THEO Kernel Shutdown Sequence ===")
        for entry in reversed(self._registry.all_entries()):
            if entry.state not in (
                SubsystemState.STARTING,
                SubsystemState.RUNNING,
            ):
                continue
            self._registry.transition(entry.name, SubsystemState.STOPPING)
            try:
                stopped = self._lifecycle.stop_subsystem(entry.name, entry.instance)
            except Exception:
                logger.exception("Failed to stop subsystem '%s'", entry.name)
                self._registry.transition(entry.name, SubsystemState.FAILED)
                continue
            if not stopped:
                logger.error("Subsystem '%s' failed to stop.", entry.name)
                self._registry.transition(entry.name, SubsystemState.FAILED)
                continue
            self._registry.transition(entry.name, SubsystemState.STOPPED)

        self._is_booted = False
        logger.info("=== THEO Kernel Shutdown Complete ===")

    @property
    def is_booted(self) -> bool:
        """Return True if the kernel has completed its boot sequence."""
        return self._is_booted
