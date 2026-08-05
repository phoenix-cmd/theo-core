"""Kernel — orchestrates boot sequence and subsystem lifecycle.

The Kernel is the central operating system engine. It executes the boot sequence,
manages subsystem registries, runs health checks, and emits SystemReady.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from theo_core.events.events import SubsystemStartedV1, SystemReadyV1

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

        Boots subsystems in start_order sequence, sets states to RUNNING,
        and publishes SystemReady event when complete.
        """
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

            try:
                self._lifecycle.start_subsystem(name, subsystem)
                started_count += 1
                self._event_bus.publish(
                    SubsystemStartedV1(
                        source="kernel",
                        subsystem_name=name,
                        version="0.2.0",
                    )
                )
            except Exception:
                logger.exception("Failed to start subsystem '%s'", name)
                raise

        self._is_booted = True
        logger.info("=== THEO Kernel Ready (%d subsystems started) ===", started_count)
        self._event_bus.publish(
            SystemReadyV1(
                source="kernel",
                subsystem_count=started_count,
            )
        )

    def shutdown(self) -> None:
        """Shutdown all registered subsystems in reverse boot order."""
        logger.info("=== THEO Kernel Shutdown Sequence ===")
        for entry in reversed(self._registry.all_entries()):
            try:
                self._lifecycle.stop_subsystem(entry.name, entry.instance)
            except Exception:
                logger.exception("Failed to stop subsystem '%s'", entry.name)

        self._is_booted = False
        logger.info("=== THEO Kernel Shutdown Complete ===")

    @property
    def is_booted(self) -> bool:
        """Return True if the kernel has completed its boot sequence."""
        return self._is_booted
