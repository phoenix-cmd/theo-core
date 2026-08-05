"""Bootstrap — builds the entire THEO object graph from configuration.

This is the ONLY file in the project that knows about all concrete
implementations. Everything else depends only on abstractions.
"""

from __future__ import annotations

from theo_core.composition.container import TheoContainer
from theo_core.events.bus import EventBus
from theo_core.infrastructure.config import TheoSettings
from theo_core.infrastructure.experiment_tracking import ExperimentTrackerFactory
from theo_core.infrastructure.logging import configure_logging
from theo_core.kernel.boot import Kernel
from theo_core.kernel.lifecycle import LifecycleManager
from theo_core.kernel.registry import SubsystemRegistry


def bootstrap(settings: TheoSettings | None = None) -> TheoContainer:
    """Build and wire the entire THEO cognitive system.

    Args:
        settings: Optional settings override. Uses defaults if not provided.

    Returns:
        A fully wired TheoContainer ready for operation.

    """
    if settings is None:
        settings = TheoSettings()

    # 1. Configure logging first — everything else may log
    configure_logging(
        level=settings.logging.level,
        format_style=settings.logging.format,
    )

    # 2. Create the event bus (central nervous system)
    event_bus = EventBus()

    # 3. Create the experiment tracker
    tracker = ExperimentTrackerFactory.create(
        backend=settings.experiment_tracking.backend,
    )

    # 4. Create the kernel components
    registry = SubsystemRegistry()
    lifecycle = LifecycleManager()

    # 5. Register core subsystems
    registry.register("event_bus", event_bus)
    registry.register("experiment_tracker", tracker)

    # 6. Assemble the kernel
    kernel = Kernel(
        registry=registry,
        event_bus=event_bus,
        lifecycle=lifecycle,
        start_order=settings.kernel.subsystem_start_order,
    )

    # 7. Build and return the container
    return TheoContainer(
        settings=settings,
        event_bus=event_bus,
        kernel=kernel,
        experiment_tracker=tracker,
    )
