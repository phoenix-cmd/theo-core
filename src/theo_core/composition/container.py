"""TheoContainer — holds all live service instances for the THEO system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theo_core.domain.research.ports.experiment_tracker import ExperimentTrackerPort
    from theo_core.events.bus import EventBus
    from theo_core.infrastructure.config import TheoSettings
    from theo_core.kernel.boot import Kernel


@dataclass
class TheoContainer:
    """Holds all live instances for the THEO cognitive system.

    Created once at startup by the bootstrap function. Passed through
    the system — never accessed as a global singleton.

    Attributes:
        settings: The root configuration.
        event_bus: The central event bus.
        kernel: The cognitive kernel.
        experiment_tracker: The active experiment tracking backend.

    """

    settings: TheoSettings
    event_bus: EventBus
    kernel: Kernel
    experiment_tracker: ExperimentTrackerPort
