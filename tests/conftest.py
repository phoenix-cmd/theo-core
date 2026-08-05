"""THEO test suite — shared fixtures and configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from theo_core.composition.bootstrap import bootstrap
from theo_core.events.bus import EventBus
from theo_core.infrastructure.config import TheoSettings

if TYPE_CHECKING:
    from theo_core.composition.container import TheoContainer


@pytest.fixture
def settings() -> TheoSettings:
    """Create default test settings with quiet logging."""
    return TheoSettings(
        logging={"level": "WARNING", "format": "console"},  # type: ignore[arg-type]
        experiment_tracking={"backend": "noop"},  # type: ignore[arg-type]
    )


@pytest.fixture
def event_bus() -> EventBus:
    """Create a fresh event bus for testing."""
    return EventBus()


@pytest.fixture
def container(settings: TheoSettings) -> TheoContainer:
    """Create a fully wired TheoContainer for integration tests."""
    return bootstrap(settings)
