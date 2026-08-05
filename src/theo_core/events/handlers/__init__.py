"""EventHandler — base interface for event handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theo_core.events.events import DomainEvent


class EventHandler(ABC):
    """Abstract base class for event handlers.

    Each handler processes a specific type of domain event.
    Handlers are registered with the EventBus at boot time.
    """

    @abstractmethod
    def handle(self, event: DomainEvent) -> None:
        """Handle a domain event.

        Args:
            event: The domain event to process.

        """

    @property
    @abstractmethod
    def event_type(self) -> type[DomainEvent]:
        """Return the type of event this handler processes.

        Returns:
            The DomainEvent subclass this handler is registered for.

        """
