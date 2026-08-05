"""EventBus — synchronous in-process publish/subscribe event bus.

The EventBus is Theo's central nervous system. All cross-subsystem
communication flows through it. No subsystem directly imports another;
they emit and consume events only.

v0.1: Synchronous, in-process. Async adapter planned for v0.2.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable

from theo_core.events.events import DomainEvent

logger = logging.getLogger(__name__)

# Type alias for handler callables
EventHandlerFunc = Callable[[DomainEvent], None]


class EventBus:
    """Synchronous publish/subscribe event bus.

    Handlers are registered per event type. When an event is published,
    all registered handlers for that event type are invoked in order.

    Thread-safety note: v0.1 is single-threaded. Future versions will
    add locking or switch to an async implementation.
    """

    def __init__(self) -> None:
        """Initialize the event bus with an empty handler registry."""
        self._handlers: dict[type[DomainEvent], list[EventHandlerFunc]] = defaultdict(list)
        self._event_log: list[DomainEvent] = []

    def subscribe(
        self,
        event_type: type[DomainEvent],
        handler: EventHandlerFunc,
    ) -> None:
        """Register a handler for a specific event type.

        Args:
            event_type: The type of event to listen for.
            handler: The callable to invoke when the event is published.

        """
        self._handlers[event_type].append(handler)
        logger.debug(
            "Handler registered for %s: %s",
            event_type.__name__,
            getattr(handler, "__name__", repr(handler)),
        )

    def unsubscribe(
        self,
        event_type: type[DomainEvent],
        handler: EventHandlerFunc,
    ) -> None:
        """Remove a handler for a specific event type.

        Args:
            event_type: The event type to unsubscribe from.
            handler: The handler to remove.

        """
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def publish(self, event: DomainEvent) -> None:
        """Publish an event to all registered handlers.

        Handlers are invoked synchronously in registration order.
        Errors in handlers are logged but do not stop other handlers.

        Args:
            event: The domain event to publish.

        """
        self._event_log.append(event)
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        logger.info(
            "Publishing %s (id=%s, handlers=%d)",
            event_type.__name__,
            event.event_id,
            len(handlers),
        )

        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Handler %s failed for event %s",
                    getattr(handler, "__name__", repr(handler)),
                    event_type.__name__,
                )

    @property
    def event_count(self) -> int:
        """Return the total number of events published."""
        return len(self._event_log)

    @property
    def handler_count(self) -> int:
        """Return the total number of registered handlers."""
        return sum(len(handlers) for handlers in self._handlers.values())

    def clear(self) -> None:
        """Clear all handlers and event history."""
        self._handlers.clear()
        self._event_log.clear()
