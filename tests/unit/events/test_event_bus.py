"""Tests for the EventBus — the central nervous system."""

from __future__ import annotations

from typing import TYPE_CHECKING

from theo_core.events.events import (
    ConversationStarted,
    DomainEvent,
    MemoryStoredV1,
    SystemReadyV1,
)

if TYPE_CHECKING:
    from theo_core.events.bus import EventBus


class TestEventBus:
    """Tests for EventBus publish/subscribe behavior."""

    def test_subscribe_and_publish(self, event_bus: EventBus) -> None:
        """A subscribed handler should receive the published event."""
        received: list[DomainEvent] = []
        event_bus.subscribe(SystemReadyV1, lambda e: received.append(e))
        event_bus.publish(SystemReadyV1(source="test", subsystem_count=5))
        assert len(received) == 1
        assert isinstance(received[0], SystemReadyV1)

    def test_handler_isolation(self, event_bus: EventBus) -> None:
        """Handlers for one event type should not fire for another."""
        received: list[DomainEvent] = []
        event_bus.subscribe(SystemReadyV1, lambda e: received.append(e))
        event_bus.publish(
            ConversationStarted(
                source="test",
                conversation_id=__import__("uuid").uuid4(),
            )
        )
        assert len(received) == 0

    def test_multiple_handlers(self, event_bus: EventBus) -> None:
        """Multiple handlers for the same event should all fire."""
        counts = {"a": 0, "b": 0}

        def handler_a(_: DomainEvent) -> None:
            counts["a"] += 1

        def handler_b(_: DomainEvent) -> None:
            counts["b"] += 1

        event_bus.subscribe(MemoryStoredV1, handler_a)
        event_bus.subscribe(MemoryStoredV1, handler_b)
        event_bus.publish(MemoryStoredV1(source="test", memory_key="m1"))
        assert counts["a"] == 1
        assert counts["b"] == 1

    def test_handler_error_does_not_stop_others(self, event_bus: EventBus) -> None:
        """A failing handler should not prevent other handlers from running."""
        received: list[str] = []

        def bad_handler(_: DomainEvent) -> None:
            msg = "intentional test failure"
            raise RuntimeError(msg)

        def good_handler(_: DomainEvent) -> None:
            received.append("ok")

        event_bus.subscribe(SystemReadyV1, bad_handler)
        event_bus.subscribe(SystemReadyV1, good_handler)
        event_bus.publish(SystemReadyV1(source="test"))
        assert received == ["ok"]

    def test_event_count(self, event_bus: EventBus) -> None:
        """Event count should track all published events."""
        event_bus.publish(SystemReadyV1(source="test"))
        event_bus.publish(SystemReadyV1(source="test"))
        assert event_bus.event_count == 2

    def test_unsubscribe(self, event_bus: EventBus) -> None:
        """Unsubscribed handlers should no longer receive events."""
        received: list[DomainEvent] = []

        def handler(e: DomainEvent) -> None:
            received.append(e)

        event_bus.subscribe(SystemReadyV1, handler)
        event_bus.unsubscribe(SystemReadyV1, handler)
        event_bus.publish(SystemReadyV1(source="test"))
        assert len(received) == 0

    def test_clear(self, event_bus: EventBus) -> None:
        """Clear should remove all handlers and history."""
        event_bus.subscribe(SystemReadyV1, lambda _: None)
        event_bus.publish(SystemReadyV1(source="test"))
        event_bus.clear()
        assert event_bus.handler_count == 0
        assert event_bus.event_count == 0
