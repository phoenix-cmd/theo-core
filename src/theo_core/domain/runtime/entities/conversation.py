"""Conversation — an ordered, append-only sequence of messages.

A Conversation is the primary aggregate for dialogue. It maintains
temporal ordering and provides methods for appending new messages.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from theo_core.domain.runtime.entities.message import Message  # noqa: TC001


class Conversation(BaseModel):
    """An ordered conversation consisting of messages.

    Conversations are append-only: messages cannot be edited or removed
    once added. This preserves the integrity of the dialogue history.

    Attributes:
        id: Unique conversation identifier.
        messages: Ordered list of messages in this conversation.
        created_at: UTC timestamp of conversation creation.
        metadata: Extensible metadata dictionary.

    """

    id: UUID = Field(default_factory=uuid4)
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def append(self, message: Message) -> None:
        """Append a message to this conversation.

        Args:
            message: The message to append.

        """
        self.messages.append(message)

    @property
    def message_count(self) -> int:
        """Return the number of messages in this conversation."""
        return len(self.messages)

    @property
    def last_message(self) -> Message | None:
        """Return the most recent message, or None if empty."""
        return self.messages[-1] if self.messages else None
