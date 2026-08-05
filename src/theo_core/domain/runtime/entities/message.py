"""Message — an immutable unit of dialogue within a conversation.

A Message represents a single utterance from a participant (user, system,
or Theo itself). Messages are append-only building blocks of a Conversation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MessageRole(StrEnum):
    """The role of a message participant."""

    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel, frozen=True):
    """An immutable message within a conversation.

    Attributes:
        id: Unique identifier for this message.
        role: The participant role that produced this message.
        content: The textual content of the message.
        timestamp: UTC timestamp of message creation.
        metadata: Extensible metadata dictionary.

    """

    id: UUID = Field(default_factory=uuid4)
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
