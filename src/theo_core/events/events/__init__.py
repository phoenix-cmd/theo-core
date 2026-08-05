"""Domain events — versioned V1 events and domain event definitions.

Events are immutable Pydantic models. They carry data but no behavior.
No subsystem imports another subsystem; they communicate only through events.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------


class DomainEvent(BaseModel, frozen=True):
    """Base class for all domain events.

    Attributes:
        event_id: Unique event identifier.
        timestamp: UTC timestamp of event creation.
        source: Name of the subsystem that emitted this event.
        schema_version: Version string for event schema stability.

    """

    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str = "unknown"
    schema_version: str = "1.0"


# ---------------------------------------------------------------------------
# Core Domain events
# ---------------------------------------------------------------------------


class ConversationStarted(DomainEvent, frozen=True):
    """Emitted when a new conversation session begins."""

    conversation_id: UUID


class MemoryStored(DomainEvent, frozen=True):
    """Emitted when a memory item is stored."""

    memory_id: str
    memory_type: str = "general"


# ---------------------------------------------------------------------------
# System V1 events
# ---------------------------------------------------------------------------


class SubsystemStartedV1(DomainEvent, frozen=True):
    """Emitted when a subsystem finishes initialization."""

    subsystem_name: str
    version: str = "0.2.0"


class SubsystemStoppedV1(DomainEvent, frozen=True):
    """Emitted when a subsystem shuts down."""

    subsystem_name: str
    reason: str = "shutdown"


class SystemReadyV1(DomainEvent, frozen=True):
    """Emitted when the kernel has finished booting all subsystems."""

    subsystem_count: int = 0


class ErrorOccurredV1(DomainEvent, frozen=True):
    """Emitted when an unhandled error occurs in any subsystem."""

    subsystem: str
    error_type: str
    message: str


# ---------------------------------------------------------------------------
# Pipeline Stage V1 events
# ---------------------------------------------------------------------------


class PerceptAnalyzedV1(DomainEvent, frozen=True):
    """Emitted when raw text is normalized into a Percept object."""

    percept_id: UUID
    intent: str
    fact_count: int = 0


class ContextUpdatedV1(DomainEvent, frozen=True):
    """Emitted when active session context is updated."""

    turn_count: int
    active_user: str = "anonymous"


class MemoryRetrievedV1(DomainEvent, frozen=True):
    """Emitted when memory items are retrieved."""

    query: str
    retrieved_count: int = 0


class KnowledgeRetrievedV1(DomainEvent, frozen=True):
    """Emitted when knowledge graph items are retrieved."""

    concept: str
    facts_count: int = 0


class GoalSelectedV1(DomainEvent, frozen=True):
    """Emitted when an active goal is selected by the Goal Engine."""

    goal_id: UUID
    goal_description: str
    priority: str = "medium"


class PlanGeneratedV1(DomainEvent, frozen=True):
    """Emitted when a cognitive plan is generated."""

    plan_id: UUID
    action_count: int = 0


class InferenceCompletedV1(DomainEvent, frozen=True):
    """Emitted when the Cognitive Inference Engine completes strategy evaluation."""

    strategy_name: str = "RuleBasedStrategy"
    confidence: float = 1.0


class ReflectionEvaluatedV1(DomainEvent, frozen=True):
    """Emitted when the Reflection Engine finishes evaluating inference quality."""

    satisfied: bool = True
    confidence: float = 1.0


class DecisionMadeV1(DomainEvent, frozen=True):
    """Emitted when the Decision Engine formulates a Decision object."""

    decision_id: UUID
    response_summary: str
    confidence: float = 1.0


class ResponseGeneratedV1(DomainEvent, frozen=True):
    """Emitted when Response Generator formats final output text."""

    response_length: int
    generator_type: str = "TemplateResponseGenerator"


class MemoryStoredV1(DomainEvent, frozen=True):
    """Emitted when a memory entry is stored or updated."""

    memory_key: str
    category: str = "semantic"


class KnowledgeValidatedV1(DomainEvent, frozen=True):
    """Emitted when a knowledge candidate is validated and committed."""

    subject: str
    predicate: str
    object: str


class TraceRecordedV1(DomainEvent, frozen=True):
    """Emitted when a CognitiveTrace is closed and saved to JSON."""

    trace_id: UUID
    file_path: str
    cognitive_depth: int = 12
