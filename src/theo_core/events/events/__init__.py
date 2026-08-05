"""Domain events — all events that flow through the event bus.

Events are immutable Pydantic models. They carry data but no behavior.
No subsystem imports another subsystem; they communicate only through events.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
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

    """

    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str = "unknown"


# ---------------------------------------------------------------------------
# System events
# ---------------------------------------------------------------------------


class SubsystemStarted(DomainEvent, frozen=True):
    """Emitted when a subsystem finishes initialization."""

    subsystem_name: str
    version: str = "0.1.0"


class SubsystemStopped(DomainEvent, frozen=True):
    """Emitted when a subsystem shuts down."""

    subsystem_name: str
    reason: str = "shutdown"


class SystemReady(DomainEvent, frozen=True):
    """Emitted when the kernel has finished booting all subsystems."""

    subsystem_count: int = 0


class ErrorOccurred(DomainEvent, frozen=True):
    """Emitted when an unhandled error occurs in any subsystem."""

    subsystem: str
    error_type: str
    message: str


class HealthCheckRequested(DomainEvent, frozen=True):
    """Emitted to request a health report from a subsystem."""

    subsystem_name: str


class PluginLoaded(DomainEvent, frozen=True):
    """Emitted when a plugin is successfully loaded."""

    plugin_name: str
    version: str = "0.1.0"


# ---------------------------------------------------------------------------
# Cognitive events
# ---------------------------------------------------------------------------


class ConversationStarted(DomainEvent, frozen=True):
    """Emitted when a new conversation begins."""

    conversation_id: UUID


class MessageReceived(DomainEvent, frozen=True):
    """Emitted when a new message is added to a conversation."""

    conversation_id: UUID
    message_id: UUID
    role: str


class ThoughtGenerated(DomainEvent, frozen=True):
    """Emitted when a new thought is produced during reasoning."""

    thought_id: UUID
    confidence: float = 0.5


class ReflectionCompleted(DomainEvent, frozen=True):
    """Emitted when a self-reflection cycle finishes."""

    reflection_id: UUID
    insights_count: int = 0


class PlanCreated(DomainEvent, frozen=True):
    """Emitted when a new plan is generated."""

    plan_id: UUID
    goal: str
    action_count: int = 0


class ActionExecuted(DomainEvent, frozen=True):
    """Emitted when an action within a plan is executed."""

    action_id: UUID
    capability: str
    result_summary: str = ""


# ---------------------------------------------------------------------------
# Memory events
# ---------------------------------------------------------------------------


class MemoryStored(DomainEvent, frozen=True):
    """Emitted when a new memory entry is stored."""

    memory_id: str
    memory_type: str = "general"


class MemoryRetrieved(DomainEvent, frozen=True):
    """Emitted when a memory entry is retrieved."""

    query_id: UUID = Field(default_factory=uuid4)
    memory_id: str
    similarity_score: float = 0.0


class MemoryConsolidated(DomainEvent, frozen=True):
    """Emitted when multiple memories are consolidated."""

    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    target_id: str = ""


class MemoryForgotten(DomainEvent, frozen=True):
    """Emitted when a memory is pruned or expired."""

    memory_id: str
    reason: str = "decay"


# ---------------------------------------------------------------------------
# Knowledge events
# ---------------------------------------------------------------------------


class FactAdded(DomainEvent, frozen=True):
    """Emitted when a new fact is added to the knowledge graph."""

    fact_id: str
    subject: str
    predicate: str
    obj: str


class RelationshipCreated(DomainEvent, frozen=True):
    """Emitted when a new relationship is created in the knowledge graph."""

    source_id: str
    target_id: str
    relation_type: str


# ---------------------------------------------------------------------------
# Research events
# ---------------------------------------------------------------------------


class ExperimentStarted(DomainEvent, frozen=True):
    """Emitted when an experiment begins."""

    experiment_id: UUID
    config_hash: str = ""


class MetricLogged(DomainEvent, frozen=True):
    """Emitted when a metric is logged during a training run."""

    run_id: UUID
    metric_name: str
    value: float
    step: int = 0


class CheckpointSaved(DomainEvent, frozen=True):
    """Emitted when a model checkpoint is saved."""

    checkpoint_id: UUID
    model_version: str = "0.1.0"


class EvaluationCompleted(DomainEvent, frozen=True):
    """Emitted when an evaluation run completes."""

    evaluation_id: UUID
    benchmark: str
    score: float = 0.0


class DatasetRegistered(DomainEvent, frozen=True):
    """Emitted when a dataset is registered in the registry."""

    dataset_id: UUID
    version: str
    checksum: str = ""


# ---------------------------------------------------------------------------
# Goal events
# ---------------------------------------------------------------------------


class GoalActivated(DomainEvent, frozen=True):
    """Emitted when a goal is added to the active stack."""

    goal_id: UUID
    description: str
    priority: str = "medium"


class GoalCompleted(DomainEvent, frozen=True):
    """Emitted when a goal is marked as completed."""

    goal_id: UUID
    metadata: dict[str, Any] = Field(default_factory=dict)
