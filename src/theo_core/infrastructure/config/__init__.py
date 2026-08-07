"""Configuration — Pydantic settings models for all THEO subsystems.

These settings are populated by the Hydra config loader at boot time.
All configurable values live here. No magic numbers anywhere.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoggingSettings(BaseModel):
    """Logging subsystem configuration."""

    level: str = "INFO"
    format: str = "json"


class MemorySettings(BaseModel):
    """Memory subsystem configuration."""

    working_memory_capacity: int = 100
    consolidation_interval_seconds: int = 3600
    forgetting_enabled: bool = True
    default_backend: str = "in_memory"


class KnowledgeSettings(BaseModel):
    """Knowledge subsystem configuration."""

    graph_backend: str = "in_memory"
    max_traversal_depth: int = 5


class IdentitySettings(BaseModel):
    """Identity subsystem configuration."""

    persona_name: str = "Theo"
    consistency_check_enabled: bool = True


class GoalSettings(BaseModel):
    """Goal subsystem configuration."""

    max_active_goals: int = 10
    default_priority: str = "medium"


class ModelSettings(BaseModel):
    """Language model configuration."""

    backend: str = "stub"
    device: str = "cpu"
    dtype: str = "float32"
    max_sequence_length: int = 2048


class TrainingSettings(BaseModel):
    """Training pipeline configuration."""

    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 10
    checkpoint_interval_steps: int = 1000
    checkpoint_dir: str = "checkpoints"


class ExperimentTrackingSettings(BaseModel):
    """Experiment tracking configuration."""

    backend: str = "noop"
    tracking_uri: str = ""
    project_name: str = "theo"


class SchedulerSettings(BaseModel):
    """Background scheduler configuration."""

    enabled: bool = True
    timezone: str = "UTC"


class SecuritySettings(BaseModel):
    """Security subsystem configuration."""

    sandbox_enabled: bool = False
    secret_backend: str = "env"  # noqa: S105


class TelemetrySettings(BaseModel):
    """Telemetry subsystem configuration."""

    enabled: bool = True
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    health_check_interval_seconds: int = 60


class EvaluationSettings(BaseModel):
    """Evaluation framework configuration."""

    default_benchmarks: list[str] = Field(default_factory=list)
    reports_dir: str = "reports"


class DatasetSettings(BaseModel):
    """Dataset pipeline configuration."""

    data_dir: str = "data"
    default_format: str = "json"


class KernelSettings(BaseModel):
    """Kernel configuration."""

    boot_timeout_seconds: int = 30
    subsystem_start_order: list[str] = Field(
        default_factory=lambda: [
            "event_bus",
            "experiment_tracker",
            "perception",
            "context_manager",
            "memory_engine",
            "memory_classifier",
            "knowledge_engine",
            "goal_manager",
            "planner",
            "inference_engine",
            "response_generator",
            "trace_recorder",
            "explain_engine",
            "replay_engine",
            "cognitive_engine",
            "symbolic_pipeline",
        ]
    )


class TheoSettings(BaseModel):
    """Root configuration for the entire THEO cognitive system.

    All subsystem settings are nested here. This is the single source
    of truth populated by the Hydra config loader.
    """

    kernel: KernelSettings = Field(default_factory=KernelSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    knowledge: KnowledgeSettings = Field(default_factory=KnowledgeSettings)
    identity: IdentitySettings = Field(default_factory=IdentitySettings)
    goals: GoalSettings = Field(default_factory=GoalSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    experiment_tracking: ExperimentTrackingSettings = Field(
        default_factory=ExperimentTrackingSettings
    )
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    dataset: DatasetSettings = Field(default_factory=DatasetSettings)
