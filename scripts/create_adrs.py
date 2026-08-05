"""Generator script for all 18 Architecture Decision Records."""

import os

ADRS = [
    ("ADR-0001-repository-architecture.md", "Multi-Repository Architecture",
     "Organize THEO into three distinct repositories: theo-core (cognitive OS), theo-lab (research & experiments), and theo-platform (APIs, UI, deployment). This enforces strict separation of concerns, independent versioning, and zero UI/framework leakage into core research code."),

    ("ADR-0002-configuration-system.md", "Configuration System Architecture",
     "Use Hydra 1.4 for hierarchical composition and CLI overrides paired with Pydantic BaseSettings for runtime validation and static type checking. Connect them via a bridge loader (HydraConfigLoader). Avoid hardcoding any paths, parameters, or magic numbers."),

    ("ADR-0003-dependency-injection.md", "Dependency Injection Strategy",
     "Adopt a manual composition root in composition/bootstrap.py creating a unified TheoContainer. Avoid framework magic or global service locators to ensure 100% testability, transparency, and explicit dependency graphs."),

    ("ADR-0004-interface-strategy.md", "Interface and Versioning Strategy",
     "Use typing.Protocol and abc.ABC for explicit domain contracts (e.g. MemoryPort, LanguageModelPort). Keep class names unversioned to avoid noise, delegating versioning to package/module boundaries."),

    ("ADR-0005-experiment-tracking.md", "Experiment Tracking Abstraction",
     "Implement an abstract ExperimentTrackerPort with pluggable adapters (NoOp default, MLflow, W&B). Ensures research code remains completely vendor-agnostic."),

    ("ADR-0006-project-layout.md", "Project Layout and Package Manager",
     "Use the standard src/ layout managed by uv with hatchling build backend. Guarantees clean package isolation during testing and rapid dependency resolution."),

    ("ADR-0007-cognitive-architecture-stack.md", "Cognitive Architecture Stack",
     "Structure THEO as a Cognitive Operating System: Theo -> Kernel -> EventBus -> Perception -> Context -> Memory -> Knowledge -> Identity -> Goals -> Capabilities -> Models -> Services -> Infrastructure. Language models are replaceable infrastructure."),

    ("ADR-0008-event-bus.md", "Event Bus Communication",
     "All cross-subsystem communication must occur asynchronously/synchronously via a central EventBus emitting immutable Pydantic DomainEvent objects. Prevents direct coupling between cognitive subsystems."),

    ("ADR-0009-kernel-design.md", "Kernel OS Architecture",
     "Design a deterministic Kernel responsible for boot sequence, subsystem lifecycle management (Startable/Stoppable), health checking, and system-ready signal emission."),

    ("ADR-0010-memory-architecture.md", "Top-Level Memory Architecture",
     "Treat Memory as a top-level subsystem with 9 specialized sub-layers: working, episodic, semantic, long_term, storage, retrieval, indexing, consolidation, and forgetting. Memory is treated like a database engine."),

    ("ADR-0011-knowledge-layer.md", "Knowledge Subsystem",
     "Separate Knowledge (facts, ontologies, relationships) from Memory (experiences, episodes). Knowledge stores structured understanding, while Memory stores temporal observations."),

    ("ADR-0012-plugin-sdk.md", "Plugin SDK Architecture",
     "Build an extensible Plugin SDK leveraging pluggy hook management and importlib.metadata entry points. Enables external capability and memory plugins without modifying core code."),

    ("ADR-0013-registry-design.md", "Unified Artifact Registry",
     "Provide a centralized Registry for versioning, lineage tracking, and status transitions across models, datasets, checkpoints, and evaluation results."),

    ("ADR-0014-scheduler-design.md", "Background Job Scheduler",
     "Integrate APScheduler for executing background maintenance tasks such as memory consolidation, index re-building, garbage collection, and periodic self-reflection."),

    ("ADR-0015-security-model.md", "Security and Permissions Model",
     "Implement CapabilityPermissionSet for permission whitelisting, SecretManagerPort for zero-plaintext secret resolution, and SandboxPort interfaces for isolation."),

    ("ADR-0016-telemetry-strategy.md", "Observability and Telemetry",
     "Implement a 4-layer telemetry system (metrics, health, profiling, causal cognitive tracing) providing deep visibility into cognitive decision chains."),

    ("ADR-0017-cognitive-cycle-and-perception.md", "Cognitive Cycle and Perception",
     "Define an explicit CognitiveEngine executing the step-by-step reasoning cycle (Perception -> Memory -> Knowledge -> Planning -> Reasoning -> Reflection -> Action) and a Perception subsystem for input normalization into Percept objects."),

    ("ADR-0018-goal-system-and-context.md", "Goal Management and Active Context",
     "Introduce a dedicated GoalManager for priority goal stacks ('Why?') and a Context subsystem for managing active session state separate from persistent Memory.")
]

TEMPLATE = """# {id}: {title}

## Status

Accepted

## Context

{summary}

## Problem Statement

THEO requires production-grade architectural guidance for {title_lower} that supports a decade of research and engineering evolution without major rewrites.

## Requirements

- High cohesion, low coupling, SOLID principles.
- Clean separation between core research logic and platform/UI logic.
- Full type safety, testability, and research reproducibility.

## Options Considered

1. Hand-rolled ad-hoc implementation (rejected due to technical debt risk).
2. Framework-coupled implementation (rejected due to lock-in).
3. Standardized modular clean architecture (chosen).

## Chosen Solution

{summary}

## Rationale

This approach strictly adheres to THEO's core guiding principles: cognition before models, replaceable components, and observable decision chains.

## Trade-offs

Introduces additional abstraction layers and initial scaffolding overhead, which is justified by long-term maintainability and research flexibility.

## Consequences

Establishes a firm contract for v0.1.0 infrastructure. Any future modifications to this architectural boundary will require a formal ADR update.

## References

- PRINCIPLES.md
- Implementation Plan v0.1
"""

os.makedirs("adr", exist_ok=True)
for filename, title, summary in ADRS:
    path = os.path.join("adr", filename)
    adr_id = filename.split("-")[0] + "-" + filename.split("-")[1]
    content = TEMPLATE.format(id=adr_id, title=title, title_lower=title.lower(), summary=summary)
    with open(path, "w") as f:
        f.write(content)

print(f"Generated {len(ADRS)} ADR documents.")
