"""Script to create all subsystem stub packages."""

import os

DIRS = [
    "perception/ports", "perception/text", "perception/image", "perception/audio",
    "perception/document", "perception/multimodal",
    "context/conversation", "context/environment", "context/execution", "context/session",
    "memory/ports", "memory/storage", "memory/retrieval", "memory/indexing",
    "memory/consolidation", "memory/forgetting", "memory/episodic", "memory/semantic",
    "memory/working", "memory/long_term",
    "knowledge/ports", "knowledge/ontology", "knowledge/facts",
    "knowledge/relationships", "knowledge/graph", "knowledge/embeddings",
    "identity/ports", "identity/persona", "identity/values", "identity/self_model",
    "goals/ports", "goals/stack", "goals/priority", "goals/manager",
    "cognitive_cycle/engine", "cognitive_cycle/stages", "cognitive_cycle/state",
    "cognition/reasoning", "cognition/reflection", "cognition/creativity",
    "capabilities/ports", "capabilities/planning", "capabilities/retrieval",
    "capabilities/summarization", "capabilities/translation", "capabilities/coding",
    "capabilities/mathematics",
    "skills/ports", "skills/registry", "skills/base",
    "models/ports", "models/language",
    "training/ports", "training/pipeline", "training/checkpointing",
    "tokenization/ports",
    "evaluation/ports", "evaluation/benchmarks", "evaluation/leaderboards",
    "evaluation/comparison", "evaluation/reports",
    "datasets/ports", "datasets/manifest", "datasets/pipeline", "datasets/validation",
    "registry/ports", "registry/models", "registry/datasets",
    "registry/checkpoints", "registry/evaluations",
    "services/conversation", "services/training", "services/memory",
    "services/reflection", "services/reasoning",
    "sdk/plugin", "sdk/plugin_loader", "sdk/plugin_api",
    "scheduler/ports", "scheduler/jobs", "scheduler/background",
    "security/ports", "security/permissions", "security/secret_manager",
    "telemetry/ports", "telemetry/metrics", "telemetry/health", "telemetry/tracing",
    "infrastructure/file_system",
    "domain/research/journals",
]

BASE = os.path.join("src", "theo_core")

count = 0
for d in DIRS:
    path = os.path.join(BASE, d)
    os.makedirs(path, exist_ok=True)
    parts = d.split("/")
    for i in range(len(parts)):
        parent = os.path.join(BASE, *parts[: i + 1])
        init = os.path.join(parent, "__init__.py")
        if not os.path.exists(init):
            mod = ".".join(parts[: i + 1])
            with open(init, "w") as f:
                f.write(f'"""theo_core.{mod} — stub module."""\n')
            count += 1

print(f"Created {count} new __init__.py files across {len(DIRS)} packages")
