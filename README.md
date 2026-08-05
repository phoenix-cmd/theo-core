# THEO — The Poet (Core Engine)

> **Version 0.2.0** — The Deterministic Cognitive Runtime

*Theo is a cognitive operating system whose purpose is to understand, reason, remember, create, and grow. Every component exists to support cognition; models, tools, and infrastructure are replaceable implementations of that purpose.*

---

## 🏛️ Architecture Philosophy

**Architecture is stable. Research is experimental.**

`theo-core` defines the cognitive architecture, kernel, domain contracts, event bus, and subsystem interfaces.
New cognitive concepts and experimental model architectures should be explored in `theo-lab`. Only after demonstrating measurable value should they be proposed via an ADR (Architecture Decision Record) for promotion into `theo-core`.

### Mental Model & Data Flow

```
Input → Perception → Context → Memory → Knowledge → Goals → Planning → Inference → Reflection → Decision → Response Generator → Learning & Trace
```

---

## 🚦 Subsystem Maturity Table (v0.2.0)

| Subsystem | Status | Implementation Details |
|---|---|---|
| **Pipeline Execution** | ✅ Functional | 12-stage sequential execution (`CognitiveEngine`) |
| **Perception** | ✅ Functional | Data-driven regex rules & entity extraction (`intents.yaml`, `preferences.yaml`) |
| **Context** | ✅ Functional | Ephemeral session context buffer (`InMemoryContextManager`) |
| **Goals** | ✅ Functional | Priority GoalStack manager (`GoalManager`) |
| **Planning** | ✅ Functional | Rule-based action sequence planner (`RuleBasedPlanner`) |
| **Inference** | ✅ Functional | Strategy-based policy evaluation (`InferenceEngine` / `RuleBasedStrategy`) |
| **Response Generation** | ✅ Functional | Template response generator (`TemplateResponseGenerator`) |
| **Memory** | ✅ Functional | 4-layer engine with append-only history & JSON persistence (`DeterministicMemoryEngine`) |
| **Knowledge** | 🚧 Stub | Graph knowledge & concept traversal (Milestone 4) |
| **Reflection** | 🚧 Minimal | Inferred confidence & satisfaction evaluation |
| **Learning & Trace** | 🚧 Minimal | Context recording & memory persistence |

---

## 🛠️ Technology Stack

- **Python**: 3.13+
- **Package Manager**: `uv`
- **Validation & Typing**: `pydantic` v2.10+
- **Configuration**: `hydra-core` v1.4+ / `pyyaml` v6.0+
- **Logging**: `structlog` v25.0+
- **Plugin Engine**: `pluggy` v1.5+
- **Scheduler**: `apscheduler` v4.0+
- **Testing & Quality**: `pytest`, `ruff`, `mypy`

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/theo-ai/theo-core.git
cd theo-core

# Sync dependencies using uv
uv sync --all-extras
```

### Interactive Cognitive Session (`theo chat`)

```bash
# Start an interactive session with THEO
uv run theo chat
```

### REPL Commands
- `/context` — View active session context snapshot.
- `/goals` — View active GoalStack items.
- `/memory` — View all active persistent memory entries stored in JSON repository.
- `/exit` or `/quit` — Exit the REPL.

---

## 🧪 Testing & Code Quality

```bash
# Run unit & integration tests
uv run pytest

# Check coverage (must be >= 80%)
uv run pytest --cov=theo_core

# Linting with Ruff
uv run ruff check src/ tests/

# Type checking with Mypy
uv run mypy src/
```
