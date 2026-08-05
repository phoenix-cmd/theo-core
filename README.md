# THEO — The Poet (Core Engine)

> **Version 0.1.0** — Cognitive Operating System Infrastructure

*Theo is a cognitive operating system whose purpose is to understand, reason, remember, create, and grow. Every component exists to support cognition; models, tools, and infrastructure are replaceable implementations of that purpose.*

---

## 🏛️ Architecture Philosophy

**Architecture is stable. Research is experimental.**

`theo-core` defines the cognitive architecture, kernel, domain contracts, event bus, and subsystem interfaces.
New cognitive concepts and experimental model architectures should be explored in `theo-lab`. Only after demonstrating measurable value should they be proposed via an ADR (Architecture Decision Record) for promotion into `theo-core`.

### Mental Model & Data Flow

```
Input → Perception → Context → Memory → Knowledge → Identity → Goals → Planning → Reasoning → Creativity → Reflection → Decision → Action → Learning
```

### Cognitive Stack

```
1. THEO (Cognitive OS Core)
2. Kernel (Boot, Lifecycle, Registry, Scheduler)
3. Event Bus (Pub/Sub Event Dispatcher)
4. Subsystem Stack (Perception, Context, Memory, Knowledge, Identity, Goals, Capabilities, Models)
5. Cognitive Cycle Engine (Iterative Reasoning Loop)
6. Services & Lab Integrations
7. Infrastructure (Config, Logging, Tracking, Security, Telemetry)
```

---

## 🛠️ Technology Stack

- **Python**: 3.13+
- **Package Manager**: `uv`
- **Validation & Typing**: `pydantic` v2.10+
- **Configuration**: `hydra-core` v1.4+
- **Logging**: `structlog` v25.0+
- **Plugin Engine**: `pluggy` v1.5+
- **Scheduler**: `apscheduler` v4.0+
- **Testing & Quality**: `pytest`, `ruff`, `mypy`

---

## 📊 Subsystem Research Success Metrics

| Subsystem | Primary Success Metric | Secondary Metric |
|---|---|---|
| **Memory** | Recall accuracy | Retrieval latency |
| **Reasoning** | Benchmark accuracy | Reasoning consistency |
| **Creativity** | Novelty & coherence score | Human evaluation |
| **Identity** | Response consistency across sessions | Alignment fidelity |
| **Goals** | Goal completion rate | Priority resolution efficiency |
| **Reflection** | Self-correction frequency | Error reduction rate |
| **Event Bus** | Throughput (events/sec) | Dispatch latency |
| **Scheduler** | Job execution success rate | Timing precision |

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

### Booting the Kernel

```bash
# Boot THEO via CLI
uv run theo boot

# Or run directly via Python module
uv run python -m theo_core
```

---

## 🧪 Testing & Code Quality

```bash
# Run unit & integration tests
uv run pytest

# Check coverage (must be >= 80%)
uv run pytest --cov=theo_core

# Linting with Ruff
uv run ruff check src/ tests/

# Strict type checking with Mypy
uv run mypy src/
```

---

## 📄 License & Principles

See [PRINCIPLES.md](PRINCIPLES.md) for the 10 foundational principles of THEO.
See `adr/` for Architecture Decision Records.
