# THEO Research Roadmap

This document outlines the living release plan for the THEO cognitive architecture from its foundation to v1.0.

*Note: THEO v0.4 Symbolic Runtime is designated as the initial reference implementation for Canon Edition C1.*

## v0.4.1 Freeze

Canon Edition C1 and the v0.4.1 Symbolic Runtime are **frozen**: the canonical laws, pipeline structure
(9 stages), decision model (Intent + ActionSpec), boundary ResponseRenderer, and the cognitive benchmark
corpus are the architectural contract. From v0.5 onward, releases deliver **content and compatibility
only** — new knowledge libraries, richer corpora, and integration layers — without modifying the frozen
contract.

Freeze posture is codified in:
- **ADR-0026** — Canon Law 4 verdict: perception enters cognition as evidence, never as a mechanical
  belief source (`BeliefSource` stays closed at MEMORY / KNOWLEDGE / INFERENCE).
- **ADR-0027** — Runtime unification: the symbolic pipeline is the single canonical runtime; the v0.2
  12-stage `CognitiveEngine` is demoted to a legacy compatibility path (`theo chat --engine legacy`),
  excluded from the frozen contract.

## Milestones

| Version | Codename | Focus | Status |
|---------|----------|-------|--------|
| **v0.1** | Foundation | Scaffolding & Core Infrastructure | ✅ Complete |
| **v0.2** | Deterministic Runtime | 12-stage Cognitive Pipeline | ✅ Complete |
| **v0.2.1** | Validation | Benchmark & Stress Testing Suite | 📋 Planned |
| **v0.3** | Constitution | Formal Cognitive Specification & Architecture Freeze | ✅ Complete |
| **v0.4** | Symbolic Runtime | Concept/Thought/Belief Graphs, Inference, Constraints, Conflicts | 🔒 FROZEN |
| **v0.4.1** | Benchmark Corpus | Knowledge Libraries, Rule Base, Cognitive Evaluation Battery | 🔒 FROZEN |
| **v0.5** | Neural Runtime | Content-only: neural proposal engines as advisory layers | 📋 Planned |
| **v0.6** | Hybrid Cognition | Content-only: belief reconciliation with frozen decision path | 📋 Planned |
| **v0.7** | Learning Runtime | Content-only: continual learning under the frozen Canon | 📋 Planned |
| **v0.8** | Metacognition | Self-Evaluation, Compute Allocation, Subsystem Trust | 📋 Planned |
| **v0.9** | Multi-Agent | Multi-Agent Cognitive Coordination | 📋 Planned |
| **v1.0** | THEO COS | THEO Cognitive Operating System | 📋 Planned |

## Detailed Focus Areas

### v0.4: Symbolic Runtime (FROZEN)
The first reference implementation of Canon Edition C1.
- Concept Graph (nodes, typed edges, activation).
- Belief System (propositions, confidence, provenance).
- Thought Graph (immutable reasoning DAG).
- Inference Engine (forward/backward chaining, rule execution).
- Hypothesis Engine (generate competing interpretations).
- Constraint & Conflict Resolver (consistency checking).
- Decision Engine v2 (evidence-driven decision selection, Intent + ActionSpec).
- ResponseRenderer (boundary-only rendering of structured ActionSpec, Canon Law 6).

### v0.4.1: Cognitive Benchmark Corpus & Knowledge Engineering (FROZEN)
Standard evaluation battery and knowledge base prior to neural integration.
- `evaluation/benchmarks/`: Standardized cognitive test suites (commonsense, taxonomy, contradiction, causal reasoning, uncertainty) — 26 cases, ≥5 per domain.
- `evaluation/harness.py`: Deterministic harness comparing expected vs actual Belief Graphs, Decisions, and GoldenTraces.
- `theo benchmark run` CLI: executable governance over the canonical pipeline.
- GoldenTrace: complete per-cycle structural trace (retrieved memory, fired rules, derived beliefs, generated hypotheses, resolved conflicts, thought DAG).

### v0.5: Neural Runtime (Content & Compatibility)
Introducing neural models to enhance, not replace, symbolic cognition.
- LLMs/SLMs act as proposal engines (Hypothesis generation) via the frozen port interface.
- Neural perception via embeddings and tokenizers.
- **Constraint:** neural predictions never bypass the frozen Decision Engine; the v0.4.1 contract is unchanged.

### v0.6: Hybrid Cognition (Content & Compatibility)
Parallel execution and cross-comparison.
- Concurrent symbolic rule execution and neural prediction.
- Belief reconciliation merging symbolic certainty with neural probability.
- **Constraint:** reconciliation must preserve the frozen decision path (Intent/ActionSpec/render boundary).

### v0.7: Learning Runtime (Content & Compatibility)
Dynamic updating of the Data Layer.
- Continual learning and memory consolidation.
- Belief revision over time.
- Automated rule extraction (learning new Knowledge from repeated Memory patterns).
- **Constraint:** learning writes through the Learning stage (Canon §6) only.

### v0.8: Metacognition
System-level self-evaluation.
- Evaluating rule failure rates and memory reliability.
- Cognitive scheduling and compute budget allocation.
- Dynamic trust weighting between symbolic and neural subsystems.

### v0.9: Multi-Agent
Scaling cognition across instances.
- Coordination between specialized THEO agents.
- Distributed thought graphs.

### v1.0: THEO Cognitive Operating System
A complete, stable, verifiably cognitive operating system.
