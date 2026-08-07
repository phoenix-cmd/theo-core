# THEO Cognitive Canon

**Edition**: C1
**Ratified**: 2026-08
**Status**: Active
**Supersedes**: None

## 1. Preamble & Normative Scope

This document serves as the constitutional foundation of the THEO cognitive architecture. 
Sections 2 through 12 of this document are **Normative**. They represent binding requirements for any system claiming conformance to this Canon.
Appendices (including Appendix A) and any designated commentary are **Informative**.

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED",  "MAY", and "OPTIONAL" in this document are to be interpreted as described in RFC 2119.

## 2. Axioms

The following philosophical assumptions form the foundation from which all laws are derived:
1. **Cognition is computation over structured state.**
2. **Language is not cognition.**
3. **Memory is evidence, not truth.**
4. **Decisions are computational artifacts.**
5. **Knowledge is organized evidence.**

## 3. Cognitive Ontology

The following terms hold formal software design definitions within THEO:

- **Thought**: A temporary computational object created during one cognitive cycle to manipulate beliefs toward satisfying an active goal. Immutable after creation.
- **Belief**: A currently accepted proposition with associated confidence, uncertainty, evidence, and provenance.
- **Knowledge**: Organized structural facts and rules that establish domain evidence.
- **Memory**: An immutable historical record that MAY be used as evidence during future cognition.
- **Goal**: A desired cognitive state target that drives pipeline priority and decision criteria.
- **Intention**: A selected plan action committed for execution within the current cycle.
- **Decision**: The final outcome selection binding a candidate response to its underlying thoughts, beliefs, and evidence.
- **Understanding**: The state where input has been fully mapped to activated concepts, resolved beliefs, and goal alignment.
- **Attention**: Dynamically allocated computational weight across concepts, memories, and inference paths.
- **Context**: Ephemeral state surrounding the active cognitive turn, discarded between sessions.

## 4. Fundamental Laws of Cognition

- **Law 1**: Every Response MUST originate from exactly one Decision.
- **Law 2**: Every Decision MUST reference one or more Thoughts.
- **Law 3**: Every Thought MUST consume zero or more Beliefs.
- **Law 4**: Beliefs MUST be derived from Memory, Knowledge, or Inference. Perception enters cognition as evidence and MUST NOT directly produce beliefs; beliefs about percepts are mechanically derived by Inference with `source=INFERENCE` and an `EvidenceTrace(source_type="perception")`.
- **Law 5**: Knowledge MUST NOT directly produce responses. Knowledge MUST only contribute evidence.
- **Law 6**: Language generation MUST NOT participate in cognitive computation. It is solely responsible for serializing a Decision into a communicable representation.
- **Law 7**: Inference MUST NOT edit Memory. Only Learning edits Memory.
- **Law 8**: Reflection MUST NOT edit Decisions. Reflection MUST only create new Thoughts.

## 5. Data vs. Computation vs. Presentation

The architecture MUST strictly separate responsibilities:
- **Data Layer**: Memory, Knowledge, Beliefs.
- **Computation Layer**: Thoughts, Perception, Inference, Learning.
- **Presentation Layer**: Language (Response Generation).

## 6. Functional Computation Model

Every cognitive stage MUST be implemented as a pure state transformation function.
The cognitive cycle is their composition over the current state $S_t$:
$$S_{t+1} = \mathcal{L}(\mathcal{D}(\mathcal{R}(\mathcal{I}(\mathcal{K}(\mathcal{M}(\mathcal{C}(\mathcal{P}(S_t))))))))$$
Side effects MUST be confined exclusively to the Learning ($\mathcal{L}$) stage.

## 7. Cognitive Invariants

1. Thoughts MUST be immutable after creation.
2. Decisions MUST reference only existing Thoughts and Beliefs.
3. Memory MUST be append-only with versioned superseding.
4. Every inference MUST record its supporting evidence and rule provenance.
5. Every Belief MUST have traceable provenance.
6. Every Response MUST be traceable to a specific Decision record.
7. Every Decision MUST reference at least one active Goal.
8. Every cognitive cycle MUST produce exactly one deterministic $S_{t+1}$ state.

## 8. Conformance Levels & Verification

Runtimes MAY declare conformance to a specific level:
- **Level 1 — Deterministic**: Rule engine, replay, traceability.
- **Level 2 — Symbolic**: Concept graph, thought graph, belief graph, symbolic inference.
- **Level 3 — Neural**: Embeddings, tokenizers, neural proposal generation.
- **Level 4 — Hybrid**: Symbolic + neural parallel execution, belief reconciliation.
- **Level 5 — Metacognitive**: Self-evaluation, strategy adaptation, compute allocation.

**Conformance Verification**: A runtime claiming conformance MUST pass the official conformance test suite corresponding to the declared Canon Edition. Conformance is demonstrated by behavior, not implementation.

## 9. Reserved Extension Points

The Canon intentionally leaves unspecified:
- Belief mathematics and calculus
- Uncertainty calculus
- Neural architectures
- Scheduler implementations
- Learning algorithms
- Optimization strategies

Implementations MAY innovate in these areas provided they preserve all normative requirements.

## 10. Governance & Compatibility

The Canon MAY be amended only if:
1. The proposed amendment preserves internal consistency.
2. It does not invalidate existing runtime conformance without explicit migration guidance.
3. It is accompanied by an ADR explaining the rationale.
4. It receives a new Canon Edition identifier.

- **Minor Amendments**: Clarify wording, add examples, fix ambiguity. Do not require a new Edition.
- **Major Amendments**: Modify Axioms, Ontology, Laws, or Invariants. MUST receive a new Edition.

## 11. Runtime Qualification Criterion

*A system qualifies as a THEO Cognitive Runtime if, and only if, it satisfies the Cognitive Ontology, the Fundamental Laws of Cognition, the Cognitive Invariants, the Functional Computation Model, and the versioned Port interfaces defined in this Canon. The internal implementation — whether deterministic, symbolic, neural, hybrid, or future paradigms — MAY vary, but these contracts are immutable.*

## 12. Glossary

- **Concept**: A unified semantic representation bridging perception and knowledge.
- **Belief**: A formal assertion of truth held by the runtime.
- **Memory**: The persisted experiential record.
- **Knowledge**: The persisted structural ruleset.
- **Context**: The active working frame.
- **Understanding**: Cognitive equilibrium regarding an input.
- **Reflection**: Metacognitive evaluation of active thoughts.
- **Inference**: The derivation of new beliefs or thoughts.
- **Percept**: The structured abstraction of raw sensory input.
- **Hypothesis**: A tentative belief awaiting evidence.
- **Constraint**: A boundary condition on valid thoughts.
- **Conflict**: A detected contradiction between beliefs or hypotheses.

---

## Appendix A — Open Research Questions (Informative)

- How SHOULD confidence and uncertainty interact mathematically?
- SHOULD concept activation decay linearly or exponentially?
- Can beliefs be partially ordered instead of totally ordered?
- How SHOULD neural evidence update symbolic beliefs?
- What scheduling policy is optimal for cognition?
- Can cognitive cost predict response quality?
- What is the minimum set of reasoning primitives sufficient for general cognition?
- How SHOULD metacognitive oversight avoid infinite regress?
