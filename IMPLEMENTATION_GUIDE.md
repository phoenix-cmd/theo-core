# THEO Implementation Guide

This living engineering document governs the implementation of THEO from v0.4 onward. Unlike the `CANON.md`, which defines timeless cognitive truths, this guide defines how the code is built, tested, and maintained.

## 1. Strict Anti-Drift Enforcement & Rule 0

**Implementation Rule 0: Prefer replacing an implementation over weakening the Canon. If an implementation fails to satisfy the Canon, the implementation is presumed incorrect unless a demonstrated contradiction or ambiguity exists in the Canon itself.**

No new cognitive concepts, architectural layers, or normative contracts shall be introduced during implementation unless an ambiguity or contradiction is discovered in Canon Edition C1. Any such discovery SHALL first be resolved through a Canon amendment (new Edition) or an ADR before implementation proceeds. The implementation serves the Canon; the Canon does not bend to convenient implementation shortcuts.

## 2. The v0.4 Symbolic Runtime Build Order

Implementation of the v0.4 Symbolic Runtime MUST follow this sequence to minimize dependencies and enable isolated validation:

| Phase | Component | Purpose |
|---|---|---|
| **1** | Concept Graph | Semantic representation |
| **2** | Belief Engine | Persistent propositions |
| **3** | Thought Graph | Immutable reasoning objects |
| **4** | Inference Engine | Rule execution |
| **5** | Hypothesis Engine | Competing interpretations |
| **6** | Constraint Engine | Consistency validation |
| **7** | Conflict Resolver | Belief reconciliation |
| **8** | Reflection Engine v2 | Meta-reasoning |
| **9** | Decision Engine v2 | Structured decisions |
| **10** | Response Generator | Surface realization |
| **11** | Integration | End-to-end symbolic runtime |
| **12** | Conformance Tests | Canon C1 verification |

## 3. Dependency Graph & Architecture Rules

- **Subsystem Isolation**: Subsystems (e.g., `InferenceEngine`, `MemoryClassifier`) MUST NOT import from each other. All interactions go through defined Port interfaces or the central `EventBus`.
- **Top-Down Dependency**: Core primitives (Thoughts, Beliefs, Concepts) sit at the lowest level. Engines depend on primitives, not on other engines.
- **Side Effects**: Side effects are strictly confined to the Learning stage. All other components MUST be pure state transformation functions.

## 4. Coding Standards

- **Language**: Python 3.11+
- **Type Hinting**: 100% strict type hinting (`mypy --strict`).
- **Formatting**: `ruff format` and `ruff check`.
- **Immutability**: Use `frozen=True` in dataclasses for objects like Thoughts and initial Percepts.

## 5. Testing Strategy & Definition of Done (DoD)

For a subsystem to be considered "Done":
1. **Unit Tests**: 90%+ code coverage for the isolated component.
2. **Type Checking**: Zero `mypy` errors.
3. **Immutability Validation**: Tests explicitly verify that objects designated as immutable (e.g., Thoughts) throw errors upon attempted modification.
4. **Integration Tests**: Component successfully receives inputs and returns outputs conforming to its Port interface without violating Canon constraints.

### Concrete Success Criteria for v0.4:
- [ ] Concepts can be activated.
- [ ] Beliefs can be created and revised.
- [ ] Thoughts form an immutable DAG.
- [ ] Inference consumes thoughts and beliefs.
- [ ] Decisions are fully traceable.
- [ ] Every Canon law passes in the conformance suite.
- [ ] Every invariant passes.
- [ ] Conformance suite (`tests/conformance/`) is completely green.

## 6. Conformance Testing Suite (`tests/conformance/`)

Phase 12 involves building the Official Conformance Test Suite for Canon Edition C1. Rather than just verifying unit functionality, the conformance suite continuously validates that the runtime satisfies the Canon laws:

- `test_law_1.py`: Verifies that every Response originates from exactly one Decision.
- `test_law_2.py`: Verifies that every Decision references one or more existing Thoughts.
- `test_law_3.py`: Verifies that every Thought references valid Beliefs.
- `test_law_4.py`: Verifies that Beliefs are derived exclusively from Memory, Knowledge, or Inference.
- `test_law_5.py`: Verifies Knowledge does not produce responses directly.
- `test_law_6.py`: Verifies language generation does not participate in cognitive computation.
- `test_law_7.py`: Verifies Inference does not mutate Memory (only Learning mutates Memory).
- `test_law_8.py`: Verifies Reflection creates new Thoughts without modifying existing Decisions.
- `test_invariant_8.py`: Verifies deterministic $S_{t+1}$ state production per cycle.

This suite treats the runtime as a black box to make the Cognitive Canon executable.

## 7. Folder Conventions

```
theo-core/
├── src/
│   └── theo_core/
│       ├── core/           # Primitives (Thoughts, Beliefs, Concepts)
│       ├── engine/         # Inference, Hypothesis, Decision engines
│       ├── memory/         # Knowledge and Memory stores
│       ├── ports/          # Interfaces between systems
│       └── utils/          # Shared helpers
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conformance/      # Canon C1 executable law test suite
```

## 8. Post-v0.4 Research Paper Milestone

Upon completion of v0.4 and successful verification by the `tests/conformance/` suite, a formal academic paper shall be drafted:
- **Structure**: Abstract, Motivation, Related Work, Cognitive Canon, Runtime Architecture, Symbolic Execution Model, Validation, Results, Limitations, Future Work.
- **Objective**: Rigorously justify THEO's design choices against existing approaches in symbolic AI, cognitive architectures (SOAR, ACT-R), and hybrid reasoning systems.
