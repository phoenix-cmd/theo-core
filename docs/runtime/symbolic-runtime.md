# THEO Symbolic Runtime — Developer Guide

## Subsystem Responsibilities

| Package | Purpose | Phase |
|---------|---------|-------|
| `_primitives/` | Shared value objects, protocols, errors, ordering | 0.5 |
| `_graph/` | Generic `Graph[N, E]` data structures and algorithms | 0.75 |
| `concepts/` | Semantic representation: typed nodes, edges, activation, taxonomy | 1 |
| `beliefs/` | Persistent propositions: confidence, provenance, revision | 2 |
| `thoughts/` | Immutable reasoning DAG: dependency edges, evidence links | 3 |
| `inference/` | Rule execution: forward/backward chaining | 4 |
| `hypotheses/` | Competing interpretations: confidence scoring, candidate pruning | 5 |
| `constraints/` | Consistency validation against Canon invariants | 6 |
| `conflict/` | Contradiction detection, multi-policy resolution | 6 |
| `decisions/` | Evidence-driven decision selection, candidate evaluation | 7 |
| `scheduler/` | OS-like priority management, compute budgets | 7 |

## Dependency Graph

```
_primitives  (no dependencies)
     ↑
  _graph     (depends only on _primitives)
     ↑
  concepts/  beliefs/  thoughts/  ...  (depend only on _primitives + _graph)
```

**Rule**: No domain package may import from another domain package. All cross-cutting concerns go through `_primitives` or `_graph`.

## Modeling Convention

- **Value objects** → `@dataclass(frozen=True, slots=True)`. Fast, hashable, no validation overhead.
- **Domain aggregates / I/O models** → `Pydantic BaseModel(frozen=True)`. Rich validation at serialization boundaries.

## Equality Semantics

- **Identity-based**: Two objects with the same `id` are equal, regardless of other fields. Applies to `ConceptId`, `SymbolicId`, and all `*Id` types.
- **Structural**: Two objects are equal if all fields match. Applies to edges, activation results, and serialized envelopes.

## Build Order

Each subsystem is built, tested, and frozen independently before the next begins.

## Testing Strategy

- `tests/unit/symbolic/` — Isolated subsystem tests (90%+ coverage)
- `tests/integration/symbolic/` — Cross-subsystem workflows
- `tests/conformance/` — Canon Edition C1 law and invariant verification
- `tests/architecture/` — Structural enforcement (isolation, layering, no circular imports)

## Conformance Strategy

The `tests/conformance/` suite treats the runtime as a black box. It validates Canon laws and invariants through behavior, not implementation inspection.
