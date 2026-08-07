# ADR 0020: Thought Graph and Defeasible Inference

## Status
Accepted (v0.3.0)

## Context
If cognition is computation over state, then "Thoughts" are the computational objects. We must define their structure, lifecycle, and how inference creates them.

## Decision
- **Immutable Thought Representations**: Once created in a cycle, a Thought cannot be edited. It serves as permanent evidence for that cognitive cycle.
- **DAG Semantics**: Thoughts form a Directed Acyclic Graph. Edges represent logical lineage (`supports`, `contradicts`, `depends_on`, `derived_from`).
- **Defeasible Inference**: The Inference Engine supports forward and backward chaining that can be "defeated" (invalidated) if new, stronger evidence contradicts a premise in the DAG.
- **Belief Propagation**: When a node's truth value changes due to new evidence, downstream Thoughts are automatically re-evaluated or flagged as invalid.

## Consequences
- Retains complete traceability for explainability.
- Prevents infinite loops via DAG cycle detection.
- Demands higher memory allocation per cognitive turn to store the graph.
