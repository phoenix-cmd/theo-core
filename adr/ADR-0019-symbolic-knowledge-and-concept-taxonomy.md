# ADR 0019: Symbolic Knowledge and Concept Taxonomy

## Status
Accepted (v0.3.0)

## Context
A purely deterministic pipeline (v0.2) passes text or raw structs through stages. For a cognitive system to reason (v0.4+), it requires an underlying representation of semantic meaning. We must define how concepts are structured and related before we can build rules that operate on them. 

## Decision
We will implement a Concept Graph featuring:
- **Concept Taxonomy**: Nodes represent discrete semantic units, connected by typed relationships (`is_a`, `part_of`, `implies`).
- **Activation Spreading Dynamics**: Perception of one concept spreads activation weight to neighborhood nodes, defining the current "Context".
- **Defeasible Inheritance**: A concept inherits properties from its parents but can override them (`Birds usually fly` vs `Penguins do not fly`), using non-monotonic logic.
- **Ontological Relationships**: Concepts map directly to the knowledge base facts (triples). 

## Consequences
- Requires a robust graph database or in-memory graph representation.
- Enables the Inference Engine to operate structurally rather than textually.
- Moves us out of NLP matching and into symbolic semantic alignment.
