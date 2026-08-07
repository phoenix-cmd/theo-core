# Concept System

Semantic representation layer. Typed concept nodes, typed edges, activation spreading, and taxonomy traversal.

Implements Canon Edition C1 §3 (Cognitive Ontology — Concepts bridge perception and knowledge) and ADR-0019.

## Contents

- `models.py` — `ConceptId`, `ConceptType`, `RelationType`, `Concept`, `ConceptEdge`
- `graph.py` — `ConceptGraph` wrapping `Graph[Concept, ConceptEdge]`
- `activation.py` — Deterministic spreading activation using `Decimal` arithmetic
