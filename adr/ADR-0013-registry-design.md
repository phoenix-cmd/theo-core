# ADR-0013: Unified Artifact Registry

## Status

Accepted

## Context

Provide a centralized Registry for versioning, lineage tracking, and status transitions across models, datasets, checkpoints, and evaluation results.

## Problem Statement

THEO requires production-grade architectural guidance for unified artifact registry that supports a decade of research and engineering evolution without major rewrites.

## Requirements

- High cohesion, low coupling, SOLID principles.
- Clean separation between core research logic and platform/UI logic.
- Full type safety, testability, and research reproducibility.

## Options Considered

1. Hand-rolled ad-hoc implementation (rejected due to technical debt risk).
2. Framework-coupled implementation (rejected due to lock-in).
3. Standardized modular clean architecture (chosen).

## Chosen Solution

Provide a centralized Registry for versioning, lineage tracking, and status transitions across models, datasets, checkpoints, and evaluation results.

## Rationale

This approach strictly adheres to THEO's core guiding principles: cognition before models, replaceable components, and observable decision chains.

## Trade-offs

Introduces additional abstraction layers and initial scaffolding overhead, which is justified by long-term maintainability and research flexibility.

## Consequences

Establishes a firm contract for v0.1.0 infrastructure. Any future modifications to this architectural boundary will require a formal ADR update.

## References

- PRINCIPLES.md
- Implementation Plan v0.1
