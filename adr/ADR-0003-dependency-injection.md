# ADR-0003: Dependency Injection Strategy

## Status

Accepted

## Context

Adopt a manual composition root in composition/bootstrap.py creating a unified TheoContainer. Avoid framework magic or global service locators to ensure 100% testability, transparency, and explicit dependency graphs.

## Problem Statement

THEO requires production-grade architectural guidance for dependency injection strategy that supports a decade of research and engineering evolution without major rewrites.

## Requirements

- High cohesion, low coupling, SOLID principles.
- Clean separation between core research logic and platform/UI logic.
- Full type safety, testability, and research reproducibility.

## Options Considered

1. Hand-rolled ad-hoc implementation (rejected due to technical debt risk).
2. Framework-coupled implementation (rejected due to lock-in).
3. Standardized modular clean architecture (chosen).

## Chosen Solution

Adopt a manual composition root in composition/bootstrap.py creating a unified TheoContainer. Avoid framework magic or global service locators to ensure 100% testability, transparency, and explicit dependency graphs.

## Rationale

This approach strictly adheres to THEO's core guiding principles: cognition before models, replaceable components, and observable decision chains.

## Trade-offs

Introduces additional abstraction layers and initial scaffolding overhead, which is justified by long-term maintainability and research flexibility.

## Consequences

Establishes a firm contract for v0.1.0 infrastructure. Any future modifications to this architectural boundary will require a formal ADR update.

## References

- PRINCIPLES.md
- Implementation Plan v0.1
