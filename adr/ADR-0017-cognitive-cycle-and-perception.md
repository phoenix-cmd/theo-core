# ADR-0017: Cognitive Cycle and Perception

## Status

Accepted

## Context

Define an explicit CognitiveEngine executing the step-by-step reasoning cycle (Perception -> Memory -> Knowledge -> Planning -> Reasoning -> Reflection -> Action) and a Perception subsystem for input normalization into Percept objects.

## Problem Statement

THEO requires production-grade architectural guidance for cognitive cycle and perception that supports a decade of research and engineering evolution without major rewrites.

## Requirements

- High cohesion, low coupling, SOLID principles.
- Clean separation between core research logic and platform/UI logic.
- Full type safety, testability, and research reproducibility.

## Options Considered

1. Hand-rolled ad-hoc implementation (rejected due to technical debt risk).
2. Framework-coupled implementation (rejected due to lock-in).
3. Standardized modular clean architecture (chosen).

## Chosen Solution

Define an explicit CognitiveEngine executing the step-by-step reasoning cycle (Perception -> Memory -> Knowledge -> Planning -> Reasoning -> Reflection -> Action) and a Perception subsystem for input normalization into Percept objects.

## Rationale

This approach strictly adheres to THEO's core guiding principles: cognition before models, replaceable components, and observable decision chains.

## Trade-offs

Introduces additional abstraction layers and initial scaffolding overhead, which is justified by long-term maintainability and research flexibility.

## Consequences

Establishes a firm contract for v0.1.0 infrastructure. Any future modifications to this architectural boundary will require a formal ADR update.

## References

- PRINCIPLES.md
- Implementation Plan v0.1
