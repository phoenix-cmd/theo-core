# Architectural Invariants of Theo

> *These architectural invariants are the non-negotiable constitution of THEO.
> They MUST never be broken unless formally superseded by an Architecture Decision Record (ADR).*

---

## 1. Subsystem Decoupling & Port Isolation
Every subsystem communicates **only** through explicit ports (abstract interfaces) or domain events published to the central `EventBus`. Direct implementation imports across subsystem boundaries are strictly forbidden.

## 2. Zero Framework Leakage in Domain
Domain code (`src/theo_core/domain/`) must remain 100% pure Python with zero dependencies on frameworks, databases, web servers, or external AI APIs.

## 3. Replaceable Cognitive Substrate
Models are replaceable implementations, not the center of cognition. The cognitive architecture defines Theo; language models, vector databases, and neural components are disposable backends.

## 4. Full Traceability and Replayability
Every cognitive cycle must produce an immutable, complete, versioned `CognitiveTrace`. Given the same trace and initial state, any cognitive execution must be 100% replayable and verifiable.

## 5. Explainable Decision Chains
Every user-visible output produced by Theo must be explainable. A researcher or user must be able to inspect the exact goal, memory, knowledge, inference rule, and reflection output that produced a given response.

## 6. Independent Persistence of Self
Memory, identity, goals, and knowledge persist independently of any language model, context window, or runtime session.

## 7. Mandatory Event Bus and Kernel Orchestration
No subsystem may bypass the Kernel lifecycle or the Event Bus dispatcher. Subsystems do not call other subsystems directly.

---

## Architecture Compliance Tests

Compliance with these invariants is automatically enforced by architectural linting tests in `tests/architecture/`. Any pull request or commit violating these invariants will fail CI.
