# ADR 0022: Decoupled Response Generation and Versioned Ports

## Status
Accepted (v0.3.0)

## Context
Per Canon Law 6, language generation MUST NOT participate in cognitive computation. In earlier prototypes, text generation and thought generation were heavily intertwined. As we prepare for v0.4 (Symbolic) and v0.5 (Neural), the subsystem interfaces must be strict and versioned.

## Decision
- **Decoupled Response Generation**: The `DecisionEngine` outputs a structured, deterministic `DecisionRecord`. The `ResponseGenerator` solely consumes this record to format surface text. It cannot query memory or run inferences.
- **Metacognitive Reflection Feedback**: The Reflection stage evaluates the Thought Graph *before* the Decision is committed, allowing the cycle to loop or re-allocate compute without surface output.
- **Versioned Ports**: All inter-subsystem communication utilizes strict versioned interfaces (e.g., `InferencePortV2`, `DecisionPortV2`).

## Consequences
- Ensures neural models introduced in v0.5 for language generation cannot hallucinate cognitive state.
- Makes the system modular; a completely different UI/presentation layer can be attached without altering the cognitive cycle.
