# ADR 0024: Cognitive Scheduler and Metacognitive Control

## Status
Accepted (v0.3.0)

## Context
As the inference, hypothesis, and reflection stages grow in complexity, a simple sequential pass is insufficient. The architecture must handle variable computation budgets and evaluate its own performance over time.

## Decision
- **Cognitive Scheduler**: An OS-like scheduler manages priority queues for inference branches. It allocates compute budgets (cycles/nodes) based on Goal urgency.
- **Metacognitive Control**: The runtime tracks its own success/failure rates. It maintains performance statistics on rules, memory reliability, and inference strategies.
- **Dynamic Compute Allocation**: When uncertainty is high, the metacognitive controller extends the budget for the Hypothesis Engine.
- **Subsystem Trust Weighting**: Designed as a reserved extension point for v0.6+ Hybrid Cognition, allowing the system to dynamically weight trust between symbolic confidence and neural probability.

## Consequences
- Requires a discrete `Scheduler` component to manage execution yielding.
- Increases the overhead of the cognitive cycle but allows for bounded execution times and self-correction.
