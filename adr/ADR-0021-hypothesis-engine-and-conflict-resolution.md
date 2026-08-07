# ADR 0021: Hypothesis Engine and Conflict Resolution

## Status
Accepted (v0.3.0)

## Context
In real-world cognitive tasks, data is often ambiguous. The runtime needs a mechanism to propose multiple interpretations before committing to a singular belief. When beliefs contradict, the system must deterministically resolve the conflict.

## Decision
- **Hypothesis Engine**: A dedicated pipeline stage generates multi-hypothesis candidate interpretations. Each hypothesis carries a confidence score and explicitly seeks evidence.
- **Candidate Pruning**: Hypotheses failing to meet confidence thresholds or violating structural constraints are pruned before becoming Beliefs.
- **Multi-Policy Conflict Resolution**: When contradiction occurs between existing beliefs or new inferences, the resolver uses policies (Recency, Epistemic Authority, Confidence Weighting) to decide which proposition remains active.
- **Non-Monotonic Reasoning**: Resolving a conflict may result in retracting a previously held Belief, cascading through the Thought Graph.

## Consequences
- Requires parallel exploration branches within a single cognitive cycle.
- Demands a robust epistemic weighting system for accurate conflict resolution.
