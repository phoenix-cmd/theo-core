# THEO Symbolic Cognitive Architecture v0.3

This document represents the architecture snapshot for THEO v0.3. It specifies the 19 subsystems that make up the symbolic runtime. This version adheres strictly to the normative contracts established in **Canon Edition C1**.

## 1. Concept System
A semantic graph representing concepts as nodes with activation dynamics. Implements defeasible inheritance and lifecycle states. Acts as the primary interface between Perception and Knowledge.

## 2. Knowledge Representation
A symbolic store of fact triples, typed predicates, and Horn clauses. Implements non-monotonic default logic (`usually`, `exception`) enabling reasoning under uncertainty.

## 3. Thought Model & Thought Graph
Thoughts are immutable representations of derived propositions. The Thought Graph is a Directed Acyclic Graph (DAG) with edge relations (`supports`, `contradicts`, `depends_on`, `replaces`). It enables cycle detection and belief propagation.

## 4. Probabilistic Epistemic Belief System
Maintains the current accepted propositions. Schema: `ID`, `Proposition`, `Confidence`, `Uncertainty`, `Support`, `Contradictions`, `Source`, `Last Verified`, `Evidence Count`, `Reasoning Depth`, `Revision History`.

## 5. Hypothesis Engine
Generates multi-hypothesis interpretations of ambiguous percepts. Performs confidence scoring, collects evidence, and prunes invalid candidates before they solidify into beliefs.

## 6. Symbolic Inference Engine
A pluggable rule engine supporting:
- Forward Chaining
- Backward Chaining
- Constraint Propagation
- Multi-step Graph Traversal

## 7. Reasoning Primitive Library
Atomic functional operations:
`activate_concept`, `retrieve_memory`, `expand_neighborhood`, `match_rule`, `infer_fact`, `generate_hypothesis`, `reject_hypothesis`, `merge_beliefs`, `detect_contradiction`, `resolve_conflict`, `commit_decision`, `learn`.

## 8. Constraint System
Validates thoughts against invariants: Truthfulness, Memory Consistency, Goal Satisfaction, Context Consistency, Identity Consistency, Knowledge Validity, and Safety Policies.

## 9. Conflict Resolution
Detects contradictions in the Thought Graph. Utilizes multi-policy resolution (recency vs. confidence), epistemic authority overrides, exception handling, and retains historical state.

## 10. Reflection Architecture
The metacognitive loop. Evaluates goal achievement, evidence completeness, assumption validity, conflict status, and proposes alternative hypotheses or resource reallocation.

## 11. Decision Architecture
Consumes Thoughts and Beliefs to produce a singular Decision. Utilizes a candidate evaluation matrix and confidence aggregation, maintaining a deep trace link for explainability.

## 12. Response Generation Architecture
Decoupled entirely from cognitive computation (per Canon Law 6). Consumes structured `DecisionRecord` objects to generate deterministic text (v0.4) or neural text (v0.5+).

## 13. Non-Linguistic Cognition Pipeline
Data flow strictly follows:
`Input → Percept → Concept Graph → Thought Graph → Belief Graph → Decision Graph → Response Generator → Text`.

## 14. Cognitive Cycle Redesign
A 13-stage pipeline encapsulating the functional model $S_{t+1} = \mathcal{L}(\dots(S_t))$. Strict I/O schemas defined for each transition.

## 15. Cognitive Scheduler Architecture
An OS-like priority scheduler managing parallel inference branches, compute budgets, and yielding to reflection interrupts.

## 16. Metacognition Architecture
System-level self-evaluation. Tracks rule failure rates, memory reliability, inference strategy performance, and dynamically adjusts subsystem trust weights.

## 17. Cognitive Complexity Metrics
Tracks computational cost per cycle: Thought Count, Belief Count, Concept Activations, Inference Depth, Branching Factor, Conflict Count, Hypothesis Count, Reflection Depth, Decision Complexity.

## 18. Public Interfaces (Versioned Ports)
Standardized integration boundaries: `ConceptPortV1`, `InferencePortV2`, `HypothesisPortV1`, `ConstraintPortV1`, `ReflectionPortV2`, `ThoughtGraphPortV1`, `DecisionPortV2`, `BeliefPortV1`, `SchedulerPortV1`, `MetacognitionPortV1`.

## 19. Research Diagrams
*Mermaid diagrams illustrating subsystem boundaries, the cognitive cycle functional flow, and hybrid neural-symbolic bridging strategies.*
