# ADR 0025: Cognitive Ontology and Fundamental Laws

## Status
Accepted (v0.3.0)

## Context
Previous iterations treated "cognition" intuitively. To build verifiable cognitive systems, we require a mathematically strict definition of what cognition is, completely independent of implementation details. 

## Decision
- **Formal Cognitive Ontology**: We strictly define 10 core concepts (Thought, Belief, Knowledge, Memory, Goal, Intention, Decision, Understanding, Attention, Context).
- **The 8 Fundamental Laws of Cognition**: Formally defines the boundaries of cognition (e.g., "Law 6: Language generation MUST NOT participate in cognitive computation").
- **RFC-style Normative Language**: We adopt RFC 2119 terminology (MUST, MUST NOT, MAY, etc.) to clearly delineate binding constraints from informative commentary.
- **Conformance Levels & Verification**: Formalizes the definition of a THEO Runtime and requires that conformance be demonstrated via a behavioral test suite.

## Consequences
- Elevates THEO from a software framework to a formal cognitive architecture standard (Canon Edition C1).
- Provides unambiguous criteria for determining whether future implementations (Symbolic, Neural, Hybrid) are "conforming".
