# ADR 0028: Neural Symbolic Interface — Controlled Experimental Boundary

## Status

Accepted (v0.5.0)

## Context

THEO v0.4.1 established deterministic symbolic cognition: the 9-stage
`SymbolicCognitivePipeline` is the canonical runtime (ADR-0027), its outputs are
fingerprinted and replayable, and the benchmark corpus and conformance suites
freeze observable behavior. The v0.4.1 audit
(`docs/research/weakest-domain-audit-v0.4.1.md`) measured that the corpus is
uniformly green but that confidence is compressed, rule engagement is sparse, and
intent diversity is low.

v0.5's objective is to **improve reasoning quality without changing symbolic
cognition**. Achieving that requires attaching statistical and neural
contribution sources to the symbolic runtime. This ADR establishes the
architectural boundary that makes such an experiment safe: it defines how
providers may interact with the symbolic runtime, what they may never do, and
how their contributions stay measurable against the frozen v0.4.1 contract.

## Problem Statement

How can the symbolic runtime consult external statistical/neural providers so
that:

- the symbolic runtime remains the source of truth for cognition;
- providers propose, score, rank, and supply evidence but never decide;
- every provider contribution is measurable against the v0.4.1 corpus;
- determinism and replay guarantees survive provider attachment;
- providers are freely replaceable without moving the cognition contract;
- accidental modification of the symbolic runtime is architecturally
  impossible.

## Decision

### Defining principle

> Neural providers are replaceable implementations, not architectural
> components.

The v0.4.1 baseline established deterministic symbolic cognition. ADR-0028
establishes a controlled experimental boundary around that cognition. v0.5 may
fail experimentally without destabilizing THEO's identity: a poorly performing
provider is removed, a surprisingly effective one is retained, and a future
model is plugged in — the cognition contract does not move.

### Four Provider Protocols with Capability Discovery

There are exactly **four** provider protocols. `ProviderCapabilities` is an
enum used for capability discovery; it is not a provider protocol.

- **`HypothesisProposalProvider`** — proposes candidate hypotheses for the
  Hypothesis Engine. Proposals are evidence, not decisions.
- **`CalibrationProvider`** — scores hypotheses and scores confidence. Contains
  exactly two methods: `score_hypotheses` and `score_confidence`.
- **`SalienceProvider`** — ranks goals and ranks rules. Ranking reorders
  candidates only; it can neither create nor delete candidates.
- **`RuleDiscoveryProvider`** — **offline only**. Driven by benchmark failures
  and `KnowledgeGapReport` during knowledge engineering. It has no runtime hook
  and its proposed rules are never auto-committed to the knowledge base.

Every provider exposes:

```python
def capabilities() -> frozenset[ProviderCapabilities]: ...
```

where `ProviderCapabilities` is an enum (e.g. `HYPOTHESIS_PROPOSAL`,
`CALIBRATION`, `SALIENCE`, `RULE_DISCOVERY`). The runtime queries capabilities
and consults only the providers that declare the capability a hook requires.
No provider is required to implement every capability, and no no-op provider is
required. A runtime with zero providers attached behaves exactly like v0.4.1.

### Provider signatures contain only snapshot DTOs

All provider inputs and outputs are frozen dataclasses — **never** internal
runtime objects. Providers live outside theo-core and MUST NOT import symbolic
runtime classes (`Hypothesis`, `Belief`, `InferenceRule`, `Goal`,
`DecisionRecord`, graph objects, etc.).

Every internal symbolic entity that crosses the boundary has a snapshot
counterpart: `ConceptSnapshot`, `BeliefSnapshot`, `RuleSnapshot`,
`HypothesisSnapshot`, `GoalSnapshot`, `DecisionSnapshot`, plus collection
wrappers (e.g. `BeliefSnapshotCollection`) where appropriate. `HypothesisProposal`
and `KnowledgeGapReport` are snapshot DTOs defined with the ports contracts.

Architectural invariant:

> If deleting `theo_core.symbolic` internals would break a provider
> implementation, the boundary is wrong.

Provider signatures reference only snapshot DTOs, primitives (`str`,
`Decimal`, ...), and `ProviderExecution`. Snapshots are immutable, semantic-only,
and carry no timestamps or mutable lifecycle fields — those belong in telemetry.

### `ProviderExecution` is the replay contract

```python
@dataclass(frozen=True, slots=True)
class ProviderExecution[T]:
    provider_name: str
    provider_version: str
    model_name: str
    model_hash: str          # sha256 over weights + tokenizer + config
    seed: int
    temperature: float
    output: T
```

`ProviderExecution` contains only what is required to reproduce the cognitive
result. It has **no free-form metadata field**. Provider-specific detail
(hyperparameters, timings, environment) belongs in `ProviderTelemetry`, a
separate, non-replay stream.

### Grounding and strict rejection

Providers may reference existing symbolic entities by identifier
(`ConceptId`, `BeliefId`, `RuleId`, `EvidenceId`). The runtime exposes a single
read-only lookup boundary:

```python
@dataclass(frozen=True, slots=True)
class GroundingSnapshot:
    belief_ids: frozenset[str]
    concept_ids: frozenset[str]
    rule_ids: frozenset[str]
    evidence_ids: frozenset[str]
```

Grounding verification is symbolic-owned:

```python
def verify_grounding(
    proposal: HypothesisProposal,
    grounding: GroundingSnapshot,
) -> bool: ...
```

A proposal whose referenced identifiers have an empty intersection with the
union of the four sets is **REJECTED** — including proposals with no references
at all. There is no `grounded=False` fallback. A proposal MUST reference at
least one existing entity.

### Determinism

Two execution modes:

| Mode | Temperature | Seed | Replay |
| --- | --- | --- | --- |
| Benchmark | 0 | fixed | exact |
| Exploration | > 0 | recorded | sampled |

**Benchmark determinism** is defined as output reproducibility, not as
"temperature 0 plus a seed":

> A benchmark provider MUST produce identical canonical provider output for
> identical provider inputs, provider version, model hash, configuration, and
> seed.

A provider that cannot guarantee this is **not benchmark-compatible** and
cannot participate in the deterministic benchmark gate. GPU kernels,
quantization, provider libraries, hardware, and tokenizer versions can
introduce variation; the contract holds the provider responsible, not the seed.

### Provider resolution (deterministic)

> Amendment (v0.5.0, Phase 1): the runtime MUST resolve at most one provider
> per capability, deterministically.

Resolution follows a single explicit rule:

```
capability
    → eligible providers (those advertising the capability)
    → sort by (priority desc, provider name asc, configuration order asc)
    → exactly one selected provider, or none
```

- Selection MUST NOT depend on set or hash iteration order (e.g. never on
  iterating a `frozenset` of providers or capabilities). Provider ordering
  comes only from the explicit configuration sequence plus the sort key above.
- A provider that does not advertise a capability is **never consulted** for
  that capability; the symbolic path executes unchanged.
- Zero providers configured ⇒ every capability resolves to none ⇒ runtime
  behavior is identical to v0.4.1.
- Every registered provider MUST expose `capabilities()`; a provider object
  that does not is a configuration error and is rejected at registration.

### Provider failure semantics

> Amendment (v0.5.0, Phase 1): a crashed provider is never silently treated as
> absent.

If a resolved provider raises while executing a hook, the runtime:

1. records the failure in provider provenance;
2. raises an explicit `ProviderFailure` (capability, provider, cause);
3. applies the **configured** fallback policy — or, when none is configured,
   fails the cycle.

There is **no silent fallback**: provider output is advisory, but a provider
error is an error. Treating a failed provider as "not there" would make
research results uninterpretable. In v0.5.0 only the fail-fast policy exists
(any provider error fails the cycle); any explicit fallback policy requires a
further ADR amendment.

### Provider provenance (trace)

> Amendment (v0.5.0, Phase 1): traces distinguish "provider not configured"
> from "provider configured but returned nothing".

The runtime trace records, per capability consulted, the resolved provider and
the invocation outcome (executed with an output summary, or failed). These two
states are scientifically different and MUST be distinguishable:

- **provider not configured** ⇒ no provider consulted, no provenance entry,
  recorded trace identical to v0.4.1;
- **provider configured but returned nothing** ⇒ provider called and
  provenance records the executed call with an empty output summary.

Provider provenance is **trace metadata only**. It is deliberately NOT part of
`GoldenTrace` or the golden fingerprint, so replay and the benchmark contract
are unchanged.

### Dependency direction (frozen)

```text
                 ┌─────────────────┐
                 │   theo-core     │
                 │                 │
                 │ protocols       │
                 │ snapshots       │
                 │ symbolic runtime│
                 └────────┬────────┘
                          │
                          │ implements
                          ▼
                 ┌─────────────────┐
                 │ theo-providers  │
                 │                 │
                 │ heuristic       │
                 │ sklearn         │
                 │ SLM             │
                 │ LLM             │
                 └─────────────────┘
```

- **theo-core MUST NOT import theo-providers.**
- **theo-providers MAY depend on theo-core contracts** (`theo_core.models.ports`).
- **theo-providers MUST NOT import `theo_core.symbolic`.**

This direction is enforced mechanically by an architectural test, kept
permanently as the firewall around the v0.4.1 baseline.

### Hook locations

Providers are consulted only at these five symbolic hooks; no rule-generation
hook exists at runtime:

| Symbolic hook | Provider consultation |
| --- | --- |
| `InferenceEngine.forward_chain` (rule selection) | `SalienceProvider.rank_rules` (reorder only) |
| `HypothesisEngine.generate_hypotheses` | `HypothesisProposalProvider.propose_hypotheses` |
| `HypothesisEngine.evaluate_hypotheses` | `CalibrationProvider.score_hypotheses` |
| `DecisionEngine.make_decision` (confidence) | `CalibrationProvider.score_confidence` |
| `GoalManager.select_top_goal` | `SalienceProvider.rank_goals` (reorder only) |

Provider output is advisory. The symbolic runtime decides; provider output never
directly produces `Belief`, `Thought`, `DecisionRecord`, or `ConflictRecord`
records, and never writes to the knowledge base.

### Gap analysis ownership

`GapAnalyzer` is owned by theo-core, in `theo_core.symbolic.analysis`. It
consumes benchmark outcomes and emits `KnowledgeGapReport` — a snapshot DTO in
the ports contracts, structured by remediation action (missing premises, weak
rule coverage, unresolved ambiguities, low-confidence regions, retrieval
failures, contradiction patterns, benchmark failures). It does not expose raw
runtime state.

## Rationale

- Snapshot-only signatures keep theo-providers decoupled from the symbolic
  model, making the "replaceable implementation" principle enforceable in code.
- A minimal `ProviderExecution` keeps the replay contract canonical: only what
  reproduces the cognitive result.
- Strict grounding rejection gives a conformance/security boundary that cannot
  be quietly weakened by a provider.
- Capability discovery lets the runtime attach exactly the providers a hook
  needs, with zero providers meaning identical v0.4.1 behavior.
- The frozen dependency direction turns a governance preference into an
  automated test.

## Trade-offs

- Snapshot DTOs duplicate shape information from internal models; the
  conversion layer is theo-core's responsibility and must be tested.
- Providers can improve reasoning quality only where a hook exposes a surface;
  v0.5 deliberately limits surface area to five hooks.
- The benchmark determinism contract is strict: some providers will be excluded
  from the benchmark gate even though they are usable in exploration mode.
- Offline rule discovery defers an in-band "learning from experience" path in
  favor of measurable knowledge engineering.

## Consequences

- theo-core declares no statistical/neural runtime dependencies; provider
  dependencies live in theo-providers.
- All pre-v0.5 tests must pass unchanged when provider contracts are added;
  the interface tests must pass on top. No existing test may be modified to
  keep the baseline green.
- Benchmark runs attach only benchmark-compatible providers; exploration runs
  record seed, temperature, and sampling in the trace.
- Any future provider kind is added by implementing a protocol, never by
  editing the symbolic runtime.

## Future Alternatives

- If belief calibration acquires a concrete consumer, `CalibrationProvider`
  may be extended deliberately via an ADR amendment rather than preemptively.
- If rule learning becomes safe, an explicit runtime rule-adoption protocol
  could be introduced behind the same grounding and determinism contracts.
- The provider boundary could generalize to non-neural advisors (external
  solvers, databases) without changing the snapshot contracts.

## Ratification

Ratified 2026-08-09 after Phase A review. No architectural objection; acceptance
covers the snapshot-only provider boundary, `ProviderExecution` as the replay
contract, strict grounding rejection, the two-operation `CalibrationProvider`,
offline `RuleDiscoveryProvider`, the frozen theo-core → theo-providers dependency
direction, and the absence of implementation choices in this ADR. The defining
principle — *neural providers are replaceable implementations, not architectural
components* — is enforced by the permanent architectural firewall test (Phase D).

Amended 2026-08-09 (v0.5.0, Phase 1) with three normative contracts verified by
the Phase 1 acceptance suite: deterministic provider resolution (one provider
per capability, explicit sort key, no set-order dependence), fail-fast provider
failure semantics (explicit `ProviderFailure`, no silent fallback), and
provider provenance as trace metadata outside the replay fingerprint.

## References

- ADR-0004: Interface & Versioning Strategy
- ADR-0012: Plugin SDK Architecture
- ADR-0026: Canon Law 4 Verdict — Perception Enters Cognition as Evidence
- ADR-0027: Runtime Unification — Symbolic Pipeline as Canonical Runtime
- `docs/reference/v0.4.1-reference.md`
- `docs/research/weakest-domain-audit-v0.4.1.md`
- `docs/implementation/v0.5-neural-interface-plan.md`
