# THEO Cognitive Benchmark Corpus (v0.4.1)

The Cognitive Benchmark Corpus is the standardized test battery for evaluating cognitive runtimes (symbolic, neural, and hybrid) against Canon Edition C1. Canon and runtime are frozen at v0.4.1; the corpus is executable governance over the canonical `SymbolicCognitivePipeline`.

## Structure

```
src/theo_core/evaluation/benchmarks/     # Corpus definitions (importable by CLI + tests)
├── __init__.py                           # Registry: DOMAIN_CASES, ALL_CASES, case_by_id
├── causal_reasoning/  (benchmarks/causal_reasoning.py)
├── commonsense.py
├── contradiction.py
├── taxonomy.py
└── uncertainty.py
tests/benchmarks/                         # Pytest battery consuming the corpus
├── test_corpus.py                        # Corpus-wide: every case must pass
├── causal_reasoning/                     # Full GoldenTrace deep-assertion tests
```

Each domain module exports a `CASES: tuple[BenchmarkCase, ...]` with at least five cases.

## Running

```bash
theo benchmark run                          # entire corpus (26 cases)
theo benchmark run --domain causal_reasoning
theo benchmark run --case bm://causal_reasoning/005
```

## Benchmark Case Schema

Each `BenchmarkCase` specifies:

1. `initial_concepts` / `initial_concept_edges`: ConceptGraph state (taxonomy).
2. `initial_beliefs` / `initial_belief_edges`: BeliefGraph state, including CONTRADICTS edges (contradiction).
3. `rules`: InferenceRule library (causal / uncertainty).
4. `percept_input`: the percept string.
5. `expected_beliefs` / `excluded_beliefs`: required / forbidden active propositions post-cycle.
6. `expected_decision_type` + `expected_action_text`: DecisionRecord outcome.
7. `min_confidence` / `max_confidence`: expected decision confidence bounds.
8. `golden_trace`: expected `GoldenTrace` fields (only non-default fields are asserted — retrieved memory, fired rules, derived beliefs, generated hypotheses, resolved conflicts, thought DAG size, decision id, response text).

## GoldenTrace

The pipeline emits a complete structural trace per cycle (Canon Invariant 2 / §6):

- `retrieved_memory_ids`: active committed beliefs entering the cycle.
- `activated_concept_ids`: concept graph nodes present post-cycle.
- `generated_hypothesis_ids`: evaluated candidate hypotheses.
- `fired_rule_ids`: inference rules that produced a step.
- `derived_belief_ids`: beliefs newly derived during the cycle (percept + inference).
- `resolved_conflict_ids`: hypothesis conflicts resolved.
- `thought_dag_node_count`: ThoughtGraph DAG size.
- `decision_id` / `response_text`: the decision reference and raw interpretation.
