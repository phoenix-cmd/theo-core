# THEO Cognitive Benchmark Corpus (v0.4.1)

The Cognitive Benchmark Corpus serves as the standardized test battery for evaluating cognitive runtimes (symbolic, neural, and hybrid) against Canon Edition C1.

## Structure

```
tests/benchmarks/
├── README.md
├── commonsense/          # Everyday reasoning and state tracking benchmarks
├── taxonomy/             # Is-A and hierarchy traversal benchmarks
├── contradiction/        # Epistemic conflict resolution benchmarks
├── uncertainty/          # Probabilistic/confidence propagation benchmarks
├── planning/             # Multi-step action sequence benchmarks
└── causal_reasoning/     # Premise-to-conclusion deduction benchmarks
```

## Benchmark Case Schema

Each benchmark test case specifies:
1. `initial_knowledge`: ConceptGraph and BeliefGraph state.
2. `input_percept`: Percept input string.
3. `expected_beliefs`: Expected active BeliefGraph state after cycle execution.
4. `expected_thoughts`: Expected ThoughtGraph DAG structure.
5. `expected_decision`: Expected DecisionRecord outcome.
6. `expected_confidence`: Expected Decimal confidence bounds.
