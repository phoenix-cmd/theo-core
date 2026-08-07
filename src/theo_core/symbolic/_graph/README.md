# Generic Graph Library

Reusable directed graph engine. Strictly structural — contains no cognitive semantics.

Every graph-based symbolic subsystem (Concepts, Thoughts, Beliefs, Hypotheses, Decisions) delegates to this library via `Graph[N, E]`.

## Contents

- `types.py` — `NodeId`, `EdgeKey`, type variables
- `graph.py` — `Graph[N, E]` data storage (no algorithms)
- `traversal.py` — BFS, DFS, shortest path, ancestors, descendants
- `validation.py` — Structural integrity checks
- `serialization.py` — Versioned JSON with checksums
- `repository.py` — Generic persistence protocol and in-memory implementation
