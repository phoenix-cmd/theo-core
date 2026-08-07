# Shared Symbolic Primitives

Shared value objects, protocols, error hierarchy, and ordering utilities used by every symbolic subsystem.

**Architectural rule**: This package has zero cognitive semantics. It provides only structural foundations.

## Contents

- `identifiers.py` — `SymbolicId` frozen value object with URI validation
- `versioning.py` — `SchemaVersion` for serialized artifact versioning
- `errors.py` — `SymbolicError` hierarchy
- `protocols.py` — `Repository[T]` and `Serializer[T]` protocols
- `ordering.py` — Deterministic iteration utilities
