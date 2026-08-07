"""ADR governance — recorded decisions MUST map to live code entities.

Each mapping asserts that the decision recorded in an ADR has a concrete
implementation that is still importable. A dropped or renamed module here
signals drift between the architecture records and the codebase.
"""

from __future__ import annotations

import importlib

MAPPINGS: tuple[tuple[str, str, str], ...] = (
    ("0017", "theo_core.symbolic.perception.models", "Percept"),
    ("0019", "theo_core.symbolic.concepts.graph", "ConceptGraph"),
    ("0020", "theo_core.symbolic.thoughts.graph", "ThoughtGraph"),
    ("0020", "theo_core.symbolic.inference.engine", "InferenceEngine"),
    ("0021", "theo_core.symbolic.hypotheses.engine", "HypothesisEngine"),
    ("0021", "theo_core.symbolic.conflict.resolver", "ConflictResolver"),
    ("0022", "theo_core.symbolic.response.port", "ResponseRendererPort"),
    ("0025", "theo_core.symbolic.pipeline", "SymbolicCognitivePipeline"),
    ("0026", "theo_core.symbolic.beliefs.models", "EvidenceTrace"),
    ("0027", "theo_core.symbolic.runtime", "SymbolicRuntime"),
)


class TestAdrDecisionMapping:
    def test_decision_symbols_are_importable(self) -> None:
        for adr, module, symbol in MAPPINGS:
            mod = importlib.import_module(module)
            assert hasattr(mod, symbol), (
                f"ADR-{adr} decision symbol {symbol!r} missing from {module}"
            )
