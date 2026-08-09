"""Port contract tests — the four provider protocols (ADR-0028)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path


def _imported_modules(path: Path) -> set[str]:
    """Return the set of module roots imported (including ``from`` targets)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def _neural_path() -> Path:
    return (
        Path(__file__).parents[2]
        / "src"
        / "theo_core"
        / "models"
        / "ports"
        / "neural.py"
    )


class TestNeuralProtocols:
    def test_neural_module_imports_no_symbolic(self) -> None:
        """Provider protocols MUST NOT import ``theo_core.symbolic``."""
        imports = _imported_modules(_neural_path())
        assert not any(name.startswith("theo_core.symbolic") for name in imports)

    def test_neural_module_type_imports_are_ports_only(self) -> None:
        """Annotation imports resolve only from ``theo_core.models.ports``."""
        imports = _imported_modules(_neural_path())
        assert any(name == "theo_core.models.ports.snapshots" for name in imports)
        assert not any(name.startswith("theo_core.symbolic") for name in imports)

    def test_all_protocol_signatures_are_snapshot_only(self) -> None:
        """No parameter annotation may name a symbolic runtime class."""
        source = _neural_path().read_text(encoding="utf-8")
        forbidden = {
            "Hypothesis",
            "Belief",
            "InferenceRule",
            "Goal",
            "DecisionRecord",
            "Concept",
            "SymbolicId",
        }
        for token in forbidden:
            assert f"{token}: " not in source, f"Symbolic type leaked: {token}"

    @staticmethod
    def _public_methods(cls: type) -> set[str]:
        members = inspect.getmembers(cls, inspect.isfunction)
        return {name for name, _ in members if not name.startswith("_")}

    def test_calibration_provider_has_only_two_methods(self) -> None:
        """CalibrationProvider contains exactly score_hypotheses/score_confidence."""
        from theo_core.models.ports.neural import CalibrationProvider

        names = self._public_methods(CalibrationProvider)
        assert names == {"capabilities", "score_hypotheses", "score_confidence"}

    def test_protocol_method_names(self) -> None:
        from theo_core.models.ports.neural import (
            HypothesisProposalProvider,
            RuleDiscoveryProvider,
            SalienceProvider,
        )

        assert self._public_methods(HypothesisProposalProvider) == {
            "capabilities",
            "propose_hypotheses",
        }
        assert self._public_methods(SalienceProvider) == {
            "capabilities",
            "rank_goals",
            "rank_rules",
        }
        assert self._public_methods(RuleDiscoveryProvider) == {
            "capabilities",
            "discover_rules",
        }
