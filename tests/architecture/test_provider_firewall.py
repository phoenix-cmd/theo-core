"""Architecture enforcement — ADR-0028 provider boundary firewall.

theo-core MUST NOT import theo-providers, and the provider-visible
``theo_core.models.ports`` surface MUST NOT pull in ``theo_core.symbolic``
(providers MAY depend on theo-core contracts and MUST NOT import symbolic
internals).
"""

from __future__ import annotations

import ast
from pathlib import Path

from theo_core.models.ports import __all__ as ports_exports


def _core_dir() -> Path:
    return Path(__file__).parents[2] / "src" / "theo_core"


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


class TestProviderFirewall:
    def test_theo_core_never_imports_theo_providers(self) -> None:
        """theo-core MUST NOT import theo-providers (ADR-0028)."""
        for py_file in sorted(_core_dir().rglob("*.py")):
            imports = _imported_modules(py_file)
            assert not any(name.startswith("theo_providers") for name in imports), (
                f"theo-providers import found in {py_file.relative_to(_core_dir())}"
            )

    def test_ports_package_does_not_import_symbolic(self) -> None:
        """Provider-visible ports MUST NOT transitively load ``theo_core.symbolic``.

        ``converters.py`` is an internal theo-core adapter and is excluded: it
        is never exported, so providers cannot reach it through the ports
        surface.
        """
        ports_dir = _core_dir() / "models" / "ports"
        for py_file in sorted(ports_dir.glob("*.py")):
            if py_file.name == "converters.py":
                continue
            imports = _imported_modules(py_file)
            assert not any(
                name.startswith("theo_core.symbolic") for name in imports
            ), f"Symbolic import found in provider-visible {py_file.name}"

    def test_converters_are_not_exported(self) -> None:
        """Converters must not appear in the provider-visible port surface."""
        assert "converters" not in ports_exports
        assert not any("converter" in name for name in ports_exports)

    def test_protocols_and_capabilities_are_exported(self) -> None:
        """The four protocols and the capability enum are exported."""
        from theo_core.models.ports.neural import (
            CalibrationProvider,
            HypothesisProposalProvider,
            RuleDiscoveryProvider,
            SalienceProvider,
        )
        from theo_core.models.ports.snapshots import ProviderCapabilities

        assert HypothesisProposalProvider is not None
        assert CalibrationProvider is not None
        assert SalienceProvider is not None
        assert RuleDiscoveryProvider is not None
        assert issubclass(ProviderCapabilities, str)
