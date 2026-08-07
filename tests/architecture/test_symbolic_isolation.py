"""Architecture enforcement test — verifies subsystem isolation for symbolic packages."""

from pathlib import Path


def _symbolic_dir() -> Path:
    return Path(__file__).parents[2] / "src" / "theo_core" / "symbolic"


class TestSymbolicArchitectureIsolation:
    def test_concepts_package_has_no_cross_domain_imports(self) -> None:
        """Verify that concepts/ package imports only from _primitives, _graph, or stdlib."""
        concepts_dir = _symbolic_dir() / "concepts"
        assert concepts_dir.exists(), f"Concepts directory {concepts_dir} does not exist"

        forbidden_imports = [
            "theo_core.cognitive_cycle",
            "theo_core.symbolic.beliefs",
            "theo_core.symbolic.thoughts",
            "theo_core.symbolic.inference",
            "theo_core.symbolic.hypotheses",
            "theo_core.symbolic.constraints",
            "theo_core.symbolic.conflict",
            "theo_core.symbolic.decisions",
            "theo_core.symbolic.scheduler",
        ]

        py_files = list(concepts_dir.glob("*.py"))
        assert len(py_files) > 0

        for py_file in py_files:
            content = py_file.read_text(encoding="utf-8")
            for forbidden in forbidden_imports:
                assert forbidden not in content, (
                    f"Forbidden import {forbidden!r} found in {py_file.name}"
                )

    def test_no_legacy_engine_imports_inside_symbolic(self) -> None:
        """The canonical symbolic stack MUST not import the legacy 12-stage engine."""
        for py_file in sorted(_symbolic_dir().rglob("*.py")):
            content = py_file.read_text(encoding="utf-8")
            assert "theo_core.cognitive_cycle" not in content, (
                f"Legacy cognitive_cycle import found in {py_file.relative_to(_symbolic_dir())}"
            )

    def test_pipeline_does_not_import_response_renderer(self) -> None:
        """Canon Law 6: the pipeline MUST not render language; rendering lives at
        the boundary (theo_core.symbolic.response), never inside the pipeline."""
        pipeline = _symbolic_dir() / "pipeline.py"
        content = pipeline.read_text(encoding="utf-8")
        assert "theo_core.symbolic.response" not in content
