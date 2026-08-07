"""Version consistency — theo_core/_version.py is the single source of truth.

Every surfaced version string (``theo_core.__version__``, the CLI banner, the
kernel boot events, and trace metadata) derives from this module, and the
hatchling build backend reads its ``__version__`` for the distribution.
"""

from __future__ import annotations

import re
from pathlib import Path

import theo_core
from theo_core._version import __version__

SRC_ROOT = Path(theo_core.__file__).parent
REPO_ROOT = SRC_ROOT.parent.parent


def test_package_version_matches_single_source() -> None:
    assert theo_core.__version__ == __version__


def test_single_source_declares_version() -> None:
    text = (SRC_ROOT / "_version.py").read_text(encoding="utf-8")
    assert f'__version__ = "{__version__}"' in text


def test_version_is_pep440_release() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_build_backend_sources_version_from_file() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in text
    assert "[tool.hatch.version]" in text
    assert 'path = "src/theo_core/_version.py"' in text
