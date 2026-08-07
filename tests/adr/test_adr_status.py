"""ADR governance — every ADR MUST declare a valid RFC-style status.

Valid status keywords follow the ADR template: Proposed, Accepted,
Deprecated, or Superseded (optionally qualified, e.g. "Accepted (v0.3.0)").
"""

from __future__ import annotations

import re
from pathlib import Path

ADRs = Path(__file__).resolve().parents[2] / "adr"
STATUS_VALUE = re.compile(r"^(Proposed|Accepted|Deprecated|Superseded)(\s*\(.+\))?$")


class TestAdrStatus:
    def test_every_adr_has_status_section(self) -> None:
        for path in sorted(ADRs.glob("ADR-*.md")):
            text = path.read_text(encoding="utf-8")
            assert "## Status" in text, f"{path.name} lacks '## Status'"

    def test_status_value_is_valid(self) -> None:
        for path in sorted(ADRs.glob("ADR-*.md")):
            lines = path.read_text(encoding="utf-8").splitlines()
            idx = lines.index("## Status")
            value = next(line.strip() for line in lines[idx + 1 :] if line.strip())
            assert STATUS_VALUE.match(value), (
                f"{path.name} has invalid status: {value!r}"
            )
