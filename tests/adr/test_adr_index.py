"""ADR governance — index completeness and numbering integrity.

Every ADR file on disk MUST be listed in the index (adr/README.md) and every
index entry MUST resolve to an existing file. ADR numbers MUST be sequential
(0001..N) with no gaps.
"""

from __future__ import annotations

import re
from pathlib import Path

ADRs = Path(__file__).resolve().parents[2] / "adr"
ADR_PATTERN = re.compile(r"^ADR-(\d{4})-[a-z0-9-]+\.md$")
INDEX_ENTRY = re.compile(r"\[ADR-(\d+): [^\]]+\]\(([^)]+\.md)\)")


def _adr_files() -> list[Path]:
    return sorted(ADRs.glob("ADR-*.md"))


def _adr_number(path: Path) -> int:
    match = ADR_PATTERN.match(path.name)
    assert match is not None, f"Bad ADR filename: {path.name}"
    return int(match.group(1))


class TestAdrIndex:
    def test_every_adr_listed_in_index(self) -> None:
        index = (ADRs / "README.md").read_text(encoding="utf-8")
        linked = {m.group(2) for m in INDEX_ENTRY.finditer(index)}
        on_disk = {p.name for p in _adr_files()}
        missing = on_disk - linked
        assert not missing, f"ADR files missing from index: {sorted(missing)}"

    def test_every_index_entry_resolves(self) -> None:
        index = (ADRs / "README.md").read_text(encoding="utf-8")
        for match in INDEX_ENTRY.finditer(index):
            target = ADRs / match.group(2)
            assert target.is_file(), f"Index links to nonexistent file: {match.group(2)}"

    def test_adr_filenames_match_convention(self) -> None:
        for path in _adr_files():
            assert ADR_PATTERN.match(path.name), f"Bad ADR filename: {path.name}"

    def test_adr_numbers_are_sequential(self) -> None:
        numbers = sorted(_adr_number(p) for p in _adr_files())
        expected = list(range(1, numbers[-1] + 1))
        assert numbers == expected, f"ADR numbering has gaps: {numbers}"
