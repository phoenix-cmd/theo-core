"""Unit tests for SymbolicStateStore checksummed persistence."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from theo_core.symbolic._primitives.errors import ChecksumMismatchError
from theo_core.symbolic.persistence.store import SymbolicStateStore
from theo_core.symbolic.pipeline import SymbolicCognitivePipeline

if TYPE_CHECKING:
    from pathlib import Path


class TestSymbolicStateStore:
    def test_load_missing_file_returns_none(self, tmp_path: Path) -> None:
        """Loading a nonexistent state file should return None."""
        store = SymbolicStateStore(tmp_path / "missing.json")
        assert store.load() is None

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """A saved committed state should roundtrip losslessly."""
        path = tmp_path / "state.json"
        pipeline = SymbolicCognitivePipeline()
        pipeline.execute_cycle("persist this input")
        state = pipeline.state

        store = SymbolicStateStore(path)
        store.save(state)
        loaded = store.load()

        assert loaded is not None
        assert {b.id.value for b in loaded.beliefs.get_active_beliefs()} == {
            b.id.value for b in state.beliefs.get_active_beliefs()
        }
        assert loaded.concepts.node_count == state.concepts.node_count
        assert loaded.thoughts.node_count == state.thoughts.node_count
        assert loaded.percept is not None
        assert state.percept is not None
        assert loaded.percept.content == state.percept.content

    def test_corrupt_payload_raises_checksum_mismatch(self, tmp_path: Path) -> None:
        """Tampered payloads MUST raise, never be silently swallowed."""
        path = tmp_path / "state.json"
        pipeline = SymbolicCognitivePipeline()
        pipeline.execute_cycle("persist this input")
        store = SymbolicStateStore(path)
        store.save(pipeline.state)

        data = json.loads(path.read_text(encoding="utf-8"))
        data["state"]["beliefs"]["nodes"][0]["id"] = "belief://tampered"
        path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ChecksumMismatchError):
            store.load()
