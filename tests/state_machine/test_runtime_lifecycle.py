"""State machine — SymbolicRuntime start/stop lifecycle behavior.

start() restores committed state and is idempotent; stop() persists and is
idempotent. Restart cycles (start -> stop -> start) are supported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from theo_core.symbolic.persistence.store import SymbolicStateStore
from theo_core.symbolic.runtime import SymbolicRuntime

if TYPE_CHECKING:
    from pathlib import Path


class TestRuntimeLifecycle:
    def test_start_then_stop_cycle(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        runtime = SymbolicRuntime(store=SymbolicStateStore(path))
        runtime.start()
        assert runtime.is_started
        runtime.stop()
        assert not runtime.is_started

    def test_double_start_is_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        runtime = SymbolicRuntime(store=SymbolicStateStore(path))
        runtime.start()
        runtime.start()
        assert runtime.is_started

    def test_double_stop_is_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        runtime = SymbolicRuntime(store=SymbolicStateStore(path))
        runtime.start()
        runtime.stop()
        runtime.stop()
        assert not runtime.is_started

    def test_stop_before_start_is_safe(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        runtime = SymbolicRuntime(store=SymbolicStateStore(path))
        runtime.stop()
        assert not runtime.is_started
        assert path.exists()

    def test_restart_cycle_restores_state(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        first = SymbolicRuntime(store=SymbolicStateStore(path))
        first.start()
        first.process("first input")
        first.stop()

        second = SymbolicRuntime(store=SymbolicStateStore(path))
        second.start()
        assert (
            second.pipeline.beliefs.node_count == first.pipeline.beliefs.node_count
        )
        second.stop()
