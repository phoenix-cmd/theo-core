"""Property tests — persistence serialization idempotence.

Saving committed symbolic state, loading it back, and saving again MUST
produce byte-identical files: canonical serialization is idempotent and
deterministic regardless of the input that produced the state.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from theo_core.symbolic.persistence.store import SymbolicStateStore
from theo_core.symbolic.pipeline import SymbolicCognitivePipeline

_INPUTS = st.text(
    alphabet=st.characters(min_codepoint=0x20, blacklist_categories={"Cs"}),
    min_size=1,
    max_size=40,
)


class TestSerializationIdempotence:
    @given(_INPUTS)
    @settings(max_examples=25)
    def test_save_load_save_is_byte_identical(self, input_text: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            pipeline = SymbolicCognitivePipeline()
            pipeline.execute_cycle(input_text)

            first_store = SymbolicStateStore(path)
            first_store.save(pipeline.state)
            first_bytes = path.read_bytes()

            loaded = first_store.load()
            assert loaded is not None

            second_store = SymbolicStateStore(path)
            second_store.save(loaded)
            second_bytes = path.read_bytes()

            assert first_bytes == second_bytes
