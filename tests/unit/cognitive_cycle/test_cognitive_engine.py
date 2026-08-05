"""Tests for CognitiveEngine 12-stage cognitive pipeline."""

from __future__ import annotations

import os

from theo_core.composition.bootstrap import bootstrap
from theo_core.domain.runtime.entities.cognitive_state import CognitiveState


class TestCognitiveEngine:
    """Tests for end-to-end 12-stage cognitive execution."""

    def test_greeting_cycle(self, tmp_path: object) -> None:
        """Greeting input should visit all stages and respond deterministically."""
        json_file = str(tmp_path) + "/test_mem.json"
        container = bootstrap(memory_file=json_file)

        state = container.cognitive_engine.process("Hello Theo")

        assert isinstance(state, CognitiveState)
        assert state.cognitive_depth == 12
        assert "Hello" in state.response_text
        assert state.active_goal is not None
        assert state.active_goal.description == "AcknowledgeGreeting"

        if os.path.exists(json_file):
            os.remove(json_file)

    def test_name_statement_cycle(self, tmp_path: object) -> None:
        """Identity introduction input should extract name into context and memory."""
        json_file = str(tmp_path) + "/test_mem_name.json"
        container = bootstrap(memory_file=json_file)

        state = container.cognitive_engine.process("My name is Falcon")

        assert state.cognitive_depth == 12
        assert "Falcon" in state.response_text
        assert container.context_manager.get("user.name") == "Falcon"

        if os.path.exists(json_file):
            os.remove(json_file)

    def test_multi_turn_name_recall(self, tmp_path: object) -> None:
        """Second greeting turn should use name stored in active context and memory."""
        json_file = str(tmp_path) + "/test_mem_recall.json"
        container = bootstrap(memory_file=json_file)

        container.cognitive_engine.process("My name is Falcon")
        state2 = container.cognitive_engine.process("Hello")

        assert "Falcon" in state2.response_text

        if os.path.exists(json_file):
            os.remove(json_file)
