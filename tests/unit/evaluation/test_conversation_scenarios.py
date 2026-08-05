"""Multi-turn conversation benchmark suite for THEO v0.2.0-final."""

from __future__ import annotations

import os

from theo_core.composition.bootstrap import bootstrap


class TestMultiTurnConversationScenarios:
    """Tests multi-turn cognitive conversation scenarios."""

    def test_5_turn_cognitive_dialogue(self, tmp_path: object) -> None:
        """5-turn cognitive conversation scenario testing perception, context, memory, and rules."""
        mem_file = str(tmp_path) + "/scen_mem.json"
        know_file = str(tmp_path) + "/scen_know.json"
        trace_dir = str(tmp_path) + "/traces"

        container = bootstrap(
            memory_file=mem_file,
            knowledge_file=know_file,
            trace_dir=trace_dir,
        )

        # Turn 1: Greeting
        s1 = container.cognitive_engine.process("Hello Theo")
        assert s1.cognitive_depth == 12
        assert "Hello" in s1.response_text

        # Turn 2: Identity statement
        s2 = container.cognitive_engine.process("My name is Falcon")
        assert "Falcon" in s2.response_text
        assert container.context_manager.get("user.name") == "Falcon"

        # Turn 3: Preference statement
        s3 = container.cognitive_engine.process("I like astronomy")
        assert "astronomy" in s3.response_text

        # Turn 4: Second greeting (context & memory recall)
        s4 = container.cognitive_engine.process("Hello again")
        assert "Falcon" in s4.response_text

        # Turn 5: Ask recommendation (concept graph traversal)
        s5 = container.cognitive_engine.process("Recommend a topic")
        assert s5.cognitive_depth == 12

        if os.path.exists(mem_file):
            os.remove(mem_file)
        if os.path.exists(know_file):
            os.remove(know_file)
