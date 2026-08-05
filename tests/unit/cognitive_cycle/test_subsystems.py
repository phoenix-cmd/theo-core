"""Unit tests for Milestone 1 and 2 deterministic cognitive subsystems."""

from __future__ import annotations

import os

from theo_core.cognition.inference.engine import InferenceEngine
from theo_core.cognition.inference.strategies.rule_based import RuleBasedStrategy
from theo_core.cognition.planning.planner import RuleBasedPlanner
from theo_core.context.session.active_context import InMemoryContextManager
from theo_core.domain.runtime.entities.cognitive_state import CognitiveState
from theo_core.domain.runtime.entities.decision import Decision
from theo_core.domain.runtime.entities.goal import GoalStatus
from theo_core.domain.runtime.entities.memory_entry import MemoryEntry
from theo_core.goals.manager.goal_manager import GoalManager
from theo_core.perception.text.data_driven_processor import DataDrivenPerceptionProcessor
from theo_core.response.template.generator import TemplateResponseGenerator


class TestDataDrivenPerceptionProcessor:
    """Tests for DataDrivenPerceptionProcessor."""

    def test_perceive_greeting(self) -> None:
        """Perceive greeting text."""
        proc = DataDrivenPerceptionProcessor()
        percept = proc.perceive("Hello Theo")
        assert percept.metadata["intent"] == "greeting"
        assert percept.confidence == 1.0

    def test_perceive_name_identity(self) -> None:
        """Perceive identity claim text."""
        proc = DataDrivenPerceptionProcessor()
        percept = proc.perceive("My name is Falcon")
        assert percept.metadata["intent"] == "introduce_identity"
        assert percept.metadata["facts"].get("user.name") == "Falcon"

    def test_perceive_preference(self) -> None:
        """Perceive user preference statement."""
        proc = DataDrivenPerceptionProcessor()
        percept = proc.perceive("I like Python")
        assert percept.metadata["intent"] == "express_preference"
        assert percept.metadata["sentiment"] == "positive"


class TestInMemoryContextManager:
    """Tests for InMemoryContextManager."""

    def test_context_lifecycle(self) -> None:
        """Context should store, increment turns, snapshot, and clear."""
        mgr = InMemoryContextManager("falcon_user")
        assert mgr.get("active_user") == "falcon_user"
        turns = mgr.increment_turns()
        assert turns == 1
        mgr.set("key1", "val1")
        snap = mgr.snapshot()
        assert snap["key1"] == "val1"
        mgr.clear()
        assert mgr.get("key1") is None
        assert mgr.get("active_user") == "falcon_user"


class TestGoalManager:
    """Tests for GoalManager."""

    def test_goal_inference(self) -> None:
        """GoalManager should infer goals from percept intent."""
        gm = GoalManager()
        proc = DataDrivenPerceptionProcessor()

        p1 = proc.perceive("Hello")
        g1 = gm.select_top_goal(p1)
        assert g1.description == "AcknowledgeGreeting"

        p2 = proc.perceive("Recommend a book")
        g2 = gm.select_top_goal(p2)
        assert g2.description == "ProvideRecommendation"

        gm.complete_goal(g1.id)
        assert g1.status == GoalStatus.COMPLETED

        gm.abandon_goal(g2.id)
        assert g2.status == GoalStatus.ABANDONED


class TestRuleBasedPlannerAndInference:
    """Tests for Planner and InferenceEngine."""

    def test_planner_actions(self) -> None:
        """Planner should generate actions for goal."""
        planner = RuleBasedPlanner()
        plan = planner.plan("ProvideRecommendation")
        assert plan.action_count > 0

    def test_inference_evaluation(self, tmp_path: object) -> None:
        """Inference strategy should produce decision summary."""
        json_file = str(tmp_path) + "/sub_infer.json"
        proc = DataDrivenPerceptionProcessor()
        gm = GoalManager()
        strategy = RuleBasedStrategy()
        engine = InferenceEngine(strategy)

        state = CognitiveState(raw_input="I like astronomy")
        state.percept = proc.perceive("I like astronomy")
        state.active_goal = gm.select_top_goal(state.percept)

        result = engine.infer(state)
        assert "candidate_response" in result
        assert engine.active_strategy_name == "RuleBasedStrategy"

        if os.path.exists(json_file):
            os.remove(json_file)


class TestTemplateResponseGenerator:
    """Tests for TemplateResponseGenerator."""

    def test_generate_response(self) -> None:
        """Generator should format Decision object."""
        gen = TemplateResponseGenerator()
        decision = Decision(response="Hello Falcon!", goal="AcknowledgeGreeting")
        out = gen.generate(decision)
        assert out == "Hello Falcon!"


class TestMemoryEntry:
    """Tests for MemoryEntry entity."""

    def test_memory_entry_confirmation(self) -> None:
        """Confirming a memory entry should increment times_confirmed."""
        mem = MemoryEntry(id="mem-000001", key="user.name", value="Falcon")
        assert mem.times_confirmed == 1
        mem.confirm()
        assert mem.times_confirmed == 2
