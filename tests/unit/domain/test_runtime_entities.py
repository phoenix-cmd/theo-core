"""Tests for runtime domain entities."""

from __future__ import annotations

from theo_core.domain.runtime.entities.cognitive_state import CognitiveState
from theo_core.domain.runtime.entities.conversation import Conversation
from theo_core.domain.runtime.entities.goal import Goal, GoalPriority, GoalStatus
from theo_core.domain.runtime.entities.message import Message, MessageRole
from theo_core.domain.runtime.entities.percept import Percept, PerceptModality
from theo_core.domain.runtime.entities.plan import Action, ActionStatus, Plan
from theo_core.domain.runtime.entities.reflection import Reflection
from theo_core.domain.runtime.entities.thought import Thought


class TestMessage:
    """Tests for the Message value object."""

    def test_create_message(self) -> None:
        """A message should store role and content."""
        msg = Message(role=MessageRole.USER, content="Hello, Theo.")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, Theo."
        assert msg.id is not None

    def test_message_is_frozen(self) -> None:
        """Messages should be immutable."""
        msg = Message(role=MessageRole.USER, content="test")
        try:
            msg.content = "changed"  # type: ignore[misc]
            was_frozen = False
        except Exception:
            was_frozen = True
        assert was_frozen


class TestConversation:
    """Tests for the Conversation aggregate."""

    def test_empty_conversation(self) -> None:
        """A new conversation should have zero messages."""
        conv = Conversation()
        assert conv.message_count == 0
        assert conv.last_message is None

    def test_append_message(self) -> None:
        """Appending a message should increase the count."""
        conv = Conversation()
        msg = Message(role=MessageRole.USER, content="Hello")
        conv.append(msg)
        assert conv.message_count == 1
        assert conv.last_message == msg


class TestThought:
    """Tests for the Thought value object."""

    def test_create_thought(self) -> None:
        """A thought should have content, confidence, and source."""
        thought = Thought(content="The sky is blue", confidence=0.9, source="reasoning")
        assert thought.confidence == 0.9
        assert thought.source == "reasoning"


class TestReflection:
    """Tests for the Reflection entity."""

    def test_create_reflection(self) -> None:
        """A reflection should contain insights."""
        thought = Thought(content="Test", confidence=0.8, source="test")
        refl = Reflection(
            source_thoughts=(thought,),
            insights=("Insight 1", "Insight 2"),
            confidence=0.7,
        )
        assert len(refl.insights) == 2
        assert len(refl.source_thoughts) == 1


class TestPlan:
    """Tests for Plan and Action entities."""

    def test_create_plan(self) -> None:
        """A plan should contain actions toward a goal."""
        action = Action(capability="summarization", parameters={"text": "test"})
        plan = Plan(goal="Summarize the document", actions=[action])
        assert plan.action_count == 1
        assert not plan.is_complete

    def test_plan_completion(self) -> None:
        """A plan is complete when all actions are in terminal states."""
        action = Action(
            capability="test",
            status=ActionStatus.COMPLETED,
        )
        plan = Plan(goal="Test", actions=[action])
        assert plan.is_complete


class TestGoal:
    """Tests for the Goal entity."""

    def test_create_goal(self) -> None:
        """A goal should have a description and priority."""
        goal = Goal(description="Learn about the user", priority=GoalPriority.HIGH)
        assert goal.status == GoalStatus.ACTIVE
        assert goal.priority == GoalPriority.HIGH

    def test_goal_id_is_derived_deterministically(self) -> None:
        """A goal should derive a deterministic GoalId from its description."""
        goal = Goal(description="AcknowledgeGreeting")
        assert goal.goal_id is not None
        assert goal.goal_id.value == "goal://acknowledgegreeting"

    def test_goal_id_is_stable_across_instances(self) -> None:
        """Identical descriptions should yield identical GoalIds."""
        g1 = Goal(description="Learn about the user")
        g2 = Goal(description="Learn about the user")
        assert g1.goal_id == g2.goal_id
        assert g1.goal_id is not None
        assert g2.goal_id is not None
        assert g1.goal_id.value == g2.goal_id.value


class TestPercept:
    """Tests for the Percept value object."""

    def test_create_text_percept(self) -> None:
        """A text percept should normalize raw text input."""
        percept = Percept(modality=PerceptModality.TEXT, content="Hello world")
        assert percept.modality == PerceptModality.TEXT
        assert percept.confidence == 1.0


class TestCognitiveState:
    """Tests for the CognitiveState entity."""

    def test_create_state(self) -> None:
        """A cognitive state should track context and visited stages."""
        state = CognitiveState(
            raw_input="hello",
            context={"active_user": "test_user"},
        )
        state.visit_stage("perception")
        assert state.context["active_user"] == "test_user"
        assert state.cognitive_depth == 1
