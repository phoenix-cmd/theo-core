"""GoalManager — goal stack engine for prioritizing cognitive objectives.

Answers 'Why am I responding?' by pushing, prioritizing, completing, and popping Goals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from theo_core.domain.runtime.entities.goal import Goal, GoalPriority, GoalStatus
from theo_core.domain.runtime.ports.goal import GoalPort

if TYPE_CHECKING:
    from uuid import UUID

    from theo_core.domain.runtime.entities.percept import Percept


class GoalManager(GoalPort):
    """Priority Goal Stack Manager.

    Maintains a list of active goals sorted by priority.
    """

    def __init__(self) -> None:
        """Initialize an empty goal stack."""
        self._goals: list[Goal] = []

    def add_goal(self, goal: Goal) -> None:
        """Add a goal to the goal stack.

        Args:
            goal: The Goal to add.

        """
        self._goals.append(goal)
        self._sort()

    def get_active_goals(self) -> list[Goal]:
        """Return all active goals sorted by priority.

        Returns:
            List of active Goal objects.

        """
        return [g for g in self._goals if g.status == GoalStatus.ACTIVE]

    def select_top_goal(self, percept: Percept | None = None) -> Goal:
        """Select or infer the highest priority active goal based on percept intent.

        Args:
            percept: Optional Percept to inform goal selection.

        Returns:
            The selected top priority Goal.

        """
        intent = percept.metadata.get("intent", "unknown") if percept else "unknown"

        # Infer goal from intent if stack has no active goal
        if intent == "greeting":
            g = Goal(description="AcknowledgeGreeting", priority=GoalPriority.HIGH)
            self.add_goal(g)
            return g

        if intent in ("introduce_identity", "express_preference"):
            g = Goal(description="RememberFact", priority=GoalPriority.HIGH)
            self.add_goal(g)
            return g

        if intent == "ask_recommendation":
            g = Goal(description="ProvideRecommendation", priority=GoalPriority.HIGH)
            self.add_goal(g)
            return g

        if intent == "ask_question":
            g = Goal(description="AnswerQuestion", priority=GoalPriority.HIGH)
            self.add_goal(g)
            return g

        active = self.get_active_goals()
        if active:
            return active[0]

        default_goal = Goal(description="MaintainConversation", priority=GoalPriority.MEDIUM)
        self.add_goal(default_goal)
        return default_goal

    def complete_goal(self, goal_id: UUID) -> None:
        """Mark a goal as completed.

        Args:
            goal_id: ID of goal to complete.

        """
        for g in self._goals:
            if g.id == goal_id:
                g.status = GoalStatus.COMPLETED

    def abandon_goal(self, goal_id: UUID) -> None:
        """Mark a goal as abandoned.

        Args:
            goal_id: ID of goal to abandon.

        """
        for g in self._goals:
            if g.id == goal_id:
                g.status = GoalStatus.ABANDONED

    def _sort(self) -> None:
        """Sort goals by priority enum value."""
        priority_order = {
            GoalPriority.CRITICAL: 0,
            GoalPriority.HIGH: 1,
            GoalPriority.MEDIUM: 2,
            GoalPriority.LOW: 3,
            GoalPriority.BACKGROUND: 4,
        }
        self._goals.sort(key=lambda g: priority_order.get(g.priority, 2))
