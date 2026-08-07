"""GoalPort — interface for the goal management system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from theo_core.domain.runtime.entities.goal import Goal
    from theo_core.domain.runtime.entities.percept import Percept


class GoalPort(ABC):
    """Abstract interface for goal management.

    Goals answer "Why?" and drive the planning subsystem.
    """

    @abstractmethod
    def select_top_goal(self, percept: Percept | None = None) -> Goal:
        """Select or infer the highest priority active goal.

        Args:
            percept: Optional Percept to inform goal selection.

        Returns:
            The selected top priority Goal.

        """

    @abstractmethod
    def add_goal(self, goal: Goal) -> None:
        """Add a goal to the active goal stack.

        Args:
            goal: The goal to add.

        """

    @abstractmethod
    def get_active_goals(self) -> list[Goal]:
        """Return all currently active goals ordered by priority.

        Returns:
            A list of active goals.

        """

    @abstractmethod
    def complete_goal(self, goal_id: UUID) -> None:
        """Mark a goal as completed.

        Args:
            goal_id: The ID of the goal to complete.

        """

    @abstractmethod
    def abandon_goal(self, goal_id: UUID) -> None:
        """Abandon a goal.

        Args:
            goal_id: The ID of the goal to abandon.

        """
