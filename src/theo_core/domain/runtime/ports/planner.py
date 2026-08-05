"""PlannerPort — interface for cognitive planning operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theo_core.domain.runtime.entities.plan import Plan


class PlannerPort(ABC):
    """Abstract interface for cognitive planning.

    The planner generates a sequence of actions to achieve a stated goal.
    """

    @abstractmethod
    def plan(self, goal: str, context: str = "") -> Plan:
        """Generate a plan to achieve the given goal.

        Args:
            goal: The objective to plan for.
            context: Optional additional context.

        Returns:
            A Plan containing ordered Actions.

        """
