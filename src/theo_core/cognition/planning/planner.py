"""Cognitive Planner — converts active Goals into Plan action sequences.

Translates 'Why am I responding?' into 'What action sequence should be executed?'.
"""

from __future__ import annotations

from theo_core.cognition.planning.actions import make_action
from theo_core.domain.runtime.entities.plan import Plan
from theo_core.domain.runtime.ports.planner import PlannerPort


class RuleBasedPlanner(PlannerPort):
    """Deterministic rule-based cognitive planner.

    Maps goal descriptions to explicit action lists.
    """

    def plan(self, goal: str, context: str = "") -> Plan:
        """Generate a Plan of actions to achieve the given goal.

        Args:
            goal: Description of the goal to achieve.
            context: Optional context string.

        Returns:
            A Plan containing ordered Action items.

        """
        del context
        actions = []

        if "AcknowledgeGreeting" in goal:
            actions.append(make_action("LookupUserIdentity"))
            actions.append(make_action("FormatGreetingResponse"))

        elif "RememberFact" in goal:
            actions.append(make_action("ExtractFactFromPercept"))
            actions.append(make_action("ClassifyMemory"))
            actions.append(make_action("StoreMemoryEntry"))
            actions.append(make_action("FormatConfirmationResponse"))

        elif "ProvideRecommendation" in goal:
            actions.append(make_action("RetrieveUserPreferences"))
            actions.append(make_action("TraverseKnowledgeGraph"))
            actions.append(make_action("FormatRecommendationResponse"))

        elif "AnswerQuestion" in goal:
            actions.append(make_action("RetrieveUserMemory"))
            actions.append(make_action("TraverseKnowledgeGraph"))
            actions.append(make_action("FormatAnswerResponse"))

        else:
            actions.append(make_action("FormatDefaultResponse"))

        return Plan(goal=goal, actions=actions)
