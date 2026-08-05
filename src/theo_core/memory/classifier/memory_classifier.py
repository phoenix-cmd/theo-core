"""MemoryClassifier — classifies percept facts into Identity, Preference, Experience, or Working."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theo_core.domain.runtime.entities.percept import Percept


class MemoryClassifier:
    """Categorizes percept facts into explicit memory categories.

    Categories:
        - Identity: Name, user identity, core personal traits.
        - Preference: Likes, dislikes, favorite topics/skills.
        - Experience: Past events, visits, temporal actions.
        - Working: Short-term contextual items.
    """

    def classify(self, percept: Percept | None) -> str:
        """Classify a percept into a primary memory category.

        Args:
            percept: The Percept object from Perception Engine.

        Returns:
            Category string: 'identity', 'preference', 'experience', or 'working'.

        """
        if percept is None:
            return "working"

        intent = percept.metadata.get("intent", "unknown")
        facts = percept.metadata.get("facts", {})

        if intent == "introduce_identity" or any("user.name" in k for k in facts):
            return "identity"

        if intent == "express_preference" or any("preference" in k for k in facts):
            return "preference"

        content_lower = percept.content.lower()
        if any(kw in content_lower for kw in ["yesterday", "went", "visited", "last week", "did"]):
            return "experience"

        return "working"
