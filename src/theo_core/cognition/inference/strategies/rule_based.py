"""RuleBasedStrategy — deterministic rule-based inference strategy for v0.2.

Evaluates data-driven rules against CognitiveState (Percept, Context, Memory, Goal, Plan).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import yaml

from theo_core.cognition.inference.strategies.base import InferenceStrategy

if TYPE_CHECKING:
    from theo_core.domain.runtime.entities.cognitive_state import CognitiveState


class RuleBasedStrategy(InferenceStrategy):
    """Deterministic rule-based cognitive inference strategy.

    Uses YAML configs and percept metadata to generate cognitive decisions.
    """

    def __init__(self, rules_dir: str = "configs/rules") -> None:
        """Initialize RuleBasedStrategy.

        Args:
            rules_dir: Path to rules directory.

        """
        self._rules_dir = rules_dir
        self._greetings: dict[str, Any] = {}
        self._recommendations: dict[str, Any] = {}
        self._load_rules()

    def _load_rules(self) -> None:
        """Load greetings and recommendations from YAML config files."""
        g_path = os.path.join(self._rules_dir, "greetings.yaml")
        if os.path.exists(g_path):
            with open(g_path, encoding="utf-8") as f:
                self._greetings = yaml.safe_load(f) or {}

        r_path = os.path.join(self._rules_dir, "recommendations.yaml")
        if os.path.exists(r_path):
            with open(r_path, encoding="utf-8") as f:
                self._recommendations = yaml.safe_load(f) or {}

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "RuleBasedStrategy"

    def evaluate(self, state: CognitiveState) -> dict[str, Any]:
        """Evaluate cognitive rules on current CognitiveState.

        Args:
            state: The current CognitiveState.

        Returns:
            Dictionary containing inferred decision details and response template parameters.

        """
        goal_desc = state.active_goal.description if state.active_goal else "MaintainConversation"
        percept = state.percept
        entities = percept.metadata.get("entities", {}) if percept else {}
        facts = percept.metadata.get("facts", {}) if percept else {}

        user_name = entities.get("name") or state.context.get("user.name")
        liked_topic = None
        used_memory_ids: list[str] = []

        # Check retrieved memories for user name and preferences
        for mem in state.retrieved_memories:
            m_id = str(mem.get("id", ""))
            key = str(mem.get("key", ""))
            val = str(mem.get("value", ""))
            if key == "user.name":
                user_name = val
                if m_id:
                    used_memory_ids.append(m_id)
            elif "preference" in key:
                liked_topic = val.lower()
                if m_id:
                    used_memory_ids.append(m_id)

        # Evaluate Greeting Goal
        if "AcknowledgeGreeting" in goal_desc:
            if user_name and liked_topic:
                resp = (
                    f"Hello {user_name}! You previously mentioned you're "
                    f"interested in {liked_topic}."
                )
            elif user_name:
                resp = f"Hello {user_name}! Great to speak with you."
            else:
                resp = "Hello! How can I assist you today?"

            summary = (
                f"Greeting rule matched. User name: '{user_name or 'None'}', "
                f"Interest: '{liked_topic or 'None'}'"
            )
            return {
                "candidate_response": resp,
                "confidence": 1.0,
                "reasoning_summary": summary,
                "actions_executed": ["LookupUserIdentity", "FormatGreetingResponse"],
                "used_memory_ids": used_memory_ids,
            }

        # Evaluate RememberFact Goal
        if "RememberFact" in goal_desc:
            if "name" in entities:
                name_val = entities["name"]
                resp = f"Nice to meet you, {name_val}."
            elif "topic" in entities:
                topic_val = entities["topic"]
                resp = f"I'll remember that you like {topic_val}."
            else:
                fact_desc = ", ".join(f"{k} = {v}" for k, v in facts.items())
                resp = (
                    f"I'll remember that {fact_desc}."
                    if fact_desc
                    else "I've recorded that information."
                )

            return {
                "candidate_response": resp,
                "confidence": 1.0,
                "reasoning_summary": f"Fact extraction matched: {facts}",
                "actions_executed": ["ExtractFactFromPercept", "StoreMemoryEntry"],
                "used_memory_ids": used_memory_ids,
            }

        # Evaluate ProvideRecommendation Goal
        if "ProvideRecommendation" in goal_desc:
            if not liked_topic and "topic" in entities:
                liked_topic = entities["topic"].lower()

            concepts = self._recommendations.get("concept_recommendations", {})
            if liked_topic and liked_topic in concepts:
                rec_info = concepts[liked_topic]
                rec = rec_info.get("recommendation", "exploring new areas")
                reason = rec_info.get("reason", "it matches your interests")
                resp = (
                    f"Based on your interest in {liked_topic}, I recommend {rec} because {reason}!"
                )
            else:
                default_rec = self._recommendations.get("default_recommendation", {})
                rec = default_rec.get("recommendation", "learning new skills")
                resp = f"I recommend {rec}!"

            return {
                "candidate_response": resp,
                "confidence": 1.0,
                "reasoning_summary": f"Recommendation matched for topic '{liked_topic or 'None'}'",
                "actions_executed": ["RetrieveUserPreferences", "FormatRecommendationResponse"],
                "used_memory_ids": used_memory_ids,
            }

        # Evaluate AnswerQuestion Goal
        if "AnswerQuestion" in goal_desc:
            raw = state.raw_input.lower()
            if "name" in raw and user_name:
                resp = f"Your name is {user_name}."
            elif liked_topic:
                resp = f"You previously mentioned you're interested in {liked_topic}."
            else:
                resp = "I do not have enough information stored yet to answer that."

            return {
                "candidate_response": resp,
                "confidence": 0.9,
                "reasoning_summary": "Question answered via memory lookup.",
                "actions_executed": ["RetrieveUserMemory", "FormatAnswerResponse"],
                "used_memory_ids": used_memory_ids,
            }

        # Default Response
        return {
            "candidate_response": "I understand. How else can I assist you?",
            "confidence": 0.8,
            "reasoning_summary": "Default conversation response.",
            "actions_executed": ["FormatDefaultResponse"],
            "used_memory_ids": used_memory_ids,
        }
