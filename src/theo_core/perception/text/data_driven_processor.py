"""DataDrivenPerceptionProcessor — YAML rule-driven perception engine.

Normalizes raw user input text into a Percept object by evaluating
intent regex patterns, entity regexes, and fact mappings loaded from configs/rules/.
"""

from __future__ import annotations

import os
import re
from typing import Any

import yaml

from theo_core.domain.runtime.entities.percept import Percept, PerceptModality
from theo_core.domain.runtime.ports.perception import PerceptionPort


class DataDrivenPerceptionProcessor(PerceptionPort):
    """Perception processor driven by YAML rules.

    No ML, transformers, or embeddings. Uses rule-based regex patterns
    to extract intent, entities, facts, and sentiment deterministically.
    """

    def __init__(self, rules_dir: str = "configs/rules") -> None:
        """Initialize the perception processor and load YAML rules.

        Args:
            rules_dir: Path to directory containing rules YAML files.

        """
        self._rules_dir = rules_dir
        self._intents: list[dict[str, Any]] = []
        self._preference_mappings: list[dict[str, Any]] = []
        self._load_rules()

    def _load_rules(self) -> None:
        """Load intent and preference regex rules from YAML configs."""
        intents_path = os.path.join(self._rules_dir, "intents.yaml")
        if os.path.exists(intents_path):
            with open(intents_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self._intents = data.get("intents", [])

        prefs_path = os.path.join(self._rules_dir, "preferences.yaml")
        if os.path.exists(prefs_path):
            with open(prefs_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self._preference_mappings = data.get("preference_mappings", [])

    def perceive(self, raw_input: str | bytes, modality: str = "text") -> Percept:
        """Process raw input text into a normalized Percept object.

        Args:
            raw_input: Raw text string (or bytes decoded as UTF-8).
            modality: Modality string (default: "text").

        Returns:
            A normalized Percept containing extracted intent, facts, and metadata.

        """
        del modality
        text = (
            raw_input.decode("utf-8") if isinstance(raw_input, bytes) else str(raw_input)
        ).strip()
        text_lower = text.lower()

        intent = "unknown"
        extracted_entities: dict[str, str] = {}
        extracted_facts: dict[str, Any] = {}

        # 1. Match intent regex patterns on original text (case insensitive)
        for rule in self._intents:
            name = rule.get("name", "unknown")
            patterns = rule.get("patterns", [])
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    intent = name
                    extracted_entities.update(match.groupdict())
                    break
            if intent != "unknown":
                break

        # 2. Match preference and entity extraction mappings
        for map_rule in self._preference_mappings:
            pat = map_rule.get("pattern", "")
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                groups = match.groupdict()
                key_fmt = map_rule.get("memory_key", "")

                try:
                    formatted_key = key_fmt.format(**groups)
                except KeyError:
                    formatted_key = key_fmt

                val = groups.get("name") or groups.get("topic") or text
                extracted_facts[formatted_key] = val
                extracted_entities.update(groups)
                if intent == "unknown":
                    is_pref = "preference" in key_fmt
                    intent = "express_preference" if is_pref else "introduce_identity"

        # 3. Rule-based sentiment detection
        sentiment = "neutral"
        if any(w in text_lower for w in ["like", "love", "great", "awesome", "prefer", "good"]):
            sentiment = "positive"
        elif any(w in text_lower for w in ["bad", "dislike", "hate", "terrible", "poor"]):
            sentiment = "negative"

        metadata: dict[str, Any] = {
            "intent": intent,
            "entities": extracted_entities,
            "facts": extracted_facts,
            "sentiment": sentiment,
            "length": len(text),
        }

        return Percept(
            modality=PerceptModality.TEXT,
            content=text,
            confidence=1.0,
            metadata=metadata,
        )
