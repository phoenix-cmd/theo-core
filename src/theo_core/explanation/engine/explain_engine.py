"""ExplainEngine — generates human-readable causal explanations of cognitive decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theo_core.domain.runtime.entities.cognitive_state import CognitiveState
    from theo_core.domain.runtime.entities.decision_record import DecisionRecord


class ExplainEngine:
    """Generates human-readable causal justifications for cognitive decisions.

    Answers 'Why did THEO make this decision?' by tracing:
    Matched Rules -> Memory Entries Used -> Goal Stack -> Decision -> Response Text.
    """

    def explain_state(self, state: CognitiveState) -> str:
        """Generate a causal explanation markdown string from a CognitiveState object.

        Args:
            state: The CognitiveState object after pipeline processing.

        Returns:
            Formatted explanation markdown string.

        """
        raw_input = state.raw_input
        intent = state.percept.metadata.get("intent", "unknown") if state.percept else "unknown"
        goal = state.active_goal.description if state.active_goal else "MaintainConversation"
        decision_dict = state.decision
        reasoning = decision_dict.get("reasoning_summary", "No reasoning summary.")
        confidence = decision_dict.get("confidence", 1.0)
        used_memories = decision_dict.get("used_memory_ids", ())
        used_rules = decision_dict.get("used_rule_ids", ())
        actions = decision_dict.get("actions_taken", ())

        lines = [
            "=== CAUSAL COGNITIVE EXPLANATION ===",
            f"Input:            '{raw_input}'",
            f"Perception Intent: {intent}",
            f"Selected Goal:     {goal}",
            f"Rule(s) Matched:   {', '.join(used_rules) if used_rules else 'RULE-0001 (Default)'}",
            f"Memories Used:     {', '.join(used_memories) if used_memories else 'None'}",
            f"Actions Executed:  {', '.join(actions) if actions else 'None'}",
            f"Policy Summary:    {reasoning}",
            f"Confidence Score:  {confidence:.2f} (100% deterministic rule policy)",
            f"Final Output:      '{state.response_text}'",
            "====================================",
        ]
        return "\n".join(lines)

    def explain_record(self, record: DecisionRecord) -> str:
        """Generate a causal explanation string from a formal DecisionRecord object.

        Args:
            record: The DecisionRecord object.

        Returns:
            Formatted explanation string.

        """
        rules_str = ", ".join(record.used_rule_ids) if record.used_rule_ids else "None"
        mems_str = ", ".join(record.used_memory_ids) if record.used_memory_ids else "None"

        lines = [
            "=== FORMAL DECISION RECORD EXPLANATION ===",
            f"Decision ID:       {record.decision_id}",
            f"Trace ID:          {record.trace_id or 'N/A'}",
            f"Goal:              {record.used_goal}",
            f"Rules Evaluated:   {rules_str}",
            f"Memories Utilized: {mems_str}",
            f"Selection Reason:  {record.selection_reason}",
            f"Confidence:        {record.confidence:.2f}",
            f"Selected Output:   '{record.selected_option}'",
            "==========================================",
        ]
        return "\n".join(lines)
