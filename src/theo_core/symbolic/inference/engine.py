"""InferenceEngine — deterministic rule-based forward and backward chaining."""

from __future__ import annotations

import time
from decimal import Decimal
from typing import TYPE_CHECKING

from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.models import (
    Belief,
    BeliefId,
    BeliefSource,
    EvidenceTrace,
)
from theo_core.symbolic.inference.models import (
    InferenceMode,
    InferenceRule,
    InferenceStep,
    InferenceTrace,
)
from theo_core.symbolic.thoughts.models import Thought, ThoughtEdge, ThoughtId, ThoughtRelation

if TYPE_CHECKING:
    from theo_core.symbolic.beliefs.graph import BeliefGraph
    from theo_core.symbolic.concepts.graph import ConceptGraph
    from theo_core.symbolic.thoughts.graph import ThoughtGraph


class InferenceEngine:
    """Deterministic forward and backward chaining inference engine.

    Enforces Canon Law 4 (Beliefs derived from inference), Law 5 (explicit traceable steps),
    Law 3 (Thoughts consume Beliefs), and Law 2 (Decisions reference Thoughts).

    Complexity Contract:
        Time: O(R * (V + E)) where R is rule count
        Memory: O(V + E)
        Deterministic: YES
    """

    @staticmethod
    def forward_chain(
        _concepts: ConceptGraph,
        beliefs: BeliefGraph,
        thoughts: ThoughtGraph,
        rules: list[InferenceRule],
        max_iterations: int = 5,
    ) -> InferenceTrace:
        """Execute forward-chaining deduction over active beliefs and rules.

        Args:
            _concepts: ConceptGraph for concept taxonomy/activation checks.
            beliefs: BeliefGraph storing active propositions.
            thoughts: ThoughtGraph storing generated reasoning DAG.
            rules: List of InferenceRule objects.
            max_iterations: Maximum iteration rounds to prevent infinite loops.

        Returns:
            An InferenceTrace recording all explicit deduction steps.

        """
        start_time = time.perf_counter()
        steps: list[InferenceStep] = []
        counter = 0
        fired_matches: set[tuple[str, tuple[str, ...]]] = set()

        # Sort rules deterministically by salience descending, then rule ID
        sorted_rules = sorted(rules, key=lambda r: (-r.salience, r.id.value))

        for _iteration in range(max_iterations):
            fired_in_iteration = False

            for rule in sorted_rules:
                matched_beliefs: list[Belief] = []
                all_conditions_met = True

                for cond in rule.conditions:
                    # Match active beliefs matching condition premise_predicate
                    matching = [
                        b
                        for b in beliefs.get_active_beliefs(min_confidence=cond.min_confidence)
                        if cond.premise_predicate.lower() in b.proposition.lower()
                    ]
                    if not matching:
                        all_conditions_met = False
                        break
                    matched_beliefs.append(matching[0])  # Take highest salience match

                if all_conditions_met and matched_beliefs:
                    matched_ids = tuple(b.id for b in matched_beliefs)
                    match_key = (rule.id.value, tuple(b.id.value for b in matched_beliefs))
                    if match_key in fired_matches:
                        continue

                    fired_matches.add(match_key)
                    counter += 1

                    # Compute combined confidence
                    min_matched_conf = min(b.confidence for b in matched_beliefs)
                    derived_confidence = min_matched_conf * rule.confidence_multiplier

                    # 1. Generate new Thought (Law 3: consumes beliefs)
                    rule_suffix = rule.id.value.replace("rule://", "")
                    thought_id = ThoughtId.of(f"thought://inf/{rule_suffix}/{counter}")
                    thought_content = (
                        rule.conclusion_template.format(*[b.proposition for b in matched_beliefs])
                        if "{" in rule.conclusion_template
                        else rule.conclusion_template
                    )

                    thought = Thought(
                        id=thought_id,
                        content=thought_content,
                        confidence=derived_confidence,
                        consumed_beliefs=matched_ids,
                        source_subsystem="inference",
                    )
                    thoughts.add_thought(thought)

                    # 2. Generate new Belief (Law 4 & Invariant 5: derived from inference)
                    belief_id = BeliefId.of(f"belief://inf/{rule_suffix}/{counter}")
                    evidence_traces = tuple(
                        EvidenceTrace(
                            evidence_id=b.id.to_symbolic_id(),
                            source_type="belief_inference",
                            weight=b.confidence,
                        )
                        for b in matched_beliefs
                    )

                    belief = Belief(
                        id=belief_id,
                        proposition=thought_content,
                        confidence=derived_confidence,
                        uncertainty=Decimal("1.0") - derived_confidence,
                        support=evidence_traces,
                        source=BeliefSource.INFERENCE,
                        evidence_count=len(evidence_traces),
                    )
                    beliefs.add_belief(belief)

                    # 3. Create ThoughtEdge link if prior thoughts exist
                    prior_thoughts = thoughts.get_thoughts()
                    if len(prior_thoughts) > 1:
                        prev_thought = prior_thoughts[-2]
                        if prev_thought.id != thought_id:
                            thoughts.add_edge(
                                ThoughtEdge(
                                    source=prev_thought.id,
                                    target=thought_id,
                                    relation=ThoughtRelation.DERIVED_FROM,
                                )
                            )

                    # 4. Record step
                    step_id = SymbolicId.of(f"step://inf/{counter}")
                    step = InferenceStep(
                        step_id=step_id,
                        rule_id=rule.id,
                        matched_belief_ids=matched_ids,
                        produced_thought_id=thought_id,
                        produced_belief_id=belief_id,
                        confidence=derived_confidence,
                    )
                    steps.append(step)
                    fired_in_iteration = True

            if not fired_in_iteration:
                break

        elapsed_ms = Decimal(str(round((time.perf_counter() - start_time) * 1000, 3)))
        return InferenceTrace(
            steps=tuple(steps),
            mode=InferenceMode.FORWARD_CHAINING,
            execution_time_ms=elapsed_ms,
        )

    @staticmethod
    def backward_chain(
        goal_predicate: str,
        beliefs: BeliefGraph,
        rules: list[InferenceRule],
        max_depth: int = 5,
    ) -> list[InferenceRule]:
        """Execute backward-chaining to find rules satisfying a goal predicate.

        Args:
            goal_predicate: The target proposition predicate string to prove.
            beliefs: Active BeliefGraph.
            rules: Available InferenceRules.
            max_depth: Maximum recursion proof depth.

        Returns:
            List of InferenceRule instances required to satisfy the goal.

        """

        def prove(target: str, depth: int, visited_rules: set[str]) -> list[InferenceRule]:
            if depth >= max_depth:
                return []

            # Check if target predicate is already satisfied by an active belief
            matching_beliefs = [
                b for b in beliefs.get_active_beliefs() if target.lower() in b.proposition.lower()
            ]
            if matching_beliefs:
                return []

            proof_chain: list[InferenceRule] = []
            for rule in rules:
                if rule.id.value in visited_rules:
                    continue

                if target.lower() in rule.conclusion_template.lower():
                    visited_rules.add(rule.id.value)
                    sub_chain_valid = True
                    sub_rules: list[InferenceRule] = []

                    for cond in rule.conditions:
                        sub = prove(cond.premise_predicate, depth + 1, visited_rules)
                        sub_rules.extend(sub)

                    if sub_chain_valid:
                        proof_chain.extend(sub_rules)
                        proof_chain.append(rule)
                        break

            return proof_chain

        return prove(goal_predicate, 0, set())
