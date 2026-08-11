"""Symbolic Derivability Oracle (ADR-0028 & Phase 6B.1 Spec).

Evaluates whether a candidate proposition P is derivable from context C
(percept, beliefs, concept taxonomy edges, inference rules) via deterministic
symbolic analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OracleResult:
    is_derivable: bool
    label: str  # "DERIVABLE" | "NON_DERIVABLE"
    derivation_type: str  # "none" | "percept_match" | "belief_echo" | "rule_chain" | "taxonomy_edge"
    derivation_trace: list[str] | None
    oracle_confidence: float = 1.0
    checked_by: str = "symbolic_oracle_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "derivation_type": self.derivation_type,
            "derivation_trace": self.derivation_trace,
            "oracle_confidence": self.oracle_confidence,
            "checked_by": self.checked_by,
        }


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip punctuation, collapse whitespace)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


# Common domain synonym mappings for paraphrase detection
SYNONYMS = {
    "shattered": "broken",
    "broken": "shattered",
    "burst": "exploded",
    "exploded": "burst",
    "vessel": "container",
    "container": "vessel",
    "heat": "temperature",
    "temperature": "heat",
    "rain": "precipitation",
    "precipitation": "rain",
}


def check_derivability(
    candidate_proposition: str,
    percept: str,
    concepts: list[dict[str, Any]],
    concept_edges: list[dict[str, Any]],
    beliefs: list[dict[str, Any]],
    rules: list[dict[str, Any]],
) -> OracleResult:
    """Run deterministic symbolic oracle checks to determine if candidate_proposition is derivable."""
    if not candidate_proposition:
        return OracleResult(
            is_derivable=False,
            label="NON_DERIVABLE",
            derivation_type="none",
            derivation_trace=None,
        )

    norm_cand = normalize_text(candidate_proposition)
    norm_percept = normalize_text(percept)
    percept_clauses = [normalize_text(c) for c in re.split(r"[.;!]\s*", percept) if c.strip()]

    # 1. Direct or clause percept match
    if norm_cand in norm_percept or norm_cand in percept_clauses:
        return OracleResult(
            is_derivable=True,
            label="DERIVABLE",
            derivation_type="percept_match",
            derivation_trace=[f"percept_input: '{percept}'"],
        )

    # 2. Token overlap and paraphrase check with synonym expansion
    cand_words = set(norm_cand.split())
    expanded_cand_words = set(cand_words)
    for w in cand_words:
        if w in SYNONYMS:
            expanded_cand_words.add(SYNONYMS[w])

    for clause in percept_clauses:
        clause_words = set(clause.split())
        expanded_clause_words = set(clause_words)
        for w in clause_words:
            if w in SYNONYMS:
                expanded_clause_words.add(SYNONYMS[w])

        if cand_words and clause_words:
            overlap = len(expanded_cand_words & expanded_clause_words) / float(len(cand_words))
            # Require high overlap (>0.75) to declare paraphrase derivability
            if overlap >= 0.75 and len(cand_words) <= len(clause_words) + 1:
                return OracleResult(
                    is_derivable=True,
                    label="DERIVABLE",
                    derivation_type="percept_match",
                    derivation_trace=[f"percept_clause: '{clause}' (overlap: {overlap:.2f})"],
                )

    # 3. Belief echo check
    for b in beliefs:
        prop = b.get("proposition", "")
        norm_b = normalize_text(prop)
        if norm_cand == norm_b or norm_cand in norm_b or norm_b in norm_cand:
            return OracleResult(
                is_derivable=True,
                label="DERIVABLE",
                derivation_type="belief_echo",
                derivation_trace=[f"belief://{b.get('id', 'unknown')}: '{prop}'"],
            )

    # 4. Taxonomy edge echo check
    concept_map = {c.get("id"): c.get("label") for c in concepts}
    for edge in concept_edges:
        src_label = concept_map.get(edge.get("source"), "").lower()
        tgt_label = concept_map.get(edge.get("target"), "").lower()
        rel = edge.get("relation", "is_a")

        if src_label and tgt_label:
            edge_patterns = [
                f"{src_label} is a {tgt_label}",
                f"a {src_label} is an {tgt_label}",
                f"{src_label} is an {tgt_label}",
                f"a {src_label} is a {tgt_label}",
                f"{src_label} is related to {tgt_label}",
            ]
            for pat in edge_patterns:
                if normalize_text(pat) == norm_cand or normalize_text(pat) in norm_cand:
                    return OracleResult(
                        is_derivable=True,
                        label="DERIVABLE",
                        derivation_type="taxonomy_edge",
                        derivation_trace=[
                            f"concept_edge: {edge.get('source')} {rel} {edge.get('target')}"
                        ],
                    )

    # 5. Rule conclusion match check
    for r in rules:
        concl = r.get("conclusion_text", "")
        if concl and normalize_text(concl) in norm_cand:
            return OracleResult(
                is_derivable=True,
                label="DERIVABLE",
                derivation_type="rule_chain",
                derivation_trace=[
                    f"rule://{r.get('id', 'unknown')} -> conclusion: '{concl}'"
                ],
            )

    return OracleResult(
        is_derivable=False,
        label="NON_DERIVABLE",
        derivation_type="none",
        derivation_trace=None,
    )
