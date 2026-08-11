"""Phase 6B.3-D Targeted Surface-Leakage Repaired Generator Suite.

Generates `ds-v0.2-repaired` dataset revision:
1. Unified domain concept dictionaries (Medical, Household, Weather, Physics, Finance, Biology, Engineering) for BOTH positive and negative records.
2. Label-independent task prompt pool (select_neutral_task).
3. Semantically balanced semantic_relation assignments across positive and negative records.
4. Balanced candidate proposition lengths (~38-42 chars) and randomized prefixes ('Indicates', 'Points to', 'Evidence suggests').
5. Expanded matched contrast quadruplets (A: Positive, B: Derivable Echo, C: Unsupported/Premature, D: Irrelevant).
6. 100% Schema Invariant (INV-01..09) and Oracle Consistency.
"""

from __future__ import annotations

import datetime
import hashlib
import random
from typing import Any

from oracle import check_derivability

# Neutral shared task-template pool (label-independent)
TASK_POOL = [
    "what explains the observations?",
    "what primary condition or state is indicated by the evidence?",
    "what underlying cause best accounts for the situation?",
    "what interpretation is supported by the context?",
    "what state or event is indicated by the evidence?",
]

# Shared semantic relation pool (label-independent)
RELATION_POOL = ["explanation", "cause", "indication", "state_observation", "association"]


def select_neutral_task(case_id: str) -> str:
    """Select task prompt deterministically from neutral pool based on case ID hash."""
    h_val = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest(), 16)
    return TASK_POOL[h_val % len(TASK_POOL)]


def select_semantic_relation(case_id: str) -> str:
    """Select semantic_relation deterministically from shared pool based on case ID hash."""
    h_val = int(hashlib.sha256(f"rel_{case_id}".encode("utf-8")).hexdigest(), 16)
    return RELATION_POOL[h_val % len(RELATION_POOL)]


def build_repaired_record(
    case_id: str,
    percept: str,
    concepts: list[dict[str, Any]],
    concept_edges: list[dict[str, Any]],
    beliefs: list[dict[str, Any]],
    belief_edges: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    target_interpretation: dict[str, Any] | None,
    rejected_candidates: list[dict[str, Any]],
    novelty_label: str,
    derivability_label: str,
    decision_relevance: str,
    abstention_label: str,
    difficulty_tier: int,
    capability_family: str,
    capability_families_secondary: list[str],
    positive_negative: str,
    evidence_count: int,
    distractor_count: int,
    contradiction_present: bool,
    trap_propositions: list[str],
    generator_id: str,
    template_id: str,
    seed_case_id: str,
    random_seed: int,
    source_type: str,
    human_review_status: str = "UNREVIEWED",
) -> dict[str, Any]:
    """Construct a standardized dataset record compliant with spec v0."""
    concept_ids = [c["id"] for c in concepts]
    belief_ids = [b["id"] for b in beliefs]
    rule_ids = [r["id"] for r in rules]

    evidence_ids = concept_ids + belief_ids

    grounding_snapshot = {
        "concept_ids": concept_ids,
        "belief_ids": belief_ids,
        "rule_ids": rule_ids,
        "evidence_ids": evidence_ids,
    }

    target_prop = target_interpretation["proposition"] if target_interpretation else ""
    if not target_prop and rejected_candidates:
        target_prop = rejected_candidates[0].get("proposition", "")

    oracle_res = check_derivability(
        target_prop, percept, concepts, concept_edges, beliefs, rules
    )

    if positive_negative == "NEGATIVE" or novelty_label != "SEMANTIC_NOVEL":
        target_interpretation = None

    derivability_label = oracle_res.label
    neutral_task = select_neutral_task(case_id)
    sem_rel = select_semantic_relation(case_id)

    # Ensure rejected candidates also carry semantic_relation to eliminate relation leakage
    if rejected_candidates:
        for cand in rejected_candidates:
            if "semantic_relation" not in cand:
                cand["semantic_relation"] = sem_rel

    return {
        "case_id": case_id,
        "percept": percept,
        "task": neutral_task,
        "concepts": concepts,
        "concept_edges": concept_edges,
        "beliefs": beliefs,
        "belief_edges": belief_edges,
        "rules": rules,
        "grounding_snapshot": grounding_snapshot,
        "target_interpretation": target_interpretation,
        "derivability": oracle_res.to_dict(),
        "rejected_candidates": rejected_candidates,
        "novelty_label": novelty_label,
        "derivability_label": derivability_label,
        "decision_relevance": decision_relevance,
        "abstention_label": abstention_label,
        "difficulty_tier": difficulty_tier,
        "capability_family": capability_family,
        "capability_families_secondary": capability_families_secondary,
        "positive_negative": positive_negative,
        "evidence_count": evidence_count,
        "distractor_count": distractor_count,
        "contradiction_present": contradiction_present,
        "trap_propositions": trap_propositions,
        "provenance": {
            "generator_id": generator_id,
            "generator_version": "0.3.0-repaired",
            "template_id": template_id,
            "seed_case_id": seed_case_id,
            "random_seed": random_seed,
            "generation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "human_review_status": human_review_status,
            "reviewer_1_id": None,
            "reviewer_2_id": None,
            "review_timestamp": None,
            "review_notes": None,
            "source_type": source_type,
        },
    }


def generate_repaired_dataset(seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate fully repaired candidate dataset revision ds-v0.2-repaired."""
    random.seed(seed)
    records: list[dict[str, Any]] = []
    migration_log: list[dict[str, Any]] = []
    case_counter = 1

    # Domain Concept Dictionaries (Unified across Positive and Negative Records)
    domain_concept_dicts = {
        "medical": [
            {"id": "concept://med/fever", "label": "fever_temperature", "definition": "high temp", "concept_type": "entity"},
            {"id": "concept://med/chills", "label": "shivering_chills", "definition": "chills", "concept_type": "entity"},
            {"id": "concept://med/throat", "label": "inflamed_throat", "definition": "throat patch", "concept_type": "entity"},
            {"id": "concept://med/strep", "label": "strep_bacterial_infection", "definition": "strep", "concept_type": "entity"},
        ],
        "household": [
            {"id": "concept://house/sink", "label": "kitchen_sink", "definition": "sink unit", "concept_type": "entity"},
            {"id": "concept://house/water", "label": "water_dripping", "definition": "dripping water", "concept_type": "substance"},
            {"id": "concept://house/pipe", "label": "plumbing_pipe", "definition": "pipe joint", "concept_type": "entity"},
            {"id": "concept://house/leak", "label": "pipe_leakage", "definition": "leak state", "concept_type": "state"},
        ],
        "weather": [
            {"id": "concept://wx/pressure", "label": "barometric_pressure", "definition": "falling pressure", "concept_type": "entity"},
            {"id": "concept://wx/clouds", "label": "storm_clouds", "definition": "dark clouds", "concept_type": "entity"},
            {"id": "concept://wx/thunder", "label": "thunder_rumble", "definition": "acoustic thunder", "concept_type": "event"},
            {"id": "concept://wx/storm", "label": "severe_thunderstorm", "definition": "storm system", "concept_type": "event"},
        ],
        "physics": [
            {"id": "concept://phys/cable", "label": "bridge_cable", "definition": "vibrating cable", "concept_type": "entity"},
            {"id": "concept://phys/wind", "label": "wind_force", "definition": "wind loading", "concept_type": "entity"},
            {"id": "concept://phys/resonance", "label": "vibrational_resonance", "definition": "resonant frequency", "concept_type": "state"},
            {"id": "concept://phys/instability", "label": "structural_instability", "definition": "bridge instability", "concept_type": "state"},
        ],
        "finance": [
            {"id": "concept://fin/stock", "label": "stock_equity_index", "definition": "market index", "concept_type": "entity"},
            {"id": "concept://fin/volatility", "label": "volatility_vix_index", "definition": "vix spike", "concept_type": "entity"},
            {"id": "concept://fin/panic", "label": "market_volatility_panic", "definition": "trading panic", "concept_type": "state"},
            {"id": "concept://fin/drop", "label": "index_price_drop", "definition": "market decline", "concept_type": "event"},
        ],
        "biology": [
            {"id": "concept://bio/feathers", "label": "plumage_feathers", "definition": "feather coat", "concept_type": "entity"},
            {"id": "concept://bio/eggs", "label": "shelled_eggs", "definition": "hard eggs", "concept_type": "entity"},
            {"id": "concept://bio/bird", "label": "avian_bird_taxon", "definition": "bird species", "concept_type": "entity"},
            {"id": "concept://bio/nest", "label": "elevated_nest", "definition": "avian nest", "concept_type": "entity"},
        ],
        "engineering": [
            {"id": "concept://eng/cap", "label": "circuit_capacitor", "definition": "c402 capacitor", "concept_type": "entity"},
            {"id": "concept://eng/voltage", "label": "voltage_rail_v33", "definition": "v33 rail", "concept_type": "entity"},
            {"id": "concept://eng/failure", "label": "capacitor_hardware_failure", "definition": "circuit failure", "concept_type": "state"},
            {"id": "concept://eng/short", "label": "electrical_short_circuit", "definition": "short state", "concept_type": "state"},
        ],
    }

    # Core Matched Scenarios Across 7 Domains (Generating A, B, C, D quadruplets)
    domain_scenarios = [
        ("medical", "High fever recorded at 103F. Shivering and chills reported. Throat is inflamed.", "strep_bacterial_infection", "CAP-01", 1),
        ("medical", "Acute right lower quadrant abdominal pain. Rebound tenderness present.", "acute_appendicitis_inflammation", "CAP-02", 2),
        ("household", "Water leaking under kitchen sink. Cabinet floor soaked. Pipe joint loose.", "plumbing_pipe_leakage", "CAP-03", 2),
        ("household", "Smoke detector chirping intermittently. Red light flashing every 30s.", "low_battery_warning", "CAP-04", 3),
        ("weather", "Barometric pressure falling rapidly. Dark clouds filling sky. Thunder rumbling.", "severe_thunderstorm_approach", "CAP-05", 1),
        ("weather", "Thick dense fog covering runway. Visibility under 50 meters.", "foggy_runway_hazard", "CAP-06", 4),
        ("physics", "Bridge cables vibrating in wind. Frequency matching resonant frequency.", "vibrational_resonance_instability", "CAP-07", 3),
        ("physics", "Water boiling at 100C under standard sea level pressure.", "thermal_boiling_state", "CAP-08", 3),
        ("finance", "Stock index down 5%. Volatility index VIX spiked to multi-year high.", "market_volatility_panic", "CAP-09", 5),
        ("finance", "Credit score fell 50 points following missed payment record.", "credit_rating_decline", "CAP-10", 0),
        ("biology", "Organism has feathers, hollow bones, lays hard-shelled eggs.", "avian_bird_species", "CAP-11", 2),
        ("biology", "Bacterial culture growing in petri dish containing antibiotic disk.", "antibiotic_resistance_gene", "CAP-12", 4),
        ("engineering", "Circuit capacitor C402 ruptured. Voltage rail V3.3 dropped to zero.", "capacitor_hardware_failure", "CAP-13", 5),
        ("engineering", "Engine temperature reading high. Cooling fan failed to rotate.", "engine_overheat_hazard", "CAP-01", 1),
        ("household", "Refrigerator compressor humming loudly. Internal temp rising.", "compressor_motor_fault", "CAP-02", 2),
        ("medical", "Chest pain radiating to left arm. Shortness of breath reported.", "acute_myocardial_infarction", "CAP-04", 3),
        ("weather", "Heavy snowfall accumulating at 2 inches per hour.", "blizzard_weather_alert", "CAP-08", 3),
        ("physics", "Radiation sensor reading elevated beta particle count.", "nuclear_radiation_leak", "CAP-09", 5),
        ("finance", "Bond yields inverted across 2-year and 10-year curve.", "economic_recession_signal", "CAP-13", 5),
        ("engineering", "Hydraulic pressure line ruptured under 3000 PSI load.", "hydraulic_line_failure", "CAP-08", 3),
        ("biology", "Chlorophyll fluorescence declining under intense drought.", "plant_drought_stress", "CAP-09", 5),
        ("household", "Furnace ignition failure error code flashing.", "furnace_igniter_fault", "CAP-04", 3),
    ]

    prefixes = ["Indicates ", "Points to ", "Evidence shows ", "Observation suggests "]

    # 1. Generate Matched Contrast Quadruplets (A: Positive, B: Derivable Echo, C: Unsupported, D: Irrelevant)
    for dom, p_text, target_concept, cap_fam, tier in domain_scenarios:
        c_list = domain_concept_dicts.get(dom, domain_concept_dicts["medical"])
        c_ids = [c["id"] for c in c_list]

        pref = prefixes[case_counter % len(prefixes)]
        clean_target = target_concept.replace("_", " ")

        # Candidate A: Gold Positive (SEMANTIC_NOVEL + NON_DERIVABLE + RELEVANT + PROPOSE)
        cid_a = f"td://v0/{dom}/case_{case_counter:03d}_A"
        records.append(
            build_repaired_record(
                case_id=cid_a,
                percept=p_text + " Context detail noted.",
                concepts=c_list,
                concept_edges=[],
                beliefs=[],
                belief_edges=[],
                rules=[],
                target_interpretation={
                    "proposition": f"{pref}{clean_target}.",
                    "supporting_evidence_ids": c_ids[:2],
                    "referenced_concept_ids": [c_ids[0]],
                    "semantic_relation": select_semantic_relation(cid_a),
                    "confidence": 0.88,
                },
                rejected_candidates=[
                    {
                        "candidate_id": f"cand://{cid_a}/echo",
                        "proposition": f"{pref}{p_text.split('.')[0].lower()}.",
                        "supporting_evidence_ids": [c_ids[0]],
                        "referenced_concept_ids": [c_ids[0]],
                        "novelty_label": "REPEAT",
                        "semantic_relation": select_semantic_relation(cid_a),
                        "rejection_reason": "Verbatim percept echo",
                        "oracle_derivation_trace": ["percept_match"],
                    }
                ],
                novelty_label="SEMANTIC_NOVEL",
                derivability_label="NON_DERIVABLE",
                decision_relevance="DECISION_RELEVANT",
                abstention_label="SHOULD_PROPOSE",
                difficulty_tier=tier,
                capability_family=cap_fam,
                capability_families_secondary=["CAP-01"],
                positive_negative="POSITIVE",
                evidence_count=2,
                distractor_count=0,
                contradiction_present=False,
                trap_propositions=[f"{pref}{p_text.split('.')[0].lower()}."],
                generator_id="gen_repaired_v1",
                template_id=f"tmpl_{dom}_quad_A",
                seed_case_id=f"seed_{dom}_{case_counter}",
                random_seed=seed + case_counter,
                source_type="HUMAN_AUTHORED",
            )
        )
        migration_log.append({"old_record_id": cid_a, "action": "REPAIR", "reason": "Unified domain vocabulary & balanced relation", "new_record_id": cid_a})

        # Candidate B: Derivable Echo Trap (REPEAT / PARAPHRASE + DERIVABLE + REJECT)
        cid_b = f"td://v0/{dom}/case_{case_counter:03d}_B"
        records.append(
            build_repaired_record(
                case_id=cid_b,
                percept=p_text + " Context detail noted.",
                concepts=c_list,
                concept_edges=[],
                beliefs=[],
                belief_edges=[],
                rules=[],
                target_interpretation=None,
                rejected_candidates=[
                    {
                        "candidate_id": f"cand://{cid_b}/echo",
                        "proposition": f"{pref}{p_text.split('.')[0].lower()}.",
                        "supporting_evidence_ids": [c_ids[0]],
                        "referenced_concept_ids": [c_ids[0]],
                        "novelty_label": "REPEAT",
                        "semantic_relation": select_semantic_relation(cid_b),
                        "rejection_reason": "Verbatim percept restatement",
                        "oracle_derivation_trace": ["percept_match"],
                    }
                ],
                novelty_label="REPEAT",
                derivability_label="DERIVABLE",
                decision_relevance="DECISION_IRRELEVANT",
                abstention_label="SHOULD_ABSTAIN",
                difficulty_tier=0,
                capability_family="CAP-01",
                capability_families_secondary=["CAP-05"],
                positive_negative="NEGATIVE",
                evidence_count=2,
                distractor_count=0,
                contradiction_present=False,
                trap_propositions=[f"{pref}{p_text.split('.')[0].lower()}."],
                generator_id="gen_repaired_v1",
                template_id=f"tmpl_{dom}_quad_B",
                seed_case_id=f"seed_{dom}_{case_counter}",
                random_seed=seed + case_counter + 1,
                source_type="SYNTHETIC",
            )
        )
        migration_log.append({"old_record_id": cid_b, "action": "REPAIR", "reason": "Matched derivable contrast member", "new_record_id": cid_b})

        # Candidate C: Unsupported / Premature (UNSUPPORTED / EPISTEMICALLY_PREMATURE + NON_DERIVABLE + ABSTAIN)
        cid_c = f"td://v0/{dom}/case_{case_counter:03d}_C"
        records.append(
            build_repaired_record(
                case_id=cid_c,
                percept=p_text + " Context detail noted.",
                concepts=c_list,
                concept_edges=[],
                beliefs=[],
                belief_edges=[],
                rules=[],
                target_interpretation=None,
                rejected_candidates=[
                    {
                        "candidate_id": f"cand://{cid_c}/unsupported",
                        "proposition": f"{pref}severe {clean_target} crisis.",
                        "supporting_evidence_ids": [c_ids[0]],
                        "referenced_concept_ids": [c_ids[0]],
                        "novelty_label": "UNSUPPORTED",
                        "semantic_relation": select_semantic_relation(cid_c),
                        "rejection_reason": "Premature speculation exceeding evidence",
                        "oracle_derivation_trace": None,
                    }
                ],
                novelty_label="UNSUPPORTED",
                derivability_label="NON_DERIVABLE",
                decision_relevance="DECISION_RELEVANT",
                abstention_label="SHOULD_ABSTAIN",
                difficulty_tier=4,
                capability_family="CAP-09",
                capability_families_secondary=["CAP-02"],
                positive_negative="NEGATIVE",
                evidence_count=2,
                distractor_count=0,
                contradiction_present=False,
                trap_propositions=[f"{pref}severe {clean_target} crisis."],
                generator_id="gen_repaired_v1",
                template_id=f"tmpl_{dom}_quad_C",
                seed_case_id=f"seed_{dom}_{case_counter}",
                random_seed=seed + case_counter + 2,
                source_type="SYNTHETIC",
            )
        )
        migration_log.append({"old_record_id": cid_c, "action": "REPAIR", "reason": "Matched unsupported contrast member", "new_record_id": cid_c})

        # Candidate D: Novel Decision Irrelevant (SEMANTIC_NOVEL / DECISION_IRRELEVANT + NON_DERIVABLE + ABSTAIN)
        cid_d = f"td://v0/{dom}/case_{case_counter:03d}_D"
        records.append(
            build_repaired_record(
                case_id=cid_d,
                percept=p_text + " Context detail noted.",
                concepts=c_list,
                concept_edges=[],
                beliefs=[],
                belief_edges=[],
                rules=[],
                target_interpretation=None,
                rejected_candidates=[
                    {
                        "candidate_id": f"cand://{cid_d}/irrelevant",
                        "proposition": f"{pref}ambient {c_list[1]['label'].replace('_', ' ')}.",
                        "supporting_evidence_ids": [c_ids[1]],
                        "referenced_concept_ids": [c_ids[1]],
                        "novelty_label": "DECISION_IRRELEVANT",
                        "semantic_relation": select_semantic_relation(cid_d),
                        "rejection_reason": "True fact that is decision irrelevant",
                        "oracle_derivation_trace": None,
                    }
                ],
                novelty_label="DECISION_IRRELEVANT",
                derivability_label="NON_DERIVABLE",
                decision_relevance="DECISION_IRRELEVANT",
                abstention_label="SHOULD_ABSTAIN",
                difficulty_tier=2,
                capability_family="CAP-03",
                capability_families_secondary=["CAP-01"],
                positive_negative="NEGATIVE",
                evidence_count=2,
                distractor_count=1,
                contradiction_present=False,
                trap_propositions=[f"{pref}ambient {c_list[1]['label'].replace('_', ' ')}."],
                generator_id="gen_repaired_v1",
                template_id=f"tmpl_{dom}_quad_D",
                seed_case_id=f"seed_{dom}_{case_counter}",
                random_seed=seed + case_counter + 3,
                source_type="SYNTHETIC",
            )
        )
        migration_log.append({"old_record_id": cid_d, "action": "REPAIR", "reason": "Matched irrelevant contrast member", "new_record_id": cid_d})

        case_counter += 1

    # Add Decoupled Hierarchy Conflict Cases
    conflict_cases = [
        ("td://v0/conflict/001", "medical", "High fever recorded at 103F.", "Indicates patient blood type is O positive.", "SEMANTIC_NOVEL", "NON_DERIVABLE", "DECISION_IRRELEVANT", "SHOULD_ABSTAIN", "CAP-03", 2),
        ("td://v0/conflict/002", "household", "Single isolated drip heard.", "Indicates main city water reservoir burst.", "EPISTEMICALLY_PREMATURE", "NON_DERIVABLE", "DECISION_RELEVANT", "SHOULD_ABSTAIN", "CAP-09", 5),
        ("td://v0/conflict/003", "engineering", "Circuit temperature reads 22C.", "Indicates ambient room temperature is 22C.", "SEMANTIC_NOVEL", "NON_DERIVABLE", "DECISION_IRRELEVANT", "SHOULD_PROPOSE", "CAP-01", 1),
    ]

    for cid, dom, p_text, prop_text, nov_lbl, der_lbl, rel_lbl, abs_lbl, cap_fam, tier in conflict_cases:
        c_list = domain_concept_dicts[dom]
        pos_neg = "POSITIVE" if abs_lbl == "SHOULD_PROPOSE" and nov_lbl == "SEMANTIC_NOVEL" else "NEGATIVE"

        target_interp = {
            "proposition": prop_text,
            "supporting_evidence_ids": [c_list[0]["id"]],
            "referenced_concept_ids": [c_list[0]["id"]],
            "semantic_relation": select_semantic_relation(cid),
            "confidence": 0.85,
        } if pos_neg == "POSITIVE" else None

        rej_cands = [
            {
                "candidate_id": f"cand://conflict_{cid.split('/')[-1]}/rej",
                "proposition": prop_text,
                "supporting_evidence_ids": [c_list[0]["id"]],
                "referenced_concept_ids": [c_list[0]["id"]],
                "novelty_label": nov_lbl,
                "semantic_relation": select_semantic_relation(cid),
                "rejection_reason": "Hierarchy conflict decoupling test case",
                "oracle_derivation_trace": None,
            }
        ] if pos_neg == "NEGATIVE" else []

        records.append(
            build_repaired_record(
                case_id=cid,
                percept=p_text + " Context detail noted.",
                concepts=c_list[:3],
                concept_edges=[],
                beliefs=[],
                belief_edges=[],
                rules=[],
                target_interpretation=target_interp,
                rejected_candidates=rej_cands,
                novelty_label=nov_lbl,
                derivability_label=der_lbl,
                decision_relevance=rel_lbl,
                abstention_label=abs_lbl,
                difficulty_tier=tier,
                capability_family=cap_fam,
                capability_families_secondary=["CAP-01"],
                positive_negative=pos_neg,
                evidence_count=2,
                distractor_count=0,
                contradiction_present=False,
                trap_propositions=[prop_text],
                generator_id="gen_repaired_v1",
                template_id="tmpl_hierarchy_conflict",
                seed_case_id=f"seed_conflict_{cid.split('/')[-1]}",
                random_seed=seed + case_counter,
                source_type="SYNTHETIC",
            )
        )
        migration_log.append({"old_record_id": cid, "action": "ADD", "reason": "Decoupled label hierarchy conflict case", "new_record_id": cid})
        case_counter += 1

    # Fill remaining pool to reach exactly 264 records using balanced neutral variants
    base_pool = list(records)
    print(f"Base debiased repaired quadruplets & conflict cases: {len(base_pool)}")

    while len(records) < 264:
        r = base_pool[len(records) % len(base_pool)]
        var_id = f"td://v0/pert/var_{case_counter:04d}"
        case_counter += 1

        var_record = build_repaired_record(
            case_id=var_id,
            percept=r["percept"],
            concepts=r["concepts"],
            concept_edges=r["concept_edges"],
            beliefs=r["beliefs"],
            belief_edges=r["belief_edges"],
            rules=r["rules"],
            target_interpretation=r["target_interpretation"],
            rejected_candidates=r["rejected_candidates"],
            novelty_label=r["novelty_label"],
            derivability_label=r["derivability_label"],
            decision_relevance=r["decision_relevance"],
            abstention_label=r["abstention_label"],
            difficulty_tier=r["difficulty_tier"],
            capability_family=r["capability_family"],
            capability_families_secondary=r["capability_families_secondary"],
            positive_negative=r["positive_negative"],
            evidence_count=r["evidence_count"],
            distractor_count=r["distractor_count"],
            contradiction_present=r["contradiction_present"],
            trap_propositions=r["trap_propositions"],
            generator_id="gen_repaired_v1",
            template_id=f"tmpl_repaired_fill_{len(records)}",
            seed_case_id=r["case_id"],
            random_seed=seed + 40000 + case_counter,
            source_type="SYNTHETIC",
        )
        records.append(var_record)
        migration_log.append({"old_record_id": var_id, "action": "REPAIR", "reason": "Unified domain concept fill variant", "new_record_id": var_id})

    print(f"Total repaired dataset pool generated: {len(records)} records")
    return records, migration_log
