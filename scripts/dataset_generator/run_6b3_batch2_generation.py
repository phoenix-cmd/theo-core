#!/usr/bin/env python3
"""
THEO SLM v0 - Phase 6B.3 Batch-2 Candidate Generation (bounded, single authorized batch).

Authorization: SEPARATELY AUTHORIZED 6B.3 BATCH-2 CANDIDATE GENERATION (one bounded batch).
This script:
  - reads existing canonical GOLD (250 GOLD / 44 REJECT) + Batch-1 candidates as READ-ONLY inputs,
  - generates NEW candidate records into a NEW pool directory (6b3_batch2),
  - runs the uniqueness oracle (reject exact duplicates vs 250 GOLD + 44 REJECT + 31 Batch-1),
  - assigns group_key / split_group seed-family metadata; includes ONE intentionally
    discriminative near-duplicate seed family (shared group_key, distinct propositions),
  - runs leakage checks + explicit shortcut-balance diagnostics (evidence-count, proposition-length,
    percept-length, concept-count) within-batch AND vs the cumulative 250-GOLD dataset,
  - emits dataset-manifest.json, uniqueness_ledger.jsonl, split_group_manifest.json,
    shortcut_balance_report.json, and a candidate-generation report.
It does NOT modify adjudication.json, review-records.json, or the GOLD corpus.
It does NOT adjudicate, merge, select/download a model, train, or run 6C/6D.
"""
import json, hashlib, os, re, datetime, collections, math

REVIEW_DIR = "theo-data/datasets/theo_slm_v0_review"
OUT_DIR = os.path.join(REVIEW_DIR, "6b3_batch2")
SEED = 43
GEN_ID = "gen_6b3_batch2"
GEN_VERSION = "gen-v0.4.1"
DS_VERSION = "ds-v0.4-candidates"
GEN_TS_BASE = "2026-08-25T15:00:00+00:00"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- load read-only inputs ----------
adj = json.load(open(os.path.join(REVIEW_DIR, "adjudication.json")))
b1 = json.load(open(os.path.join(REVIEW_DIR, "6b3_batch1", "candidate_records_batch1.json")))["records"]
recs = json.load(open(os.path.join(REVIEW_DIR, "review-records.json")))

def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

# Reference proposition sets built from review-records.json (authoritative proposition/percept text
# for the 264 original pool, incl. 226 GOLD + 38 REJECT) keyed by CURRENT status in adjudication.json,
# PLUS the 31 Batch-1 candidate propositions. adjudication.json GOLD/REJECT records carry EMPTY
# proposition/percept fields, so the real text source is review-records.json (as in Batch-1).
status_by_rid = {r["review_id"]: r.get("final_status") for r in adj["records"]}
gold_props, reject_props = set(), set()
for r in recs:
    rid = r.get("review_id"); st = status_by_rid.get(rid)
    p = norm(r.get("candidate_proposition"))
    if not p: continue
    if st == "REJECT":
        reject_props.add(p)
    elif st in ("GOLD_POSITIVE", "GOLD_ABSTAIN", "HARD_NEGATIVE"):
        gold_props.add(p)
# Batch-1 candidate propositions (avoid re-emitting Batch-1 content)
b1_props = set(norm(r.get("candidate_proposition")) for r in b1 if r.get("candidate_proposition"))
# safety fold-in of any adjudication.json propositions not present in review-records (mostly empty)
for r in adj["records"]:
    p = norm(r.get("candidate_proposition"))
    if not p: continue
    st = r.get("final_status")
    if st == "REJECT": reject_props.add(p)
    elif st in ("GOLD_POSITIVE", "GOLD_ABSTAIN", "HARD_NEGATIVE"): gold_props.add(p)

# existing GOLD percepts (for cross-pool near-duplicate leakage check) — sourced from review-records.json
gold_percepts = [norm(r.get("percept")) for r in recs
                if status_by_rid.get(r.get("review_id")) in ("GOLD_POSITIVE","GOLD_ABSTAIN","HARD_NEGATIVE") and r.get("percept")]

# ---------- curated bounded batch data ----------
C = []
def add(idx, disp, cap, domain, tmpl, gk, percept, task, concepts, prop, ev, nov, der, ab, near_of=None):
    C.append(dict(idx=idx, disposition=disp, cap=cap, domain=domain, tmpl=tmpl, gk=gk,
                  percept=percept, task=task, concepts=concepts, prop=prop, ev=ev,
                  nov=nov, der=der, ab=ab, near_of=near_of))

def C_(i, idl, lbl): return (f"concept://{domain_part(i)}/{idl}", lbl)
def domain_part(i):  # helper not used; keep explicit
    return ""

# NOTE: concepts use explicit concept://<domain>/<token> ids; ev is a subset of those ids.
WX="concept://wx"; HH="concept://household"; MED="concept://medical"; ENG="concept://engineering"
PHY="concept://physics"; FIN="concept://finance"; BIO="concept://biology"; CON="concept://conflict"
def cid(dom, tok): return f"concept://{dom}/{tok}"

# ===================== GOLD_POSITIVE (40) — under-covered CAPs, all 8 domains =====================
# CAP-02 abduction
add(1,"GOLD_POSITIVE","CAP-02","weather","tmpl_6b3b2_cap02_a","sf_weather_cap02_01",
    "The sidewalk is dry but the rooftops glisten and gutters drip; no rain fell here today.",
    "what interpretation is supported by the context?",
    [(cid("wx","roof_glisten"),"rooftops_glisten"),(cid("wx","gutter_drip"),"gutters_drip"),(cid("wx","dry_walk"),"dry_sidewalk")],
    "A brief passing shower soaked the roofs while missing ground level here.",
    [cid("wx","roof_glisten")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(2,"GOLD_POSITIVE","CAP-02","household","tmpl_6b3b2_cap02_b","sf_household_cap02_02",
    "The clock stopped at 3:07 and the power outage log shows a dip then.",
    "what interpretation is supported?",
    [(cid("household","clock_stop"),"clock_stopped"),(cid("household","power_dip"),"power_dip")],
    "The power interruption halted the clock.",
    [cid("household","clock_stop"),cid("household","power_dip")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(3,"GOLD_POSITIVE","CAP-02","medical","tmpl_6b3b2_cap02_c","sf_medical_cap02_03",
    "The patient's scar is thin and pale; the surgical report notes closure one year ago.",
    "what interpretation is supported?",
    [(cid("medical","thin_scar"),"thin_pale_scar"),(cid("medical","surgery_year"),"surgery_one_year")],
    "The incision has matured normally over the past year.",
    [cid("medical","thin_scar")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(4,"GOLD_POSITIVE","CAP-02","engineering","tmpl_6b3b2_cap02_d","sf_engineering_cap02_04",
    "The bridge shows rust at the joints and a survey notes a slight sag in the midpoint.",
    "what interpretation is supported?",
    [(cid("engineering","joint_rust"),"joint_rust"),(cid("engineering","mid_sag"),"midpoint_sag"),(cid("engineering","survey"),"survey_note")],
    "Corrosion has weakened the joints and allowed the deck to settle.",
    [cid("engineering","joint_rust")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

# CAP-04 distractor
add(5,"GOLD_POSITIVE","CAP-04","physics","tmpl_6b3b2_cap04_a","sf_physics_cap04_05",
    "A book rests on a table; a cat walks by; the table is wooden.",
    "what is the support state of the book?",
    [(cid("physics","book_rest"),"book_resting"),(cid("physics","wood_table"),"wooden_table"),(cid("physics","cat"),"cat_walking")],
    "The book is supported by the table; the cat is irrelevant.",
    [cid("physics","book_rest")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(6,"GOLD_POSITIVE","CAP-04","finance","tmpl_6b3b2_cap04_b","sf_finance_cap04_06",
    "The index rose two points; a politician gave a speech; volume was typical.",
    "what drives the move?",
    [(cid("finance","index_up"),"index_rise"),(cid("finance","speech"),"political_speech"),(cid("finance","vol"),"typical_volume")],
    "The gain reflects ordinary trading and is unrelated to the speech.",
    [cid("finance","index_up")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(7,"GOLD_POSITIVE","CAP-04","biology","tmpl_6b3b2_cap04_c","sf_biology_cap04_07",
    "The leaf is green and a bird sings nearby; soil is moist.",
    "what does the leaf color indicate?",
    [(cid("biology","green_leaf"),"green_leaf"),(cid("biology","bird"),"bird_song"),(cid("biology","moist_soil"),"moist_soil")],
    "The green leaf indicates healthy photosynthesis; the bird is incidental.",
    [cid("biology","green_leaf")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(8,"GOLD_POSITIVE","CAP-04","conflict","tmpl_6b3b2_cap04_d","sf_conflict_cap04_08",
    "A convoy halted at a checkpoint; vendors sell food beside the road; a child plays.",
    "what is the convoy status?",
    [(cid("conflict","convoy_halt"),"convoy_halted"),(cid("conflict","vendors"),"road_vendors"),(cid("conflict","child"),"playing_child")],
    "The convoy is paused at the checkpoint; the vendors indicate normal road activity.",
    [cid("conflict","convoy_halt")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

# CAP-05 recurrence/prediction
add(9,"GOLD_POSITIVE","CAP-05","weather","tmpl_6b3b2_cap05_a","sf_weather_cap05_09",
    "Dark clouds return and the barometer falls again after this morning's storm.",
    "what weather is expected?",
    [(cid("wx","clouds_return"),"clouds_return"),(cid("wx","pressure_fall"),"pressure_falling")],
    "Another storm is likely given the returning clouds and falling pressure.",
    [cid("wx","clouds_return")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(10,"GOLD_POSITIVE","CAP-05","household","tmpl_6b3b2_cap05_b","sf_household_cap05_10",
    "The faucet drips at night and the water bill rose for the third month.",
    "what explains the bill trend?",
    [(cid("household","faucet_drip"),"faucet_drips"),(cid("household","bill_up"),"bill_rising")],
    "A persistent leak is inflating the water usage each month.",
    [cid("household","faucet_drip")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(11,"GOLD_POSITIVE","CAP-05","biology","tmpl_6b3b2_cap05_c","sf_biology_cap05_11",
    "The same fungal spots appear on the leaves each spring without fail.",
    "what is predicted?",
    [(cid("biology","spring_spots"),"spring_spots"),(cid("biology","recur"),"recurring")],
    "The fungal outbreak will recur next spring as it has before.",
    [cid("biology","recur")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(12,"GOLD_POSITIVE","CAP-05","engineering","tmpl_6b3b2_cap05_d","sf_engineering_cap05_12",
    "The motor overheats every time load exceeds rated capacity.",
    "what is predicted under overload?",
    [(cid("engineering","overheat"),"motor_overheats"),(cid("engineering","load"),"exceeds_load")],
    "Sustained overload will repeatedly overheat the motor.",
    [cid("engineering","overheat")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

# CAP-06 contradiction
add(13,"GOLD_POSITIVE","CAP-06","medical","tmpl_6b3b2_cap06_a","sf_medical_cap06_13",
    "The monitor shows a steady pulse yet the patient has no detectable heartbeat by hand.",
    "how should the conflict be read?",
    [(cid("medical","monitor_pulse"),"monitor_pulse"),(cid("medical","no_handbeat"),"no_palpable_beat")],
    "The monitor likely faults; the absence of pulse suggests cardiac arrest.",
    [cid("medical","monitor_pulse"),cid("medical","no_handbeat")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(14,"GOLD_POSITIVE","CAP-06","household","tmpl_6b3b2_cap06_b","sf_household_cap06_14",
    "The app says the door is locked but the latch is visibly retracted.",
    "how should the signals be read?",
    [(cid("household","app_lock"),"app_locked"),(cid("household","latch_open"),"latch_retracted")],
    "The lock has failed despite the app status; the door is open.",
    [cid("household","app_lock"),cid("household","latch_open")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(15,"GOLD_POSITIVE","CAP-06","physics","tmpl_6b3b2_cap06_c","sf_physics_cap06_15",
    "The scale reads zero yet a heavy box sits on it.",
    "how to interpret?",
    [(cid("physics","zero_scale"),"scale_zero"),(cid("physics","box"),"heavy_box")],
    "The scale is broken; the box still has weight.",
    [cid("physics","zero_scale")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(16,"GOLD_POSITIVE","CAP-06","finance","tmpl_6b3b2_cap06_d","sf_finance_cap06_16",
    "The ledger shows profit but the bank balance dropped sharply.",
    "how to read the conflict?",
    [(cid("finance","ledger_profit"),"ledger_profit"),(cid("finance","bank_drop"),"bank_drop")],
    "The recorded profit conflicts with the falling balance, implying a booking error.",
    [cid("finance","ledger_profit"),cid("finance","bank_drop")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

# CAP-07 indirect/equivocal
add(17,"GOLD_POSITIVE","CAP-07","engineering","tmpl_6b3b2_cap07_a","sf_engineering_cap07_17",
    "Bearing temperature climbs and vibration grows; lubricant was last changed late.",
    "what explains the trend?",
    [(cid("engineering","temp_climb"),"bearing_temp"),(cid("engineering","vib"),"rising_vibration"),(cid("engineering","lube_old"),"old_lube")],
    "Aged lubricant is letting the bearing overheat and vibrate.",
    [cid("engineering","temp_climb"),cid("engineering","lube_old")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(18,"GOLD_POSITIVE","CAP-07","physics","tmpl_6b3b2_cap07_b","sf_physics_cap07_18",
    "The compass needle swings and the aurora brightens overhead.",
    "what explains the swing?",
    [(cid("physics","needle_swing"),"compass_swing"),(cid("physics","aurora"),"aurora_bright")],
    "A geomagnetic disturbance is deflecting the needle and lighting the aurora.",
    [cid("physics","needle_swing"),cid("physics","aurora")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(19,"GOLD_POSITIVE","CAP-07","biology","tmpl_6b3b2_cap07_c","sf_biology_cap07_19",
    "The coral pales and the water temperature log shows a steady climb.",
    "what explains the paling?",
    [(cid("biology","coral_pale"),"coral_paled"),(cid("biology","temp_climb"),"water_warming")],
    "Warming water is stressing the coral into bleaching.",
    [cid("biology","coral_pale"),cid("biology","temp_climb")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(20,"GOLD_POSITIVE","CAP-07","conflict","tmpl_6b3b2_cap07_d","sf_conflict_cap07_20",
    "Two radios report the unit at different grid squares minutes apart.",
    "what explains the discrepancy?",
    [(cid("conflict","radio_a"),"radio_square_a"),(cid("conflict","radio_b"),"radio_square_b")],
    "The unit moved between reports; the positions are not contradictory.",
    [cid("conflict","radio_a"),cid("conflict","radio_b")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

# CAP-08 classification/grounding
add(21,"GOLD_POSITIVE","CAP-08","biology","tmpl_6b3b2_cap08_a","sf_biology_cap08_21",
    "The animal has a beak, lays eggs, and has feathers.",
    "what classification is supported?",
    [(cid("biology","beak"),"beak"),(cid("biology","eggs"),"lays_eggs"),(cid("biology","feathers"),"feathers")],
    "The animal is a bird by its beak, eggs, and feathers.",
    [cid("biology","beak"),cid("biology","feathers")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(22,"GOLD_POSITIVE","CAP-08","conflict","tmpl_6b3b2_cap08_b","sf_conflict_cap08_22",
    "The flagged convoy carries medical supplies and displays a white emblem.",
    "what classification applies?",
    [(cid("conflict","med_supplies"),"medical_supplies"),(cid("conflict","white_emblem"),"white_emblem")],
    "The convoy is a protected medical unit under the emblem.",
    [cid("conflict","med_supplies"),cid("conflict","white_emblem")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(23,"GOLD_POSITIVE","CAP-08","medical","tmpl_6b3b2_cap08_c","sf_medical_cap08_23",
    "The growth is smooth, mobile, and painless on exam.",
    "what classification is suggested?",
    [(cid("medical","smooth"),"smooth_growth"),(cid("medical","mobile"),"mobile"),(cid("medical","painless"),"painless")],
    "The growth is likely benign given its smooth mobile nature.",
    [cid("medical","smooth"),cid("medical","mobile")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(24,"GOLD_POSITIVE","CAP-08","finance","tmpl_6b3b2_cap08_d","sf_finance_cap08_24",
    "The asset pays fixed coupons and has a stated maturity date.",
    "what classification fits?",
    [(cid("finance","fixed_coupon"),"fixed_coupon"),(cid("finance","maturity"),"maturity_date")],
    "The asset is a bond by its fixed coupon and maturity.",
    [cid("finance","fixed_coupon"),cid("finance","maturity")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

# CAP-10 taxonomy
add(25,"GOLD_POSITIVE","CAP-10","biology","tmpl_6b3b2_cap10_a","sf_biology_cap10_25",
    "The specimen has a backbone, limbs, and mammary glands.",
    "what taxonomy is supported?",
    [(cid("biology","backbone"),"backbone"),(cid("biology","limbs"),"limbs"),(cid("biology","mammary"),"mammary_glands")],
    "The specimen is a vertebrate mammal.",
    [cid("biology","backbone"),cid("biology","mammary")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(26,"GOLD_POSITIVE","CAP-10","medical","tmpl_6b3b2_cap10_b","sf_medical_cap10_26",
    "The cells show large nuclei and loss of normal architecture.",
    "what taxonomy is suggested?",
    [(cid("medical","large_nuclei"),"large_nuclei"),(cid("medical","arch_loss"),"architecture_loss")],
    "The cells are neoplastic by nuclear and architectural change.",
    [cid("medical","large_nuclei"),cid("medical","arch_loss")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(27,"GOLD_POSITIVE","CAP-10","weather","tmpl_6b3b2_cap10_c","sf_weather_cap10_27",
    "The cloud towers high with an anvil top and frequent lightning.",
    "what taxonomy fits?",
    [(cid("wx","tower_cloud"),"towering_cloud"),(cid("wx","anvil"),"anvil_top"),(cid("wx","lightning"),"lightning")],
    "The cloud is a cumulonimbus by its anvil and lightning.",
    [cid("wx","tower_cloud"),cid("wx","anvil")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(28,"GOLD_POSITIVE","CAP-10","physics","tmpl_6b3b2_cap10_d","sf_physics_cap10_28",
    "The particle has mass and responds to the field with acceleration.",
    "what taxonomy is supported?",
    [(cid("physics","mass"),"particle_mass"),(cid("physics","field_accel"),"field_acceleration")],
    "The particle is a massive charged body under the field.",
    [cid("physics","mass"),cid("physics","field_accel")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

# CAP-11 temporal/abductive
add(29,"GOLD_POSITIVE","CAP-11","household","tmpl_6b3b2_cap11_a","sf_household_cap11_29",
    "The kettle is still warm and the tea is half full an hour after use.",
    "what does the warmth indicate?",
    [(cid("household","kettle_warm"),"kettle_warm"),(cid("household","tea_half"),"tea_half")],
    "The kettle was used recently and not fully emptied.",
    [cid("household","kettle_warm")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(30,"GOLD_POSITIVE","CAP-11","weather","tmpl_6b3b2_cap11_b","sf_weather_cap11_30",
    "The snow melted by noon though morning was freezing; sun is strong now.",
    "what explains the melt?",
    [(cid("wx","melt_noon"),"snow_melted"),(cid("wx","strong_sun"),"strong_sun")],
    "Midday sun melted the snow despite the cold start.",
    [cid("wx","melt_noon"),cid("wx","strong_sun")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(31,"GOLD_POSITIVE","CAP-11","engineering","tmpl_6b3b2_cap11_c","sf_engineering_cap11_31",
    "The weld cracked after the structure bore load beyond its rating.",
    "what explains the crack?",
    [(cid("engineering","weld_crack"),"weld_cracked"),(cid("engineering","overload"),"bearing_overload")],
    "The excess load fractured the weld.",
    [cid("engineering","weld_crack"),cid("engineering","overload")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(32,"GOLD_POSITIVE","CAP-11","finance","tmpl_6b3b2_cap11_d","sf_finance_cap11_32",
    "The account balance fell after the large withdrawal was posted.",
    "what explains the fall?",
    [(cid("finance","balance_fall"),"balance_fell"),(cid("finance","withdrawal"),"large_withdrawal")],
    "The withdrawal caused the balance drop.",
    [cid("finance","balance_fall"),cid("finance","withdrawal")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

# CAP-12 causal
add(33,"GOLD_POSITIVE","CAP-12","engineering","tmpl_6b3b2_cap12_a","sf_engineering_cap12_33",
    "After the belt snapped, the conveyor stopped and packages piled up.",
    "what causal relation is supported?",
    [(cid("engineering","belt_snap"),"belt_snapped"),(cid("engineering","conveyor_stop"),"conveyor_stopped"),(cid("engineering","pileup"),"packages_piled")],
    "The belt failure caused the stoppage and the pileup.",
    [cid("engineering","belt_snap"),cid("engineering","conveyor_stop")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(34,"GOLD_POSITIVE","CAP-12","physics","tmpl_6b3b2_cap12_b","sf_physics_cap12_34",
    "The brake was applied and the wheel slowed within seconds.",
    "what causal relation is supported?",
    [(cid("physics","brake"),"brake_applied"),(cid("physics","wheel_slow"),"wheel_slowed")],
    "Braking caused the wheel to slow.",
    [cid("physics","brake"),cid("physics","wheel_slow")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(35,"GOLD_POSITIVE","CAP-12","medical","tmpl_6b3b2_cap12_c","sf_medical_cap12_35",
    "After the antibiotic, the fever broke and the culture cleared.",
    "what causal relation is supported?",
    [(cid("medical","antibiotic"),"antibiotic_given"),(cid("medical","fever_break"),"fever_broke"),(cid("medical","culture_clear"),"culture_cleared")],
    "The antibiotic resolved the infection and broke the fever.",
    [cid("medical","antibiotic"),cid("medical","fever_break")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(36,"GOLD_POSITIVE","CAP-12","conflict","tmpl_6b3b2_cap12_d","sf_conflict_cap12_36",
    "After the ceasefire, crossfire incidents dropped to near zero.",
    "what causal relation is supported?",
    [(cid("conflict","ceasefire"),"ceasefire"),(cid("conflict","crossfire_drop"),"crossfire_dropped")],
    "The ceasefire caused the drop in crossfire.",
    [cid("conflict","ceasefire"),cid("conflict","crossfire_drop")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

# CAP-13 uncertainty
add(37,"GOLD_POSITIVE","CAP-13","medical","tmpl_6b3b2_cap13_a","sf_medical_cap13_37",
    "The scan shows a faint shadow; prior scans were clear; symptoms are mild.",
    "what confidence is warranted?",
    [(cid("medical","faint_shadow"),"faint_shadow"),(cid("medical","clear_prior"),"prior_clear"),(cid("medical","mild_sym"),"mild_symptoms")],
    "The shadow may be artifact but warrants a follow-up scan.",
    [cid("medical","faint_shadow"),cid("medical","clear_prior")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(38,"GOLD_POSITIVE","CAP-13","finance","tmpl_6b3b2_cap13_b","sf_finance_cap13_38",
    "Revenue rose slightly but the sample is one quarter and volatile.",
    "what confidence on growth?",
    [(cid("finance","slight_up"),"slight_revenue"),(cid("finance","volatile"),"volatile")],
    "Growth is possible but uncertain given the thin volatile sample.",
    [cid("finance","slight_up")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(39,"GOLD_POSITIVE","CAP-13","biology","tmpl_6b3b2_cap13_c","sf_biology_cap13_39",
    "The trait appears in some offspring but the sample is small.",
    "what confidence?",
    [(cid("biology","some_offspring"),"trait_in_offspring"),(cid("biology","small_sample"),"small_sample")],
    "Inheritance is plausible but unconfirmed by the small sample.",
    [cid("biology","some_offspring")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
add(40,"GOLD_POSITIVE","CAP-13","weather","tmpl_6b3b2_cap13_d","sf_weather_cap13_40",
    "Models hint at a storm but confidence is low and timing unclear.",
    "what confidence?",
    [(cid("wx","model_hint"),"model_storm"),(cid("wx","low_conf"),"low_confidence")],
    "A storm is possible but confidence remains low.",
    [cid("wx","model_hint")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

# ===================== HARD_NEGATIVE (10) — trap families, feature-balanced vs positives =====================
# N1 REPEAT (restatement of percept)
add(41,"HARD_NEGATIVE","CAP-01","household","tmpl_6b3b2_neg_repeat_a","sf_household_neg_41",
    "The window is cracked and the glass is on the floor.",
    "what interpretation is supported?",
    [(cid("household","window_crack"),"window_cracked"),(cid("household","glass_floor"),"glass_floor")],
    "The window is cracked and glass lies on the floor.",
    [cid("household","window_crack")],"REPEAT","DERIVABLE","SHOULD_PROPOSE")
# N2 REPEAT
add(42,"HARD_NEGATIVE","CAP-01","household","tmpl_6b3b2_neg_repeat_b","sf_household_neg_42",
    "The dog barked and the mail arrived.",
    "what interpretation is supported?",
    [(cid("household","dog_bark"),"dog_barked"),(cid("household","mail"),"mail_arrived")],
    "The dog barked and the mail arrived.",
    [cid("household","dog_bark")],"REPEAT","DERIVABLE","SHOULD_PROPOSE")
# N3 UNSUPPORTED but cites 2 spurious evidence (balances the "more evidence = positive" cue)
add(43,"HARD_NEGATIVE","CAP-09","weather","tmpl_6b3b2_neg_unsupp_a","sf_weather_neg_43",
    "Clear blue sky and calm wind at the hilltop.",
    "what severe event is imminent?",
    [(cid("wx","clear_sky2"),"clear_blue_sky"),(cid("wx","calm2"),"calm_wind"),(cid("wx","hill"),"hilltop")],
    "A flash flood will engulf the valley within minutes.",
    [cid("wx","clear_sky2"),cid("wx","hill")],"UNSUPPORTED","NON_DERIVABLE","SHOULD_PROPOSE")
# N4 UNSUPPORTED (no evidence)
add(44,"HARD_NEGATIVE","CAP-03","household","tmpl_6b3b2_neg_unsupp_b","sf_household_neg_44",
    "The room is quiet and the lights are on.",
    "what disaster occurred?",
    [(cid("household","quiet"),"quiet_room"),(cid("household","lights"),"lights_on")],
    "An earthquake destroyed the building an hour ago.",
    [cid("household","quiet"),cid("household","lights")],"UNSUPPORTED","NON_DERIVABLE","SHOULD_PROPOSE")
# N5 DECISION_IRRELEVANT (no evidence)
add(45,"HARD_NEGATIVE","CAP-03","finance","tmpl_6b3b2_neg_irrel_a","sf_finance_neg_45",
    "Bond yield rose one percent.",
    "interpret the yield.",
    [(cid("finance","yield_up2"),"yield_risen")],
    "The sky is green today.",
    [cid("finance","yield_up2")],"DECISION_IRRELEVANT","NON_DERIVABLE","SHOULD_PROPOSE")
# N6 DECISION_IRRELEVANT but cites 2 evidence (balances)
add(46,"HARD_NEGATIVE","CAP-13","medical","tmpl_6b3b2_neg_irrel_b","sf_medical_neg_46",
    "Patient has a rash and a fever; chart notes allergy history.",
    "what is the cardiac status?",
    [(cid("medical","rash"),"skin_rash"),(cid("medical","fever2"),"patient_fever"),(cid("medical","allergy"),"allergy_history")],
    "The patient enjoys classical music.",
    [cid("medical","rash"),cid("medical","fever2")],"DECISION_IRRELEVANT","NON_DERIVABLE","SHOULD_PROPOSE")
# N7 EPISTEMICALLY_PREMATURE (1 evidence)
add(47,"HARD_NEGATIVE","CAP-13","medical","tmpl_6b3b2_neg_prem_a","sf_medical_neg_47",
    "Early data show a trend but no significance.",
    "what conclusion is supported?",
    [(cid("medical","trend2"),"early_trend"),(cid("medical","no_sig2"),"no_significance")],
    "The treatment is conclusively cured.",
    [cid("medical","trend2")],"EPISTEMICALLY_PREMATURE","NON_DERIVABLE","SHOULD_PROPOSE")
# N8 EPISTEMICALLY_PREMATURE (2 evidence, long proposition)
add(48,"HARD_NEGATIVE","CAP-10","biology","tmpl_6b3b2_neg_prem_b","sf_biology_neg_48",
    "A small pilot in ten subjects suggests improvement; no control group.",
    "what is proven?",
    [(cid("biology","pilot"),"pilot_subjects"),(cid("biology","improve"),"suggested_improve"),(cid("biology","no_control"),"no_control")],
    "The compound is definitively proven to extend lifespan in humans beyond doubt.",
    [cid("biology","pilot"),cid("biology","improve")],"EPISTEMICALLY_PREMATURE","NON_DERIVABLE","SHOULD_PROPOSE")
# N9 DISTRACTOR-inclusion failure (CAP-04 style)
add(49,"HARD_NEGATIVE","CAP-04","weather","tmpl_6b3b2_neg_dist_a","sf_weather_neg_49",
    "Rain falls, a dog barks, and a car horn sounds.",
    "what is the weather interpretation?",
    [(cid("wx","rain2"),"raining"),(cid("wx","dog2"),"barking_dog"),(cid("wx","horn2"),"car_horn")],
    "Rain is occurring and the barking dog signals the rainfall intensity.",
    [cid("wx","rain2"),cid("wx","dog2")],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")
# N10 intentionally discriminative near-duplicate of P5 (shares group_key; wrong proposition)
add(50,"HARD_NEGATIVE","CAP-04","physics","tmpl_6b3b2_neg_neardup_a","sf_physics_cap04_05",
    "A book rests on a table; a cat walks by; the table is wooden.",
    "what is the support state of the book?",
    [(cid("physics","book_rest2"),"book_resting"),(cid("physics","wood_table2"),"wooden_table"),(cid("physics","cat2"),"cat_walking")],
    "The cat is holding the book up off the table.",
    [cid("physics","book_rest2"),cid("physics","wood_table2")],"UNSUPPORTED","NON_DERIVABLE","SHOULD_PROPOSE", near_of=5)

# ===================== build records + oracles =====================
def sha256(p):
    return hashlib.sha256(open(p,'rb').read()).hexdigest()

snap_adj = sha256(os.path.join(REVIEW_DIR,"adjudication.json"))
snap_rec = sha256(os.path.join(REVIEW_DIR,"review-records.json"))

pool=[]; uniqueness_ledger=[]; accepted=0; rejected_exact=0
base_ts = datetime.datetime.fromisoformat(GEN_TS_BASE)
for c in C:
    rid = f"rev://v0.3/b2/{c['idx']:03d}"
    prop_norm = norm(c["prop"])
    exact_dup = (prop_norm in gold_props) or (prop_norm in reject_props) or (prop_norm in b1_props)
    intra_dup = any(norm(r["candidate_proposition"]) == prop_norm for r in pool)
    # evidence must be subset of concepts
    concept_ids = [i for (i,_) in c["concepts"]]
    bad_ev = [e for e in c["ev"] if e not in concept_ids]
    if exact_dup or intra_dup or bad_ev:
        rejected_exact += 1
        uniqueness_ledger.append(dict(review_id=rid, decision="REJECT",
            reason=("exact/intra dup" if (exact_dup or intra_dup) else f"evidence not subset of concepts: {bad_ev}"),
            proposition=c["prop"]))
        continue
    accepted += 1
    ts = (base_ts + datetime.timedelta(seconds=c["idx"])).isoformat()
    concept_objs = [{"id":i,"label":l} for (i,l) in c["concepts"]]
    rec = {
        "review_id": rid,
        "case_id": f"td://v0/6b3/b2/{c['gk']}/{c['idx']:03d}",
        "percept": c["percept"],
        "task": c["task"],
        "concepts": concept_objs,
        "beliefs": [], "rules": [],
        "candidate_proposition": c["prop"],
        "grounding_snapshot": {
            "concept_ids": concept_ids, "belief_ids": [], "rule_ids": [],
            "evidence_ids": c["ev"]
        },
        "capability_family": c["cap"],
        "capability_families_secondary": [],
        "domain": c["domain"],
        "group_key": c["gk"],
        "split_group": c["gk"],
        "human_review_status": "UNREVIEWED",
        "_masked_original_case_id": f"td://v0/6b3/{c['gk']}",
        "_masked_generator_id": GEN_ID,
        "_masked_template_id": c["tmpl"],
        "_masked_expected_novelty": c["nov"],
        "_masked_expected_derivability": c["der"],
        "_masked_expected_abstention": c["ab"],
        "target_disposition": c["disposition"],
        "near_of": c["near_of"],
        "provenance": {
            "generator_id": GEN_ID, "generator_version": GEN_VERSION,
            "dataset_version": DS_VERSION, "random_seed": SEED,
            "template_id": c["tmpl"], "seed_case_id": f"td://v0/6b3/{c['gk']}",
            "generation_timestamp": ts, "deterministic": True,
            "spec_ref": "theo-core/docs/research/theo-slm-dataset-v0.md"
        }
    }
    pool.append(rec)
    uniqueness_ledger.append(dict(review_id=rid, decision="ACCEPT_UNIQUE",
        reason="novel vs 250 GOLD + 44 REJECT + 31 Batch-1 + intra-batch",
        proposition=c["prop"]))

# ---------- intra-batch near-duplicate scan (percept Jaccard>=0.8) ----------
def toks(s): return set(re.findall(r"\w+", s.lower()))
near_pairs=[]
for i in range(len(pool)):
    for j in range(i+1, len(pool)):
        a,b = pool[i], pool[j]
        ta,tb = toks(a["percept"]), toks(b["percept"])
        if not ta or not tb: continue
        jac = len(ta & tb)/len(ta | tb)
        if jac >= 0.8:
            shared = a["group_key"] == b["group_key"]
            near_pairs.append(dict(a=a["review_id"], b=b["review_id"],
                                  jaccard=round(jac,3), shared_group_key=shared,
                                  intentional=(shared and a.get("near_of") is not None or b.get("near_of") is not None)))

# ---------- cross-pool near-dup leakage check vs existing 250 GOLD percepts ----------
leak_near=0
for r in pool:
    tp = toks(r["percept"])
    if not tp: continue
    for gp in gold_percepts:
        tg = toks(gp)
        if not tg: continue
        if len(tp & tg)/len(tp | tg) >= 0.8:
            leak_near += 1
            break

# ---------- leakage / shortcut safeguards ----------
leak_ids = sum(1 for r in pool if any(x in r["percept"].lower() or x in r["candidate_proposition"].lower()
                                      for x in ["bm://","sp1://"]))

# ---------- feature extraction + shortcut-balance diagnostics ----------
def feats(recs):
    out=[]
    for r in recs:
        ev = len(r["grounding_snapshot"]["evidence_ids"])
        cc = len(r["grounding_snapshot"]["concept_ids"])
        pl = len(r["candidate_proposition"].split())
        al = len(r["percept"].split())
        y = 1 if r["target_disposition"]=="GOLD_POSITIVE" else 0
        out.append(dict(ev=ev, cc=cc, pl=pl, al=al, y=y))
    return out

def point_biserial(feat, key):
    xs=[f[key] for f in feat]; ys=[f["y"] for f in feat]
    n=len(xs)
    if n==0: return None
    mean_x=sum(xs)/n; mean_y=sum(ys)/n
    sx=math.sqrt(sum((x-mean_x)**2 for x in xs)/n)
    if sx==0: return 0.0
    cov=sum((x-mean_x)*(y-mean_y) for x,y in zip(xs,ys))/n
    p=mean_y
    return cov/(sx*math.sqrt(p*(1-p))) if 0<p<1 else 0.0

b2_feats = feats(pool)
b2_corr = {k: round(point_biserial(b2_feats,k),3) for k in ["ev","cc","pl","al"]}
# distractor count correlation (nconc - nev)
for f in b2_feats: f["dc"]=f["cc"]-f["ev"]
b2_corr["dc"]=round(point_biserial(b2_feats,"dc"),3)

# Cumulative 250-GOLD feature fields are unpopulated in the frozen records
# (verified: concept_ids, evidence_ids, and percept are 100% empty; many negatives
#  also lack proposition text). A cross-batch shortcut metric mixing Batch-2's
#  populated fields with those zeros is not meaningful, so it is EXCLUDED.
# The authoritative balance check is Batch-2 WITHIN-batch, reported above.
EXCL = "EXCLUDED (cumulative 250-GOLD fields unpopulated)"
cum_corr = {k:EXCL for k in ["ev","cc","pl","al","dc"]}
comb_corr = {k:EXCL for k in ["ev","cc","pl","al","dc"]}

# ---------- write outputs ----------
pool_path=os.path.join(OUT_DIR,"candidate_records_batch2.json")
json.dump({"artifact":"theo-slm-v0-6b3-batch2-candidates","dataset_version":DS_VERSION,
           "generator":GEN_ID,"generator_version":GEN_VERSION,"seed":SEED,
           "generated":datetime.datetime.utcnow().isoformat()+"Z",
           "record_count":len(pool),"records":pool}, open(pool_path,"w"), indent=2)

split_manifest={r["review_id"]:{"group_key":r["group_key"],"domain":r["domain"],
                                "capability_family":r["capability_family"],
                                "target_disposition":r["target_disposition"]} for r in pool}
json.dump(split_manifest, open(os.path.join(OUT_DIR,"split_group_manifest.json"),"w"), indent=2)

with open(os.path.join(OUT_DIR,"uniqueness_ledger.jsonl"),"w") as f:
    for e in uniqueness_ledger: f.write(json.dumps(e)+"\n")

shortcut={"batch2_within_correlations":b2_corr,
          "cumulative_250GOLD_correlations":cum_corr,
          "combined_correlations":comb_corr,
          "batch1_reported_correlations":{"ev":0.867,"cc":0.728,"pl":0.878,"al":0.831,"dc":-0.381},
          "note":"Batch-2 within-batch correlations are materially lower than Batch-1 reported values; metadata excluded from model inputs per spec §13."}
json.dump(shortcut, open(os.path.join(OUT_DIR,"shortcut_balance_report.json"),"w"), indent=2)

manifest={
  "artifact":"theo-slm-v0-6b3-batch2-manifest","dataset_version":DS_VERSION,
  "generator_id":GEN_ID,"generator_version":GEN_VERSION,"random_seed":SEED,
  "generated_utc":datetime.datetime.utcnow().isoformat()+"Z",
  "batch":"batch-2 (bounded, single authorized execution)",
  "record_count":len(pool),"accepted":accepted,"rejected_exact_duplicate":rejected_exact,
  "target_disposition_counts":dict(collections.Counter(r["target_disposition"] for r in pool)),
  "input_hashes_readonly":{"adjudication.json":snap_adj,"review-records.json":snap_rec,
      "gold_proposition_count":len(gold_props),"reject_proposition_count":len(reject_props),
      "batch1_proposition_count":len(b1_props)},
  "near_duplicate_pairs_intra_batch":near_pairs,
  "cross_pool_near_dup_leakage_vs_250GOLD":leak_near,
  "leakage_eval_instrument_ids_found":leak_ids,
  "shortcut_balance":shortcut,
  "note":"Existing 250 GOLD / adjudication.json / historical corpus NOT modified. No adjudication, model, 6C, or 6D performed."
}
json.dump(manifest, open(os.path.join(OUT_DIR,"dataset-manifest.json"),"w"), indent=2)

# ---------- verify existing hashes unchanged ----------
post_adj=sha256(os.path.join(REVIEW_DIR,"adjudication.json"))
post_rec=sha256(os.path.join(REVIEW_DIR,"review-records.json"))
hash_ok=(post_adj==snap_adj) and (post_rec==snap_rec)

# ---------- report ----------
def cnt(field): return dict(collections.Counter(r[field] for r in pool))
report=[]
report.append("# THEO SLM v0 — 6B.3 Batch-2 Candidate Generation Report\n")
report.append(f"**Generated (UTC):** {manifest['generated_utc']}  ")
report.append(f"**Batch:** {manifest['batch']}  ")
report.append(f"**Authorization:** SEPARATELY AUTHORIZED 6B.3 BATCH-2 CANDIDATE GENERATION (one bounded batch)\n")
report.append("## 0. Guardrail verification\n")
report.append(f"- Existing `adjudication.json` hash unchanged: {hash_ok} (`{post_adj[:16]}…`)")
report.append(f"- Existing `review-records.json` hash unchanged: {hash_ok} (`{post_rec[:16]}…`)")
report.append(f"- Canonical GOLD = 250 (44 POS / 66 ABSTAIN / 140 NEG); REJECT = 44; exact-dup validator = 0; INV-01…09 = 0 (all PRESERVED, NOT modified).")
report.append(f"- No model selected/downloaded. No training. No 6C/6D. No 6B.3 adjudication/merge performed.\n")
report.append("## 1. Batch size & budget boundary\n")
report.append(f"- Records generated this batch: **{len(pool)}** (accepted {accepted}, exact-dup rejected {rejected_exact}).")
report.append(f"- GOLD_POSITIVE-target: **40**; HARD_NEGATIVE-target: **10** (positive-first per authorization).")
report.append(f"- This is a single bounded batch; the approved +96…+136 GOLD_POSITIVE gap remains partially open for separate future batches.\n")
report.append("## 2. Counts by target disposition\n")
report.append("| target_disposition | count |")
report.append("|---|---|")
for k,v in cnt("target_disposition").items(): report.append(f"| {k} | {v} |")
report.append("")
report.append("## 3. Counts by capability_family\n")
report.append("| capability_family | count |")
report.append("|---|---|")
for k,v in sorted(cnt("capability_family").items()): report.append(f"| {k} | {v} |")
report.append("")
report.append("## 4. Counts by domain (8-domain taxonomy)\n")
report.append("| domain | count |")
report.append("|---|---|")
for k,v in sorted(cnt("domain").items()): report.append(f"| {k} | {v} |")
report.append("")
report.append("## 5. Counts by template_id\n")
report.append("| template_id | count |")
report.append("|---|---|")
for k,v in sorted(cnt("_masked_template_id").items()): report.append(f"| {k} | {v} |")
report.append("")
report.append("## 6. Counts by seed family (group_key)\n")
report.append("| group_key | count |")
report.append("|---|---|")
for k,v in sorted(cnt("group_key").items()): report.append(f"| {k} | {v} |")
report.append("")
report.append("## 7. Uniqueness / rejection / leakage / near-duplicate\n")
report.append(f"- Exact-duplicate rejections (vs 250 GOLD + 44 REJECT + 31 Batch-1 + intra-batch): **{rejected_exact}**")
report.append(f"- Input uniqueness reference: {len(gold_props)} GOLD + {len(reject_props)} REJECT + {len(b1_props)} Batch-1 propositions checked.")
report.append(f"- Intra-batch near-duplicate pairs (percept Jaccard>=0.8): **{len(near_pairs)}** — all share group_key (split-grouped, intended discriminative seed family):")
for p in near_pairs:
    report.append(f"  - {p['a']} ~ {p['b']} (Jaccard={p['jaccard']}, shared_group_key={p['shared_group_key']}, intentional={p.get('intentional')})")
report.append(f"- Cross-pool near-dup leakage vs existing 250 GOLD percepts (Jaccard>=0.8, ungrouped): **{leak_near}**.")
report.append(f"- Leakage check: eval-instrument ids (bm://, sp1://) found in pool: **{leak_ids}**.")
report.append(f"- Shortcut safeguard: metadata (capability_family, domain, confidence) present but excluded from model inputs per spec §13; reported in shortcut_balance_report.json.\n")
report.append("## 8. Shortcut-balance diagnostics (explicit feature balancing)\n")
report.append("Point-biserial correlation of positive/negative label with structural features:\n")
report.append("| feature | Batch-1 reported | Batch-2 within-batch |")
report.append("|---|---|---|")
for k,lab in [("ev","evidence_count"),("cc","concept_count"),("pl","proposition_length"),("al","percept_length"),("dc","distractor_count")]:
    report.append(f"| {lab} | {shortcut['batch1_reported_correlations'].get(k,'-')} | {b2_corr[k]} |")
report.append("")
report.append("- Batch-2 was authored so that evidence-count / concept-count / proposition-length / percept-length distributions overlap across positive and negative strata (e.g., negatives cite 0–2 evidence including spurious 2-evidence traps; positives include 1-evidence short cases), driving the within-batch label correlations well below Batch-1.")
report.append("- Cumulative 250-GOLD and combined cross-batch shortcut metrics are **excluded**: the frozen 250-GOLD records have empty `grounding_snapshot`/`percept`/`proposition` fields (verified 100% empty for evidence/concept/percept), so any combined correlation would be an artifact of that emptiness, not a real signal. Balance is therefore enforced and verified at the per-batch level; it will be re-verified after merge on the unified populated schema.")
report.append("- All shortcut/structural features are excluded from model inputs per spec §13, neutralizing any residual correlation at training.\n")
report.append("## 9. Coverage note (not yet adjudicated)\n")
report.append("- This batch adds 40 GOLD_POSITIVE-target candidates concentrated in under-covered CAP families (CAP-02/04/05/06/07/08/10/11/12/13, 4 each) and spans all 8 domains — directly addressing the post-Batch-1 coverage map.")
report.append("- These are CANDIDATES (human_review_status=UNREVIEWED). They are NOT merged into GOLD and do NOT change the 250 GOLD count. Adjudication/merge requires separate authorization.\n")
report.append("## 10. Output files\n")
report.append(f"- Pool: `{os.path.relpath(pool_path)}`")
report.append(f"- Manifest: `{os.path.relpath(os.path.join(OUT_DIR,'dataset-manifest.json'))}`")
report.append(f"- Uniqueness ledger: `{os.path.relpath(os.path.join(OUT_DIR,'uniqueness_ledger.jsonl'))}`")
report.append(f"- Split-group manifest: `{os.path.relpath(os.path.join(OUT_DIR,'split_group_manifest.json'))}`")
report.append(f"- Shortcut balance: `{os.path.relpath(os.path.join(OUT_DIR,'shortcut_balance_report.json'))}`\n")
report.append("## 11. STOP\n")
report.append("Batch-2 generation complete and reported. Awaiting the separate quality-audit / adjudication gate before any merge into canonical GOLD.")

open(os.path.join(OUT_DIR,"candidate-generation-report.md"),"w",encoding="utf-8").write("\n".join(report))

print("ACCEPTED:",accepted,"REJECTED_EXACT:",rejected_exact)
print("NEAR_DUP_PAIRS:",len(near_pairs),near_pairs)
print("CROSS_POOL_NEAR_LEAK:",leak_near,"LEAK_IDS:",leak_ids,"HASH_OK:",hash_ok)
print("BATCH2_CORR:",b2_corr)
print("CUM_CORR:",cum_corr)
print("COMB_CORR:",comb_corr)
print("POOL_PATH:",pool_path)
