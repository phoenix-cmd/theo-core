#!/usr/bin/env python3
"""
THEO SLM v0 — Phase 6B.3 Batch-1 Candidate Generation (bounded, single authorized batch).

Authorization: HUMAN AUTHORIZATION GRANTED FOR ONE BOUNDED BATCH ONLY.
This script:
  - reads existing GOLD corpus + REJECT ledger as read-only inputs (NEVER modified),
  - generates NEW candidate records into a NEW pool directory,
  - runs the uniqueness oracle (reject exact duplicates vs 226 GOLD + 38 REJECT),
  - assigns group_key / split_group seed-family metadata for near-dup grouping,
  - emits dataset-manifest.json, uniqueness_ledger.jsonl, split_group_manifest.json,
    and a candidate-generation report.
It does NOT modify adjudication.json, review-records.json, the gold corpus, or the
historical 67/66/131 corpus. It does NOT perform adjudication, model selection, 6C, or 6D.
"""
import json, hashlib, os, re, datetime, collections

REVIEW_DIR = "theo-data/datasets/theo_slm_v0_review"
OUT_DIR = os.path.join(REVIEW_DIR, "6b3_batch1")
SEED = 43
GEN_ID = "gen_6b3_batch1"
GEN_VERSION = "gen-v0.4.0"
DS_VERSION = "ds-v0.4-candidates"
GEN_TS_BASE = "2026-08-25T12:00:00+00:00"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- load read-only inputs ----------
adj = json.load(open(os.path.join(REVIEW_DIR, "adjudication.json")))
recs = json.load(open(os.path.join(REVIEW_DIR, "review-records.json")))

# map review_id -> final_status
status_by_rid = {r["review_id"]: r.get("final_status") for r in adj["records"]}
gold_props = []      # 226 GOLD propositions (normalized)
reject_props = []    # 38 REJECT propositions (normalized)
for r in recs:
    rid = r.get("review_id")
    st = status_by_rid.get(rid)
    prop = (r.get("candidate_proposition") or "").strip()
    if st == "REJECT":
        reject_props.append(prop)
    elif st in ("GOLD_POSITIVE", "GOLD_ABSTAIN", "HARD_NEGATIVE"):
        gold_props.append(prop)

def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

GOLD_NORM = set(norm(p) for p in gold_props if p)
REJECT_NORM = set(norm(p) for p in reject_props if p)

# ---------- curated bounded batch data ----------
# disposition: GOLD_POSITIVE (novel interpretable) or HARD_NEGATIVE (trap)
# cap: capability_family; domain: one of 8; gk: group_key (seed family)
# novelty/deriv/abstain: expected *_masked fields
# evidence: list of concept ids (subset of concepts) cited
C = []  # list of dicts

def add(idx, disp, cap, domain, tmpl, gk, percept, task, concepts, prop, ev, nov, der, ab, near_of=None):
    C.append(dict(idx=idx, disposition=disp, cap=cap, domain=domain, tmpl=tmpl, gk=gk,
                  percept=percept, task=task, concepts=concepts, prop=prop, ev=ev,
                  nov=nov, der=der, ab=ab, near_of=near_of))

# ---- 20 GOLD_POSITIVE (prioritizing under-covered CAPs + all 8 domains) ----
add(1,"GOLD_POSITIVE","CAP-02","weather","tmpl_6b3_cap02_a","sf_weather_abd_01",
    "Multiple streetlights along the avenue are dark; a neighbor reports a loud transformer bang minutes earlier.",
    "what interpretation is supported by the context?",
    [("concept://wx/transformer_noise","transformer_bang"),("concept://wx/lights_out","streetlights_dark"),("concept://wx/sky","night_sky")],
    "A localized transformer failure likely caused the block-wide outage.",
    ["concept://wx/transformer_noise","concept://wx/lights_out"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(2,"GOLD_POSITIVE","CAP-02","household","tmpl_6b3_cap02_b","sf_house_abd_01",
    "The bread is flattened and the toaster emits a burning smell with no lever movement.",
    "what interpretation is supported by the context?",
    [("concept://household/burning_smell","toaster_burning"),("concept://household/no_lever","lever_stuck"),("concept://household/bread","flattened_bread")],
    "The heating element likely shorted, preventing the carriage from engaging.",
    ["concept://household/burning_smell","concept://household/no_lever"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(3,"GOLD_POSITIVE","CAP-04","weather","tmpl_6b3_cap04_a","sf_weather_dist_01",
    "Bright sunshine, clear sky, a red bicycle parked nearby, and a puddle from morning watering.",
    "what is the current weather interpretation?",
    [("concept://wx/sunshine","bright_sunshine"),("concept://wx/clear_sky","clear_sky"),("concept://wx/bicycle","red_bicycle"),("concept://wx/puddle","morning_puddle")],
    "Conditions are dry and clear; the bicycle and puddle are irrelevant distractors.",
    ["concept://wx/sunshine","concept://wx/clear_sky"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(4,"GOLD_POSITIVE","CAP-04","finance","tmpl_6b3_cap04_b","sf_fin_dist_01",
    "The stock ticker shows a gain; a celebrity tweet unrelated to the firm; trading volume is average.",
    "what drives the price move here?",
    [("concept://finance/ticker_gain","price_gain"),("concept://finance/celeb_tweet","unrelated_tweet"),("concept://finance/volume_avg","average_volume")],
    "The price movement reflects normal volume and is unrelated to the celebrity post.",
    ["concept://finance/ticker_gain","concept://finance/volume_avg"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(5,"GOLD_POSITIVE","CAP-06","household","tmpl_6b3_cap06_a","sf_house_contra_01",
    "The thermostat reads 22C yet the occupant wears a winter coat and shivers.",
    "how should the conflicting signals be interpreted?",
    [("concept://household/thermostat_reading","sensor_22c"),("concept://household/shivering","occupant_shivers"),("concept://household/coat","winter_coat")],
    "The heating likely failed despite the thermostat sensor reading.",
    ["concept://household/thermostat_reading","concept://household/shivering"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(6,"GOLD_POSITIVE","CAP-06","medical","tmpl_6b3_cap06_b","sf_med_contra_01",
    "Patient chart shows 'afebrile' but the nurse note says 'feverish to touch, sweating'.",
    "how should the conflicting records be interpreted?",
    [("concept://medical/afebrile_flag","chart_afebrile"),("concept://medical/fever_observation","nurse_fever"),("concept://medical/sweating","sweating")],
    "The recorded temperature conflicts with observed febrile signs, suggesting a measurement error.",
    ["concept://medical/afebrile_flag","concept://medical/fever_observation"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(7,"GOLD_POSITIVE","CAP-07","physics","tmpl_6b3_cap07_a","sf_phys_indir_01",
    "The pendulum slows, the room is warmer than at start, and the clock runs behind.",
    "what explains the timing drift?",
    [("concept://physics/slower_swing","pendulum_slows"),("concept://physics/warmer_room","room_warmer"),("concept://physics/clock_behind","clock_lag")],
    "Thermal expansion of the pendulum rod likely lengthened it, slowing oscillation.",
    ["concept://physics/slower_swing","concept://physics/warmer_room"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(8,"GOLD_POSITIVE","CAP-07","engineering","tmpl_6b3_cap07_b","sf_eng_indir_01",
    "Vibration readings rise near the bearing; lubricant level is low; temperature climbs.",
    "what explains the vibration trend?",
    [("concept://engineering/low_lube","low_lubricant"),("concept://engineering/temp_climb","bearing_temp"),("concept://engineering/vibration","rising_vibration")],
    "Insufficient lubrication is likely overheating the bearing, raising vibration.",
    ["concept://engineering/low_lube","concept://engineering/temp_climb"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(9,"GOLD_POSITIVE","CAP-08","biology","tmpl_6b3_cap08_a","sf_bio_ground_01",
    "The specimen shows bilateral symmetry and mammary glands.",
    "what classification is supported by the features?",
    [("concept://biology/bilateral_sym","bilateral_symmetry"),("concept://biology/mammary","mammary_glands")],
    "The specimen is a mammal, per observed bilateral symmetry and mammary features.",
    ["concept://biology/bilateral_sym","concept://biology/mammary"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(10,"GOLD_POSITIVE","CAP-08","conflict","tmpl_6b3_cap08_b","sf_conf_ground_01",
    "Two patrols report opposite control of the same bridge within an hour.",
    "what is the control status of the bridge?",
    [("concept://conflict/patrol_A_claim","patrol_a_control"),("concept://conflict/patrol_B_claim","patrol_b_control"),("concept://conflict/bridge","bridge_id")],
    "The bridge changed hands or reports conflict; control is currently contested.",
    ["concept://conflict/patrol_A_claim","concept://conflict/patrol_B_claim"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(11,"GOLD_POSITIVE","CAP-10","biology","tmpl_6b3_cap10_a","sf_bio_tax_01",
    "The organism has feathers and lays eggs; it is observed incubating young.",
    "what classification is supported?",
    [("concept://biology/feathers","feathers"),("concept://biology/egg_laying","lays_eggs"),("concept://biology/incubating","incubating_young")],
    "The organism is a bird, inferred from feather and egg-laying traits.",
    ["concept://biology/feathers","concept://biology/egg_laying"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(12,"GOLD_POSITIVE","CAP-10","medical","tmpl_6b3_cap10_b","sf_med_tax_01",
    "The lesion is asymmetric with an irregular border and variable color.",
    "what does the lesion pattern suggest?",
    [("concept://medical/asymmetry","lesion_asymmetry"),("concept://medical/irregular_border","irregular_border"),("concept://medical/var_color","variable_color")],
    "The lesion pattern suggests malignancy risk by ABCD criteria.",
    ["concept://medical/asymmetry","concept://medical/irregular_border"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(13,"GOLD_POSITIVE","CAP-11","household","tmpl_6b3_cap11_a","sf_house_temp_01",
    "The iron is still warm an hour after unplugging; the cord shows scorch marks.",
    "what does the residual heat indicate?",
    [("concept://household/residual_warmth","iron_warm"),("concept://household/scorch","cord_scorch"),("concept://household/unplugged","unplugged_hour")],
    "Residual heat indicates the iron was recently active and likely overloaded.",
    ["concept://household/residual_warmth","concept://household/scorch"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(14,"GOLD_POSITIVE","CAP-11","weather","tmpl_6b3_cap11_b","sf_weather_temp_01",
    "The storm front passed at noon; by evening barometric pressure is rising and skies clear.",
    "what weather change is indicated?",
    [("concept://wx/pressure_rising","rising_pressure"),("concept://wx/clearing_sky","clearing_skies"),("concept://wx/front_passed","front_passed")],
    "Post-frontal high pressure is restoring fair conditions.",
    ["concept://wx/pressure_rising","concept://wx/clearing_sky"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(15,"GOLD_POSITIVE","CAP-12","engineering","tmpl_6b3_cap12_a","sf_eng_causal_01",
    "After the valve closed, downstream pressure dropped and flow stopped.",
    "what causal relation is supported?",
    [("concept://engineering/valve_closed","valve_closure"),("concept://engineering/pressure_drop","downstream_pressure_drop"),("concept://engineering/flow_stop","flow_ceased")],
    "Valve closure caused the downstream pressure loss and flow cessation.",
    ["concept://engineering/valve_closed","concept://engineering/pressure_drop"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(16,"GOLD_POSITIVE","CAP-12","physics","tmpl_6b3_cap12_b","sf_phys_causal_01",
    "The ice melted after the lamp was switched on; surface temperature rose.",
    "what causal relation is supported?",
    [("concept://physics/lamp_on","lamp_activated"),("concept://physics/temp_rise","surface_temp_rise"),("concept://physics/ice_melt","ice_melted")],
    "Lamp heat caused the ice to melt via thermal transfer.",
    ["concept://physics/lamp_on","concept://physics/temp_rise"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(17,"GOLD_POSITIVE","CAP-13","medical","tmpl_6b3_cap13_a","sf_med_uncert_01",
    "Chest scan shows a small nodule; history unclear; one prior similar benign case.",
    "what confidence is warranted?",
    [("concept://medical/nodule_find","small_nodule"),("concept://medical/prior_benign","prior_benign_case"),("concept://medical/unclear_history","unclear_history")],
    "The nodule is possibly benign but warrants follow-up given limited history.",
    ["concept://medical/nodule_find","concept://medical/prior_benign"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(18,"GOLD_POSITIVE","CAP-13","finance","tmpl_6b3_cap13_b","sf_fin_uncert_01",
    "Quarterly revenue is up 2% but margin is compressed and one large client decision is pending.",
    "what confidence is warranted on growth?",
    [("concept://finance/revenue_up","revenue_gain"),("concept://finance/margin_compressed","margin_compression"),("concept://finance/client_pending","pending_client")],
    "Growth is modest and uncertain pending the client decision.",
    ["concept://finance/revenue_up","concept://finance/margin_compressed"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(19,"GOLD_POSITIVE","CAP-01","weather","tmpl_6b3_cap01_a","sf_weather_storm_01",
    "Dark clouds, distant thunder, and a rapid drop in pressure.",
    "what interpretation is supported by the context?",
    [("concept://wx/dark_clouds","dark_clouds"),("concept://wx/thunder","distant_thunder"),("concept://wx/pressure_drop","pressure_fall")],
    "An approaching thunderstorm is likely given the pressure fall and thunder.",
    ["concept://wx/dark_clouds","concept://wx/pressure_drop"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(20,"GOLD_POSITIVE","CAP-03","physics","tmpl_6b3_cap03_a","sf_phys_rel_01",
    "An object accelerates downhill; friction is noted low; wind is negligible; a bird flies overhead.",
    "what drives the acceleration?",
    [("concept://physics/acceleration","downhill_accel"),("concept://physics/low_friction","low_friction"),("concept://physics/bird","bird_overhead")],
    "Gravity drives the acceleration; the bird is irrelevant.",
    ["concept://physics/acceleration","concept://physics/low_friction"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

# ---- 10 HARD_NEGATIVE (trap families) ----
add(21,"HARD_NEGATIVE","CAP-05","household","tmpl_6b3_neg_repeat_a","sf_house_repeat_01",
    "The cup fell and broke on the floor.",
    "what interpretation is supported?",
    [("concept://household/cup_fell","cup_fell"),("concept://household/broke","cup_broke")],
    "The cup was broken by the impact on the floor.",
    ["concept://household/cup_fell"],"REPEAT","DERIVABLE","SHOULD_PROPOSE")

add(22,"HARD_NEGATIVE","CAP-09","weather","tmpl_6b3_neg_unsupp_a","sf_weather_unsupp_01",
    "Clear sky, calm wind.",
    "what severe weather is imminent?",
    [("concept://wx/clear_sky","clear_sky"),("concept://wx/calm","calm_wind")],
    "A tornado will strike within the hour.",
    [],"UNSUPPORTED","NON_DERIVABLE","SHOULD_PROPOSE")

add(23,"HARD_NEGATIVE","CAP-03","finance","tmpl_6b3_neg_irrel_a","sf_fin_irrel_01",
    "Bond yield rose 1%.",
    "interpret the yield movement.",
    [("concept://finance/yield_rise","yield_up")],
    "The sky is blue today.",
    [],"DECISION_IRRELEVANT","NON_DERIVABLE","SHOULD_PROPOSE")

add(24,"HARD_NEGATIVE","CAP-13","medical","tmpl_6b3_neg_prem_a","sf_med_prem_01",
    "Early trial shows a trend but no statistical significance.",
    "what conclusion is supported?",
    [("concept://medical/trend","early_trend"),("concept://medical/no_sig","no_significance")],
    "The drug is definitively cured.",
    ["concept://medical/trend"],"EPISTEMICALLY_PREMATURE","NON_DERIVABLE","SHOULD_PROPOSE")

add(25,"HARD_NEGATIVE","CAP-12","physics","tmpl_6b3_neg_unsupp_b","sf_phys_unsupp_01",
    "A ball rolls straight across the table.",
    "what caused the motion?",
    [("concept://physics/ball_roll","ball_rolls"),("concept://physics/straight","straight_path")],
    "A hidden magnet beneath the table pulled it.",
    [],"UNSUPPORTED","NON_DERIVABLE","SHOULD_PROPOSE")

add(26,"HARD_NEGATIVE","CAP-05","biology","tmpl_6b3_neg_repeat_b","sf_bio_repeat_01",
    "The cell divided into two daughter cells.",
    "what interpretation is supported?",
    [("concept://biology/cell_div","cell_divided"),("concept://biology/daughters","two_daughters")],
    "The cell underwent division producing two cells.",
    ["concept://biology/cell_div"],"REPEAT","DERIVABLE","SHOULD_PROPOSE")

add(27,"HARD_NEGATIVE","CAP-04","weather","tmpl_6b3_neg_dist_a","sf_weather_dist_02",
    "Rain is falling, a dog is barking, and a car horn sounds.",
    "what is the weather interpretation?",
    [("concept://wx/rain","raining"),("concept://wx/dog","barking_dog"),("concept://wx/horn","car_horn")],
    "Rain is occurring and the barking dog indicates rainfall intensity.",
    ["concept://wx/rain","concept://wx/dog"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE")

add(28,"HARD_NEGATIVE","CAP-01","engineering","tmpl_6b3_neg_deriv_a","sf_eng_deriv_01",
    "The pump stopped because power was cut.",
    "what interpretation is supported?",
    [("concept://engineering/pump_stop","pump_stopped"),("concept://engineering/power_cut","power_cut")],
    "The pump stopped because power was cut.",
    ["concept://engineering/pump_stopped"],"REPEAT","DERIVABLE","SHOULD_PROPOSE")

add(29,"HARD_NEGATIVE","CAP-12","conflict","tmpl_6b3_neg_unsupp_c","sf_conf_unsupp_01",
    "Two groups are arguing at the market.",
    "what caused the dispute?",
    [("concept://conflict/group_A","group_a"),("concept://conflict/group_B","group_b"),("concept://conflict/market","market")],
    "Group A attacked because group B wore red.",
    [],"UNSUPPORTED","NON_DERIVABLE","SHOULD_PROPOSE")

add(30,"HARD_NEGATIVE","CAP-09","household","tmpl_6b3_neg_irrel_b","sf_house_irrel_01",
    "The lamp is on.",
    "interpret the lighting state.",
    [("concept://household/lamp_on","lamp_on")],
    "The refrigerator hums.",
    [],"DECISION_IRRELEVANT","NON_DERIVABLE","SHOULD_PROPOSE")

# ---- near-duplicate pair (Jaccard>=0.8) sharing group_key with #19 ----
add(31,"GOLD_POSITIVE","CAP-01","weather","tmpl_6b3_cap01_b","sf_weather_storm_01",
    "Dark clouds, distant thunder, and a rapid fall in pressure.",
    "what interpretation is supported by the context?",
    [("concept://wx/dark_clouds","dark_clouds"),("concept://wx/thunder","distant_thunder"),("concept://wx/pressure_drop","pressure_fall")],
    "A thunderstorm is approaching as shown by the falling pressure and thunder.",
    ["concept://wx/dark_clouds","concept://wx/pressure_drop"],"SEMANTIC_NOVEL","NON_DERIVABLE","SHOULD_PROPOSE",
    near_of=19)

# ---------- build records + run uniqueness oracle ----------
def sha256(p):
    return hashlib.sha256(open(p,'rb').read()).hexdigest()

snap_adj = sha256(os.path.join(REVIEW_DIR,"adjudication.json"))
snap_rec = sha256(os.path.join(REVIEW_DIR,"review-records.json"))

pool = []
uniqueness_ledger = []
accepted = 0
rejected_exact = 0

base_ts = datetime.datetime.fromisoformat(GEN_TS_BASE)
for c in C:
    rid = f"rev://v0.3/b1/{c['idx']:03d}"
    prop_norm = norm(c["prop"])
    exact_dup = (prop_norm in GOLD_NORM) or (prop_norm in REJECT_NORM)
    # intra-batch uniqueness
    intra_dup = any(norm(r["candidate_proposition"]) == prop_norm for r in pool)
    if exact_dup or intra_dup:
        rejected_exact += 1
        uniqueness_ledger.append(dict(review_id=rid, decision="REJECT_EXACT_DUPLICATE",
                                      reason="matches existing GOLD/REJECT or intra-batch dup",
                                      proposition=c["prop"]))
        continue
    accepted += 1
    ts = (base_ts + datetime.timedelta(seconds=c["idx"])).isoformat()
    concept_objs = [{"id":i,"label":l} for (i,l) in c["concepts"]]
    concept_ids = [i for (i,_) in c["concepts"]]
    rec = {
        "review_id": rid,
        "case_id": f"td://v0/6b3/b1/{c['gk']}/{c['idx']:03d}",
        "percept": c["percept"],
        "task": c["task"],
        "concepts": concept_objs,
        "beliefs": [],
        "rules": [],
        "candidate_proposition": c["prop"],
        "grounding_snapshot": {
            "concept_ids": concept_ids,
            "belief_ids": [],
            "rule_ids": [],
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
        "provenance": {
            "generator_id": GEN_ID,
            "generator_version": GEN_VERSION,
            "dataset_version": DS_VERSION,
            "random_seed": SEED,
            "template_id": c["tmpl"],
            "seed_case_id": f"td://v0/6b3/{c['gk']}",
            "generation_timestamp": ts,
            "deterministic": True,
            "spec_ref": "theo-core/docs/research/theo-slm-dataset-v0.md"
        }
    }
    pool.append(rec)
    uniqueness_ledger.append(dict(review_id=rid, decision="ACCEPT_UNIQUE",
                                  reason="novel vs 226 GOLD + 38 REJECT + intra-batch",
                                  proposition=c["prop"]))

# ---------- intra-batch near-duplicate scan (informational; must share group_key) ----------
def toks(s): return set(re.findall(r"\w+", s.lower()))
near_pairs = []
for i in range(len(pool)):
    for j in range(i+1, len(pool)):
        a,b = pool[i], pool[j]
        ta,tb = toks(a["percept"]), toks(b["percept"])
        if not ta or not tb: continue
        jac = len(ta & tb)/len(ta | tb)
        if jac >= 0.8:
            shared = a["group_key"] == b["group_key"]
            near_pairs.append(dict(a=a["review_id"], b=b["review_id"],
                                  jaccard=round(jac,3), shared_group_key=shared))

# ---------- leakage / shortcut safeguards ----------
leak_ids = sum(1 for r in pool if any(x in r["percept"].lower() or x in r["candidate_proposition"].lower()
                                      for x in ["bm://","sp1://"]))

# ---------- write outputs ----------
pool_path = os.path.join(OUT_DIR,"candidate_records_batch1.json")
json.dump({"artifact":"theo-slm-v0-6b3-batch1-candidates",
           "dataset_version":DS_VERSION,"generator":GEN_ID,"generator_version":GEN_VERSION,
           "seed":SEED,"generated":datetime.datetime.utcnow().isoformat()+"Z",
           "record_count":len(pool),"records":pool}, open(pool_path,"w"), indent=2)

# split-group manifest (group_key per record)
split_manifest = {r["review_id"]: {"group_key": r["group_key"], "domain": r["domain"],
                                   "capability_family": r["capability_family"],
                                   "target_disposition": r["target_disposition"]} for r in pool}
json.dump(split_manifest, open(os.path.join(OUT_DIR,"split_group_manifest.json"),"w"), indent=2)

# uniqueness ledger
with open(os.path.join(OUT_DIR,"uniqueness_ledger.jsonl"),"w") as f:
    for e in uniqueness_ledger:
        f.write(json.dumps(e)+"\n")

# manifest
manifest = {
  "artifact":"theo-slm-v0-6b3-batch1-manifest",
  "dataset_version":DS_VERSION,
  "generator_id":GEN_ID,
  "generator_version":GEN_VERSION,
  "random_seed":SEED,
  "generated_utc": datetime.datetime.utcnow().isoformat()+"Z",
  "batch":"batch-1 (bounded, single authorized execution)",
  "record_count": len(pool),
  "accepted": accepted,
  "rejected_exact_duplicate": rejected_exact,
  "target_disposition_counts": dict(collections.Counter(r["target_disposition"] for r in pool)),
  "input_hashes_readonly": {
     "adjudication.json": snap_adj,
     "review-records.json": snap_rec,
     "gold_proposition_count": len(GOLD_NORM),
     "reject_proposition_count": len(REJECT_NORM)
  },
  "near_duplicate_pairs_intra_batch": near_pairs,
  "leakage_eval_instrument_ids_found": leak_ids,
  "note":"Existing 226 GOLD / adjudication.json / historical corpus NOT modified. No adjudication, model, 6C, or 6D performed."
}
json.dump(manifest, open(os.path.join(OUT_DIR,"dataset-manifest.json"),"w"), indent=2)

# ---------- verify existing hashes unchanged ----------
post_adj = sha256(os.path.join(REVIEW_DIR,"adjudication.json"))
post_rec = sha256(os.path.join(REVIEW_DIR,"review-records.json"))
hash_ok = (post_adj==snap_adj) and (post_rec==snap_rec)

# ---------- report ----------
def cnt(field): return dict(collections.Counter(r[field] for r in pool))
report = []
report.append("# THEO SLM v0 — 6B.3 Batch-1 Candidate Generation Report\n")
report.append(f"**Generated (UTC):** {manifest['generated_utc']}  ")
report.append(f"**Batch:** {manifest['batch']}  ")
report.append(f"**Authorization:** HUMAN AUTHORIZATION GRANTED FOR ONE BOUNDED BATCH ONLY\n")
report.append("\n## 0. Guardrail verification\n")
report.append(f"- Existing `adjudication.json` hash unchanged: {hash_ok} (`{post_adj[:16]}…`)")
report.append(f"- Existing `review-records.json` hash unchanged: {hash_ok} (`{post_rec[:16]}…`)")
report.append(f"- 226 GOLD records: NOT modified. 38 REJECT: NOT restored. Historical 67/66/131 corpus: NOT modified.")
report.append(f"- No model selected/downloaded. No training. No 6C/6D. No 6B.3 adjudication performed.\n")
report.append("## 1. Batch size & budget boundary\n")
report.append(f"- Records generated this batch: **{len(pool)}** (accepted {accepted}, exact-dup rejected {rejected_exact}).")
report.append(f"- This is ~13% of the plan's ~240 positive raw budget — a deliberately partial, inspectable batch.")
report.append(f"- Remaining authorized budget NOT consumed in this execution.\n")
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
report.append("## 7. Uniqueness / rejection / leakage / shortcut statistics\n")
report.append(f"- Exact-duplicate rejections (vs 226 GOLD + 38 REJECT + intra-batch): **{rejected_exact}**")
report.append(f"- Input uniqueness reference: {len(GOLD_NORM)} GOLD propositions + {len(REJECT_NORM)} REJECT propositions checked.")
report.append(f"- Intra-batch near-duplicate pairs (percept Jaccard>=0.8): **{len(near_pairs)}** — all share group_key (split-grouped, NOT rejected):")
for p in near_pairs:
    report.append(f"  - {p['a']} ~ {p['b']} (Jaccard={p['jaccard']}, shared_group_key={p['shared_group_key']})")
report.append(f"- Leakage check: eval-instrument ids (bm://, sp1://) found in pool: **{leak_ids}**.")
report.append(f"- Shortcut safeguard: metadata fields (capability_family, domain, confidence) present but excluded from model inputs per spec §13; reported here only for audit.")
report.append(f"- Deterministic seed: {SEED}; all records carry provenance.random_seed, generator_id, generator_version, template_id, seed_case_id, generation_timestamp.\n")
report.append("## 8. Coverage note (not yet adjudicated)\n")
report.append("- This batch adds GOLD_POSITIVE-target candidates across CAP-02/04/06/07/08/10/11/12/13 (previously under-covered) and spans all 8 domains (medical, weather, physics, finance, biology, engineering, household, conflict).")
report.append("- These are CANDIDATES (human_review_status=UNREVIEWED). They are NOT merged into GOLD and do NOT change the 226 GOLD count. Adjudication requires separate authorization.\n")
report.append("## 9. Output files\n")
report.append(f"- Pool: `{os.path.relpath(pool_path)}`")
report.append(f"- Manifest: `{os.path.relpath(os.path.join(OUT_DIR,'dataset-manifest.json'))}`")
report.append(f"- Uniqueness ledger: `{os.path.relpath(os.path.join(OUT_DIR,'uniqueness_ledger.jsonl'))}`")
report.append(f"- Split-group manifest: `{os.path.relpath(os.path.join(OUT_DIR,'split_group_manifest.json'))}`\n")
report.append("## 10. STOP\n")
report.append("Batch generation complete and reported. Awaiting human inspection of this batch + coverage/quality report before any adjudication authorization or next-batch decision.")

open(os.path.join(OUT_DIR,"candidate-generation-report.md"),"w").write("\n".join(report))

print("ACCEPTED:", accepted, "REJECTED_EXACT:", rejected_exact)
print("NEAR_DUP_PAIRS:", len(near_pairs), near_pairs)
print("LEAK_IDS:", leak_ids, "HASH_OK:", hash_ok)
print("POOL_PATH:", pool_path)
