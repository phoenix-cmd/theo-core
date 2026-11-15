"""
6C.5 — Normalized experiment specification (PREPARATION ONLY — NOT EXECUTED).

Single source of truth for the controlled B/C comparison. Both B and C import
identical prompts, identical data, identical split/seed, identical PRIMARY
supervision from this module. The ONLY intended difference is the architectural
mechanism package (B = single decoder + snapshot reranker; C = shared encoder +
two heads + Ports gate), with C's auxiliary evidence-head loss pre-registered and
LOCKED (see AUX_LOSS_WEIGHT). Nothing here edits b_architecture.py /
c_architecture.py or any historical 6C.4 artifact.
"""
from __future__ import annotations
import os, json, hashlib, re

ROOT = os.environ.get("THEO_ROOT", r"C:\Users\bs162\Desktop\THEO")
M1_DIR = os.path.join(ROOT, "theo-data/models/6c_acquired/M1_Qwen2.5-0.5B-Instruct")
BATCH2 = os.path.join(ROOT, "theo-data/datasets/theo_slm_v0_review/6b3_batch2")
MAPPING = os.path.join(BATCH2, "6c4_mapping_manifest.json")
DEDUP = os.path.join(ROOT, "theo-data/datasets/theo_slm_v0_deduplicated/candidate_records.json")
BATCH1 = os.path.join(ROOT, "theo-data/datasets/theo_slm_v0_review/6b3_batch1/candidate_records_batch1.json")
BATCH2R = os.path.join(ROOT, "theo-data/datasets/theo_slm_v0_review/6b3_batch2/candidate_records_batch2_remediated.json")
FIXTURE = os.path.join(ROOT, "theo-data/datasets/theo_slm_v0_artifacts/phase-6e3/populated-evidence-corpus.json")
ADJUDICATION = os.path.join(ROOT, "theo-data/datasets/theo_slm_v0_review/adjudication.json")

SEED = 20260826
OUT_B = os.environ.get("THEO_OUT_B", os.path.join(BATCH2, "M1_Qwen2.5-0.5B-Instruct-6c5-B"))
OUT_C = os.environ.get("THEO_OUT_C", os.path.join(BATCH2, "M1_Qwen2.5-0.5B-Instruct-6c5-C"))

# --- FROZEN fixture integrity (pre-registered; eval asserts equality at runtime) ---
FIXTURE_SHA256 = "f99ca025eaaa3073e296e7cb734581f0871315553b5c145bb7f2fc27a4ac6457"

# --- Locked auxiliary-loss weight (R1). C includes its evidence-head target at this
#     weight relative to the primary envelope target. PROHIBITED from tuning. ---
AUX_LOSS_WEIGHT = 1.0

# --- Normalized prompt (SINGLE shared constant; copied verbatim from c_arch so it is a
#     real, previously-used prompt). Identical for B and C. ---
NORM_SYS = ('You are a semantic interpretation engine. Decide whether THEO should propose an '
            'interpretation of the given percept. Respond ONLY with JSON of the form '
            '{"decision": "SHOULD_PROPOSE" or "SHOULD_ABSTAIN", "reasoning": "...", '
            '"evidence_refs": ["E1"]}.')

NORM_USER = (
    'Percept: {percept}\n\n'
    'Available evidence (cite the bracketed IDs in "evidence_refs"; only cite IDs that '
    'genuinely support your decision):\n{evlist}\n\n'
    'Respond ONLY with JSON: {{"decision": "SHOULD_PROPOSE" or "SHOULD_ABSTAIN", '
    '"reasoning": "...", "evidence_refs": ["E1"]}}')

NORM_EVID_SYS = ('You are an evidence-relevance encoder. Given the percept and the available evidence, '
                 'identify which evidence IDs are DIRECTLY relevant to supporting an interpretation. '
                 'Respond ONLY with JSON of the form {"relevant_evidence": ["E1"]}. '
                 'Cite ONLY IDs from the provided evidence list.')

NORM_EVID_USER = (
    '<<EVIDENCE_HEAD>>\n'
    'Percept: {percept}\n\n'
    'Available evidence:\n{evlist}\n\n'
    'Respond ONLY with JSON: {{"relevant_evidence": ["E1"]}}. Cite ONLY IDs from the list above.')

# Firewall (R2/R3): the reserved marker <<EVIDENCE_HEAD>> delimits C's auxiliary evidence
# channel. The PRIMARY (envelope) user turn for BOTH B and C must NEVER contain it, so that
# evidence is delivered identically via the shared envelope and C's extra head is the only
# architectural delta. The Phase-2 audit asserts byte-equality of evidence_payload across B/C.
EVIDENCE_TOKEN = "<<EVIDENCE_HEAD>>"
def firewall_user_turn(s): return EVIDENCE_TOKEN not in s

FIXED_REASONING = "Decision derived from provided evidence per THEO semantic-interpretation contract."

# Architecture-agnostic prompt neutrality: the prompt must not encode any B/C-specific
# mechanism. Used by the Phase-2 audit (semantic-neutrality review).
FORBIDDEN_PROMPT_TOKENS = ["rerank", "ports", "snapshot", "two-head", "two head",
                           "shared encoder", "evidence head", "evidence-head",
                           "architecture b", "architecture c", "decoder-only", "baseline"]

def prompt_is_neutral():
    blob = (NORM_SYS + " " + NORM_USER + " " + NORM_EVID_SYS + " " + NORM_EVID_USER).lower()
    hits = [t for t in FORBIDDEN_PROMPT_TOKENS if t in blob]
    return (len(hits) == 0), hits

def _load_records(path):
    j = json.load(open(path))
    if isinstance(j, list):
        return j
    for v in j.values():
        if isinstance(v, list):
            return v
    return []

def merged_source_records():
    """deduplicated -> batch1 -> batch2_remediated (first match wins)."""
    recs = {}
    for path in (DEDUP, BATCH1, BATCH2R):
        for r in _load_records(path):
            if isinstance(r, dict) and "case_id" in r:
                recs.setdefault(r["case_id"], r)
    return recs

def build_examples():
    """Return (examples, data_sha). B and C receive IDENTICAL primary supervision
    (envelope). C additionally receives the auxiliary evidence-head target (locked)."""
    mp = json.load(open(MAPPING))
    rec_by_id = merged_source_records()
    h = hashlib.sha256()
    out = []
    for r in mp["rows"]:
        if not r.get("included_in_training"):
            continue
        cid = r["case_id"]
        rec = rec_by_id.get(cid)
        if not rec:
            continue
        percept = rec.get("percept") or rec.get("text")
        status = r["final_status"]
        tgt = "SHOULD_PROPOSE" if status == "GOLD_POSITIVE" else "SHOULD_ABSTAIN"
        pos = list(r.get("evidence_ids") or []) if status == "GOLD_POSITIVE" else []
        textmap = {}
        for key in ("concepts", "beliefs", "rules"):
            for item in (rec.get(key) or []):
                if isinstance(item, dict) and "id" in item:
                    textmap[item["id"]] = item.get("label") or item.get("definition") or item["id"]
        ev_list = "\n".join(f"[{pid}] {textmap.get(pid, pid)}" for pid in pos) if pos else "(no evidence provided)"
        envelope = json.dumps({"decision": tgt, "reasoning": FIXED_REASONING,
                               "evidence_refs": pos}, ensure_ascii=False)
        ev_head = json.dumps({"relevant_evidence": pos}, ensure_ascii=False)
        out.append({"case_id": cid, "percept": percept, "ev_list": ev_list,
                    "evidence": [{"id": pid, "text": textmap.get(pid, pid)} for pid in pos],
                    "tgt": tgt, "pos": pos, "envelope": envelope, "evidence_head": ev_head,
                    "capability_family": rec.get("capability_family")})
        h.update(f"{cid}|{tgt}|{envelope}".encode())
    return out, h.hexdigest()

def split_examples(examples, seed=SEED):
    """Deterministic stratified 90/10 by tgt. IDENTICAL procedure for B and C.
    Returns (train, dev) so the audit can verify the exact partitions used at training."""
    import random
    rnd = random.Random(seed)
    by_tgt = {}
    for e in examples:
        by_tgt.setdefault(e["tgt"], []).append(e)
    train, dev = [], []
    for t, lst in by_tgt.items():
        rnd.shuffle(lst)
        k = max(1, int(0.9 * len(lst)))
        train += lst[:k]; dev += lst[k:]
    return train, dev

def evidence_payload(case):
    """SINGLE source of external evidence for a fixture case. Returned payload is passed
    IDENTICALLY to B and C at inference (firewall test asserts byte-equality)."""
    ev = case.get("evidence") or []
    ev_list = "\n".join(f"[{e['id']}] {e['text']}" for e in ev)
    snapshot = {e["id"]: e["text"] for e in ev}
    return ev_list, snapshot

def fixture_sha256():
    return hashlib.sha256(open(FIXTURE, "rb").read()).hexdigest()

def adjudication_sha256():
    return hashlib.sha256(open(ADJUDICATION, "rb").read()).hexdigest()
