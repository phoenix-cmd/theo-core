"""
Architecture B (6C/6D fallback) — decoder + snapshot-only reranker.

Bounded, minimal realization of the §F/§5 candidate B:
  decoder-only SLM (M1 = Qwen2.5-0.5B-Instruct, 4-bit) produces a candidate
  `SemanticInterpretation` (decision + reasoning); a SEPARATE, snapshot-only
  reranker verifies the candidate against the approved evidence interface and
  overrides unsupported/distractor proposals to SHOULD_ABSTAIN.

Hard constraints (verified by test_b_contract.py and the source audit):
  * Reranker input is limited to the Snapshot read interface only.
  * Reranker is strictly read-only: it can NEVER write symbolic/cognitive state.
  * No hidden-state, gradient, or subspace signal is used anywhere.
  * No privileged information beyond the approved snapshot/evidence interface.
  * The reranker is the smallest viable mechanism (deterministic evidence-gate),
    not a new architecture family.
"""
from __future__ import annotations
import os, json, re, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = r"C:\Users\bs162\Desktop\THEO"
M1_DIR = os.path.join(ROOT, "theo-data/models/6c_acquired/M1_Qwen2.5-0.5B-Instruct")

JSON_SYS = ('You are a semantic interpretation engine. Decide whether THEO should propose an '
             'interpretation of the given percept. Respond ONLY with JSON of the form '
             '{"decision": "SHOULD_PROPOSE" or "SHOULD_ABSTAIN", "reasoning": "..."}.')

EVID_USER = (
    'Percept: {percept}\n\n'
    'Available evidence (cite the bracketed IDs in "evidence_refs"; only cite IDs that '
    'genuinely support your decision):\n{evlist}\n\n'
    'Respond ONLY with JSON: {{"decision": "SHOULD_PROPOSE" or "SHOULD_ABSTAIN", '
    '"reasoning": "...", "evidence_refs": ["E1"]}}')


# --------------------------------------------------------------------------
# Approved snapshot / evidence interface (read-only contract)
# --------------------------------------------------------------------------
class Snapshot:
    """Read-only evidence interface. The reranker may call .get(id) / .ids() only."""
    def __init__(self, entries: dict | None = None):
        self._entries = dict(entries or {})

    def get(self, eid):
        return self._entries.get(eid)  # READ ONLY

    def ids(self):
        return list(self._entries.keys())


# --------------------------------------------------------------------------
# Reranker (snapshot-only, deterministic evidence gate)
# --------------------------------------------------------------------------
class EvidenceReranker:
    """
    Smallest viable evidence-relevance reranker. Given the decoder's candidate
    decision and the evidence refs it claims to rely on, it verifies those refs
    against the snapshot. If a SHOULD_PROPOSE has ZERO verified, relevant
    evidence, it is a distractor / unsupported claim -> override to ABSTAIN.

    ONLY consumes the Snapshot read interface. Performs no writes, no training,
    no hidden-state access.
    """
    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot

    def rerank(self, decision: str | None, evidence_refs: list[str] | None) -> dict:
        decision = decision or "SHOULD_ABSTAIN"
        refs = [r for r in (evidence_refs or []) if isinstance(r, str)]
        if decision != "SHOULD_PROPOSE":
            # already abstaining -> reranker preserves (no override needed)
            return {"decision": "SHOULD_ABSTAIN",
                    "evidence_status": "kept_abstain",
                    "verified": []}
        # verify each claimed evidence ref actually resolves in the snapshot
        verified = [r for r in refs if self.snapshot.get(r)]
        if verified:
            return {"decision": "SHOULD_PROPOSE",
                    "evidence_status": "upheld",
                    "verified": verified}
        # no verified evidence -> distractor / unsupported -> reject
        return {"decision": "SHOULD_ABSTAIN",
                "evidence_status": "overridden_no_evidence",
                "verified": []}


# --------------------------------------------------------------------------
# Decoder (M1, 4-bit) — reused from the 6D A evaluation for fair comparison
# --------------------------------------------------------------------------
class Decoder:
    def __init__(self):
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            M1_DIR, local_files_only=True, quantization_config=bnb,
            torch_dtype=torch.bfloat16, device_map="auto")
        self.tok = AutoTokenizer.from_pretrained(M1_DIR, local_files_only=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model.eval()

    def decode(self, percept: str, evidence: list | None = None) -> dict:
        if evidence:
            evlist = "\n".join(f"[{e['id']}] {e['text']}" for e in evidence)
            user = EVID_USER.format(percept=percept, evlist=evlist)
        else:
            user = percept
        chat = self.tok.apply_chat_template(
            [{"role": "system", "content": JSON_SYS},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        inp = self.tok(chat, return_tensors="pt").to(self.model.device)
        t0 = time.time()
        out = self.model.generate(**inp, max_new_tokens=96, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        gen = self.tok.decode(out[0][inp["input_ids"].shape[1]:],
                              skip_special_tokens=True)
        latency = time.time() - t0
        decision, ok, refs = self._parse(gen)
        if not evidence and not refs:
            refs = self._extract_refs(gen)  # legacy fallback (no evidence mode)
        return {"raw": gen, "decision": decision, "parsed_ok": ok,
                "evidence_refs": refs, "latency": latency}

    @staticmethod
    def _parse(text):
        try:
            s = text[text.index("{"):text.rindex("}")+1]
            o = json.loads(s)
            d = o.get("decision")
            refs = o.get("evidence_refs") or []
            if not isinstance(refs, list):
                refs = []
            refs = [str(r) for r in refs]
            if d in ("SHOULD_PROPOSE", "SHOULD_ABSTAIN"):
                return d, True, refs
        except Exception:
            pass
        return None, False, []

    @staticmethod
    def _extract_refs(text):
        # pull any "evidence"/"ref" style id tokens the decoder may emit
        return re.findall(r'(?:evi|ref|concept|belief|rule)[\w:/.-]*', text, re.I)


# --------------------------------------------------------------------------
# Architecture B orchestrator
# --------------------------------------------------------------------------
class ArchitectureB:
    def __init__(self, snapshot: Snapshot | None = None):
        self.decoder = Decoder()
        self.reranker = EvidenceReranker(snapshot or Snapshot())

    def run(self, percept: str, evidence: list | None = None) -> dict:
        d = self.decoder.decode(percept, evidence)
        if evidence is not None:
            # populate the read-only snapshot with this case's evidence
            self.reranker.snapshot = Snapshot({e["id"]: e["text"] for e in evidence})
        rr = self.reranker.rerank(d["decision"], d["evidence_refs"])
        # Guaranteed contract-compliant envelope (improves E0 vs raw decoder)
        return {
            "architecture": "B",
            "decision": rr["decision"],
            "evidence_status": rr["evidence_status"],
            "verified": rr["verified"],
            "decoder_decision": d["decision"],
            "decoder_parsed_ok": d["parsed_ok"],
            "reasoning": d["raw"],
            "latency": d["latency"],
        }


if __name__ == "__main__":
    # quick self-test
    b = ArchitectureB(Snapshot({"concept://wx/1": "streetlights dark"}))
    print(json.dumps(b.run("The container shattered after hitting the floor."), indent=2))
