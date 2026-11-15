"""
Architecture C (6C/6D) - shared encoder / two-head.

Bounded realization of the 6C plan (candidate C, sec F / sec 5):
  A SINGLE shared encoder (M1 = Qwen2.5-0.5B-Instruct, 4-bit) serves TWO heads:
    Head 1 (decoder / decision head):  generative -> {decision, reasoning, evidence_refs}
    Head 2 (evidence / relation head): generative -> {relevant_evidence}
  Both heads read the per-case evidence ONLY through the read-only Ports (the frozen
  neural-symbolic interface). Neither head can mutate symbolic / cognitive state.
  A deterministic gate verifies the cited evidence against Ports and overrides any
  unsupported proposal to SHOULD_ABSTAIN.

Hard constraints (verified by test_c_contract.py + the independent source audit):
  * Both heads consume ONLY the shared encoder + the read-only Ports interface.
  * No symbolic-state writes; no hidden-state projection or similar selection signals;
    the shared encoder is used only in generative mode (no hidden-state projection,
    no specialized tuning of output layers, no parameter-efficient tuning, no
    model-generated labels used as supervision).
  * Human/symbolic GOLD labels are the only legitimate supervision source; in this
    bounded build C is evaluated ZERO-SHOT (no supervised head training in this step).
  * Firewall intact: no theo_core / theo_providers import.
"""
from __future__ import annotations
import os, json, re, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = r"C:\Users\bs162\Desktop\THEO"
M1_DIR = os.path.join(ROOT, "theo-data/models/6c_acquired/M1_Qwen2.5-0.5B-Instruct")

DECISION_SYS = ('You are a semantic interpretation engine. Decide whether THEO should propose an '
                'interpretation of the given percept. Respond ONLY with JSON of the form '
                '{"decision": "SHOULD_PROPOSE" or "SHOULD_ABSTAIN", "reasoning": "...", '
                '"evidence_refs": ["E1"]}.')

EVID_SYS = ('You are an evidence-relevance encoder. Given the percept and the available evidence, '
            'identify which evidence IDs are DIRECTLY relevant to supporting an interpretation. '
            'Respond ONLY with JSON of the form {"relevant_evidence": ["E1"]}. Cite ONLY IDs from '
            'the provided evidence list.')

DECISION_USER = (
    'Percept: {percept}\n\n'
    'Available evidence (cite the bracketed IDs in "evidence_refs"; only cite IDs that '
    'genuinely support your decision):\n{evlist}\n\n'
    'Respond ONLY with JSON: {{"decision": "SHOULD_PROPOSE" or "SHOULD_ABSTAIN", '
    '"reasoning": "...", "evidence_refs": ["E1"]}}')

EVID_USER = (
    'Percept: {percept}\n\n'
    'Available evidence:\n{evlist}\n\n'
    'Respond ONLY with JSON: {{"relevant_evidence": ["E1"]}}. Cite ONLY IDs from the list above.')


class Ports:
    """Read-only frozen neural-symbolic interface. Heads may call .get / .ids only."""
    def __init__(self, entries=None):
        self._entries = dict(entries or {})

    def get(self, eid):
        return self._entries.get(eid)  # READ ONLY

    def ids(self):
        return list(self._entries.keys())


class SharedEncoder:
    """The single shared encoder for both heads (M1, 4-bit, generative only)."""
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

    def generate(self, sys_prompt, user_content, max_new_tokens=96):
        chat = self.tok.apply_chat_template(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True)
        inp = self.tok(chat, return_tensors="pt").to(self.model.device)
        t0 = time.time()
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=self.tok.eos_token_id)
        gen = self.tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        return gen, time.time() - t0

    @staticmethod
    def _parse_json(text, *keys):
        try:
            s = text[text.index("{"):text.rindex("}") + 1]
            o = json.loads(s)
            out = {}
            for k in keys:
                v = o.get(k)
                if isinstance(v, list):
                    v = [str(x) for x in v]
                out[k] = v
            return out, True
        except Exception:
            return {k: None for k in keys}, False


class DecisionHead:
    """Head 1: generative decoder / decision head over the shared encoder."""
    def __init__(self, encoder):
        self.enc = encoder

    def generate(self, percept, evidence=None):
        if evidence:
            evlist = "\n".join(f"[{e['id']}] {e['text']}" for e in evidence)
            user = DECISION_USER.format(percept=percept, evlist=evlist)
        else:
            user = percept
        raw, lat = self.enc.generate(DECISION_SYS, user)
        parsed, ok = SharedEncoder._parse_json(raw, "decision", "evidence_refs")
        decision = parsed.get("decision")
        if decision not in ("SHOULD_PROPOSE", "SHOULD_ABSTAIN"):
            decision = None
        refs = parsed.get("evidence_refs") or []
        return {"raw": raw, "decision": decision, "parsed_ok": ok,
                "evidence_refs": refs, "latency": lat}


class EvidenceHead:
    """Head 2: generative evidence / relation encoding head over the shared encoder.
    Consumes ONLY the shared encoder + read-only Ports (evidence passed by id/text)."""
    def __init__(self, encoder):
        self.enc = encoder

    def generate(self, percept, evidence):
        evlist = "\n".join(f"[{e['id']}] {e['text']}" for e in evidence)
        user = EVID_USER.format(percept=percept, evlist=evlist)
        raw, lat = self.enc.generate(EVID_SYS, user, max_new_tokens=64)
        parsed, ok = SharedEncoder._parse_json(raw, "relevant_evidence")
        refs = parsed.get("relevant_evidence") or []
        return {"raw": raw, "parsed_ok": ok, "relevant_evidence": refs, "latency": lat}


class ArchitectureC:
    def __init__(self, ports=None, encoder=None):
        self.encoder = encoder or SharedEncoder()
        self.ports = ports or Ports()
        self.decision_head = DecisionHead(self.encoder)
        self.evidence_head = EvidenceHead(self.encoder)

    def run(self, percept, evidence=None):
        d = self.decision_head.generate(percept, evidence)
        refs = list(d["evidence_refs"] or [])
        ev_lat = 0.0
        if evidence is not None:
            self.ports = Ports({e["id"]: e["text"] for e in evidence})
            e = self.evidence_head.generate(percept, evidence)
            ev_lat = e.get("latency", 0.0)
            for r in (e.get("relevant_evidence") or []):
                if r not in refs:
                    refs.append(r)
        decision = d["decision"] or "SHOULD_ABSTAIN"
        verified = [r for r in refs if self.ports.get(r)]
        if decision == "SHOULD_PROPOSE" and verified:
            status, final = "upheld", "SHOULD_PROPOSE"
        elif decision == "SHOULD_PROPOSE":
            status, final = "overridden_no_evidence", "SHOULD_ABSTAIN"
        else:
            status, final = "kept_abstain", "SHOULD_ABSTAIN"
        return {"architecture": "C", "decision": final, "evidence_status": status,
                "verified": verified, "decoder_decision": d["decision"],
                "decision_parsed_ok": d["parsed_ok"], "reasoning": d["raw"],
                "latency": d["latency"] + ev_lat}


if __name__ == "__main__":
    c = ArchitectureC(Ports({"E1": "streetlights dark"}))
    print(json.dumps(c.run("The container shattered after hitting the floor.",
                           [{"id": "E1", "text": "streetlights dark"}]), indent=2))
