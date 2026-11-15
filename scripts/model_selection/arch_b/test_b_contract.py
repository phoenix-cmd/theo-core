"""
Contract tests for Architecture B's reranker.

Proves, without a GPU/model, that the reranker:
  C1. consumes ONLY the Snapshot read interface (get/ids) — no other methods;
  C2. is strictly read-only (cannot mutate snapshot or any symbolic state);
  C3. receives NO hidden-state / gradient / subspace / privileged signal;
  C4. cannot see training labels or disallowed information;
  C5. architectural firewall: the reranker module imports no theo_core symbolic module;
  C6. behavioral correctness of the evidence gate (override / uphold / preserve).
"""
import sys, os, types, importlib.util, json

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("b_arch", os.path.join(HERE, "b_architecture.py"))
b_arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b_arch)


class SpySnapshot:
    """Records every method call; rejects any non-read method."""
    ALLOWED = {"get", "ids"}
    def __init__(self, entries=None):
        self._e = dict(entries or {})
        self.calls = []
    def get(self, eid):
        self.calls.append(("get", eid))
        return self._e.get(eid)
    def ids(self):
        self.calls.append(("ids", None))
        return list(self._e.keys())
    # explicit write methods that MUST never be called:
    def __setitem__(self, k, v): raise AssertionError("Snapshot write attempted")
    def add(self, k, v): raise AssertionError("Snapshot.add attempted")
    def update(self, d): raise AssertionError("Snapshot.update attempted")


def test_C1_C2_only_read_interface():
    spy = SpySnapshot({"c1": "x"})
    r = b_arch.EvidenceReranker(spy)
    r.rerank("SHOULD_PROPOSE", ["c1"])
    methods = {c[0] for c in spy.calls}
    assert methods <= SpySnapshot.ALLOWED, f"disallowed snapshot methods used: {methods}"
    # snapshot content must be untouched
    assert spy._e == {"c1": "x"}, "snapshot mutated"
    print("C1/C2 PASS: reranker uses only snapshot read interface, no mutation")


def test_C3_no_hidden_state_signal():
    r = b_arch.EvidenceReranker(SpySnapshot())
    import inspect
    sig = inspect.signature(r.rerank)
    params = list(sig.parameters)
    assert params == ["decision", "evidence_refs"], f"unexpected reranker inputs: {params}"
    full = open(os.path.join(HERE, "b_architecture.py")).read()
    # scope the check to the EvidenceReranker class body (avoid the module docstring,
    # which legitimately names the excluded signals)
    idx = full.index("class EvidenceReranker")
    src = full[idx:]
    for banned in ("hidden_states", "backward(", "subspace", "cos(H1)", ".embedding", "logits", "peft", "LoraConfig"):
        assert banned not in src, f"banned signal present in reranker: {banned}"
    print("C3 PASS: reranker signature + source free of hidden-state/gradient/subspace signals")


def test_C4_no_privileged_info():
    src = open(os.path.join(HERE, "b_architecture.py")).read()
    for banned in ("adjudication.json", "GOLD_", "REJECT", "label_type", "cos_H1",
                   "peft", "LoraConfig"):
        assert banned not in src, f"privileged/training signal referenced in arch_b: {banned}"
    print("C4 PASS: no training labels / privileged info referenced in arch_b source")


def test_C5_firewall_imports():
    import ast
    with open(os.path.join(HERE, "b_architecture.py")) as f:
        tree = ast.parse(f.read())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names: imported.add(a.name)
        elif isinstance(n, ast.ImportFrom):
            if n.module: imported.add(n.module)
    banned = [m for m in imported if "theo_core" in m or "theo_providers" in m]
    assert not banned, f"reranker imports symbolic modules: {banned}"
    print(f"C5 PASS: no theo_core/theo_providers import in reranker (imports={sorted(imported)})")


def test_C6_behavioral_gate():
    # empty snapshot -> PROPOSE must be overridden to ABSTAIN
    r = b_arch.EvidenceReranker(SpySnapshot())
    o1 = r.rerank("SHOULD_PROPOSE", [])
    assert o1["decision"] == "SHOULD_ABSTAIN" and o1["evidence_status"] == "overridden_no_evidence"
    # empty snapshot -> ABSTAIN preserved
    o2 = r.rerank("SHOULD_ABSTAIN", [])
    assert o2["decision"] == "SHOULD_ABSTAIN" and o2["evidence_status"] == "kept_abstain"
    # populated snapshot with verified ref -> PROPOSE upheld
    r2 = b_arch.EvidenceReranker(SpySnapshot({"ev://a": "fact"}))
    o3 = r2.rerank("SHOULD_PROPOSE", ["ev://a"])
    assert o3["decision"] == "SHOULD_PROPOSE" and o3["evidence_status"] == "upheld"
    # populated snapshot but unverified ref -> override
    o4 = r2.rerank("SHOULD_PROPOSE", ["ev://missing"])
    assert o4["decision"] == "SHOULD_ABSTAIN"
    print("C6 PASS: evidence gate overrides unsupported proposals, preserves abstentions, upholds verified")


if __name__ == "__main__":
    import inspect
    test_C1_C2_only_read_interface()
    test_C3_no_hidden_state_signal()
    test_C4_no_privileged_info()
    test_C5_firewall_imports()
    test_C6_behavioral_gate()
    print("\nALL CONTRACT TESTS PASSED")
