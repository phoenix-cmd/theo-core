"""
Contract tests for Architecture C (shared encoder / two-head).
Run WITHOUT loading the 4-bit model (uses a FakeEncoder).
Proves: Ports read-only, heads cannot mutate symbolic state, no prohibited
signals in source, firewall intact, two distinct heads over one shared encoder,
and the deterministic gate logic.
"""
import os, sys, json, importlib.util, inspect

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("c_arch", os.path.join(HERE, "c_architecture.py"))
c_arch = importlib.util.module_from_spec(spec); spec.loader.exec_module(c_arch)


class FakeEncoder:
    """Stand-in for SharedEncoder; returns queued JSON, counts generate() calls."""
    def __init__(self):
        self.queue = []
        self.calls = 0
    def generate(self, sys_prompt, user_content, max_new_tokens=96):
        self.calls += 1
        if self.queue:
            return self.queue.pop(0), 0.0
        return '{"decision": "SHOULD_ABSTAIN"}', 0.0


class SpyPorts(c_arch.Ports):
    def __init__(self, entries=None):
        super().__init__(entries)
        self.get_calls = 0
        self.ids_calls = 0
        self.write_calls = 0
    def get(self, eid):
        self.get_calls += 1
        return super().get(eid)
    def ids(self):
        self.ids_calls += 1
        return super().ids()


def test_C1_ports_read_only():
    methods = [m for m in dir(c_arch.Ports) if not m.startswith("__")]
    for banned in ("set", "add", "update", "put", "delete", "remove", "mutate", "write", "insert"):
        assert banned not in methods, f"Ports exposes mutating method: {banned}"
    p = c_arch.Ports({"E1": "x"})
    n0 = len(p._entries)
    for _ in range(5):
        p.get("E1"); p.get("missing"); p.ids()
    assert len(p._entries) == n0, "Ports.get mutated internal entries"
    print("C1 PASS: Ports is strictly read-only (no mutating methods; get/ids do not mutate)")


def test_C2_heads_cannot_mutate_symbolic_state():
    enc = FakeEncoder()
    enc.queue = ['{"decision": "SHOULD_PROPOSE", "evidence_refs": ["E1"]}']
    sp = SpyPorts({"E1": "x"})
    c = c_arch.ArchitectureC(ports=sp, encoder=enc)
    out = c.run("some percept")  # evidence=None path keeps injected ports
    assert sp.write_calls == 0
    assert len(sp._entries) == 1, "symbolic state mutated"
    assert sp.get_calls >= 1, "heads did not read through Ports"
    print("C2 PASS: heads only READ via Ports; no symbolic-state mutation")


def test_C3_no_prohibited_signals_in_source():
    src = open(os.path.join(HERE, "c_architecture.py")).read()
    for banned in ("hidden_states", "subspace", "cos(H1)", "cos_h1", "aligned readout",
                  "backward(", "gradient", "peft", "LoraConfig", "embedding", "logits"):
        assert banned not in src, f"banned signal present in C source: {banned}"
    print("C3 PASS: no hidden-state/subspace/gradient/cos(H1)/peft signals in C source")


def test_C4_no_qwen_label_or_privileged_signals():
    src = open(os.path.join(HERE, "c_architecture.py")).read()
    for banned in ("Qwen-generated", "label_type", "adjudication.json", "GOLD_", "REJECT", "peft"):
        assert banned not in src, f"privileged/training signal referenced in C source: {banned}"
    print("C4 PASS: no Qwen-generated-label / privileged / training-signal references in C source")


def test_C5_firewall_imports():
    import ast
    tree = ast.parse(open(os.path.join(HERE, "c_architecture.py")).read())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names: imported.add(a.name)
        elif isinstance(n, ast.ImportFrom):
            if n.module: imported.add(n.module)
    assert "theo_core" not in " ".join(imported), "theo_core imported"
    assert "theo_providers" not in " ".join(imported), "theo_providers imported"
    print(f"C5 PASS: no theo_core/theo_providers import in C (imports={sorted(imported)})")


def test_C6_gate_logic():
    # Case 1: PROPOSE with verified evidence -> upheld
    enc = FakeEncoder()
    enc.queue = ['{"decision": "SHOULD_PROPOSE", "evidence_refs": ["E1"]}',
                 '{"relevant_evidence": ["E1"]}']
    c = c_arch.ArchitectureC(ports=c_arch.Ports(), encoder=enc)
    o1 = c.run("p", evidence=[{"id": "E1", "text": "x"}])
    assert o1["decision"] == "SHOULD_PROPOSE" and o1["evidence_status"] == "upheld"

    # Case 2: PROPOSE with NO verified evidence -> overridden
    enc = FakeEncoder()
    enc.queue = ['{"decision": "SHOULD_PROPOSE", "evidence_refs": ["E9"]}',
                 '{"relevant_evidence": []}']
    c = c_arch.ArchitectureC(ports=c_arch.Ports(), encoder=enc)
    o2 = c.run("p", evidence=[{"id": "E1", "text": "x"}])
    assert o2["decision"] == "SHOULD_ABSTAIN" and o2["evidence_status"] == "overridden_no_evidence"

    # Case 3: ABSTAIN preserved
    enc = FakeEncoder()
    enc.queue = ['{"decision": "SHOULD_ABSTAIN", "evidence_refs": []}',
                 '{"relevant_evidence": []}']
    c = c_arch.ArchitectureC(ports=c_arch.Ports(), encoder=enc)
    o3 = c.run("p", evidence=[{"id": "E1", "text": "x"}])
    assert o3["decision"] == "SHOULD_ABSTAIN" and o3["evidence_status"] == "kept_abstain"
    print("C6 PASS: gate upholds verified PROPOSE, overrides unsupported, preserves ABSTAIN")


def test_C7_two_heads_over_shared_encoder():
    enc = FakeEncoder()
    enc.queue = ['{"decision": "SHOULD_PROPOSE", "evidence_refs": ["E1"]}',
                 '{"relevant_evidence": ["E1"]}']
    c = c_arch.ArchitectureC(ports=c_arch.Ports(), encoder=enc)
    assert c.decision_head is not c.evidence_head, "heads must be distinct"
    assert c.decision_head.enc is c.evidence_head.enc, "heads must share one encoder"
    c.run("p", evidence=[{"id": "E1", "text": "x"}])
    assert enc.calls == 2, f"expected 2 generate calls (two heads), got {enc.calls}"
    print("C7 PASS: two distinct heads share one encoder (2 generate calls in populated mode)")


if __name__ == "__main__":
    test_C1_ports_read_only()
    test_C2_heads_cannot_mutate_symbolic_state()
    test_C3_no_prohibited_signals_in_source()
    test_C4_no_qwen_label_or_privileged_signals()
    test_C5_firewall_imports()
    test_C6_gate_logic()
    test_C7_two_heads_over_shared_encoder()
    print("\nALL C CONTRACT TESTS PASSED")
