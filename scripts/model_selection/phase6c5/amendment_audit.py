"""
6C.5 — Independent amendment audit (READ-ONLY; trains nothing).

Implements the corrected, platform-specific provenance/dependency hard-gates required by the
cloud amendment review. Mirrors the gates that cloud_6c5.py enforces at execution time.

Run:  python phase6c5/amendment_audit.py
Exit 0 = all scientific + dependency-baseline gates pass (local = historical baseline).
        Cloud-only fields are captured here as the HISTORICAL WINDOWS BASELINE and explicitly
        flagged so they are NEVER reused as the cloud artifact hash.

This is the audit to re-run after the platform-specific dependency correction.
"""
import os, sys, json, hashlib, subprocess, platform, importlib.util
import kaggle_6c5_common as C

ROOT = C.normalize_root(os.environ.get("THEO_ROOT", r"C:\Users\bs162\Desktop\THEO"))
HERE = ROOT / "theo-core" / "scripts" / "model_selection" / "phase6c5"
os.environ["THEO_ROOT"] = str(ROOT)
spec = importlib.util.spec_from_file_location("exp", str(HERE / "experiment_spec.py"))
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)

PINNED = {"python": "3.14.3", "torch": "2.13.0+cu126", "cuda_runtime": "12.6",
          "transformers": "5.8.0", "peft": "0.20.0", "bitsandbytes": "0.50.1"}
ADJ_SHA = "97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4"

FAILS = []
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if (detail and not cond) else ""))
    if not cond: FAILS.append(name)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def dir_hash(d):
    h = hashlib.sha256()
    for root, _, files in os.walk(d):
        for fn in sorted(files):
            fp = os.path.join(root, fn); h.update(fn.encode()); h.update(sha256_file(fp).encode())
    return h.hexdigest()

def _glibc():
    try:
        return os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, ValueError, OSError):
        return "n/a (non-Linux)"

# ---- scientific hash gates (reproducible on any host) ----
check("adjudication SHA 97bc6bc6", exp.adjudication_sha256() == ADJ_SHA, exp.adjudication_sha256())
check("fixture SHA f99ca025", exp.fixture_sha256() == exp.FIXTURE_SHA256, exp.fixture_sha256())
examples, data_sha = exp.build_examples()
train_ex, dev_ex = exp.split_examples(examples)
import json as _j, tempfile
# Reproduce bind_6c5.py's exact split-manifest serialization (indent=2, key order) so the
# split SHA matches the frozen binding (eda4cb66…).
split_doc = {"n_train": len(train_ex), "n_dev": len(dev_ex),
             "train": [e["case_id"] for e in train_ex], "dev": [e["case_id"] for e in dev_ex], "seed": exp.SEED}
_tp = os.path.join(str(HERE), "_split_audit_tmp.json")
_j.dump(split_doc, open(_tp, "w"), indent=2)
split_sha = sha256_file(_tp); os.remove(_tp)
check("split SHA eda4cb66", split_sha == "eda4cb6695b6651a7cef8bdee0a7a990f4f799a6bf9148269ee758d0704ee232", split_sha)
prompt_blob = "||".join([exp.NORM_SYS, exp.NORM_USER, exp.NORM_EVID_SYS, exp.NORM_EVID_USER])
check("prompt hash a29e12a2", hashlib.sha256(prompt_blob.encode()).hexdigest() ==
      "a29e12a2839b106d3a3632e1f353961d74182245a41602b445d06f71abb097ba")

# M1 revision
binding = _j.load(open(os.path.join(exp.BATCH2, "6c5_binding_manifest.json")))
for fname, fh in binding["base_model"]["file_hashes"].items():
    p = os.path.join(exp.M1_DIR, fname)
    check(f"M1 weight {fname}", os.path.exists(p) and sha256_file(p) == fh,
          sha256_file(p) if os.path.exists(p) else "MISSING")

# ---- dependency baseline hard-gate (must be reproduced on cloud) ----
py = sys.version.split()[0]
tf = __import__("torch").__version__; cuda = __import__("torch").version.cuda
tx = __import__("transformers").__version__; pf = __import__("peft").__version__
bn = __import__("bitsandbytes").__version__
print(f"[dep] python={py} torch={tf} cuda={cuda} transformers={tx} peft={pf} bitsandbytes={bn}")
check("python==3.14.3", py == PINNED["python"])
check("torch==2.13.0+cu126", tf == PINNED["torch"])
check("cuda_runtime==12.6", cuda == PINNED["cuda_runtime"])
check("transformers==5.8.0", tx == PINNED["transformers"])
check("peft==0.20.0", pf == PINNED["peft"])
check("bitsandbytes==0.50.1", bn == PINNED["bitsandbytes"])

# ---- cloud-only fields: captured as HISTORICAL WINDOWS BASELINE (never reused as cloud hash) ----
import torch
gpus = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpus.append(f"{p.name} cc{p.major}.{p.minor}")
else:
    gpus = ["NO_CUDA"]
bnb_loc = os.path.dirname(__import__("bitsandbytes").__file__)
bnb_win_sha = dir_hash(bnb_loc)
print(f"[baseline] OS={platform.system()} release={platform.release()} glibc={_glibc()} gpus={gpus}")
print(f"[baseline] LOCAL Windows bitsandbytes package SHA = {bnb_win_sha}  <-- HISTORICAL EVIDENCE ONLY; do NOT use as cloud artifact hash")
print(f"[baseline] Cloud runner must bind its own Linux bitsandbytes wheel SHA at execution time.")

print("\n=== AMENDMENT AUDIT RESULT ===")
if FAILS:
    print(f"FAIL: {len(FAILS)} gate(s) failed -> HALT"); print(FAILS); sys.exit(1)
print("ALL GATES PASS. Local stack = historical Windows baseline; cloud must reproduce pinned versions or HALT.")
sys.exit(0)
