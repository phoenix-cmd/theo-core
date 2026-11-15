"""
6C.5 — Cloud execution script (Kaggle / external GPU). PREPARATION ONLY — DO NOT RUN
until the cloud environment amendment is reviewed and a fresh execution authorization binds
the cloud runtime assumptions.

When authorized (env vars THEO_ROOT=read-only inputs, THEO_OUT_B/THEO_OUT_C=writable adapter
dirs):
  1. Hard-gates the dependency baseline (Python/torch/CUDA/transformers/peft/bitsandbytes).
     If the cloud cannot reproduce the required stack -> HALT (never silently substitute).
  2. Verifies ALL scientific hashes against 6c5_binding_manifest.json + frozen adjudication
     SHA -> HALT on mismatch.
  3. Re-runs the 16 Phase-2 checks as the final gate (subprocess, clean exit).
  4. Captures complete, platform-specific provenance (GPU model/count/compute-cap, CUDA,
     NVIDIA driver, OS/glibc, dependency lock hash, and the ACTUAL Linux bitsandbytes
     wheel/package SHA bound at runtime).
  5. Trains B once, then C once (identical frozen spec via train_6c5.train_one).
  6. Writes 6c5_cloud_execution_manifest.json.

Latency gates are NOT decided here — measured locally on GTX 1650 after retrieval (amendment §7).
The local Windows bitsandbytes wheel/hash is HISTORICAL EVIDENCE ONLY; the cloud binds its own
Linux bitsandbytes wheel SHA and never reuses the Windows one.
"""
import os, sys, json, hashlib, subprocess, datetime, platform
import torch

# ---- REQUIRED dependency baseline (must be reproduced exactly on the cloud) ----
PINNED = {
    "python": "3.14.3",
    "torch": "2.13.0+cu126",
    "cuda_runtime": "12.6",
    "transformers": "5.8.0",
    "peft": "0.20.0",
    "bitsandbytes": "0.50.1",
}
ADJ_SHA   = "97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4"
M1_REV    = "7ae55760"

THEO_ROOT = os.environ.get("THEO_ROOT")
OUT_B = os.environ.get("THEO_OUT_B")
OUT_C = os.environ.get("THEO_OUT_C")
if not THEO_ROOT or not OUT_B or not OUT_C:
    sys.exit("ERROR: set THEO_ROOT, THEO_OUT_B, THEO_OUT_C env vars")

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("exp", os.path.join(HERE, "experiment_spec.py"))
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def dir_hash(d):
    h = hashlib.sha256()
    for root, _, files in os.walk(d):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            h.update(fn.encode()); h.update(sha256_file(fp).encode())
    return h.hexdigest()

def glibc_version():
    try:
        return os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, ValueError, OSError):
        return "n/a"

def nvidia_driver():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else "n/a"
    except Exception:
        return "n/a"

def dependency_lock_sha():
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60)
        return hashlib.sha256(out.stdout.encode()).hexdigest() if out.returncode == 0 else "n/a"
    except Exception:
        return "n/a"

# ---- 0. hard-gate the dependency baseline ----
py = sys.version.split()[0]
tf = torch.__version__
cuda = torch.version.cuda
tx = __import__("transformers").__version__
pf = __import__("peft").__version__
bn = __import__("bitsandbytes").__version__
dep_ok = (py == PINNED["python"] and tf == PINNED["torch"] and cuda == PINNED["cuda_runtime"]
          and tx == PINNED["transformers"] and pf == PINNED["peft"] and bn == PINNED["bitsandbytes"])
print(f"[dep] python={py} torch={tf} cuda={cuda} transformers={tx} peft={pf} bitsandbytes={bn}")
if not dep_ok:
    sys.exit(f"HALT: dependency baseline not reproduced. Required {PINNED}; cloud resolved a different stack.")

# ---- 1. verify the ACTUAL bitsandbytes package/wheel SHA on THIS (cloud) host ----
bnb_loc = os.path.dirname(__import__("bitsandbytes").__file__)
bnb_wheel_sha = dir_hash(bnb_loc)
print(f"[dep] bitsandbytes package SHA (Linux cloud artifact, bound now) = {bnb_wheel_sha}")

# ---- 2. scientific hash verification against binding ----
BINDING = os.path.join(THEO_ROOT, "theo-data/datasets/theo_slm_v0_review/6b3_batch2/6c5_binding_manifest.json")
if not os.path.exists(BINDING):
    sys.exit("ERROR: 6c5_binding_manifest.json not found at " + BINDING)
binding = json.load(open(BINDING))
binding_sha = sha256_file(BINDING)

examples, data_sha = exp.build_examples()
ok = True
for fname, fh in binding["base_model"]["file_hashes"].items():
    p = os.path.join(exp.M1_DIR, fname)
    if not os.path.exists(p) or sha256_file(p) != fh:
        print(f"  MISMATCH M1 {fname}: expected {fh}, got {sha256_file(p) if os.path.exists(p) else 'MISSING'}"); ok = False
if data_sha != binding["dataset_sha256"]:
    print(f"  MISMATCH dataset: expected {binding['dataset_sha256']}, got {data_sha}"); ok = False
if exp.fixture_sha256() != binding["fixture_sha256_expected"]:
    print(f"  MISMATCH fixture: expected {binding['fixture_sha256_expected']}, got {exp.fixture_sha256()}"); ok = False
if exp.adjudication_sha256() != ADJ_SHA:
    print(f"  MISMATCH adjudication: expected {ADJ_SHA}, got {exp.adjudication_sha256()}"); ok = False
train_ex, dev_ex = exp.split_examples(examples)
split_doc = {"train": [e["case_id"] for e in train_ex], "dev": [e["case_id"] for e in dev_ex], "seed": exp.SEED}
tp = os.path.join(os.path.dirname(OUT_B), "6c5_split_manifest.json")
json.dump(split_doc, open(tp, "w"), indent=2)
if sha256_file(tp) != binding.get("split_manifest_sha256"):
    print(f"  MISMATCH split: expected {binding.get('split_manifest_sha256')}, got {sha256_file(tp)}"); ok = False
if not ok:
    sys.exit("HALT: scientific hash verification failed — do not train")
print(f"[verify] all scientific hashes match binding (binding_sha={binding_sha[:16]}...)")

# ---- 3. re-run 16 Phase-2 checks as final gate (subprocess, inherits env) ----
print("[gate] re-running Phase-2 audit as final gate...")
r = subprocess.run([sys.executable, os.path.join(HERE, "phase2_audit.py")], env=os.environ, cwd=HERE)
if r.returncode != 0:
    sys.exit("HALT: Phase-2 audit failed before training")
print("[gate] Phase-2 audit PASS")

# ---- 4. provenance capture (platform-specific) ----
gpus = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpus.append({"index": i, "name": p.name,
                     "compute_capability": f"{p.major}.{p.minor}",
                     "total_memory_mb": p.total_memory // (1024*1024)})
else:
    gpus = ["NO_CUDA"]

provenance = {
    "model_revision": M1_REV,
    "python_version": py,
    "torch": tf, "cuda_runtime": cuda,
    "transformers": tx, "peft": pf,
    "bitsandbytes_version": bn,
    "bitsandbytes_wheel_sha256": bnb_wheel_sha,   # Linux cloud artifact, bound at runtime
    "nvidia_driver": nvidia_driver(),
    "gpu_topology": gpus,
    "os": platform.system(), "os_release": platform.release(), "glibc": glibc_version(),
    "dependency_lock_sha256": dependency_lock_sha(),
    "binding_manifest_sha256": binding_sha,
    "windows_bitsandbytes_note": "local Windows bnb wheel/hash is HISTORICAL EVIDENCE ONLY; not used as cloud artifact hash",
}

# ---- 5/6. training (B then C) ----
import train_6c5
artifacts = {}
for arch in ("B", "C"):
    t0 = datetime.datetime.now()
    train_6c5.train_one(arch)
    dt = (datetime.datetime.now() - t0).total_seconds()
    adir = exp.OUT_C if arch == "C" else exp.OUT_B
    artifacts[arch] = {"adapter_dir": adir,
                       "adapter_dir_sha256": dir_hash(adir) if os.path.isdir(adir) else "MISSING",
                       "wallclock_s": dt}

manifest = {
    "experiment": "6C.5 cloud training (Kaggle/equivalent)",
    "executed_at": datetime.datetime.now().isoformat(),
    "dependency_baseline_pinned": PINNED,
    "provenance": provenance,
    "binding_reference": {"adjudication_sha256": ADJ_SHA,
                          "dataset_sha256": binding["dataset_sha256"],
                          "fixture_sha256": binding["fixture_sha256_expected"],
                          "split_manifest_sha256": binding.get("split_manifest_sha256"),
                          "prompt_hash": binding["prompt_hash"],
                          "aux_loss_weight": binding["aux_loss_weight"]},
    "artifacts": artifacts,
    "note": "Latency gates are measured locally on GTX 1650 after retrieval; cloud speed does not redefine the <=0.5s criterion. Windows bnb hash is historical only.",
}
out_path = os.path.join(os.path.dirname(OUT_B), "6c5_cloud_execution_manifest.json")
json.dump(manifest, open(out_path, "w"), indent=2)
print(f"[done] cloud manifest -> {out_path}")
print(f"[done] adapter hashes: B={artifacts['B']['adapter_dir_sha256'][:16]} C={artifacts['C']['adapter_dir_sha256'][:16]}")
print(f"[done] bnb Linux wheel SHA: {bnb_wheel_sha}")
