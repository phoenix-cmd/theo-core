"""
6C.5 cloud orchestrator — shared constants and helpers (orchestration only).

This module MUST NOT modify architecture, dataset, supervision, gates, model revisions,
seed, hyperparameters, or evaluation logic. It only knows paths, the frozen expected
hashes, and the Kaggle CLI invocation shape.
"""
import os, sys, json, hashlib, subprocess, re
from pathlib import Path


def normalize_root(raw):
    """Normalize a repository root into exactly one native pathlib.Path.

    Accepts Windows (C:\\...), Git-Bash / MSYS (/c/..., /cygdrive/c/...), or a
    relative path, and renders it in the CURRENT interpreter's native form
    (backslashes on Windows Python, forward slashes under MSYS/Git-Bash Python).
    Resolves once, verifies it is the actual THEO repository root, and never
    prepends __file__, the script directory, or another root to an absolute path.
    """
    s = str(raw).strip()
    if not s:
        s = r"C:\Users\bs162\Desktop\THEO"
    # Reduce any drive form (/c/, /cygdrive/c/, C:\) to (drive, rest), then render
    # in the native form for the running interpreter.
    m = (re.match(r"^/cygdrive/([a-zA-Z])/(.*)$", s)
         or re.match(r"^/([a-zA-Z])/(.*)$", s)
         or re.match(r"^([a-zA-Z]):[\\/]?(.*)$", s))
    if m:
        drive, rest = m.group(1).upper(), m.group(2)
        if os.name == "nt":
            s = f"{drive}:\\{rest.replace('/', '\\')}"
        else:
            s = f"/{drive.lower()}/{rest}"
    p = Path(s)
    if not p.is_absolute():
        p = Path.cwd() / p
    try:
        p = p.resolve()
    except Exception:
        pass
    norm = str(p).replace("\\", "/")
    # Hard invariant: a single drive root, no mixed git-bash/windows forms.
    if len(re.findall(r"[A-Za-z]:", norm)) > 1:
        halt(f"normalize_root: multiple drive roots detected in '{norm}'")
    if "/c/" in norm and "c:" in norm.lower():
        halt(f"normalize_root: mixed git-bash and windows paths in '{norm}'")
    if not ((p / "theo-core").exists() or (p / "theo-data").exists()):
        halt(f"normalize_root: {p} is not a THEO repository root (missing theo-core/theo-data)")
    return p


# ---- THEO root (overridable for tests / cloud); normalized exactly once ----
ROOT = normalize_root(os.environ.get("THEO_ROOT", r"C:\Users\bs162\Desktop\THEO"))

# ---- frozen expected hashes (from 6c5_binding_manifest.json / L5 authorization) ----
EXPECTED = {
    "adjudication_sha256": "97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4",
    "fixture_sha256": "f99ca025eaaa3073e296e7cb734581f0871315553b5c145bb7f2fc27a4ac6457",
    "split_manifest_sha256": "eda4cb6695b6651a7cef8bdee0a7a990f4f799a6bf9148269ee758d0704ee232",
    "prompt_hash": "a29e12a2839b106d3a3632e1f353961d74182245a41602b445d06f71abb097ba",
    "m1_revision": "7ae55760",
    "bundle_sha256": "454cfc42f70b1457cf8837f8bd55c392e443f572640d6d26114d736e14b6ed31",
    "cloud_6c5_sha256": "c03f5d0eef95187225c5126621ceea414e719b1f1a58cc4f50d1751792bb4071",
    "aux_loss_weight": 1.0,
    "seed": 20260826,
}

# ---- allowlisted input bundle (relative to ROOT); nothing else is packaged ----
ALLOWLIST = [
    "theo-data/models/6c_acquired/M1_Qwen2.5-0.5B-Instruct/model.safetensors",
    "theo-data/models/6c_acquired/M1_Qwen2.5-0.5B-Instruct/config.json",
    "theo-data/datasets/theo_slm_v0_review/adjudication.json",
    "theo-data/datasets/theo_slm_v0_review/6b3_batch2/6c4_mapping_manifest.json",
    "theo-data/datasets/theo_slm_v0_review/6b3_batch2/candidate_records_batch2_remediated.json",
    "theo-data/datasets/theo_slm_v0_review/6b3_batch2/6c5_binding_manifest.json",
    "theo-data/datasets/theo_slm_v0_review/6b3_batch2/6c5_split_manifest.json",
    "theo-data/datasets/theo_slm_v0_deduplicated/candidate_records.json",
    "theo-data/datasets/theo_slm_v0_review/6b3_batch1/candidate_records_batch1.json",
    "theo-data/datasets/theo_slm_v0_artifacts/phase-6e3/populated-evidence-corpus.json",
    "theo-core/scripts/model_selection/phase6c5/experiment_spec.py",
    "theo-core/scripts/model_selection/phase6c5/train_6c5.py",
    "theo-core/scripts/model_selection/phase6c5/cloud_6c5.py",
    "theo-core/scripts/model_selection/phase6c5/phase2_audit.py",
    "theo-core/scripts/model_selection/phase6c5/amendment_audit.py",
    "theo-core/scripts/model_selection/phase6c5/bind_6c5.py",
    "theo-core/scripts/model_selection/phase6c5/requirements.txt",
]

# Patterns that must NEVER enter the bundle (6E.20.1 checkpoints + unrelated clutter).
EXCLUDE_PATTERNS = ["6e20", "6E20", "6e.20", "checkpoint", ".git", "__pycache__",
                    "node_modules", ".ipynb_checkpoints", "6c5_cloud_execution_manifest.json"]

# ---- Kaggle identifiers (override via env) ----
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "<username>")
KAGGLE_DATASET_SLUG = os.environ.get("KAGGLE_DATASET_SLUG", "theo-6c5-inputs")
KAGGLE_KERNEL_SLUG = os.environ.get("KAGGLE_KERNEL_SLUG", "theo-6c5-bc-training")
# Owner of the input dataset (defaults to the pushing username; override if different).
# Used to bind the kernel's dataset_sources to the exact authorized slug (provider #5 fix).
KAGGLE_DATASET_OWNER = os.environ.get("KAGGLE_DATASET_OWNER", KAGGLE_USERNAME)
L5_AUTH_DOC = ROOT / "theo-core/docs/research/theo-slm-phase-6c5-l5-cloud-execution-authorization.md"

BINDING = ROOT / "theo-data/datasets/theo_slm_v0_review/6b3_batch2/6c5_binding_manifest.json"


def halt(msg):
    print(f"[HALT] {msg}")
    sys.exit(1)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dir_sha256(d):
    h = hashlib.sha256()
    for root, _, files in os.walk(d):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            h.update(fn.encode()); h.update(sha256_file(fp).encode())
    return h.hexdigest()


def normalize_rel(path):
    return path.replace("\\", "/")


def print_mount_diagnostics(input_root="/kaggle/input", max_depth=3):
    """Scan and print the structure of input_root up to max_depth
    with file/dir type annotations BEFORE any resolution attempt.
    """
    input_root_str = str(input_root).replace("\\", "/").rstrip("/")
    print("=" * 60)
    print(f"[mount_diagnostics] Scanning input root: {input_root_str}")
    if not os.path.exists(input_root_str):
        print(f"[mount_diagnostics] WARNING: {input_root_str} does not exist!")
        print("=" * 60)
        return
    if not os.path.isdir(input_root_str):
        print(f"[mount_diagnostics] WARNING: {input_root_str} is not a directory!")
        print("=" * 60)
        return

    try:
        immediate = sorted(os.listdir(input_root_str))
        print(f"[mount_diagnostics] Immediate children of {input_root_str}: {immediate}")
    except Exception as e:
        print(f"[mount_diagnostics] Error listing immediate children of {input_root_str}: {e}")

    print(f"[mount_diagnostics] Recursive tree (depth <= {max_depth}):")
    base_parts_len = len(Path(input_root_str).resolve().parts) if os.path.exists(input_root_str) else 0
    for root_dir, dirs, files in os.walk(input_root_str):
        curr_parts_len = len(Path(root_dir).resolve().parts)
        depth = curr_parts_len - base_parts_len
        if depth > max_depth:
            dirs.clear()
            continue
        rel = os.path.relpath(root_dir, input_root_str).replace("\\", "/")
        indent = "  " * depth
        if rel != ".":
            print(f"[mount_diagnostics] {indent}[DIR]  {rel}/")
        for d in sorted(dirs):
            if depth + 1 <= max_depth:
                sub_rel = os.path.normpath(os.path.join(rel, d)).replace("\\", "/")
                print(f"[mount_diagnostics] {indent}  [DIR]  {sub_rel}/")
        for f in sorted(files):
            f_rel = os.path.normpath(os.path.join(rel, f)).replace("\\", "/")
            f_path = os.path.join(root_dir, f)
            try:
                sz = os.path.getsize(f_path)
            except Exception:
                sz = "n/a"
            print(f"[mount_diagnostics] {indent}  [FILE] {f_rel} ({sz} bytes)")
    print("=" * 60)


def locate_dataset_mount(input_root="/kaggle/input", dataset_slug="theo-6c5-inputs", dataset_owner="Anakinistic", max_depth=4):
    """Locate the unambiguous dataset mount directory matching dataset_slug and dataset_owner.
    Searches input_root tree up to max_depth (e.g. /kaggle/input/datasets/anakinistic/theo-6c5-inputs
    or /kaggle/input/theo-6c5-inputs). Requires exactly one matching mount folder.
    """
    input_root_str = str(input_root).replace("\\", "/").rstrip("/")
    if not os.path.exists(input_root_str):
        halt(f"locate_dataset_mount: input root does not exist: {input_root_str}")

    norm_slug = str(dataset_slug).lower().replace("_", "-").strip()
    norm_owner = str(dataset_owner).lower().replace("_", "-").strip() if dataset_owner else ""

    # Check if input_root itself is the dataset mount
    base_name = os.path.basename(input_root_str).lower().replace("_", "-")
    if base_name == norm_slug or (norm_owner and base_name == f"{norm_owner}-{norm_slug}"):
        return input_root_str

    candidates = []
    base_parts_len = len(Path(input_root_str).resolve().parts) if os.path.exists(input_root_str) else 0

    for root_dir, dirs, _ in os.walk(input_root_str):
        curr_parts_len = len(Path(root_dir).resolve().parts)
        depth = curr_parts_len - base_parts_len
        if depth > max_depth:
            dirs.clear()
            continue

        dir_name = os.path.basename(root_dir).lower().replace("_", "-")
        if dir_name == norm_slug or (norm_owner and dir_name == f"{norm_owner}-{norm_slug}"):
            parent_name = os.path.basename(os.path.dirname(root_dir)).lower().replace("_", "-")
            norm_path = root_dir.replace("\\", "/")
            is_owner_match = bool(norm_owner and (parent_name == norm_owner or dir_name.startswith(f"{norm_owner}-")))
            candidates.append((norm_path, is_owner_match))

    # Disambiguate if owner match is present
    owner_matched = [c[0] for c in candidates if c[1]]
    if owner_matched:
        final_candidates = owner_matched
    else:
        final_candidates = [c[0] for c in candidates]

    if len(final_candidates) == 0:
        halt(f"locate_dataset_mount: no mount matching '{dataset_slug}' (owner={dataset_owner}) found under {input_root_str} (depth <= {max_depth})")
    if len(final_candidates) > 1:
        halt(f"locate_dataset_mount: ambiguous dataset mounts for '{dataset_slug}': {final_candidates}")

    return final_candidates[0]


def resolve_cloud_bundle_root(input_root="/kaggle/input", dataset_slug="theo-6c5-inputs",
                               dataset_owner="Anakinistic", expected_bundle_sha=EXPECTED["bundle_sha256"], max_depth=3):
    """Discover and verify the directory containing bundled theo-core/, theo-data/,
    and bundle_manifest.json (with matching bundle_sha256) inside the matched dataset mount.

    Strategy:
    1. Print mount diagnostics unconditionally (forensic visibility on any outcome).
    2. Identify the single matching dataset mount under input_root (depth <= 4).
    3. Search ONLY inside that dataset mount up to max_depth (<= 3) for directories
       containing theo-core/, theo-data/, and valid bundle_manifest.json.
    4. Fail-closed:
       - 0 candidates  -> HALT
       - >1 candidates -> HALT
       - exactly 1     -> return candidate POSIX path
    """
    # 1. Unconditionally emit diagnostics first
    print_mount_diagnostics(input_root=input_root, max_depth=max_depth + 1)

    # 2. Locate the unambiguous dataset mount
    dataset_mount = locate_dataset_mount(input_root=input_root, dataset_slug=dataset_slug, dataset_owner=dataset_owner, max_depth=4)
    print(f"[resolver] Identified dataset mount root: {dataset_mount}")

    # 3. Search ONLY inside dataset_mount up to max_depth
    candidates = []
    base_parts_len = len(Path(dataset_mount).resolve().parts) if os.path.exists(dataset_mount) else 0

    for root_dir, dirs, _ in os.walk(dataset_mount):
        curr_parts_len = len(Path(root_dir).resolve().parts)
        depth = curr_parts_len - base_parts_len
        if depth > max_depth:
            dirs.clear()
            continue

        has_tc = os.path.isdir(os.path.join(root_dir, "theo-core"))
        has_td = os.path.isdir(os.path.join(root_dir, "theo-data"))
        manifest_path = os.path.join(root_dir, "bundle_manifest.json")
        has_manifest = os.path.isfile(manifest_path)

        if has_tc and has_td:
            if not has_manifest:
                halt(f"resolve_cloud_bundle_root: candidate {root_dir} has theo-core/ + theo-data/ but missing bundle_manifest.json")
            try:
                mf = json.load(open(manifest_path))
                actual_b_sha = mf.get("bundle_sha256")
            except Exception as e:
                halt(f"resolve_cloud_bundle_root: cannot parse {manifest_path}: {e}")
            if expected_bundle_sha and actual_b_sha != expected_bundle_sha:
                halt(f"resolve_cloud_bundle_root: bundle_manifest.json SHA mismatch: expected {expected_bundle_sha}, got {actual_b_sha}")
            norm = root_dir.replace("\\", "/")
            if norm not in candidates:
                candidates.append(norm)

    if len(candidates) == 0:
        halt(f"resolve_cloud_bundle_root: no valid bundle root (theo-core + theo-data + valid bundle_manifest.json) found inside dataset mount {dataset_mount} (depth <= {max_depth})")
    if len(candidates) > 1:
        halt(f"resolve_cloud_bundle_root: ambiguous -- {len(candidates)} candidates under dataset mount {dataset_mount}: {candidates}")

    resolved = candidates[0]
    print(f"[resolver] Resolved bundle root: {resolved}")
    return resolved


def scan_exclusions(root):
    """Return list of files under root matching EXCLUDE_PATTERNS (proof of exclusion)."""
    root = normalize_root(root)
    hits = []
    for dp, _, fns in os.walk(str(root)):
        for fn in fns:
            p = normalize_rel(os.path.join(dp, fn))
            if any(pat.lower() in p.lower() for pat in EXCLUDE_PATTERNS):
                hits.append(p)
    return sorted(hits)


def build_allowlist_paths(root):
    root = normalize_root(root)
    paths = []
    for rel in ALLOWLIST:
        src = root / rel
        if not os.path.exists(src):
            halt(f"allowlisted input missing: {rel}")
        paths.append((src, normalize_rel(rel)))
    return paths


def validate_binding_hashes(root):
    """Recompute the frozen scientific hashes from the real tree and compare to EXPECTED.
    Mirrors amendment_audit.py. Returns (ok, details)."""
    canonical = normalize_root(root)
    expected_root = normalize_root(r"C:\Users\bs162\Desktop\THEO")
    if canonical != expected_root:
        halt(f"validate_binding_hashes: root {canonical} != canonical THEO {expected_root}")
    spec_path = (canonical / "theo-core" / "scripts" / "model_selection"
                 / "phase6c5" / "experiment_spec.py")
    sp_norm = str(spec_path).replace("\\", "/")
    if not sp_norm.startswith(str(canonical).replace("\\", "/")):
        halt(f"validate_binding_hashes: spec path not under canonical root: {spec_path}")
    if "/c/" in sp_norm and "c:" in sp_norm.lower():
        halt(f"validate_binding_hashes: mixed git-bash/windows path in spec: {spec_path}")
    print(f"[validate] canonical_root = {canonical}")
    print(f"[validate] spec_path      = {spec_path}")
    import importlib.util, json as _j, os as _os
    os.environ["THEO_ROOT"] = str(canonical)
    try:
        spec = importlib.util.spec_from_file_location("exp", str(spec_path))
        exp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp)
    except Exception as e:
        halt(f"cannot load experiment_spec.py from root {canonical}: {e}")
    details, ok = {}, True

    adj = exp.adjudication_sha256()
    details["adjudication_sha256"] = adj
    if adj != EXPECTED["adjudication_sha256"]: ok = False

    fix = exp.fixture_sha256()
    details["fixture_sha256"] = fix
    if fix != EXPECTED["fixture_sha256"]: ok = False

    examples, _ = exp.build_examples()
    train_ex, dev_ex = exp.split_examples(examples)
    split_doc = {"n_train": len(train_ex), "n_dev": len(dev_ex),
                 "train": [e["case_id"] for e in train_ex],
                 "dev": [e["case_id"] for e in dev_ex], "seed": exp.SEED}
    tp = _os.path.join(_os.path.dirname(str(spec_path)), "_split_tmp.json")
    _j.dump(split_doc, open(tp, "w"), indent=2)
    sp = sha256_file(tp); _os.remove(tp)
    details["split_manifest_sha256"] = sp
    if sp != EXPECTED["split_manifest_sha256"]: ok = False

    pb = "||".join([exp.NORM_SYS, exp.NORM_USER, exp.NORM_EVID_SYS, exp.NORM_EVID_USER])
    ph = hashlib.sha256(pb.encode()).hexdigest()
    details["prompt_hash"] = ph
    if ph != EXPECTED["prompt_hash"]: ok = False

    bh = _j.load(open(canonical / "theo-data/datasets/theo_slm_v0_review/6b3_batch2/6c5_binding_manifest.json"))
    for fname, fh in bh["base_model"]["file_hashes"].items():
        p = os.path.join(exp.M1_DIR, fname)
        cur = sha256_file(p) if os.path.exists(p) else "MISSING"
        details[f"m1_{fname}"] = cur
        if cur != fh: ok = False
    return ok, details


def check_l5_authorization():
    """L5 authorization is active iff the L5 doc exists AND amendment_audit.py passes (exit 0)."""
    if not os.path.exists(L5_AUTH_DOC):
        return False, "L5 authorization doc missing"
    r = subprocess.run([sys.executable,
                        str(ROOT / "theo-core/scripts/model_selection/phase6c5/amendment_audit.py")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False, f"amendment_audit.py failed (rc={r.returncode})"
    return True, "L5 active (doc present + amendment_audit PASS)"


def _kaggle_cmd_prefix():
    """Resolve the Kaggle CLI invocation prefix for the running interpreter.

    Prefers '<sys.executable> -m kaggle' when the kaggle module is importable.
    Falls back to a directly-resolved kaggle.exe (under sys.prefix/Scripts, or
    located via shutil.which); it is NOT required to be on PATH. Returns the
    command prefix list, or None if no kaggle CLI can be resolved.
    """
    try:
        import importlib.util
        if importlib.util.find_spec("kaggle") is not None:
            return [sys.executable, "-m", "kaggle"]
    except Exception:
        pass
    # fallback: directly-resolved kaggle.exe (not required to be on PATH)
    import shutil
    candidate = os.path.join(sys.prefix, "Scripts", "kaggle.exe")
    if os.path.exists(candidate):
        return [candidate]
    found = shutil.which("kaggle")
    if found:
        return [found]
    return None


def resolve_kaggle_command(sub_args):
    """Return the full Kaggle CLI command ([exe, ...] + sub_args), or None if unavailable."""
    prefix = _kaggle_cmd_prefix()
    if prefix is None:
        return None
    return prefix + list(sub_args)


def run_kaggle_cmd(args, dry_run, halt_on_fail=True):
    """Run a kaggle CLI command. `args` should start with the logical 'kaggle' token.
    In dry-run, print only (no subprocess). HALT on failure unless told otherwise."""
    tag = "DRY-RUN " if dry_run else ""
    # The caller passes a logical 'kaggle' first token; resolve it to a real command.
    sub = args[1:] if (args and args[0] == "kaggle") else list(args)
    prefix = _kaggle_cmd_prefix()
    if prefix is None and not dry_run:
        halt("kaggle CLI not available (module not importable and no kaggle executable found)")
        return 1
    cmd = (prefix if prefix is not None else ["kaggle"]) + sub
    print(f"[kaggle] {tag}{' '.join(cmd)}")
    if dry_run:
        return 0
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout: print(r.stdout)
    if r.stderr: print(r.stderr)
    if r.returncode != 0 and halt_on_fail:
        halt(f"kaggle command failed: {' '.join(cmd)} (exit {r.returncode})")
    return r.returncode
