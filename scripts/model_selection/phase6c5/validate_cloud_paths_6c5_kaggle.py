"""
validate_cloud_paths_6c5_kaggle.py — Dedicated non-training Kaggle path & bundle validator.

Pushes a lightweight script kernel to Kaggle that mounts the dataset, runs the
diagnostics-first bounded resolver, verifies file presence, and validates scientific
file hashes against the frozen binding manifest without loading models or executing
cloud_6c5.py.

Usage:
  python validate_cloud_paths_6c5_kaggle.py [--bundle DIR] [--execute] [--dry-run]
"""
import os, sys, json, argparse, subprocess
import kaggle_6c5_common as C


def _validator_kernel_slug():
    return "theo-6c5-bc-path-validate"


def _validator_kernel_metadata(user):
    owner = os.environ.get("KAGGLE_DATASET_OWNER") or (user if user and user != "<username>" else C.KAGGLE_DATASET_OWNER)
    slug = _validator_kernel_slug()
    return {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [f"{owner}/{C.KAGGLE_DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }


def _validator_script_py(user=None):
    owner = os.environ.get("KAGGLE_DATASET_OWNER") or (user if user and user != "<username>" else C.KAGGLE_DATASET_OWNER)
    slug = C.KAGGLE_DATASET_SLUG
    expected_bundle_sha = C.EXPECTED["bundle_sha256"]
    expected_cloud_sha = C.EXPECTED["cloud_6c5_sha256"]
    expected_adj = C.EXPECTED["adjudication_sha256"]
    expected_fix = C.EXPECTED["fixture_sha256"]
    expected_split = C.EXPECTED["split_manifest_sha256"]

    return (
        "import os, sys, json, hashlib\n"
        "from pathlib import Path\n"
        "\n"
        "# ---- Dedicated Kaggle Path & Binding Validator (Non-Training, Remediation #15) ----\n"
        f'DATASET_SLUG = "{slug}"\n'
        f'DATASET_OWNER = "{owner}"\n'
        f'EXPECTED_BUNDLE_SHA = "{expected_bundle_sha}"\n'
        f'EXPECTED_CLOUD_SHA = "{expected_cloud_sha}"\n'
        'INPUT_ROOT = "/kaggle/input"\n'
        f'EXPECTED_ADJ = "{expected_adj}"\n'
        f'EXPECTED_FIX = "{expected_fix}"\n'
        f'EXPECTED_SPLIT = "{expected_split}"\n'
        "\n"
        "def sha256_file(path):\n"
        "    h = hashlib.sha256()\n"
        "    with open(path, 'rb') as f:\n"
        "        for b in iter(lambda: f.read(1 << 20), b''):\n"
        "            h.update(b)\n"
        "    return h.hexdigest()\n"
        "\n"
        "# ---- 1. Mount diagnostics (emitted unconditionally BEFORE resolution) ----\n"
        "def print_mount_diagnostics(input_root=INPUT_ROOT, max_depth=4):\n"
        '    input_root_str = str(input_root).replace("\\\\", "/").rstrip("/")\n'
        '    print("=" * 60)\n'
        '    print(f"[mount_diagnostics] Scanning input root: {input_root_str}")\n'
        "    if not os.path.exists(input_root_str):\n"
        '        print(f"[mount_diagnostics] WARNING: {input_root_str} does not exist!")\n'
        '        print("=" * 60)\n'
        "        return\n"
        "    if not os.path.isdir(input_root_str):\n"
        '        print(f"[mount_diagnostics] WARNING: {input_root_str} is not a directory!")\n'
        '        print("=" * 60)\n'
        "        return\n"
        "    try:\n"
        "        immediate = sorted(os.listdir(input_root_str))\n"
        '        print(f"[mount_diagnostics] Immediate children of {input_root_str}: {immediate}")\n'
        "    except Exception as e:\n"
        '        print(f"[mount_diagnostics] Error listing immediate children: {e}")\n'
        '    print(f"[mount_diagnostics] Recursive tree (depth <= {max_depth}):")\n'
        "    base_parts_len = len(Path(input_root_str).resolve().parts) if os.path.exists(input_root_str) else 0\n"
        "    for root_dir, dirs, files in os.walk(input_root_str):\n"
        "        curr_parts_len = len(Path(root_dir).resolve().parts)\n"
        "        depth = curr_parts_len - base_parts_len\n"
        "        if depth > max_depth:\n"
        "            dirs.clear()\n"
        "            continue\n"
        '        rel = os.path.relpath(root_dir, input_root_str).replace("\\\\", "/")\n'
        '        indent = "  " * depth\n'
        '        if rel != ".":\n'
        '            print(f"[mount_diagnostics] {indent}[DIR]  {rel}/")\n'
        "        for d in sorted(dirs):\n"
        "            if depth + 1 <= max_depth:\n"
        '                sub_rel = os.path.normpath(os.path.join(rel, d)).replace("\\\\", "/")\n'
        '                print(f"[mount_diagnostics] {indent}  [DIR]  {sub_rel}/")\n'
        "        for f in sorted(files):\n"
        '            f_rel = os.path.normpath(os.path.join(rel, f)).replace("\\\\", "/")\n'
        "            f_path = os.path.join(root_dir, f)\n"
        "            try:\n"
        "                sz = os.path.getsize(f_path)\n"
        "            except Exception:\n"
        '                sz = "n/a"\n'
        '            print(f"[mount_diagnostics] {indent}  [FILE] {f_rel} ({sz} bytes)")\n'
        '    print("=" * 60)\n'
        "\n"
        "# ---- 2. Unambiguous dataset mount locator (depth <= 4) ----\n"
        "def locate_dataset_mount(input_root=INPUT_ROOT, dataset_slug=DATASET_SLUG, dataset_owner=DATASET_OWNER, max_depth=4):\n"
        '    input_root_str = str(input_root).replace("\\\\", "/").rstrip("/")\n'
        "    if not os.path.exists(input_root_str):\n"
        '        sys.exit(f"HALT: input root does not exist: {input_root_str}")\n'
        '    norm_slug = str(dataset_slug).lower().replace("_", "-").strip()\n'
        '    norm_owner = str(dataset_owner).lower().replace("_", "-").strip() if dataset_owner else ""\n'
        '    base_name = os.path.basename(input_root_str).lower().replace("_", "-")\n'
        '    if base_name == norm_slug or (norm_owner and base_name == f"{norm_owner}-{norm_slug}"):\n'
        "        return input_root_str\n"
        "    candidates = []\n"
        "    base_parts_len = len(Path(input_root_str).resolve().parts) if os.path.exists(input_root_str) else 0\n"
        "    for root_dir, dirs, _ in os.walk(input_root_str):\n"
        "        curr_parts_len = len(Path(root_dir).resolve().parts)\n"
        "        depth = curr_parts_len - base_parts_len\n"
        "        if depth > max_depth:\n"
        "            dirs.clear()\n"
        "            continue\n"
        '        dir_name = os.path.basename(root_dir).lower().replace("_", "-")\n'
        '        if dir_name == norm_slug or (norm_owner and dir_name == f"{norm_owner}-{norm_slug}"):\n'
        '            parent_name = os.path.basename(os.path.dirname(root_dir)).lower().replace("_", "-")\n'
        '            norm_path = root_dir.replace("\\\\", "/")\n'
        '            is_owner_match = bool(norm_owner and (parent_name == norm_owner or dir_name.startswith(f"{norm_owner}-")))\n'
        "            candidates.append((norm_path, is_owner_match))\n"
        "    owner_matched = [c[0] for c in candidates if c[1]]\n"
        "    final_candidates = owner_matched if owner_matched else [c[0] for c in candidates]\n"
        "    if len(final_candidates) == 0:\n"
        '        sys.exit(f"HALT: no mount matching \'{dataset_slug}\' (owner={dataset_owner}) found under {input_root_str} (depth <= {max_depth})")\n'
        "    if len(final_candidates) > 1:\n"
        '        sys.exit(f"HALT: ambiguous dataset mounts for \'{dataset_slug}\': {final_candidates}")\n'
        "    return final_candidates[0]\n"
        "\n"
        "# ---- 3. Bounded bundle root resolver & manifest check (depth <= 3) ----\n"
        "def resolve_cloud_bundle_root(input_root=INPUT_ROOT, dataset_slug=DATASET_SLUG, dataset_owner=DATASET_OWNER, expected_bundle_sha=EXPECTED_BUNDLE_SHA, max_depth=3):\n"
        "    print_mount_diagnostics(input_root=input_root, max_depth=max_depth + 1)\n"
        "    dataset_mount = locate_dataset_mount(input_root=input_root, dataset_slug=dataset_slug, dataset_owner=dataset_owner, max_depth=4)\n"
        '    print(f"[resolver] Identified dataset mount root: {dataset_mount}")\n'
        "    candidates = []\n"
        "    base_parts_len = len(Path(dataset_mount).resolve().parts) if os.path.exists(dataset_mount) else 0\n"
        "    for root_dir, dirs, _ in os.walk(dataset_mount):\n"
        "        curr_parts_len = len(Path(root_dir).resolve().parts)\n"
        "        depth = curr_parts_len - base_parts_len\n"
        "        if depth > max_depth:\n"
        "            dirs.clear()\n"
        "            continue\n"
        '        has_tc = os.path.isdir(os.path.join(root_dir, "theo-core"))\n'
        '        has_td = os.path.isdir(os.path.join(root_dir, "theo-data"))\n'
        '        manifest_path = os.path.join(root_dir, "bundle_manifest.json")\n'
        "        has_manifest = os.path.isfile(manifest_path)\n"
        "        if has_tc and has_td:\n"
        "            if not has_manifest:\n"
        '                sys.exit(f"HALT: candidate {root_dir} has theo-core/ + theo-data/ but missing bundle_manifest.json")\n'
        "            try:\n"
        "                mf = json.load(open(manifest_path))\n"
        '                actual_b_sha = mf.get("bundle_sha256")\n'
        "            except Exception as e:\n"
        '                sys.exit(f"HALT: cannot parse {manifest_path}: {e}")\n'
        "            if expected_bundle_sha and actual_b_sha != expected_bundle_sha:\n"
        '                sys.exit(f"HALT: bundle_manifest.json SHA mismatch: expected {expected_bundle_sha}, got {actual_b_sha}")\n'
        '            norm = root_dir.replace("\\\\", "/")\n'
        "            if norm not in candidates:\n"
        "                candidates.append(norm)\n"
        "    if len(candidates) == 0:\n"
        '        sys.exit(f"HALT: no valid bundle root (theo-core + theo-data + valid bundle_manifest.json) found inside dataset mount {dataset_mount} (depth <= {max_depth})")\n'
        "    if len(candidates) > 1:\n"
        '        sys.exit(f"HALT: ambiguous -- {len(candidates)} candidates under dataset mount {dataset_mount}: {candidates}")\n'
        "    resolved = candidates[0]\n"
        '    print(f"[resolver] Resolved bundle root: {resolved}")\n'
        "    return dataset_mount, resolved\n"
        "\n"
        "DATASET_MOUNT_ROOT, RESOLVED_THEO_ROOT = resolve_cloud_bundle_root(INPUT_ROOT, DATASET_SLUG, DATASET_OWNER, EXPECTED_BUNDLE_SHA)\n"
        "\n"
        "# ---- 4. Diagnostic path manifest ----\n"
        'THEO_CORE = os.path.join(RESOLVED_THEO_ROOT, "theo-core")\n'
        'THEO_DATA = os.path.join(RESOLVED_THEO_ROOT, "theo-data")\n'
        'PHASE6C5 = os.path.join(THEO_CORE, "scripts", "model_selection", "phase6c5")\n'
        'SPEC_PATH = os.path.join(PHASE6C5, "experiment_spec.py")\n'
        'CLOUD_ENTRY = os.path.join(PHASE6C5, "cloud_6c5.py")\n'
        'BINDING_PATH = os.path.join(THEO_DATA, "datasets", "theo_slm_v0_review", "6b3_batch2", "6c5_binding_manifest.json")\n'
        'SPLIT_PATH = os.path.join(THEO_DATA, "datasets", "theo_slm_v0_review", "6b3_batch2", "6c5_split_manifest.json")\n'
        'ADJ_PATH = os.path.join(THEO_DATA, "datasets", "theo_slm_v0_review", "adjudication.json")\n'
        'FIXTURE_PATH = os.path.join(THEO_DATA, "datasets", "theo_slm_v0_artifacts", "phase-6e3", "populated-evidence-corpus.json")\n'
        'M1_DIR = os.path.join(THEO_DATA, "models", "6c_acquired", "M1_Qwen2.5-0.5B-Instruct")\n'
        "\n"
        'print("=" * 60)\n'
        'print(f"DATASET_MOUNT_ROOT  = {DATASET_MOUNT_ROOT}")\n'
        'print(f"RESOLVED_THEO_ROOT  = {RESOLVED_THEO_ROOT}")\n'
        'print(f"THEO_CORE           = {THEO_CORE}")\n'
        'print(f"THEO_DATA           = {THEO_DATA}")\n'
        'print(f"experiment_spec.py  = {SPEC_PATH}")\n'
        'print(f"cloud_6c5.py        = {CLOUD_ENTRY}")\n'
        'print(f"binding_manifest    = {BINDING_PATH}")\n'
        'print("=" * 60)\n'
        "\n"
        "# ---- 5. Verify required files exist ----\n"
        "reqs = [SPEC_PATH, CLOUD_ENTRY, BINDING_PATH, SPLIT_PATH, ADJ_PATH, FIXTURE_PATH]\n"
        "for p in reqs:\n"
        "    if not os.path.isfile(p):\n"
        '        sys.exit(f"HALT: required file missing: {p}")\n'
        "assert os.path.isdir(THEO_CORE), f\"HALT: {THEO_CORE} is not a directory\"\n"
        "assert os.path.isdir(THEO_DATA), f\"HALT: {THEO_DATA} is not a directory\"\n"
        "assert os.path.isdir(M1_DIR), f\"HALT: {M1_DIR} is not a directory\"\n"
        "\n"
        "# ---- 6. Verify scientific SHA-256 hashes against frozen bindings ----\n"
        "binding = json.load(open(BINDING_PATH))\n"
        "adj_sha = sha256_file(ADJ_PATH)\n"
        "fix_sha = sha256_file(FIXTURE_PATH)\n"
        "split_sha = sha256_file(SPLIT_PATH)\n"
        "cloud_sha = sha256_file(CLOUD_ENTRY)\n"
        "\n"
        'print(f"[hash_check] adjudication SHA-256 = {adj_sha}")\n'
        "if adj_sha != EXPECTED_ADJ:\n"
        '    sys.exit(f"HALT: adjudication SHA mismatch: expected {EXPECTED_ADJ}, got {adj_sha}")\n'
        "\n"
        'print(f"[hash_check] fixture SHA-256      = {fix_sha}")\n'
        "if fix_sha != EXPECTED_FIX:\n"
        '    sys.exit(f"HALT: fixture SHA mismatch: expected {EXPECTED_FIX}, got {fix_sha}")\n'
        "\n"
        'print(f"[hash_check] split SHA-256        = {split_sha}")\n'
        "if split_sha != EXPECTED_SPLIT:\n"
        '    sys.exit(f"HALT: split SHA mismatch: expected {EXPECTED_SPLIT}, got {split_sha}")\n'
        "\n"
        'print(f"[hash_check] cloud_6c5.py SHA-256  = {cloud_sha}")\n'
        "if cloud_sha != EXPECTED_CLOUD_SHA:\n"
        '    sys.exit(f"HALT: cloud_6c5.py SHA mismatch: expected {EXPECTED_CLOUD_SHA}, got {cloud_sha}")\n'
        "\n"
        "for fname, fh in binding.get('base_model', {}).get('file_hashes', {}).items():\n"
        "    p = os.path.join(M1_DIR, fname)\n"
        "    if not os.path.isfile(p):\n"
        '        sys.exit(f"HALT: M1 model file missing: {p}")\n'
        "    cur = sha256_file(p)\n"
        '    print(f"[hash_check] M1 {fname} = {cur}")\n'
        "    if cur != fh:\n"
        '        sys.exit(f"HALT: M1 {fname} SHA mismatch: expected {fh}, got {cur}")\n'
        "\n"
        "# ---- 7. Pre-training import smoke test ----\n"
        "os.chdir(PHASE6C5)\n"
        "sys.path.insert(0, PHASE6C5)\n"
        "os.environ['THEO_ROOT'] = RESOLVED_THEO_ROOT\n"
        "os.environ['THEO_OUT_B'] = '/kaggle/working/M1_Qwen2.5-0.5B-Instruct-6c5-B'\n"
        "os.environ['THEO_OUT_C'] = '/kaggle/working/M1_Qwen2.5-0.5B-Instruct-6c5-C'\n"
        "import cloud_6c5\n"
        'print("[import_smoke] cloud_6c5 imported successfully.")\n'
        "\n"
        'print("=" * 60)\n'
        'print("[PATH_VALIDATION_PASS] All cloud paths resolved and scientific hashes verified on Kaggle mount.")\n'
        'print("=" * 60)\n'
    )


def build_path_validation_kernel(bundle_dir, user):
    kd = os.path.join(bundle_dir, "_path_validation_kernel")
    os.makedirs(kd, exist_ok=True)
    open(os.path.join(kd, "script.py"), "w").write(_validator_script_py(user))
    meta_path = os.path.join(kd, "kernel-metadata.json")
    json.dump(_validator_kernel_metadata(user), open(meta_path, "w"), indent=2)
    return kd


def validate_paths(bundle_dir, execute, dry_run):
    l5_ok, l5_msg = C.check_l5_authorization()
    print(f"[validate_paths] L5 authorization: {l5_msg}")
    if execute and not l5_ok:
        C.halt("execution requested but L5 authorization is NOT active")

    if not os.path.exists(os.path.join(bundle_dir, "bundle_manifest.json")):
        C.halt(f"bundle manifest not found in {bundle_dir}; run prepare first")

    user = os.environ.get("KAGGLE_USERNAME", "<username>")
    if user == "<username>":
        C.halt("KAGGLE_USERNAME not set (export KAGGLE_USERNAME / KAGGLE_DATASET_SLUG / KAGGLE_KERNEL_SLUG)")

    kd = build_path_validation_kernel(bundle_dir, user)
    slug = _validator_kernel_slug()
    owner = os.environ.get("KAGGLE_DATASET_OWNER") or user
    print(f"[validate_paths] Built path validation kernel at {kd}")
    print(f"[validate_paths] Planned kernel: {user}/{slug} mounting {owner}/{C.KAGGLE_DATASET_SLUG}")

    cmd = ["kaggle", "kernels", "push", "-p", kd]
    C.run_kaggle_cmd(cmd, dry_run)

    if dry_run:
        print("[validate_paths] DRY-RUN: no Kaggle contact made.")
        return

    print("[validate_paths] Polling path validation kernel status (single check)...")
    st_cmd = ["kaggle", "kernels", "status", f"{user}/{slug}"]
    full_cmd = C.resolve_kaggle_command(st_cmd[1:])
    if full_cmd is None:
        C.halt("kaggle CLI not available")
    r = subprocess.run(full_cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        C.halt("kernel status check failed after submission")
    print(f"[validate_paths] Kernel submitted. Check output log for [PATH_VALIDATION_PASS].")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(C.ROOT, "6c5_cloud_bundle"))
    ap.add_argument("--execute", action="store_true", help="push kernel to Kaggle (else dry-run)")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", help="dry-run mode")
    a = ap.parse_args()
    dry_run = not a.execute
    validate_paths(a.bundle, a.execute, dry_run)
