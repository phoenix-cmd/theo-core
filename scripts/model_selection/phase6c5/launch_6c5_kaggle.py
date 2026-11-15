"""
launch_6c5_kaggle.py — submit the bundle and run cloud_6c5.py on Kaggle.

Orchestration only. Requires explicit --execute (otherwise dry-run). Execution also requires the
active L5 authorization (present doc + amendment_audit PASS). No automatic retry; any
authorization / runtime / hash failure HALTs.

Usage:
  python launch_6c5_kaggle.py --bundle DIR [--execute] [--dry-run]
"""
import os, sys, json, argparse, subprocess
import kaggle_6c5_common as C


def _write_dataset_metadata(bundle_dir, user):
    meta = {
        "title": "THEO 6C.5 cloud inputs",
        "id": f"{user}/{C.KAGGLE_DATASET_SLUG}",
        "licenses": [{"name": "other"}],
    }
    p = os.path.join(bundle_dir, "dataset-metadata.json")
    json.dump(meta, open(p, "w"), indent=2)
    return p


def _kernel_metadata(user, kernel_slug=None):
    """Kernel metadata for the Kaggle script kernel.

    Defect B fix: language='python', kernel_type='script'.
    Provider-failure #5 fix: dataset_sources binds the exact authorized dataset slug
    (KAGGLE_DATASET_OWNER / KAGGLE_DATASET_SLUG), independent of the kernel owner username.
    Provider-failure #6 fix: title is derived from the canonical slug so Kaggle's
    title->slug normalization resolves to the same identifier (avoids 409 'title does not
    resolve to id').
    Remediation #8 fix: code_file is the *Python* entry `script.py` (valid Python only);
    the shell wrapper `run.sh` is auxiliary and never executed as Python source.
    """
    owner = C.KAGGLE_DATASET_OWNER
    slug = kernel_slug or C.KAGGLE_KERNEL_SLUG
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


def write_kernel_metadata(user, kernel_dir, kernel_slug=None):
    """Write kernel-metadata.json for the given (optionally overridden) kernel slug."""
    os.makedirs(kernel_dir, exist_ok=True)
    path = os.path.join(kernel_dir, "kernel-metadata.json")
    json.dump(_kernel_metadata(user, kernel_slug), open(path, "w"), indent=2)
    return path


def _datasets_create_args(bundle_dir):
    """Kaggle dataset-create args (Defect A fix: --dir-mode zip preserves nested tree)."""
    return ["kaggle", "datasets", "create", "-p", bundle_dir, "--dir-mode", "zip"]
def _kernel_script_py(user=None):
    """Valid-Python Kaggle entry point. Shell/environment setup is done in Python;
    this file must NEVER contain shell assignments (e.g. SLUG=...).

    The resolver is inlined because the kernel cannot import local orchestration
    modules before THEO_ROOT is established. cloud_6c5.py is the authoritative
    hash/binding verifier; script.py only resolves paths and checks file existence.
    """
    owner = os.environ.get("KAGGLE_DATASET_OWNER") or (user if user and user != "<username>" else C.KAGGLE_DATASET_OWNER)
    slug = C.KAGGLE_DATASET_SLUG
    expected_bundle_sha = C.EXPECTED["bundle_sha256"]
    expected_cloud_sha = C.EXPECTED["cloud_6c5_sha256"]
    return (
        "import os, sys, json, hashlib\n"
        "from pathlib import Path\n"
        "\n"
        "# ---- Kaggle script-kernel entry point (Remediation #15) ----\n"
        "# Valid Python only. No shell assignments. No hard-coded /THEO prefix.\n"
        "# Bounded resolver discovers the matched dataset mount (depth <= 4) and bundle root (depth <= 3).\n"
        "\n"
        f'DATASET_SLUG = "{slug}"\n'
        f'DATASET_OWNER = "{owner}"\n'
        f'EXPECTED_BUNDLE_SHA = "{expected_bundle_sha}"\n'
        f'EXPECTED_CLOUD_SHA = "{expected_cloud_sha}"\n'
        'INPUT_ROOT = "/kaggle/input"\n'
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
        'BINDING_PATH = os.path.join(THEO_DATA, "datasets", "theo_slm_v0_review",\n'
        '                            "6b3_batch2", "6c5_binding_manifest.json")\n'
        'CLOUD_ENTRY = os.path.join(PHASE6C5, "cloud_6c5.py")\n'
        "\n"
        'print("=" * 60)\n'
        'print(f"DATASET_MOUNT_ROOT  = {DATASET_MOUNT_ROOT}")\n'
        'print(f"RESOLVED_THEO_ROOT  = {RESOLVED_THEO_ROOT}")\n'
        'print(f"THEO_CORE           = {THEO_CORE}")\n'
        'print(f"THEO_DATA           = {THEO_DATA}")\n'
        'print(f"experiment_spec.py  = {SPEC_PATH}")\n'
        'print(f"binding_manifest    = {BINDING_PATH}")\n'
        'print(f"cloud_6c5.py        = {CLOUD_ENTRY}")\n'
        'print("=" * 60)\n'
        "\n"
        "# ---- 5. Pre-import required file checks & cloud_6c5 SHA verification ----\n"
        "for label, path in [(\"experiment_spec.py\", SPEC_PATH),\n"
        '                    ("6c5_binding_manifest.json", BINDING_PATH),\n'
        '                    ("cloud_6c5.py", CLOUD_ENTRY)]:\n'
        "    if not os.path.isfile(path):\n"
        '        sys.exit(f"HALT: required file missing: {label} -> {path}")\n'
        "\n"
        'assert os.path.isdir(THEO_CORE), f"HALT: {THEO_CORE} is not a directory"\n'
        'assert os.path.isdir(THEO_DATA), f"HALT: {THEO_DATA} is not a directory"\n'
        "\n"
        "cur_cloud_sha = sha256_file(CLOUD_ENTRY)\n"
        'print(f"[preflight] cloud_6c5.py SHA-256 = {cur_cloud_sha}")\n'
        "if cur_cloud_sha != EXPECTED_CLOUD_SHA:\n"
        '    sys.exit(f"HALT: cloud_6c5.py SHA-256 mismatch: expected {EXPECTED_CLOUD_SHA}, got {cur_cloud_sha}")\n'
        "\n"
        "# ---- 6. Environment configuration ----\n"
        'THEO_OUT_B = "/kaggle/working/M1_Qwen2.5-0.5B-Instruct-6c5-B"\n'
        'THEO_OUT_C = "/kaggle/working/M1_Qwen2.5-0.5B-Instruct-6c5-C"\n'
        'os.environ["THEO_ROOT"] = RESOLVED_THEO_ROOT\n'
        'os.environ["THEO_OUT_B"] = THEO_OUT_B\n'
        'os.environ["THEO_OUT_C"] = THEO_OUT_C\n'
        'os.environ.setdefault("KAGGLE_DATASET_SLUG", DATASET_SLUG)\n'
        "os.makedirs(THEO_OUT_B, exist_ok=True)\n"
        "os.makedirs(THEO_OUT_C, exist_ok=True)\n"
        "os.chdir(PHASE6C5)\n"
        "sys.path.insert(0, PHASE6C5)\n"
        "\n"
        "# ---- 7. Gated B/C training pipeline invocation ----\n"
        "# cloud_6c5.py performs authoritative hash/binding verification.\n"
        "# HALTS if the dependency baseline (Python 3.14.3) is not reproduced.\n"
        "import cloud_6c5\n"
    )



def _write_kernel(bundle_dir, kernel_dir, user):
    os.makedirs(kernel_dir, exist_ok=True)
    # Auxiliary shell wrapper (shell syntax only; not the Kaggle Python entry).
    run_sh = (
        "#!/bin/bash\nset -e\n"
        f"SLUG={C.KAGGLE_DATASET_SLUG}\n"
        "export THEO_ROOT=/kaggle/input/$SLUG/THEO\n"
        "export THEO_OUT_B=/kaggle/working/M1_Qwen2.5-0.5B-Instruct-6c5-B\n"
        "export THEO_OUT_C=/kaggle/working/M1_Qwen2.5-0.5B-Instruct-6c5-C\n"
        "mkdir -p $THEO_OUT_B $THEO_OUT_C\n"
        "cd $THEO_ROOT/theo-core/scripts/model_selection/phase6c5\n"
        "python script.py\n"
    )
    open(os.path.join(kernel_dir, "run.sh"), "w").write(run_sh)
    # Kaggle executes this Python file (code_file in kernel-metadata.json).
    open(os.path.join(kernel_dir, "script.py"), "w").write(_kernel_script_py(user))
    write_kernel_metadata(user, kernel_dir)
    return kernel_dir


def launch(bundle_dir, execute, dry_run):
    # Guard: execution requires the L5 authorization to be active.
    l5_ok, l5_msg = C.check_l5_authorization()
    print(f"[launch] L5 authorization: {l5_msg}")
    if execute and not l5_ok:
        C.halt("execution requested but L5 authorization is NOT active")

    if not os.path.exists(os.path.join(bundle_dir, "bundle_manifest.json")):
        C.halt(f"bundle manifest not found in {bundle_dir}; run prepare first")

    user = os.environ.get("KAGGLE_USERNAME", "<username>")
    if user == "<username>":
        C.halt("KAGGLE_USERNAME not set (export KAGGLE_USERNAME / KAGGLE_DATASET_SLUG / KAGGLE_KERNEL_SLUG)")

    _write_dataset_metadata(bundle_dir, user)
    kernel_dir = os.path.join(bundle_dir, "_kernel")
    _write_kernel(bundle_dir, kernel_dir, user)

    print(f"[launch] would submit dataset '{user}/{C.KAGGLE_DATASET_SLUG}' and "
          f"kernel '{user}/{C.KAGGLE_KERNEL_SLUG}' (GPU on, internet off)")

    # 1. dataset: create or reuse
    ds_slug = f"{user}/{C.KAGGLE_DATASET_SLUG}"
    st = C.run_kaggle_cmd(["kaggle", "datasets", "status", ds_slug], dry_run, halt_on_fail=False)
    if st != 0:
        C.run_kaggle_cmd(_datasets_create_args(bundle_dir), dry_run)
    else:
        print("[launch] dataset already exists; reusing")

    # 2. push kernel (do NOT auto-retry)
    C.run_kaggle_cmd(["kaggle", "kernels", "push", "-p", kernel_dir], dry_run)

    if dry_run:
        print("[launch] DRY-RUN: no Kaggle contact made. Above is the planned submission.")
        return

    # 3. optional single status poll (no retry on failure)
    print("[launch] polling kernel status (single check, no auto-retry)...")
    st_cmd = ["kaggle", "kernels", "status", f"{user}/{C.KAGGLE_KERNEL_SLUG}"]
    cmd = C.resolve_kaggle_command(st_cmd[1:])
    if cmd is None:
        C.halt("kaggle CLI not available (module not importable and no kaggle executable found)")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        C.halt("kernel status check failed after submission (no auto-retry)")
    print("[launch] SUBMITTED. Retrieve results with retrieve_6c5_kaggle_results.py (requires --execute).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(C.ROOT, "6c5_cloud_bundle"))
    ap.add_argument("--execute", action="store_true", help="actually contact Kaggle (else dry-run)")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", help="no Kaggle contact (default)")
    a = ap.parse_args()
    dry_run = not a.execute
    launch(a.bundle, a.execute, dry_run)
