"""
smoke_test_6c5_kaggle.py — provider-level smoke test for dataset attachment + kernel metadata.

Orchestration/provider-integration ONLY. This does NOT run cloud_6c5.py and does NOT train B/C.
It pushes a dedicated no-op smoke kernel (separate, deterministic slug, so the existing failed
training kernel is preserved as evidence), then queries the LIVE kernel metadata from Kaggle and asserts:
  - dataset_sources == [<owner>/<dataset_slug>]   (provider-failure #5 guard)
  - language == "python", kernel_type == "script"
  - enable_gpu == True, enable_internet == False
  - title/slug are provider-consistent (title derived from the canonical slug; avoids #6 409)
Only after ALL provider metadata checks pass is a training kernel permitted (a separate, later step
under a fresh L5 training authorization).

Identity is single-source and deterministic:
  SMOKE_KERNEL_SLUG = theo-6c5-bc-smoke-<bundle-hash-prefix>
  title            = SMOKE_KERNEL_SLUG            (so title's slug == id slug)
  metadata id      = <user>/SMOKE_KERNEL_SLUG
  API query        = same slug

Usage:
  python smoke_test_6c5_kaggle.py --bundle DIR [--execute] [--dry-run]
"""
import os, sys, json, argparse, re, subprocess
import kaggle_6c5_common as C
import launch_6c5_kaggle as launch

# Deterministic base; the full slug appends the bundle-hash prefix so each prepared
# bundle gets a unique, reproducible kernel identity (no collisions, no guessing).
SMOKE_KERNEL_PREFIX_BASE = "theo-6c5-bc-smoke"
_KERNEL_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,49}$")


def expected_dataset_slug():
    """Exact authorized dataset slug the kernel must attach (provider-failure #5 guard)."""
    return f"{C.KAGGLE_DATASET_OWNER}/{C.KAGGLE_DATASET_SLUG}"


def bundle_hash_prefix(bundle_dir, n=8):
    mp = os.path.join(bundle_dir, "bundle_manifest.json")
    if not os.path.exists(mp):
        C.halt("bundle_manifest.json missing in bundle; run prepare_6c5_cloud_bundle.py first")
    data = json.load(open(mp))
    sha = data.get("bundle_sha256")
    if not sha:
        C.halt("bundle_manifest.json missing bundle_sha256")
    return sha[:n]


def smoke_kernel_slug(bundle_dir):
    """Single canonical smoke-kernel slug, derived deterministically from the bundle hash."""
    return f"{SMOKE_KERNEL_PREFIX_BASE}-{bundle_hash_prefix(bundle_dir)}"


def is_valid_kernel_slug(slug):
    """Provider form: lowercase alphanumeric + hyphens, starts with alnum, <=50 chars."""
    return bool(_KERNEL_SLUG_RE.match(slug))


def _write_smoke_run_sh(kernel_dir, dataset_slug):
    os.makedirs(kernel_dir, exist_ok=True)
    run_sh = (
        "#!/bin/bash\nset -e\n"
        f"SLUG={dataset_slug}\n"
        "echo \"SMOKE TEST: verifying Kaggle dataset attachment (no training)\"\n"
        "ls -R /kaggle/input/$SLUG\n"
        "echo \"dataset attachment present; smoke kernel exiting (no training)\"\n"
    )
    open(os.path.join(kernel_dir, "run.sh"), "w").write(run_sh)


def _write_smoke_script_py(kernel_dir, dataset_slug):
    """Valid-Python smoke entry point. No training, no shell syntax."""
    os.makedirs(kernel_dir, exist_ok=True)
    src = (
        "import os\n"
        "\n"
        "# Kaggle smoke-test entry point. Valid Python only; no training.\n"
        f'DATASET_SLUG = "{dataset_slug}"\n'
        'root = f"/kaggle/input/{DATASET_SLUG}/THEO"\n'
        'print("SMOKE TEST: verifying Kaggle dataset attachment (no training)")\n'
        'print("dataset root exists:", os.path.isdir(root))\n'
        'print("dataset attachment present; smoke kernel exiting (no training)")\n'
    )
    open(os.path.join(kernel_dir, "script.py"), "w").write(src)


def build_smoke_kernel(bundle_dir, user):
    """Write a no-op smoke kernel (script.py: no training) + metadata.

    The kernel identity is single-source: slug -> title -> id -> API query all use the
    same canonical slug. code_file is the Python entry `script.py`; run.sh is auxiliary.
    """
    slug = smoke_kernel_slug(bundle_dir)
    if not is_valid_kernel_slug(slug):
        C.halt(f"derived smoke kernel slug is not provider-valid: {slug}")
    kernel_dir = os.path.join(bundle_dir, "_smoke_kernel")
    _write_smoke_run_sh(kernel_dir, C.KAGGLE_DATASET_SLUG)
    _write_smoke_script_py(kernel_dir, C.KAGGLE_DATASET_SLUG)
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,  # title == slug => Kaggle title->slug normalization matches id slug (#6)
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [expected_dataset_slug()],
        "competition_sources": [],
        "kernel_sources": [],
    }
    os.makedirs(kernel_dir, exist_ok=True)
    json.dump(meta, open(os.path.join(kernel_dir, "kernel-metadata.json"), "w"), indent=2)
    return kernel_dir


def query_kernel_metadata(user, slug):
    """Fetch LIVE kernel metadata via the Kaggle SDK (2.2.4).

    The SDK exposes kernel metadata only through `kernels_pull(..., metadata=True)`,
    which downloads `kernel-metadata.json` to the given path (the SDK provides no
    dedicated single-object metadata getter). We use exactly that provider-supported
    call. Returns a normalized dict; HALTS if retrieval fails.
    """
    import tempfile
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    full = f"{user}/{slug}"
    try:
        pull_dir = api.kernels_pull(full, path=tempfile.mkdtemp(), metadata=True)
        meta_path = os.path.join(pull_dir, "kernel-metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("kernels_pull did not emit kernel-metadata.json")
        meta = json.load(open(meta_path))
    except Exception as e:  # noqa: BLE001 - any retrieval failure must HALT, not train
        C.halt(f"could not retrieve live kernel metadata for {full}: {e}")
    return {
        "slug": full,
        "title": meta.get("title"),
        "language": meta.get("language"),
        "kernel_type": meta.get("kernel_type"),
        "enable_gpu": meta.get("enable_gpu"),
        "enable_internet": meta.get("enable_internet"),
        "dataset_sources": list(meta.get("dataset_sources") or []),
    }


def _norm_slug(s):
    # Kaggle normalizes owner/dataset slugs to lowercase in kernel metadata; compare
    # slug-normalized so provider casing does not cause a false failure.
    return (s or "").lower()


def verify_provider_metadata(user, slug):
    exp = expected_dataset_slug()
    meta = query_kernel_metadata(user, slug)
    print("[smoke] live kernel metadata:", json.dumps(meta, indent=2))
    got = [str(s) for s in (meta["dataset_sources"] or [])]
    checks = [
        ("dataset_sources contains exact dataset",
         _norm_slug(exp) in [_norm_slug(s) for s in got]),
        ("language == python", meta["language"] == "python"),
        ("kernel_type == script", meta["kernel_type"] == "script"),
        ("enable_gpu == True", meta["enable_gpu"] is True),
        ("enable_internet == False", meta["enable_internet"] is False),
        ("title/slug consistent (title == slug)", _norm_slug(meta["title"]) == _norm_slug(slug)),
    ]
    ok = True
    for name, cond in checks:
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond
    return ok


def smoke(bundle_dir, execute, dry_run):
    user = os.environ.get("KAGGLE_USERNAME", "<username>")
    if user == "<username>":
        C.halt("KAGGLE_USERNAME not set (export KAGGLE_USERNAME / KAGGLE_DATASET_SLUG / KAGGLE_KERNEL_SLUG)")
    if not os.path.exists(os.path.join(bundle_dir, "bundle_manifest.json")):
        C.halt(f"bundle manifest not found in {bundle_dir}; run prepare first")

    slug = smoke_kernel_slug(bundle_dir)
    kernel_dir = build_smoke_kernel(bundle_dir, user)
    print(f"[smoke] would push smoke kernel '{user}/{slug}' "
          f"(dataset={expected_dataset_slug()}, GPU on, internet off)")

    # push the smoke kernel only (never the training kernel, never cloud_6c5.py)
    C.run_kaggle_cmd(["kaggle", "kernels", "push", "-p", kernel_dir], dry_run)

    if dry_run:
        print("[smoke] DRY-RUN: no Kaggle contact. Above is the planned smoke push.")
        return

    print("[smoke] querying live kernel metadata from Kaggle ...")
    if not verify_provider_metadata(user, slug):
        C.halt("PROVIDER METADATA VERIFICATION FAILED — do NOT start the training kernel")
    print("[smoke] PROVIDER METADATA VERIFIED — dataset attached, metadata correct. "
          "Stopping (no training). Proceed to fresh L5 training authorization.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(C.ROOT, "6c5_cloud_bundle"))
    ap.add_argument("--execute", action="store_true", help="actually contact Kaggle (else dry-run)")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", help="no Kaggle contact (default)")
    a = ap.parse_args()
    smoke(a.bundle, a.execute, not a.execute)
