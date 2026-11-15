"""
prepare_6c5_cloud_bundle.py — build the allowlisted Kaggle input bundle.

Orchestration only: copies only ALLOWLIST files, explicitly excludes 6E.20.1 checkpoints
and unrelated files, verifies all binding hashes, and emits a bundle manifest + SHA-256.
No Kaggle contact (safe in dry-run and normal mode).

Usage:
  python prepare_6c5_cloud_bundle.py [--root ROOT] [--out BUNDLE_DIR] [--force]
"""
import os, sys, json, shutil, argparse
import kaggle_6c5_common as C


def build_bundle(root, out, force=False):
    # 1. exclude 6E.20.1 checkpoints + unrelated files (proof of exclusion)
    excl = C.scan_exclusions(root)
    print(f"[prepare] scanned exclusions: {len(excl)} file(s) matched EXCLUDE_PATTERNS (none will be bundled)")
    for e in excl[:20]:
        print(f"          excluded: {e}")

    # 2. verify all binding hashes before packaging
    ok, details = C.validate_binding_hashes(root)
    print("[prepare] binding hash validation:")
    for k, v in details.items():
        print(f"          {k} = {v}")
    if not ok:
        C.halt("binding hash validation FAILED — do not bundle")

    # 3. assemble allowlisted files
    paths = C.build_allowlist_paths(root)
    print(f"[prepare] {len(paths)} allowlisted file(s) to bundle")

    files = []
    for src, rel in paths:
        # extra guard: never bundle an excluded path
        if any(p.lower() in str(src).lower() for p in C.EXCLUDE_PATTERNS):
            C.halt(f"exclude pattern matched allowlisted path: {rel}")
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        files.append({"relpath": rel, "sha256": C.sha256_file(dst), "size": os.path.getsize(dst)})

    # 4. bundle manifest + SHA-256
    import hashlib
    canon = "".join(f"{f['relpath']}{f['sha256']}" for f in sorted(files, key=lambda x: x["relpath"]))
    bundle_sha = hashlib.sha256(canon.encode()).hexdigest()
    manifest = {
        "generated_by": "prepare_6c5_cloud_bundle.py",
        "root": str(root),
        "bundle_sha256": bundle_sha,
        "excluded_count": len(excl),
        "files": files,
    }
    mpath = os.path.join(out, "bundle_manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=2)
    print(f"[prepare] wrote bundle_manifest.json ({len(files)} files)")
    print(f"[prepare] BUNDLE SHA-256 = {bundle_sha}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(C.ROOT))
    ap.add_argument("--out", default=str(C.ROOT / "6c5_cloud_bundle"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    root = C.normalize_root(a.root)
    out = C.normalize_root(a.out)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(out / "bundle_manifest.json") and not a.force:
        print("[prepare] bundle already exists (use --force to rebuild)")
        sys.exit(0)
    build_bundle(root, out)
    print("[prepare] DONE (no Kaggle contact)")
