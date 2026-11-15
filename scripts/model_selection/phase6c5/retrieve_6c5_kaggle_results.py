"""
retrieve_6c5_kaggle_results.py — download B/C adapters, logs, manifest, and hashes.

Orchestration only. Preserves cloud outputs unchanged (verbatim copy + recorded hashes).
Requires --execute to contact Kaggle; default is dry-run.

Usage:
  python retrieve_6c5_kaggle_results.py --out DIR [--execute]
"""
import os, sys, json, argparse, shutil
import kaggle_6c5_common as C


def retrieve(out_dir, execute, dry_run):
    user = os.environ.get("KAGGLE_USERNAME", "<username>")
    if user == "<username>":
        C.halt("KAGGLE_USERNAME not set")
    os.makedirs(out_dir, exist_ok=True)
    kernel_slug = f"{user}/{C.KAGGLE_KERNEL_SLUG}"
    print(f"[retrieve] would download kernel output for {kernel_slug} into {out_dir}")
    C.run_kaggle_cmd(["kaggle", "kernels", "output", "-p", out_dir, kernel_slug], dry_run)
    if dry_run:
        print("[retrieve] DRY-RUN: no Kaggle contact. Above is the planned retrieval.")
        return
    # Record fetched artifact hashes (preserve unchanged)
    manifest = {}
    for name in ("6c5_cloud_execution_manifest.json",):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            manifest[name] = C.sha256_file(p)
    for sub in ("M1_Qwen2.5-0.5B-Instruct-6c5-B", "M1_Qwen2.5-0.5B-Instruct-6c5-C"):
        d = os.path.join(out_dir, sub)
        if os.path.isdir(d):
            manifest[sub] = C.dir_sha256(d)
    json.dump(manifest, open(os.path.join(out_dir, "retrieved_artifacts.json"), "w"), indent=2)
    print(f"[retrieve] DONE. Preserved cloud outputs unchanged; hashes in retrieved_artifacts.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(C.ROOT, "6c5_cloud_results"))
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    a = ap.parse_args()
    dry_run = not a.execute
    retrieve(a.out, a.execute, dry_run)
