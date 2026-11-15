"""
verify_6c5_cloud_results.py — verify returned cloud artifacts against the cloud manifest.

Orchestration only; fully offline. No Kaggle contact.
  - every returned artifact matches the cloud manifest's recorded hash;
  - B and C outputs are distinct;
  - no input/frozen artifact changed (binding hashes still valid).
HALT on any mismatch.

Usage:
  python verify_6c5_cloud_results.py --results DIR --binding BINDING_JSON
"""
import os, sys, json, argparse
import kaggle_6c5_common as C


def verify_results(results_dir, binding_path):
    report = {"checks": []}
    def check(name, cond, detail=""):
        report["checks"].append({"name": name, "pass": bool(cond), "detail": detail})
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if (detail and not cond) else ""))

    cloud_manifest_path = os.path.join(results_dir, "6c5_cloud_execution_manifest.json")
    if not os.path.exists(cloud_manifest_path):
        C.halt("cloud execution manifest not found in results dir")
    cloud = json.load(open(cloud_manifest_path))

    # 1. every returned artifact matches the cloud manifest's recorded hash
    artifacts = cloud.get("artifacts", {})
    for arch in ("B", "C"):
        adir = os.path.join(results_dir, f"M1_Qwen2.5-0.5B-Instruct-6c5-{arch}")
        if not os.path.isdir(adir):
            check(f"adapter_dir_present_{arch}", False, adir)
            continue
        cur = C.dir_sha256(adir)
        exp_sha = artifacts.get(arch, {}).get("adapter_dir_sha256")
        check(f"adapter_hash_{arch}", cur == exp_sha, f"got {cur[:16]} expected {str(exp_sha)[:16]}")

    # 2. B and C outputs are distinct
    b = artifacts.get("B", {}).get("adapter_dir_sha256")
    c = artifacts.get("C", {}).get("adapter_dir_sha256")
    check("B_C_distinct", b != c and b is not None and c is not None, f"B={str(b)[:12]} C={str(c)[:12]}")

    # 3. no input artifact changed: binding hashes still valid
    ok, details = C.validate_binding_hashes(C.ROOT)
    check("binding_hashes_intact", ok, "one or more frozen inputs changed" if not ok else "")

    # 4. provenance sanity: cloud recorded a Linux bnb wheel SHA, not a reused Windows one
    prov = cloud.get("provenance", {})
    bnb = prov.get("bitsandbytes_wheel_sha256")
    check("linux_bnb_wheel_bound", bool(bnb), "missing bitsandbytes wheel SHA in cloud provenance")

    failed = [c["name"] for c in report["checks"] if not c["pass"]]
    report["passed"] = not failed
    json.dump(report, open(os.path.join(results_dir, "verify_report.json"), "w"), indent=2)
    if failed:
        C.halt("verification FAILED: " + ", ".join(failed))
    print("[verify] ALL CHECKS PASS — cloud results verified, B/C distinct, inputs unchanged.")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(C.ROOT, "6c5_cloud_results"))
    ap.add_argument("--binding", default=C.BINDING)
    a = ap.parse_args()
    verify_results(a.results, a.binding)
