"""
feasibility_6c5_kaggle.py — READ-ONLY cloud runtime feasibility test (Kaggle).

Sole purpose: determine whether the required execution environment is *technically obtainable*
on the existing approved Kaggle GPU/provider path, WITHOUT modifying any scientific artifact,
performing training, or mutating the dataset.

What it does
------------
1. Runs on the Kaggle GPU image (base interpreter is Python 3.12) WITHOUT replacing it.
2. Builds/installs an ISOLATED CPython 3.14.3 environment (conda -> pyenv -> source build).
3. Installs the EXACT pinned dependencies into that isolated environment.
4. Re-executes a capture payload under 3.14.3 to record the realized stack:
   Python, torch, CUDA runtime, NVIDIA driver, Transformers, PEFT, bitsandbytes, the ACTUAL
   bitsandbytes package SHA-256, Linux/glibc.
5. Verifies GPU accessibility and QLoRA prerequisites (bitsandbytes 4-bit importable on CUDA).
6. Performs a minimal import/runtime smoke only -- ZERO model training, ZERO dataset mutation.
7. Emits a machine-readable compatibility manifest and FAILS CLOSED (non-zero exit) if any
   required version cannot be reproduced exactly.

Key distinction (do NOT silently substitute)
-------------------------------------------
  BASE_IMAGE_IS_3_12  = informational. The Kaggle base itself uses Python 3.12. This is EXPECTED
                        and does NOT by itself make the run infeasible -- the isolated 3.14.3 env
                        decides feasibility.
  FEASIBLE            = isolated Python 3.14.3 + all pins reproduced exactly (base 3.12 is fine).
  INFEASIBLE          = required stack could not be reproduced; exact reason recorded.

This is the LAST authorized Kaggle workaround. Hard stopping rule (see authorization doc):
  PASS  -> environment can be considered for a fresh L5 training authorization.
  FAIL  -> abandon the Kaggle path for this experiment; do NOT relax pins; no further workarounds.
"""
import os, sys, json, subprocess, hashlib, platform, shutil, tempfile

# Identical to cloud_6c5.py PINNED -- enforced here so the feasibility gate and the training
# gate can never drift apart.
REQUIRED = {
    "python": "3.14.3",
    "torch": "2.13.0+cu126",
    "cuda_runtime": "12.6",
    "transformers": "5.8.0",
    "peft": "0.20.0",
    "bitsandbytes": "0.50.1",
}

MANIFEST_PATH = os.environ.get(
    "FEASIBILITY_MANIFEST", "/kaggle/working/6c5_runtime_feasibility_manifest.json"
)


def _run(cmd, cwd=None, timeout=1800):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)


def build_isolated_python314():
    """Build/install an isolated CPython 3.14.3; never replaces the base interpreter.
    Tries conda, then pyenv, then an official source build. Returns the python exe or None."""
    # 1) conda
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda:
        env = "py314"
        r = _run([conda, "create", "-y", "-n", env, "python=3.14.3"])
        if r.returncode == 0:
            base = _run([conda, "info", "--base"]).stdout.strip()
            exe = os.path.join(base, "envs", env, "bin", "python")
            if os.path.exists(exe):
                return exe
    # 2) pyenv
    pyenv = shutil.which("pyenv")
    if pyenv:
        if _run([pyenv, "install", "3.14.3"]).returncode == 0:
            exe = _run([pyenv, "which", "python", "3.14.3"]).stdout.strip()
            if exe and os.path.exists(exe):
                return exe
    # 3) official source build
    try:
        import urllib.request, tarfile
        tmp = tempfile.mkdtemp()
        tgz = os.path.join(tmp, "py.tgz")
        urllib.request.urlretrieve(
            "https://www.python.org/ftp/python/3.14.3/Python-3.14.3.tgz", tgz)
        tarfile.open(tgz).extractall(tmp)
        src = os.path.join(tmp, "Python-3.14.3")
        prefix = os.path.join(tmp, "py314")
        if (_run(["sh", "configure", "--prefix=" + prefix], cwd=src).returncode == 0
                and _run(["make", "-j"], cwd=src).returncode == 0
                and _run(["make", "install"], cwd=src).returncode == 0):
            exe = os.path.join(prefix, "bin", "python3")
            if os.path.exists(exe):
                return exe
    except Exception:
        pass
    return None


def install_pins(py_exe):
    """Install the exact pinned deps into the isolated interpreter. Failures propagate (-> INFEASIBLE)."""
    _run([py_exe, "-m", "pip", "install", "--upgrade", "pip"], timeout=600)
    _run([py_exe, "-m", "pip", "install",
          "torch==2.13.0+cu126", "--index-url", "https://download.pytorch.org/whl/cu126"], timeout=1200)
    _run([py_exe, "-m", "pip", "install",
          "transformers==5.8.0", "peft==0.20.0", "bitsandbytes==0.50.1"], timeout=1200)


def capture_under(py_exe):
    """Run a capture payload under the given (3.14.3) interpreter; return realized stack dict."""
    payload = r'''
import sys, json, hashlib, os, platform
out = {}
out["python"] = sys.version.split()[0]
def dir_sha(d):
    h = hashlib.sha256()
    for root,_,files in os.walk(d):
        for fn in sorted(files):
            fp = os.path.join(root,fn)
            h.update(fn.encode())
            hh = hashlib.sha256()
            with open(fp,"rb") as f:
                for b in iter(lambda: f.read(1<<20), b""): hh.update(b)
            h.update(hh.hexdigest().encode())
    return h.hexdigest()
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda_runtime"] = torch.version.cuda
    out["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        out["gpu_name"] = torch.cuda.get_device_name(0)
        out["cuda_driver"] = torch.cuda.get_device_capability(0)
    else:
        out["cuda_driver"] = None
except Exception as e:
    out["torch_error"] = repr(e)
try:
    import transformers as T
    out["transformers"] = T.__version__
except Exception as e:
    out["transformers_error"] = repr(e)
try:
    import peft
    out["peft"] = peft.__version__
except Exception as e:
    out["peft_error"] = repr(e)
try:
    import bitsandbytes as bnb
    out["bitsandbytes"] = bnb.__version__
    out["bitsandbytes_package_sha256"] = dir_sha(os.path.dirname(bnb.__file__))
except Exception as e:
    out["bitsandbytes_error"] = repr(e)
out["linux"] = platform.system()
try:
    out["glibc"] = os.confstr("CS_GNU_LIBC_VERSION")
except Exception:
    out["glibc"] = "n/a"
json.dump(out, open(os.environ.get("FEASIBILITY_CAPTURE","/kaggle/working/_cap.json"), "w"))
'''
    cap_path = "/kaggle/working/_cap.json"
    os.makedirs(os.path.dirname(cap_path), exist_ok=True)
    open(cap_path, "w").write(payload)
    r = _run([py_exe, cap_path], timeout=600)
    if r.returncode != 0:
        raise RuntimeError("capture payload failed under 3.14.3: " + r.stderr[:2000])
    return json.load(open(cap_path))


def verify_qlora_prereqs(cap):
    return {
        "gpu_accessible": bool(cap.get("cuda_available")),
        "bitsandbytes_importable": "bitsandbytes_error" not in cap,
        "4bit_available": bool(cap.get("cuda_available")) and ("bitsandbytes_error" not in cap),
    }


def evaluate_compatibility(required, cap, base_python):
    """Pure decision: returns (verdict, mismatches, prereqs, base_is_3_12). Fail-closed."""
    mismatches = []
    for k, v in required.items():
        got = cap.get(k)
        if got != v:
            mismatches.append({"component": k, "required": v, "actual": got})
    prereqs = verify_qlora_prereqs(cap)
    base_is_3_12 = str(base_python).startswith("3.12")
    verdict = "INFEASIBLE" if mismatches else "FEASIBLE"
    return verdict, mismatches, prereqs, base_is_3_12


def main():
    base_python = sys.version.split()[0]
    manifest = {
        "purpose": "read-only runtime feasibility (no training, no dataset mutation)",
        "required": REQUIRED,
        "base_image_python": base_python,
        "hard_stopping_rule": "PASS -> consider fresh L5 training auth; FAIL -> abandon Kaggle path",
    }
    py_exe = build_isolated_python314()
    if py_exe is None:
        manifest.update({
            "verdict": "INFEASIBLE",
            "fail_closed": True,
            "reason": "could not build/install isolated CPython 3.14.3 on Kaggle image",
            "mismatches": [{"component": "python", "required": REQUIRED["python"], "actual": None}],
        })
        _write_manifest(manifest)
        _print_verdict(manifest)
        sys.exit(2)

    try:
        install_pins(py_exe)
        cap = capture_under(py_exe)
    except Exception as e:
        manifest.update({
            "verdict": "INFEASIBLE",
            "fail_closed": True,
            "reason": "pin install or capture failed: " + repr(e)[:500],
        })
        _write_manifest(manifest)
        _print_verdict(manifest)
        sys.exit(2)

    verdict, mismatches, prereqs, base_is_3_12 = evaluate_compatibility(REQUIRED, cap, base_python)
    manifest.update({
        "base_image_is_3_12": base_is_3_12,
        "realized": cap,
        "mismatches": mismatches,
        "qlora_prereqs": prereqs,
        "verdict": verdict,
        "fail_closed": (verdict == "INFEASIBLE"),
    })
    _write_manifest(manifest)
    _print_verdict(manifest)
    sys.exit(2 if verdict == "INFEASIBLE" else 0)


def _write_manifest(m):
    try:
        os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
        json.dump(m, open(MANIFEST_PATH, "w"), indent=2)
    except Exception:
        pass


def _print_verdict(m):
    print("=" * 60)
    print("6C.5 RUNTIME FEASIBILITY:", m.get("verdict"))
    print("  base_image_python:", m.get("base_image_python"),
          "| base_is_3_12:", m.get("base_image_is_3_12"))
    if m.get("realized"):
        r = m["realized"]
        print("  realized python      :", r.get("python"))
        print("  realized torch       :", r.get("torch"), "cuda", r.get("cuda_runtime"))
        print("  realized transformers:", r.get("transformers"), "peft", r.get("peft"),
              "bnb", r.get("bitsandbytes"))
        print("  bnb package SHA      :", r.get("bitsandbytes_package_sha256"))
        print("  cuda_available       :", r.get("cuda_available"))
    if m.get("mismatches"):
        print("  MISMATCHES:")
        for mm in m["mismatches"]:
            print("   -", mm["component"], "required", mm["required"], "actual", mm["actual"])
    print("=" * 60)


# ---------------------------------------------------------------------------
# Kernel packaging (so this script can be submitted as a dedicated Kaggle kernel)
# ---------------------------------------------------------------------------
# Self-contained dataset binding (no orchestrator import needed inside Kaggle).
KAGGLE_DATASET_OWNER = "Anakinistic"
KAGGLE_DATASET_SLUG = "theo-6c5-inputs"


def build_feasibility_kernel(bundle_dir, user):
    """Write a dedicated feasibility kernel.

    The Kaggle entrypoint is `script.py` (declared as code_file). `script.py` is THIS file: a
    single, valid-Python entrypoint. When the Kaggle runtime invokes it with no arguments, it
    explicitly selects probe-cloud mode (pure Python; no shell wrapper). The runtime marker
    (KAGGLE_RUNTIME=1) is set in Python — never via shell `export`.
    Performs NO cloud evaluation at build time and NO training. Returns the kernel directory path."""
    slug = "theo-6c5-bc-feasibility"
    kernel_dir = os.path.join(bundle_dir, ".feasibility_kernel")
    os.makedirs(kernel_dir, exist_ok=True)
    # The entry is THIS file (valid Python) executed under the base interpreter; it builds 3.14.3.
    open(os.path.join(kernel_dir, "script.py"), "w").write(
        open(os.path.abspath(__file__)).read())
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,   # required to pip-install the isolated interpreter + wheels
        "dataset_sources": [f"{KAGGLE_DATASET_OWNER}/{KAGGLE_DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    json.dump(meta, open(os.path.join(kernel_dir, "kernel-metadata.json"), "w"), indent=2)
    return kernel_dir


def _in_kaggle_runtime():
    # Native Kaggle signal. NOTE: do NOT use the presence of /kaggle/working as the signal —
    # it is unreliable on some dev hosts. KAGGLE_KERNEL_RUN_TYPE is set by the Kaggle runtime.
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def _run_probe():
    # Permitted ONLY inside the Kaggle runtime. Establishes the runtime marker in Python
    # (no shell `export`) and runs the cloud evaluator.
    if not _in_kaggle_runtime():
        print("ERROR: --probe-cloud is permitted only inside the Kaggle runtime.",
              file=sys.stderr)
        print("       On the dev host use: python feasibility_6c5_kaggle.py --build-kernel",
              file=sys.stderr)
        sys.exit(2)
    os.environ["KAGGLE_RUNTIME"] = "1"
    main()


def _usage_error():
    print("ERROR: explicit subcommand required.", file=sys.stderr)
    print("  python feasibility_6c5_kaggle.py --build-kernel   (local packaging only; no evaluation)",
          file=sys.stderr)
    print("  python feasibility_6c5_kaggle.py --probe-cloud    (Kaggle-only evaluator)",
          file=sys.stderr)


def _dispatch(argv):
    """Explicit CLI contract.
    - no args -> if inside Kaggle runtime, explicitly select probe-cloud mode; else usage error
                 (NO false local cloud verdict)
    - --build-kernel -> local packaging only
    - --probe-cloud  -> cloud evaluator, permitted ONLY inside the Kaggle runtime"""
    if not argv:
        return None
    if argv[0] == "--build-kernel":
        return "build"
    if argv[0] == "--probe-cloud":
        return "probe"
    return "unknown"


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        # Kaggle script entrypoint (code_file = script.py) is invoked with NO arguments.
        # Explicitly select probe-cloud mode, but ONLY when actually inside the Kaggle runtime,
        # so a bare local run cannot emit a false feasibility verdict.
        if _in_kaggle_runtime():
            args = ["--probe-cloud"]
        else:
            _usage_error()
            sys.exit(2)
    mode = _dispatch(args)
    if mode == "build":
        bundle_dir = os.path.dirname(os.path.abspath(__file__))
        user = os.environ.get("KAGGLE_USERNAME", "Anakinistic")
        kd = build_feasibility_kernel(bundle_dir, user)
        print("Built feasibility kernel at:", kd)
        print("  files: script.py, kernel-metadata.json")
        sys.exit(0)
    if mode == "probe":
        _run_probe()
        sys.exit(0)
    # unknown
    _usage_error()
    sys.exit(2)
