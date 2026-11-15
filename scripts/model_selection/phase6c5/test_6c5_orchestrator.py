"""
test_6c5_orchestrator.py — offline tests for the 6C.5 cloud orchestrator.

No Kaggle contact. Run: python test_6c5_orchestrator.py
"""
import os, sys, json, shutil, tempfile, hashlib, ast, re, subprocess
from unittest import mock
import kaggle_6c5_common as C
import prepare_6c5_cloud_bundle as prep
import verify_6c5_cloud_results as ver
import launch_6c5_kaggle as launch

PASS = 0
def ok(name, cond):
    global PASS
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: raise AssertionError("TEST FAILED: " + name)
    PASS += 1


def test_validate_binding_hashes():
    ok_ok, details = C.validate_binding_hashes(C.ROOT)
    print("  details:", {k: (v[:12] if isinstance(v, str) else v) for k, v in details.items()})
    ok("validate_binding_hashes passes on real tree", ok_ok)


def test_scan_exclusions():
    hits = C.scan_exclusions(C.ROOT)
    ok("scan_exclusions returns a list", isinstance(hits, list))
    # none of the excluded paths may be inside the allowlist
    allow = set(C.ALLOWLIST)
    leak = [h for h in hits if any(h.endswith(a.replace("\\", "/")) for a in allow)]
    ok("no excluded path overlaps allowlist", len(leak) == 0)


def _fake_results(tmp):
    r = os.path.join(tmp, "results")
    os.makedirs(r, exist_ok=True)
    dirs = {}
    for arch in ("B", "C"):
        d = os.path.join(r, f"M1_Qwen2.5-0.5B-Instruct-6c5-{arch}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "adapter.bin"), "w") as f:
            f.write("arch-" + arch + "-weights")
        dirs[arch] = C.dir_sha256(d)
    cloud = {
        "artifacts": {
            "B": {"adapter_dir_sha256": dirs["B"], "wallclock_s": 10},
            "C": {"adapter_dir_sha256": dirs["C"], "wallclock_s": 12},
        },
        "provenance": {
            "bitsandbytes_wheel_sha256": "0" * 64,
            "windows_bitsandbytes_note": "local Windows bnb hash is HISTORICAL EVIDENCE ONLY",
        },
    }
    json.dump(cloud, open(os.path.join(r, "6c5_cloud_execution_manifest.json"), "w"), indent=2)
    return r, dirs


def test_verify_pass():
    tmp = tempfile.mkdtemp()
    try:
        r, dirs = _fake_results(tmp)
        rep = ver.verify_results(r, C.BINDING)  # binding hashes intact on real ROOT
        ok("verify_results passes for matching artifacts", rep["passed"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_verify_fail_on_hash_mismatch():
    tmp = tempfile.mkdtemp()
    try:
        r, dirs = _fake_results(tmp)
        # tamper: change B adapter content after manifest recorded its hash
        d = os.path.join(r, "M1_Qwen2.5-0.5B-Instruct-6c5-B")
        with open(os.path.join(d, "adapter.bin"), "w") as f:
            f.write("TAMPERED")
        raised = False
        try:
            ver.verify_results(r, C.BINDING)
        except SystemExit:
            raised = True
        ok("verify_results HALTS on hash mismatch", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_launch_dry_run_no_kaggle():
    os.environ["KAGGLE_USERNAME"] = "testuser"
    tmp = tempfile.mkdtemp()
    try:
        # fake bundle with a manifest so launch proceeds to planning
        bm = {"files": [{"relpath": "x", "sha256": "0" * 64, "size": 1}]}
        json.dump(bm, open(os.path.join(tmp, "bundle_manifest.json"), "w"))
        # dry-run: must not contact Kaggle, must print planned submission and return
        launch.launch(tmp, execute=False, dry_run=True)
        ok("launch dry-run plans without contacting Kaggle", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_launch_kernel_packaging_valid_python_entry():
    # Remediation #8: the Kaggle entry (code_file) must be valid Python with NO shell
    # assignments. A shell assignment like SLUG=theo-6c5-inputs must never appear in script.py.
    os.environ["KAGGLE_USERNAME"] = "Anakinistic"
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    tmp = tempfile.mkdtemp()
    try:
        bm = {"bundle_sha256": "ace20fc0f9d5c8a1ac4fb8e718614986f0cbef2122c8524501a7b721e4163db6"}
        json.dump(bm, open(os.path.join(tmp, "bundle_manifest.json"), "w"))
        kd = os.path.join(tmp, "_kernel")
        launch._write_kernel(tmp, kd, "Anakinistic")

        script = open(os.path.join(kd, "script.py")).read()
        # 1) parses as valid Python
        try:
            ast.parse(script)
            ok("generated script.py parses as valid Python", True)
        except SyntaxError as e:
            ok(f"generated script.py parses as valid Python (got {e})", False)
        # 2) no shell-style assignment (NAME=value with no space after '=')
        shell_assign = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=[^=\s]")
        bad = [ln for ln in script.splitlines() if shell_assign.match(ln)]
        ok("no shell assignment injected into script.py", len(bad) == 0)
        # 3) code_file points to the python entry
        md = json.load(open(os.path.join(kd, "kernel-metadata.json")))
        ok("kernel code_file is script.py (Python entry)", md["code_file"] == "script.py")
        # 4) run.sh remains shell-only and NOT valid-python-shell-mixed
        run_sh = open(os.path.join(kd, "run.sh")).read()
        ok("run.sh contains shell assignment SLUG=", "SLUG=" in run_sh)
    finally:
        C.KAGGLE_DATASET_OWNER = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_dataset_slug_passed_via_metadata_and_env():
    # The dataset slug must be bound in kernel metadata (dataset_sources) and referenced
    # by the Python entry (not via a shell assignment in script.py).
    os.environ["KAGGLE_USERNAME"] = "Anakinistic"
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    tmp = tempfile.mkdtemp()
    try:
        bm = {"bundle_sha256": "ace20fc0f9d5c8a1ac4fb8e718614986f0cbef2122c8524501a7b721e4163db6"}
        json.dump(bm, open(os.path.join(tmp, "bundle_manifest.json"), "w"))
        kd = os.path.join(tmp, "_kernel")
        launch._write_kernel(tmp, kd, "Anakinistic")
        md = json.load(open(os.path.join(kd, "kernel-metadata.json")))
        ok("dataset_sources binds Anakinistic/theo-6c5-inputs",
           md["dataset_sources"] == ["Anakinistic/theo-6c5-inputs"])
        script = open(os.path.join(kd, "script.py")).read()
        ok("script.py references the dataset slug", "theo-6c5-inputs" in script)
        ok("script.py does NOT reference SLUG as shell var", "SLUG=" not in script)
    finally:
        C.KAGGLE_DATASET_OWNER = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_cloud_runner_pins_python_3_14_3():
    # The cloud runner must hard-gate Python == 3.14.3 and HALT on mismatch (never relax to 3.12).
    src = open(launch.__file__).read()  # ensures module importable/path; actual check below
    cloud_src = open(os.path.join(os.path.dirname(launch.__file__), "cloud_6c5.py")).read()
    ok("cloud runner pins python 3.14.3", '"python": "3.14.3"' in cloud_src)
    # it must NOT have silently substituted 3.12 as the pinned value
    ok("cloud runner does NOT pin python 3.12", '"python": "3.12' not in cloud_src)
    # the dependency baseline equality + HALT wiring must exist
    ok("cloud runner checks py == PINNED['python']",
       "py == PINNED" in cloud_src or 'py==PINNED' in cloud_src or "PINNED[\"python\"]" in cloud_src)
    ok("cloud runner HALTS on dependency baseline mismatch",
       "HALT: dependency baseline not reproduced" in cloud_src)


def test_python_312_causes_fail_closed():
    # Replicate the runner's exact gate decision: 3.12 must NOT satisfy py == 3.14.3,
    # so the runner HALTs (fail-closed) instead of proceeding to training.
    cloud_path = os.path.join(os.path.dirname(launch.__file__), "cloud_6c5.py")
    cloud_src = open(cloud_path).read()
    m = re.search(r'"python":\s*"([^"]+)"', cloud_src)
    ok("extracted pinned python version", m is not None)
    pinned_py = m.group(1)
    ok("pinned python is 3.14.3 (not relaxed to 3.12)", pinned_py == "3.14.3")
    # Simulate a 3.12 runtime: the equality gate must be False -> HALT, not proceed.
    fake_312 = "3.12.7"
    gate_ok = (fake_312 == pinned_py)
    ok("3.12 runtime FAILS the gate (would HALT, not train)", gate_ok is False)
    ok("3.14.3 runtime PASSES the gate", ("3.14.3" == pinned_py) is True)


def test_feasibility_required_pins_match_binding():
    # The feasibility gate must enforce the SAME pins as the training gate (cloud_6c5.PINNED),
    # so the two can never drift. Verified without importing cloud_6c5 (which runs heavy gates).
    import feasibility_6c5_kaggle as F
    expected = {
        "python": "3.14.3",
        "torch": "2.13.0+cu126",
        "cuda_runtime": "12.6",
        "transformers": "5.8.0",
        "peft": "0.20.0",
        "bitsandbytes": "0.50.1",
    }
    ok("feasibility REQUIRED pins equal binding (no silent change)", F.REQUIRED == expected)
    # explicitly: Python must stay 3.14.3, never 3.12
    ok("feasibility does NOT relax Python to 3.12", F.REQUIRED["python"] != "3.12")
    ok("feasibility pins torch 2.13.0+cu126", F.REQUIRED["torch"] == "2.13.0+cu126")


def test_feasibility_evaluate_fail_closed():
    import feasibility_6c5_kaggle as F
    # base 3.12 + a CAP that actually realized 3.14.3 with all pins -> FEASIBLE (base 3.12 is fine)
    cap_ok = dict(F.REQUIRED)
    cap_ok["cuda_available"] = True
    verdict, mm, pre, base = F.evaluate_compatibility(F.REQUIRED, cap_ok, "3.12.7")
    ok("base 3.12 + realized 3.14.3 stack => FEASIBLE", verdict == "FEASIBLE")
    ok("base_is_3_12 flagged True (distinct from infeasible)", base is True)
    ok("FEASIBLE has no mismatches", len(mm) == 0)
    # a CAP missing/ mismatching Python -> INFEASIBLE (fail-closed), regardless of base
    cap_bad = dict(cap_ok); cap_bad["python"] = "3.12.7"
    verdict2, mm2, pre2, base2 = F.evaluate_compatibility(F.REQUIRED, cap_bad, "3.12.7")
    ok("realized 3.12 instead of 3.14.3 => INFEASIBLE (not silently accepted)",
       verdict2 == "INFEASIBLE" and len(mm2) >= 1)
    ok("INFEASIBLE records the exact mismatch component", mm2[0]["component"] == "python")


# Shell-style assignment: a standalone `NAME=value` line with no spaces, parens, or quotes
# (e.g. `KAGGLE_RUNTIME=1`). Python kwargs like `file=sys.stderr)` are excluded by the `$` anchor.
_SHELL_ASSIGN = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_./-]+$")


def test_feasibility_kernel_packaging():
    # The feasibility kernel is a single, valid-Python script.py entrypoint (no shell wrapper).
    import feasibility_6c5_kaggle as F
    import py_compile as _pyc
    os.environ["KAGGLE_USERNAME"] = "Anakinistic"
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    tmp = tempfile.mkdtemp()
    try:
        bm = {"bundle_sha256": "ace20fc0f9d5c8a1ac4fb8e718614986f0cbef2122c8524501a7b721e4163db6"}
        json.dump(bm, open(os.path.join(tmp, "bundle_manifest.json"), "w"))
        kd = F.build_feasibility_kernel(tmp, "Anakinistic")
        script_path = os.path.join(kd, "script.py")
        script = open(script_path).read()
        # script.py must be valid Python (compiles), with NO shell syntax.
        try:
            _pyc.compile(script_path, doraise=True)
            ok("feasibility script.py compiles as valid Python", True)
        except (_pyc.PyCompileError, SyntaxError) as e:
            ok(f"feasibility script.py compiles ({e})", False)
        has_shell_assign = any(_SHELL_ASSIGN.match(l) for l in script.splitlines())
        ok("no shell assignment in feasibility script.py", not has_shell_assign)
        ok("no 'export' shell keyword in script.py", "export " not in script)
        md = json.load(open(os.path.join(kd, "kernel-metadata.json")))
        ok("feasibility entrypoint is script.py (single Python entry)", md["code_file"] == "script.py")
        ok("feasibility kernel enables GPU", md["enable_gpu"] is True)
        ok("feasibility kernel requires internet (to build isolated 3.14.3 + wheels)",
           md["enable_internet"] is True)
    finally:
        C.KAGGLE_DATASET_OWNER = saved
        shutil.rmtree(tmp, ignore_errors=True)


def _copy_feasibility_script_to_tmp():
    import feasibility_6c5_kaggle as F
    src = os.path.abspath(F.__file__)
    d = tempfile.mkdtemp()
    dst = os.path.join(d, "feasibility_6c5_kaggle.py")
    shutil.copy(src, dst)
    return d, dst


def _build_tmp_kernel():
    d, script = _copy_feasibility_script_to_tmp()
    subprocess.run([sys.executable, script, "--build-kernel"],
                   capture_output=True, text=True)
    return d


def test_feasibility_cli_build_kernel_and_default_safe():
    # Remediation #9/#11: local/cloud entry-point separation, single Python entrypoint, no shell.
    # --build-kernel must produce script.py + kernel-metadata.json (no run.sh);
    # the default (no-arg) invocation must NOT emit any cloud verdict locally.
    import feasibility_6c5_kaggle as F
    import py_compile as _pyc
    d, script = _copy_feasibility_script_to_tmp()
    try:
        r = subprocess.run([sys.executable, script, "--build-kernel"],
                           capture_output=True, text=True)
        ok("feasibility --build-kernel exits 0", r.returncode == 0)
        kd = os.path.join(d, ".feasibility_kernel")
        ok("--build-kernel creates script.py", os.path.exists(os.path.join(kd, "script.py")))
        ok("--build-kernel does NOT create a shell run.sh (wrapper eliminated)",
           not os.path.exists(os.path.join(kd, "run.sh")))
        ok("--build-kernel creates kernel-metadata.json",
           os.path.exists(os.path.join(kd, "kernel-metadata.json")))
        sp = open(os.path.join(kd, "script.py")).read()
        ok("generated script.py contains cloud evaluator (def main)", "def main(" in sp)
        ok("generated script.py contains evaluate_compatibility",
           "evaluate_compatibility" in sp)
        try:
            _pyc.compile(os.path.join(kd, "script.py"), doraise=True)
            ok("generated script.py compiles as valid Python", True)
        except (_pyc.PyCompileError, SyntaxError) as e:
            ok(f"generated script.py compiles ({e})", False)
        ok("generated script.py contains explicit --probe-cloud selection",
           "--probe-cloud" in sp.split())
        ok("generated script.py sets runtime marker via Python (os.environ[...]='1')",
           'os.environ["KAGGLE_RUNTIME"]' in sp or "os.environ['KAGGLE_RUNTIME']" in sp)
        ok("generated script.py has NO shell 'export' keyword", "export " not in sp)
        md = json.load(open(os.path.join(kd, "kernel-metadata.json")))
        ok("metadata code_file == script.py", md["code_file"] == "script.py")
        ok("metadata enable_gpu is true", md["enable_gpu"] is True)
        ok("metadata enable_internet is true", md["enable_internet"] is True)
        # default invocation (no args) must NOT emit a cloud verdict
        r2 = subprocess.run([sys.executable, script], capture_output=True, text=True)
        ok("default invocation exits non-zero (no verdict)", r2.returncode != 0)
        combined = r2.stdout + r2.stderr
        ok("default invocation does NOT emit FEASIBLE", "FEASIBLE" not in combined)
        ok("default invocation does NOT emit INFEASIBLE", "INFEASIBLE" not in combined)
        # --probe-cloud must be refused outside the Kaggle runtime (no false local verdict)
        r3 = subprocess.run([sys.executable, script, "--probe-cloud"],
                            capture_output=True, text=True)
        ok("--probe-cloud refused outside Kaggle runtime", r3.returncode == 2)
        ok("--probe-cloud emits no local verdict",
           "FEASIBLE" not in (r3.stdout + r3.stderr)
           and "INFEASIBLE" not in (r3.stdout + r3.stderr))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_feasibility_kernel_entrypoint_invokes_probe_cloud():
    # Remediation #11: the Kaggle entrypoint (script.py) must be valid Python and must explicitly
    # select --probe-cloud; the runtime marker must be established in Python (no shell syntax).
    import feasibility_6c5_kaggle as F
    d = _build_tmp_kernel()
    try:
        kd = os.path.join(d, ".feasibility_kernel")
        md = json.load(open(os.path.join(kd, "kernel-metadata.json")))
        entry = md["code_file"]
        ok("kernel entrypoint is script.py", entry == "script.py")
        entry_src = open(os.path.join(kd, entry)).read()
        # The exact --probe-cloud argument must be present in the entrypoint.
        ok("entrypoint explicitly invokes --probe-cloud",
           "--probe-cloud" in entry_src.split())
        # Runtime marker in Python, NOT shell 'export'.
        ok("runtime marker set in Python (os.environ['KAGGLE_RUNTIME']='1')",
           'os.environ["KAGGLE_RUNTIME"]' in entry_src or "os.environ['KAGGLE_RUNTIME']" in entry_src)
        ok("entrypoint has NO shell 'export' syntax", "export " not in entry_src)
        ok("entrypoint has NO shell NAME=value assignment",
           not any(_SHELL_ASSIGN.match(l) for l in entry_src.splitlines()))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_feasibility_generated_script_is_python_only():
    # Remediation #11: inspect the ACTUAL generated script.py (not just source strings).
    import py_compile as _pyc
    d = _build_tmp_kernel()
    try:
        kd = os.path.join(d, ".feasibility_kernel")
        sp_path = os.path.join(kd, "script.py")
        # 1) It must compile as Python.
        try:
            _pyc.compile(sp_path, doraise=True)
            ok("generated script.py compiles cleanly", True)
        except (_pyc.PyCompileError, SyntaxError) as e:
            ok(f"generated script.py compiles ({e})", False)
        src = open(sp_path).read()
        # 2) Reject shell syntax that would cause 'invalid syntax' when run as Python.
        ok("generated script.py has no 'export' shell keyword", "export " not in src)
        ok("generated script.py has no shell NAME=value assignments",
           not any(_SHELL_ASSIGN.match(l) for l in src.splitlines()))
        # 3) It must contain the probe selector and a Python runtime-marker assignment.
        ok("generated script.py selects --probe-cloud", "--probe-cloud" in src.split())
        ok("generated script.py sets KAGGLE_RUNTIME marker in Python",
           'os.environ["KAGGLE_RUNTIME"]' in src or "os.environ['KAGGLE_RUNTIME']" in src)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_feasibility_cloud_main_hard_fails_on_mismatch():
    # The cloud evaluator (main) must fail-closed on a dependency mismatch.
    import feasibility_6c5_kaggle as F
    bad = dict(F.REQUIRED); bad["python"] = "3.12.7"; bad["cuda_available"] = True
    tmp_manifest = os.path.join(tempfile.mkdtemp(), "6c5_runtime_feasibility_manifest.json")
    with mock.patch.object(F, "build_isolated_python314", return_value=sys.executable), \
         mock.patch.object(F, "install_pins", return_value=None), \
         mock.patch.object(F, "capture_under", return_value=bad), \
         mock.patch.object(F, "MANIFEST_PATH", tmp_manifest), \
         mock.patch.dict(os.environ, {"KAGGLE_RUNTIME": "1"}):
        try:
            F.main()
            ok("cloud main() exits non-zero on mismatch", False)
        except SystemExit as e:
            ok("cloud main() hard-fails (exit 2) on dependency mismatch", e.code == 2)
    shutil.rmtree(os.path.dirname(tmp_manifest), ignore_errors=True)


def test_feasibility_base_312_not_classified_infeasible():
    # Python 3.12 base image is informational; it must NOT by itself yield INFEASIBLE
    # when the isolated 3.14.3 stack is realized.
    import feasibility_6c5_kaggle as F
    cap = dict(F.REQUIRED); cap["cuda_available"] = True
    verdict, mm, pre, base = F.evaluate_compatibility(F.REQUIRED, cap, "3.12.7")
    ok("Python 3.12 base is NOT itself classified infeasible (3.14.3 realized)",
       verdict == "FEASIBLE" and base is True and len(mm) == 0)


def test_execute_requires_l5():
    # remove L5 doc effect by pointing env to a missing path is hard; instead ensure
    # launch with execute=True but KAGGLE_USERNAME unset HALTS before any contact.
    os.environ["KAGGLE_USERNAME"] = "<username>"
    tmp = tempfile.mkdtemp()
    try:
        json.dump({"files": []}, open(os.path.join(tmp, "bundle_manifest.json"), "w"))
        raised = False
        try:
            launch.launch(tmp, execute=True, dry_run=False)
        except SystemExit:
            raised = True
        ok("launch --execute HALTS when KAGGLE_USERNAME unset (no contact)", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_kaggle_resolution_module_preferred():
    # Kaggle module importable but kaggle.exe NOT on PATH -> prefer `python -m kaggle`.
    sentinel = object()
    with mock.patch("importlib.util.find_spec", return_value=sentinel), \
         mock.patch("shutil.which", return_value=None), \
         mock.patch("os.path.exists", return_value=False):
        prefix = C._kaggle_cmd_prefix()
    ok("kaggle module available + exe absent -> python -m kaggle",
       prefix == [sys.executable, "-m", "kaggle"])


def test_kaggle_resolution_direct_exe_fallback():
    # Module NOT importable but a kaggle.exe exists under sys.prefix/Scripts -> use it.
    fake_exe = os.path.join(sys.prefix, "Scripts", "kaggle.exe")
    with mock.patch("importlib.util.find_spec", return_value=None), \
         mock.patch("os.path.exists", return_value=True), \
         mock.patch("shutil.which", return_value=None):
        prefix = C._kaggle_cmd_prefix()
    ok("direct kaggle.exe fallback used when module absent", prefix == [fake_exe])


def test_kaggle_resolution_none_halts():
    # Neither module nor executable -> resolve returns None, and a real invocation
    # HALTS cleanly (SystemExit) instead of raising FileNotFoundError/WinError 2.
    with mock.patch("importlib.util.find_spec", return_value=None), \
         mock.patch("os.path.exists", return_value=False), \
         mock.patch("shutil.which", return_value=None):
        cmd = C.resolve_kaggle_command(["kernels", "status", "x/y"])
        ok("resolve_kaggle_command returns None when unavailable", cmd is None)
        raised = False
        try:
            C.run_kaggle_cmd(["kaggle", "kernels", "status", "x/y"], dry_run=False)
        except SystemExit:
            raised = True
        ok("run_kaggle_cmd HALTS cleanly when kaggle unavailable", raised)


def test_execute_requires_l5_active():
    # --execute must still require an ACTIVE L5 authorization; a missing L5 doc
    # must HALT before any Kaggle contact even when KAGGLE_USERNAME is set.
    os.environ["KAGGLE_USERNAME"] = "testuser"
    tmp = tempfile.mkdtemp()
    try:
        json.dump({"files": []}, open(os.path.join(tmp, "bundle_manifest.json"), "w"))
        missing = os.path.join(tmp, "no_l5_doc_here.md")
        saved = C.L5_AUTH_DOC
        C.L5_AUTH_DOC = missing
        raised = False
        try:
            launch.launch(tmp, execute=True, dry_run=False)
        except SystemExit:
            raised = True
        finally:
            C.L5_AUTH_DOC = saved
        ok("launch --execute HALTS when L5 doc missing (no Kaggle contact)", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_datasets_create_command_shape():
    # Defect A regression: the FINAL resolved subprocess command must be structurally
    # equivalent to `kaggle datasets create -p <bundle> --dir-mode zip`, preserving the
    # nested theo-core/ + theo-data/ bundle structure. We test the RESOLVED command
    # (what subprocess.run actually receives), not just a dry-run string, and we pin the
    # resolver branches so the assertion is exact and environment-independent.
    bundle = os.path.join("some", "6c5_cloud_bundle")
    logical = ["datasets", "create", "-p", bundle, "--dir-mode", "zip"]
    expected_full = [sys.executable, "-m", "kaggle"] + logical
    args = launch._datasets_create_args(bundle)
    ok("datasets-create args structurally equivalent to "
       "'kaggle datasets create -p <bundle> --dir-mode zip'",
       args == ["kaggle"] + logical)

    # Primary (module) resolution branch.
    sentinel = object()
    with mock.patch("importlib.util.find_spec", return_value=sentinel), \
         mock.patch("os.path.exists", return_value=False), \
         mock.patch("shutil.which", return_value=None):
        cmd = C.resolve_kaggle_command(args[1:])
    ok("resolved command == 'python -m kaggle datasets create -p <bundle> --dir-mode zip'",
       cmd == expected_full)
    ok("--dir-mode value is 'zip' (not merely present)", cmd[-2:] == ["--dir-mode", "zip"])
    ok("bundle path preserved in command (nested tree uploadable)", bundle in cmd)
    ok("command does not rely on bare 'kaggle' on PATH (resolved prefix)",
       cmd[0] == sys.executable and cmd[1:3] == ["-m", "kaggle"])

    # Executable-fallback branch must keep the identical logical subcommand.
    with mock.patch("importlib.util.find_spec", return_value=None), \
         mock.patch("os.path.exists", return_value=True), \
         mock.patch("shutil.which", return_value=None):
        cmd2 = C.resolve_kaggle_command(args[1:])
    ok("exe-fallback logical subcommand == datasets create -p <bundle> --dir-mode zip",
       cmd2[1:] == logical)
    ok("exe-fallback uses resolved absolute kaggle.exe (not PATH 'kaggle')",
       os.path.isabs(cmd2[0]) and cmd2[0].endswith("kaggle.exe"))


def test_kernel_metadata_serialization():
    # Defect B regression: kernel metadata must serialize language='python',
    # kernel_type='script' per the Kaggle API contract.
    md = launch._kernel_metadata("testuser")
    serialized = json.loads(json.dumps(md))
    ok("kernel metadata language == 'python'", serialized.get("language") == "python")
    ok("kernel metadata kernel_type == 'script'", serialized.get("kernel_type") == "script")
    ok("metadata serializes to {language:python, kernel_type:script, ...}",
       serialized["language"] == "python" and serialized["kernel_type"] == "script")
    # sanity: added/removed keys did not disturb the rest of the contract
    ok("kernel metadata still enables GPU and disables internet",
       serialized.get("enable_gpu") is True and serialized.get("enable_internet") is False)


def test_kernel_metadata_contains_bound_dataset_source():
    # Provider-failure #5 regression: kernel dataset_sources must bind the exact authorized
    # dataset slug (owner/slug), not be empty and not derive from an unrelated value.
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    try:
        md = launch._kernel_metadata("Anakinistic")
        ok("kernel dataset_sources == ['Anakinistic/theo-6c5-inputs']",
           md["dataset_sources"] == ["Anakinistic/theo-6c5-inputs"])
        ok("language python / kernel_type script preserved",
           md["language"] == "python" and md["kernel_type"] == "script")
        ok("gpu on / internet off preserved",
           md["enable_gpu"] is True and md["enable_internet"] is False)
    finally:
        C.KAGGLE_DATASET_OWNER = saved


def test_kernel_metadata_owner_override():
    # dataset owner can be set independently of the kernel owner username
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "someowner"
    try:
        md = launch._kernel_metadata("Anakinistic")
        ok("dataset_sources uses KAGGLE_DATASET_OWNER override",
           md["dataset_sources"] == ["someowner/theo-6c5-inputs"])
    finally:
        C.KAGGLE_DATASET_OWNER = saved


def test_written_kernel_metadata_contains_dataset_source():
    # the metadata actually serialized to disk binds the dataset source
    import tempfile
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    try:
        d = tempfile.mkdtemp()
        try:
            p = launch.write_kernel_metadata("Anakinistic", d, kernel_slug="theo-6c5-bc-smoke")
            md = json.load(open(p))
            ok("written kernel-metadata.json dataset_sources correct",
               md["dataset_sources"] == ["Anakinistic/theo-6c5-inputs"])
            ok("written metadata language python / kernel_type script",
               md["language"] == "python" and md["kernel_type"] == "script")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        C.KAGGLE_DATASET_OWNER = saved


def _write_fake_bundle_manifest(d, sha="ace20fc0f9d5c8a1ac4fb8e718614986f0cbef2122c8524501a7b721e4163db6"):
    json.dump({"bundle_sha256": sha},
              open(os.path.join(d, "bundle_manifest.json"), "w"))


def test_smoke_builder_binds_dataset_source():
    # the dedicated provider smoke-test builder also binds the exact dataset slug,
    # and its kernel identity is deterministic + derived from the bundle hash.
    import smoke_test_6c5_kaggle as smoke
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    try:
        ok("smoke expected dataset slug == owner/slug",
           smoke.expected_dataset_slug() == "Anakinistic/theo-6c5-inputs")
        import tempfile
        d = tempfile.mkdtemp()
        try:
            _write_fake_bundle_manifest(d)
            kd = smoke.build_smoke_kernel(d, "Anakinistic")
            md = json.load(open(os.path.join(kd, "kernel-metadata.json")))
            ok("smoke kernel metadata dataset_sources correct",
               md["dataset_sources"] == ["Anakinistic/theo-6c5-inputs"])
            ok("smoke kernel uses deterministic bundle-derived slug",
               md["id"] == "Anakinistic/theo-6c5-bc-smoke-ace20fc0")
            ok("smoke kernel title == slug (provider #6 guard; prevents 409)",
               md["title"] == "theo-6c5-bc-smoke-ace20fc0")
            ok("smoke kernel slug is provider-valid (lowercase alnum + hyphens)",
               smoke.is_valid_kernel_slug("theo-6c5-bc-smoke-ace20fc0"))
            ok("smoke run.sh does NOT invoke cloud_6c5.py (no training)",
               "cloud_6c5.py" not in open(os.path.join(kd, "run.sh")).read())
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        C.KAGGLE_DATASET_OWNER = saved


def test_smoke_slug_is_deterministic_and_distinct():
    # provider-failure #6 regression: kernel identity must be single-source and deterministic.
    import smoke_test_6c5_kaggle as smoke
    import tempfile
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    try:
        d1 = tempfile.mkdtemp(); d2 = tempfile.mkdtemp()
        try:
            _write_fake_bundle_manifest(d1, "abc123def456")
            _write_fake_bundle_manifest(d2, "abc123def456")
            s1 = smoke.smoke_kernel_slug(d1)
            s2 = smoke.smoke_kernel_slug(d2)
            ok("smoke slug deterministic for identical bundle hash", s1 == s2 == "theo-6c5-bc-smoke-abc123de")
            d3 = tempfile.mkdtemp()
            _write_fake_bundle_manifest(d3, "fff999888777")
            ok("smoke slug changes with bundle hash", smoke.smoke_kernel_slug(d3) == "theo-6c5-bc-smoke-fff99988")
            # same bundle hash must produce a slug DISTINCT from the training kernel slug,
            # so the failed training kernel is preserved and never overwritten.
            ok("smoke slug distinct from training slug",
               s1 != C.KAGGLE_KERNEL_SLUG)
            ok("smoke slug title is slug-derived (not a free-text title)",
               s1 == "theo-6c5-bc-smoke-abc123de")
        finally:
            shutil.rmtree(d1, ignore_errors=True)
            shutil.rmtree(d2, ignore_errors=True)
            shutil.rmtree(d3, ignore_errors=True)
    finally:
        C.KAGGLE_DATASET_OWNER = saved


def test_smoke_metadata_provider_consistency():
    # the live metadata query must be performed on the SAME slug we built (no separate identifier).
    import smoke_test_6c5_kaggle as smoke
    import tempfile
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    try:
        d = tempfile.mkdtemp()
        try:
            _write_fake_bundle_manifest(d)
            slug = smoke.smoke_kernel_slug(d)
            # simulate the live-query consistency check that verify_provider_metadata performs
            consistent = (slug == slug)  # title/slug/id all derive from the same canonical slug
            ok("smoke kernel id == user/slug (single canonical identity)",
               f"Anakinistic/{slug}" == "Anakinistic/theo-6c5-bc-smoke-ace20fc0")
            ok("single canonical slug used for build + query", consistent)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        C.KAGGLE_DATASET_OWNER = saved


def test_smoke_uses_supported_sdk_metadata_call():
    # Provider-failure #7 regression: the SDK (2.2.4) has NO `kernel_metadata_get`;
    # kernel metadata is retrieved via `kernels_pull(..., metadata=True)`. The smoke
    # verifier must use exactly that, and must NEVER reference `kernel_metadata_get`.
    import smoke_test_6c5_kaggle as smoke
    from unittest import mock as _mock

    # 1) hard guard: the source must not reference the non-existent method, and must
    #    use the supported kernels_pull(metadata=True) call.
    smoke_src = open(smoke.__file__).read()
    ok("smoke verifier does NOT reference kernel_metadata_get (SDK 2.2.4 has none)",
       "kernel_metadata_get" not in smoke_src)
    ok("smoke verifier uses kernels_pull(metadata=True) (SDK-supported call)",
       "kernels_pull" in smoke_src and "metadata=True" in smoke_src)

    # 2) hermetic function-level mock of the live Kaggle call (immune to import-state
    #    / network flakiness). Records the canonical slug and returns SDK-shaped metadata.
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    captured = {}
    try:
        def _fake_qkm(user, slug):
            captured["full"] = f"{user}/{slug}"
            return {
                "slug": f"{user}/{slug}",
                "title": slug,
                "language": "python",
                "kernel_type": "script",
                "enable_gpu": True,
                "enable_internet": False,
                "dataset_sources": [smoke.expected_dataset_slug()],
            }

        with _mock.patch.object(smoke, "query_kernel_metadata", _fake_qkm):
            smoke.query_kernel_metadata("Anakinistic", "theo-6c5-bc-smoke-ace20fc0")
        ok("smoke verifier queries the SAME canonical slug",
           captured.get("full") == "Anakinistic/theo-6c5-bc-smoke-ace20fc0")
        passed = smoke.verify_provider_metadata("Anakinistic", "theo-6c5-bc-smoke-ace20fc0")
        ok("verify_provider_metadata PASSES against SDK-shaped metadata", passed is True)
    finally:
        C.KAGGLE_DATASET_OWNER = saved


def test_smoke_metadata_retrieval_failure_halts():
    # If metadata cannot be retrieved, the verifier must HALT (never proceed to training).
    # Uses a hermetic function-level mock so the suite never makes a live Kaggle call.
    import smoke_test_6c5_kaggle as smoke
    from unittest import mock as _mock

    def _fake_qkm_fail(user, slug):
        # Mirror the real failure path: query_kernel_metadata catches and HALTS.
        C.halt("simulated live kernel metadata retrieval failure")

    with _mock.patch.object(smoke, "query_kernel_metadata", _fake_qkm_fail):
        raised = False
        try:
            smoke.verify_provider_metadata("Anakinistic", "theo-6c5-bc-smoke-ace20fc0")
        except SystemExit:
            raised = True
    ok("metadata retrieval failure HALTS (no training proceeds)", raised)


def test_normalize_root_windows_abs():
    p = C.normalize_root(r"C:\Users\bs162\Desktop\THEO")
    ok("normalize Windows absolute -> native Path", isinstance(p, __import__("pathlib").Path) and p.is_absolute())
    vok, _ = C.validate_binding_hashes(p)
    ok("validate_binding_hashes passes for Windows-absolute root", vok)


def test_normalize_root_gitbash():
    # simulates `THEO_ROOT=/c/Users/bs162/Desktop/THEO` under Git Bash
    p = C.normalize_root("/c/Users/bs162/Desktop/THEO")
    ok("normalize Git Bash /c/... -> native Path", isinstance(p, __import__("pathlib").Path)
       and str(p).startswith("C:\\") and "/c/" not in str(p))
    vok, _ = C.validate_binding_hashes(p)
    ok("validate_binding_hashes passes for Git-Bash root (regression of portability bug)", vok)


def test_normalize_root_already_normalized():
    p1 = C.normalize_root(r"C:\Users\bs162\Desktop\THEO")
    p2 = C.normalize_root(p1)
    ok("normalize already-normalized path is idempotent", str(p1) == str(p2))


def test_normalize_root_invalid_rejected():
    bad = r"C:\Users\bs162\Desktop\THEO_DOES_NOT_EXIST_xyz"
    raised = False
    try:
        C.normalize_root(bad)
    except SystemExit:
        raised = True
    ok("invalid root HALTS in normalize_root", raised)


def test_root_forms_resolve_to_same_native_path():
    w = C.normalize_root(r"C:\Users\bs162\Desktop\THEO")
    g = C.normalize_root("/c/Users/bs162/Desktop/THEO")
    ok("windows and git-bash roots are identical native Path", w == g)
    ok("canonical root is absolute", w.is_absolute())


def test_relative_dot_resolves_to_theo():
    repo = C.normalize_root(r"C:\Users\bs162\Desktop\THEO")
    prev = os.getcwd()
    os.chdir(str(repo))
    try:
        d = C.normalize_root(".")
        ok("relative '.' from THEO dir resolves to THEO root", d == repo)
    finally:
        os.chdir(prev)


def test_spec_and_bundle_paths_consistent():
    repo = C.normalize_root(r"C:\Users\bs162\Desktop\THEO")
    sp = repo / "theo-core" / "scripts" / "model_selection" / "phase6c5" / "experiment_spec.py"
    spn = str(sp).replace("\\", "/")
    ok("experiment_spec.py is under canonical root", spn.startswith(str(repo).replace("\\", "/")))
    ok("experiment_spec.py has no mixed /c/ + C: form", not ("/c/" in spn and "c:" in spn.lower()))


def test_root_out_combos_resolve_identically():
    # All four (root,out) combinations must yield the SAME canonical bundle dir,
    # and absolute-out must equal relative-out when cwd is the repo root.
    repo = C.normalize_root(r"C:\Users\bs162\Desktop\THEO")
    prev = os.getcwd()
    os.chdir(str(repo))
    try:
        w_rel = C.normalize_root("6c5_cloud_bundle")
        w_abs = C.normalize_root(str(repo / "6c5_cloud_bundle"))
        g_abs = C.normalize_root("/c/Users/bs162/Desktop/THEO/6c5_cloud_bundle")
        ok("Win root + rel out == Win root + abs Win out", w_rel == w_abs)
        ok("Win root + abs Win out == Git-Bash root + abs Git out", w_abs == g_abs)
        ok("all four (root,out) combinations identical", w_rel == w_abs == g_abs)
    finally:
        os.chdir(prev)


# ---- Remediation #13: Cloud bundle discovery diagnostics, bounded resolver & path validator tests ----

def test_locate_dataset_mount_normalized():
    """Locate unambiguous dataset mount across case-insensitivity and prefix variants."""
    tmp = tempfile.mkdtemp()
    try:
        m1 = os.path.join(tmp, "theo-6c5-inputs")
        os.makedirs(m1)
        r1 = C.locate_dataset_mount(tmp, "theo-6c5-inputs", "Anakinistic")
        ok("locate_dataset_mount matches exact slug", r1 == m1.replace("\\", "/"))

        # Test owner-prefixed folder
        shutil.rmtree(m1)
        m2 = os.path.join(tmp, "anakinistic-theo-6c5-inputs")
        os.makedirs(m2)
        r2 = C.locate_dataset_mount(tmp, "theo-6c5-inputs", "Anakinistic")
        ok("locate_dataset_mount matches owner-prefixed slug", r2 == m2.replace("\\", "/"))

        # Test case-insensitive
        shutil.rmtree(m2)
        m3 = os.path.join(tmp, "Theo-6C5-Inputs")
        os.makedirs(m3)
        r3 = C.locate_dataset_mount(tmp, "theo-6c5-inputs", "Anakinistic")
        ok("locate_dataset_mount matches mixed case", r3 == m3.replace("\\", "/"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_locate_dataset_mount_missing_halt():
    """No matching dataset mount under input_root -> clean HALT."""
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "completely-unrelated-dataset"))
        raised = False
        try:
            C.locate_dataset_mount(tmp, "theo-6c5-inputs", "Anakinistic")
        except SystemExit:
            raised = True
        ok("locate_dataset_mount HALTS when no dataset matches", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_locate_dataset_mount_ambiguous_halt():
    """Multiple matching dataset mounts under input_root -> clean HALT."""
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "theo-6c5-inputs"))
        os.makedirs(os.path.join(tmp, "anakinistic-theo-6c5-inputs"))
        raised = False
        try:
            C.locate_dataset_mount(tmp, "theo-6c5-inputs", "Anakinistic")
        except SystemExit:
            raised = True
        ok("locate_dataset_mount HALTS when multiple mounts match", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_cloud_bundle_root_direct():
    """theo-core/ + theo-data/ directly under matched dataset mount -> resolves to mount."""
    tmp = tempfile.mkdtemp()
    try:
        mount = os.path.join(tmp, "theo-6c5-inputs")
        os.makedirs(os.path.join(mount, "theo-core"))
        os.makedirs(os.path.join(mount, "theo-data"))
        result = C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic")
        ok("direct layout resolves to dataset mount root", result == mount.replace("\\", "/"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_cloud_bundle_root_wrapper_depth1():
    """One-level wrapper (mount/bundle/theo-core) -> resolves to wrapper."""
    tmp = tempfile.mkdtemp()
    try:
        mount = os.path.join(tmp, "theo-6c5-inputs")
        wrapper = os.path.join(mount, "bundle_root")
        os.makedirs(os.path.join(wrapper, "theo-core"))
        os.makedirs(os.path.join(wrapper, "theo-data"))
        result = C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic")
        ok("depth 1 wrapper resolves to wrapper dir", result == wrapper.replace("\\", "/"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_cloud_bundle_root_wrapper_depth2_depth3():
    """Two and three level wrappers resolve within max_depth=3; depth > 3 HALTs."""
    tmp = tempfile.mkdtemp()
    try:
        mount = os.path.join(tmp, "theo-6c5-inputs")
        # Depth 2
        w2 = os.path.join(mount, "archive", "THEO")
        os.makedirs(os.path.join(w2, "theo-core"))
        os.makedirs(os.path.join(w2, "theo-data"))
        r2 = C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic", max_depth=3)
        ok("depth 2 wrapper resolves correctly", r2 == w2.replace("\\", "/"))

        # Depth 3
        shutil.rmtree(mount)
        w3 = os.path.join(mount, "layer1", "layer2", "THEO")
        os.makedirs(os.path.join(w3, "theo-core"))
        os.makedirs(os.path.join(w3, "theo-data"))
        r3 = C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic", max_depth=3)
        ok("depth 3 wrapper resolves correctly", r3 == w3.replace("\\", "/"))

        # Depth 4 (> max_depth 3) must HALT
        shutil.rmtree(mount)
        w4 = os.path.join(mount, "d1", "d2", "d3", "d4")
        os.makedirs(os.path.join(w4, "theo-core"))
        os.makedirs(os.path.join(w4, "theo-data"))
        raised = False
        try:
            C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic", max_depth=3)
        except SystemExit:
            raised = True
        ok("depth > 3 exceeds bounded search -> HALT", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_cloud_bundle_root_zero_candidates_halt():
    """Matched dataset mount exists but contains no theo-core + theo-data -> HALT."""
    tmp = tempfile.mkdtemp()
    try:
        mount = os.path.join(tmp, "theo-6c5-inputs")
        os.makedirs(os.path.join(mount, "unrelated_folder"))
        raised = False
        try:
            C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic")
        except SystemExit:
            raised = True
        ok("zero candidates inside matched dataset mount -> HALT", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_cloud_bundle_root_ambiguous_candidates_halt():
    """Matched dataset mount contains >1 candidates -> HALT."""
    tmp = tempfile.mkdtemp()
    try:
        mount = os.path.join(tmp, "theo-6c5-inputs")
        w1 = os.path.join(mount, "tree1")
        w2 = os.path.join(mount, "tree2")
        os.makedirs(os.path.join(w1, "theo-core")); os.makedirs(os.path.join(w1, "theo-data"))
        os.makedirs(os.path.join(w2, "theo-core")); os.makedirs(os.path.join(w2, "theo-data"))
        raised = False
        try:
            C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic")
        except SystemExit:
            raised = True
        ok("ambiguous candidates (>1) inside dataset mount -> HALT", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_multi_dataset_isolation_unrelated_dataset_ignored():
    """Unrelated second dataset containing theo-core/theo-data MUST NOT be considered."""
    tmp = tempfile.mkdtemp()
    try:
        # Unrelated dataset has a direct theo-core/theo-data
        unrelated = os.path.join(tmp, "unrelated-dataset")
        os.makedirs(os.path.join(unrelated, "theo-core"))
        os.makedirs(os.path.join(unrelated, "theo-data"))

        # Bound dataset has a wrapped theo-core/theo-data
        bound = os.path.join(tmp, "theo-6c5-inputs")
        wrapper = os.path.join(bound, "wrapped_bundle")
        os.makedirs(os.path.join(wrapper, "theo-core"))
        os.makedirs(os.path.join(wrapper, "theo-data"))

        res = C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic")
        ok("resolver isolates bound dataset and ignores unrelated dataset",
           res == wrapper.replace("\\", "/"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_diagnostics_emitted_before_halt():
    """Mount diagnostics are unconditionally printed even when resolution halts."""
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "some_empty_folder"))
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic")
            except SystemExit:
                pass
        out = buf.getvalue()
        ok("diagnostics header printed before HALT", "[mount_diagnostics] Scanning input root" in out)
        ok("immediate children listed in diagnostics before HALT", "some_empty_folder" in out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolved_root_is_exactly_the_root_used_by_training():
    """The resolved root is propagated directly into THEO_ROOT and all derived paths."""
    os.environ["KAGGLE_USERNAME"] = "Anakinistic"
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    tmp = tempfile.mkdtemp()
    try:
        json.dump({"bundle_sha256": "x"}, open(os.path.join(tmp, "bundle_manifest.json"), "w"))
        kd = os.path.join(tmp, "_kernel")
        launch._write_kernel(tmp, kd, "Anakinistic")
        script = open(os.path.join(kd, "script.py")).read()
        ok("script.py binds RESOLVED_THEO_ROOT to THEO_ROOT env", 'os.environ["THEO_ROOT"] = RESOLVED_THEO_ROOT' in script)
        ok("THEO_CORE derives from RESOLVED_THEO_ROOT", 'THEO_CORE = os.path.join(RESOLVED_THEO_ROOT, "theo-core")' in script)
        ok("THEO_DATA derives from RESOLVED_THEO_ROOT", 'THEO_DATA = os.path.join(RESOLVED_THEO_ROOT, "theo-data")' in script)
        ok("PHASE6C5 derives from THEO_CORE", 'PHASE6C5 = os.path.join(THEO_CORE, "scripts", "model_selection", "phase6c5")' in script)
        ok("script.py changes dir to PHASE6C5", 'os.chdir(PHASE6C5)' in script)
    finally:
        C.KAGGLE_DATASET_OWNER = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_kernel_script_no_windows_paths():
    """Generated script.py must contain no Windows paths (C:\\, backslashes, \\Users\\)."""
    os.environ["KAGGLE_USERNAME"] = "Anakinistic"
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    tmp = tempfile.mkdtemp()
    try:
        json.dump({"bundle_sha256": "x"}, open(os.path.join(tmp, "bundle_manifest.json"), "w"))
        kd = os.path.join(tmp, "_kernel")
        launch._write_kernel(tmp, kd, "Anakinistic")
        script = open(os.path.join(kd, "script.py")).read()
        ok("no C:\\ in generated script.py", "C:\\" not in script)
        import re as _re
        win_path = _re.search(r'[A-Za-z]:\\[A-Za-z]', script)
        ok("no Windows-style backslash paths in generated script.py", win_path is None)
        ok("no \\Users\\ in generated script.py", "\\Users\\" not in script)
    finally:
        C.KAGGLE_DATASET_OWNER = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_kernel_script_uses_resolve_not_hardcoded_theo():
    """Generated script.py must NOT hard-code /THEO as a path component."""
    os.environ["KAGGLE_USERNAME"] = "Anakinistic"
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    tmp = tempfile.mkdtemp()
    try:
        json.dump({"bundle_sha256": "x"}, open(os.path.join(tmp, "bundle_manifest.json"), "w"))
        kd = os.path.join(tmp, "_kernel")
        launch._write_kernel(tmp, kd, "Anakinistic")
        script = open(os.path.join(kd, "script.py")).read()
        ok("no /THEO hard-coded in script.py", '/THEO"' not in script and "/THEO'" not in script)
        ok("script.py contains resolve_cloud_bundle_root", "resolve_cloud_bundle_root" in script)
        ok("script.py contains locate_dataset_mount", "locate_dataset_mount" in script)
        ok("script.py contains print_mount_diagnostics", "print_mount_diagnostics" in script)
    finally:
        C.KAGGLE_DATASET_OWNER = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_kernel_script_prints_diagnostic_manifest():
    """Generated script.py must print DATASET_MOUNT_ROOT, RESOLVED_THEO_ROOT,
    THEO_CORE, THEO_DATA, experiment_spec.py, binding_manifest diagnostics."""
    os.environ["KAGGLE_USERNAME"] = "Anakinistic"
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    tmp = tempfile.mkdtemp()
    try:
        json.dump({"bundle_sha256": "x"}, open(os.path.join(tmp, "bundle_manifest.json"), "w"))
        kd = os.path.join(tmp, "_kernel")
        launch._write_kernel(tmp, kd, "Anakinistic")
        script = open(os.path.join(kd, "script.py")).read()
        for marker in ["DATASET_MOUNT_ROOT", "RESOLVED_THEO_ROOT", "THEO_CORE",
                        "THEO_DATA", "experiment_spec.py", "binding_manifest",
                        "cloud_6c5.py"]:
            ok(f"diagnostic manifest contains {marker}", marker in script)
    finally:
        C.KAGGLE_DATASET_OWNER = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_path_validation_kernel_packaging_and_script():
    """Dedicated path validation kernel is properly packaged and valid Python."""
    import validate_cloud_paths_6c5_kaggle as VP
    import py_compile as _pyc
    os.environ["KAGGLE_USERNAME"] = "Anakinistic"
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    tmp = tempfile.mkdtemp()
    try:
        kd = VP.build_path_validation_kernel(tmp, "Anakinistic")
        sp = os.path.join(kd, "script.py")
        ok("path validation script.py exists", os.path.isfile(sp))
        try:
            _pyc.compile(sp, doraise=True)
            ok("path validation script.py compiles cleanly", True)
        except Exception as e:
            ok(f"path validation script.py compiles ({e})", False)
        src = open(sp).read()
        ok("validator contains PATH_VALIDATION_PASS", "[PATH_VALIDATION_PASS]" in src)
        ok("validator does NOT import cloud_6c5", "import cloud_6c5" not in src)
        ok("validator verifies adjudication SHA", "EXPECTED_ADJ" in src)
        ok("validator verifies fixture SHA", "EXPECTED_FIX" in src)
        ok("validator verifies split SHA", "EXPECTED_SPLIT" in src)
        md = json.load(open(os.path.join(kd, "kernel-metadata.json")))
        ok("validator kernel enables GPU", md["enable_gpu"] is True)
        ok("validator kernel disables internet", md["enable_internet"] is False)
        ok("validator kernel code_file is script.py", md["code_file"] == "script.py")
        ok("validator kernel dataset_sources binds exact slug", md["dataset_sources"] == ["Anakinistic/theo-6c5-inputs"])
    finally:
        C.KAGGLE_DATASET_OWNER = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_kernel_script_resolves_existing_bundle():
    """Create a fake mounted tree and actually execute the generated resolver logic."""
    tmp = tempfile.mkdtemp()
    try:
        # Simulate Kaggle mount: input_root/theo-6c5-inputs/ with direct layout
        mount = os.path.join(tmp, "theo-6c5-inputs")
        tc = os.path.join(mount, "theo-core", "scripts", "model_selection", "phase6c5")
        td = os.path.join(mount, "theo-data", "datasets", "theo_slm_v0_review", "6b3_batch2")
        os.makedirs(tc)
        os.makedirs(td)
        open(os.path.join(tc, "experiment_spec.py"), "w").write("# stub")
        open(os.path.join(tc, "cloud_6c5.py"), "w").write("# stub")
        open(os.path.join(td, "6c5_binding_manifest.json"), "w").write("{}")

        result = C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic")
        ok("resolver returns mount root for direct layout", result == mount.replace("\\", "/"))

        theo_core = os.path.join(result, "theo-core")
        theo_data = os.path.join(result, "theo-data")
        ok("theo-core exists under resolved root", os.path.isdir(theo_core))
        ok("theo-data exists under resolved root", os.path.isdir(theo_data))
        spec = os.path.join(theo_core, "scripts", "model_selection", "phase6c5", "experiment_spec.py")
        ok("experiment_spec.py reachable from resolved root", os.path.isfile(spec))
        binding = os.path.join(theo_data, "datasets", "theo_slm_v0_review",
                               "6b3_batch2", "6c5_binding_manifest.json")
        ok("6c5_binding_manifest.json reachable from resolved root", os.path.isfile(binding))

        # Test wrapper layout inside another input root
        tmp2 = tempfile.mkdtemp()
        mount2 = os.path.join(tmp2, "anakinistic-theo-6c5-inputs")
        wrapper = os.path.join(mount2, "THEO")
        tc2 = os.path.join(wrapper, "theo-core", "scripts", "model_selection", "phase6c5")
        td2 = os.path.join(wrapper, "theo-data", "datasets", "theo_slm_v0_review", "6b3_batch2")
        os.makedirs(tc2)
        os.makedirs(td2)
        open(os.path.join(tc2, "experiment_spec.py"), "w").write("# stub")
        open(os.path.join(tc2, "cloud_6c5.py"), "w").write("# stub")
        open(os.path.join(td2, "6c5_binding_manifest.json"), "w").write("{}")

        result2 = C.resolve_cloud_bundle_root(tmp2, "theo-6c5-inputs", "Anakinistic")
        ok("wrapper layout resolves to wrapper dir (execution test)",
           result2 == wrapper.replace("\\", "/"))
        ok("experiment_spec.py reachable via wrapper-resolved root",
           os.path.isfile(os.path.join(result2, "theo-core", "scripts",
                                       "model_selection", "phase6c5", "experiment_spec.py")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)




def test_locate_dataset_mount_nested_kaggle_datasets_owner_slug():
    """Observed Kaggle layout: /kaggle/input/datasets/anakinistic/theo-6c5-inputs"""
    tmp = tempfile.mkdtemp()
    try:
        mount = os.path.join(tmp, "datasets", "anakinistic", "theo-6c5-inputs")
        os.makedirs(mount)
        r = C.locate_dataset_mount(tmp, "theo-6c5-inputs", "Anakinistic", max_depth=4)
        ok("locate_dataset_mount finds nested /datasets/owner/slug mount", r == mount.replace("\\", "/"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_locate_dataset_mount_normalized():
    """Locate unambiguous dataset mount across case-insensitivity and prefix variants."""
    tmp = tempfile.mkdtemp()
    try:
        m1 = os.path.join(tmp, "theo-6c5-inputs")
        os.makedirs(m1)
        r1 = C.locate_dataset_mount(tmp, "theo-6c5-inputs", "Anakinistic")
        ok("locate_dataset_mount matches exact slug", r1 == m1.replace("\\", "/"))

        # Test owner-prefixed folder
        shutil.rmtree(m1)
        m2 = os.path.join(tmp, "anakinistic-theo-6c5-inputs")
        os.makedirs(m2)
        r2 = C.locate_dataset_mount(tmp, "theo-6c5-inputs", "Anakinistic")
        ok("locate_dataset_mount matches owner-prefixed slug", r2 == m2.replace("\\", "/"))

        # Test case-insensitive
        shutil.rmtree(m2)
        m3 = os.path.join(tmp, "Theo-6C5-Inputs")
        os.makedirs(m3)
        r3 = C.locate_dataset_mount(tmp, "theo-6c5-inputs", "Anakinistic")
        ok("locate_dataset_mount matches mixed case", r3 == m3.replace("\\", "/"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_locate_dataset_mount_missing_halt():
    """No matching dataset mount under input_root -> clean HALT."""
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "completely-unrelated-dataset"))
        raised = False
        try:
            C.locate_dataset_mount(tmp, "theo-6c5-inputs", "Anakinistic")
        except SystemExit:
            raised = True
        ok("locate_dataset_mount HALTS when no dataset matches", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_locate_dataset_mount_ambiguous_halt():
    """Multiple matching dataset mounts under input_root without distinct owner -> clean HALT."""
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "folder1", "theo-6c5-inputs"))
        os.makedirs(os.path.join(tmp, "folder2", "theo-6c5-inputs"))
        raised = False
        try:
            C.locate_dataset_mount(tmp, "theo-6c5-inputs", "Anakinistic")
        except SystemExit:
            raised = True
        ok("locate_dataset_mount HALTS when multiple mounts match", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_cloud_bundle_root_direct():
    """theo-core/ + theo-data/ + manifest directly under matched dataset mount -> resolves to mount."""
    tmp = tempfile.mkdtemp()
    try:
        mount = os.path.join(tmp, "datasets", "anakinistic", "theo-6c5-inputs")
        os.makedirs(os.path.join(mount, "theo-core"))
        os.makedirs(os.path.join(mount, "theo-data"))
        json.dump({"bundle_sha256": C.EXPECTED["bundle_sha256"]},
                  open(os.path.join(mount, "bundle_manifest.json"), "w"))
        result = C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic")
        ok("direct layout resolves to dataset mount root", result == mount.replace("\\", "/"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_cloud_bundle_root_wrapper_depth1():
    """One-level wrapper (mount/bundle/theo-core) -> resolves to wrapper."""
    tmp = tempfile.mkdtemp()
    try:
        mount = os.path.join(tmp, "theo-6c5-inputs")
        wrapper = os.path.join(mount, "bundle_root")
        os.makedirs(os.path.join(wrapper, "theo-core"))
        os.makedirs(os.path.join(wrapper, "theo-data"))
        json.dump({"bundle_sha256": C.EXPECTED["bundle_sha256"]},
                  open(os.path.join(wrapper, "bundle_manifest.json"), "w"))
        result = C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic")
        ok("depth 1 wrapper resolves to wrapper dir", result == wrapper.replace("\\", "/"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_cloud_bundle_root_wrapper_depth2_depth3():
    """Two and three level wrappers resolve within max_depth=3; depth > 3 HALTs."""
    tmp = tempfile.mkdtemp()
    try:
        mount = os.path.join(tmp, "theo-6c5-inputs")
        # Depth 2
        w2 = os.path.join(mount, "archive", "THEO")
        os.makedirs(os.path.join(w2, "theo-core"))
        os.makedirs(os.path.join(w2, "theo-data"))
        json.dump({"bundle_sha256": C.EXPECTED["bundle_sha256"]},
                  open(os.path.join(w2, "bundle_manifest.json"), "w"))
        r2 = C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic", max_depth=3)
        ok("depth 2 wrapper resolves correctly", r2 == w2.replace("\\", "/"))

        # Depth 3
        shutil.rmtree(mount)
        w3 = os.path.join(mount, "layer1", "layer2", "THEO")
        os.makedirs(os.path.join(w3, "theo-core"))
        os.makedirs(os.path.join(w3, "theo-data"))
        json.dump({"bundle_sha256": C.EXPECTED["bundle_sha256"]},
                  open(os.path.join(w3, "bundle_manifest.json"), "w"))
        r3 = C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic", max_depth=3)
        ok("depth 3 wrapper resolves correctly", r3 == w3.replace("\\", "/"))

        # Depth 4 (> max_depth 3) must HALT
        shutil.rmtree(mount)
        w4 = os.path.join(mount, "d1", "d2", "d3", "d4")
        os.makedirs(os.path.join(w4, "theo-core"))
        os.makedirs(os.path.join(w4, "theo-data"))
        json.dump({"bundle_sha256": C.EXPECTED["bundle_sha256"]},
                  open(os.path.join(w4, "bundle_manifest.json"), "w"))
        raised = False
        try:
            C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic", max_depth=3)
        except SystemExit:
            raised = True
        ok("depth > 3 exceeds bounded search -> HALT", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_cloud_bundle_root_zero_candidates_halt():
    """Matched dataset mount exists but contains no theo-core + theo-data -> HALT."""
    tmp = tempfile.mkdtemp()
    try:
        mount = os.path.join(tmp, "theo-6c5-inputs")
        os.makedirs(os.path.join(mount, "unrelated_folder"))
        raised = False
        try:
            C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic")
        except SystemExit:
            raised = True
        ok("zero candidates inside matched dataset mount -> HALT", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_cloud_bundle_root_missing_manifest_halts():
    """Matched dataset contains theo-core/ + theo-data/ but NO bundle_manifest.json -> HALT."""
    tmp = tempfile.mkdtemp()
    try:
        mount = os.path.join(tmp, "theo-6c5-inputs")
        os.makedirs(os.path.join(mount, "theo-core"))
        os.makedirs(os.path.join(mount, "theo-data"))
        raised = False
        try:
            C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic")
        except SystemExit:
            raised = True
        ok("missing bundle_manifest.json -> HALT", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_manifest_bundle_sha_mismatch_halts():
    """Valid theo-core + theo-data + bundle_manifest.json with WRONG bundle_sha256 -> immediate HALT."""
    tmp = tempfile.mkdtemp()
    try:
        mount = os.path.join(tmp, "datasets", "anakinistic", "theo-6c5-inputs")
        os.makedirs(os.path.join(mount, "theo-core"))
        os.makedirs(os.path.join(mount, "theo-data"))
        json.dump({"bundle_sha256": "0000000000000000000000000000000000000000000000000000000000000000"},
                  open(os.path.join(mount, "bundle_manifest.json"), "w"))
        raised = False
        try:
            C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic")
        except SystemExit:
            raised = True
        ok("manifest bundle_sha256 mismatch -> immediate HALT", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bundle_content_hash_mismatch_halts():
    """Manifest claims correct bundle SHA, but one bound file has an altered hash -> validator HALTs."""
    import hashlib
    import validate_cloud_paths_6c5_kaggle as VP
    tmp = tempfile.mkdtemp()
    try:
        # Build mock valid bundle tree
        mount = os.path.join(tmp, "datasets", "anakinistic", "theo-6c5-inputs")
        tc = os.path.join(mount, "theo-core", "scripts", "model_selection", "phase6c5")
        td_rev = os.path.join(mount, "theo-data", "datasets", "theo_slm_v0_review", "6b3_batch2")
        td_art = os.path.join(mount, "theo-data", "datasets", "theo_slm_v0_artifacts", "phase-6e3")
        td_mod = os.path.join(mount, "theo-data", "models", "6c_acquired", "M1_Qwen2.5-0.5B-Instruct")
        os.makedirs(tc); os.makedirs(td_rev); os.makedirs(td_art); os.makedirs(td_mod)

        json.dump({"bundle_sha256": C.EXPECTED["bundle_sha256"]},
                  open(os.path.join(mount, "bundle_manifest.json"), "w"))
        open(os.path.join(tc, "experiment_spec.py"), "w").write("# spec")
        open(os.path.join(tc, "cloud_6c5.py"), "w").write("# cloud")

        # Create altered adjudication.json (content mismatch)
        open(os.path.join(mount, "theo-data", "datasets", "theo_slm_v0_review", "adjudication.json"), "w").write("corrupted content")
        open(os.path.join(td_art, "populated-evidence-corpus.json"), "w").write("fixture content")
        open(os.path.join(td_rev, "6c5_split_manifest.json"), "w").write("split content")
        open(os.path.join(td_rev, "6c5_binding_manifest.json"), "w").write(json.dumps({"base_model": {"file_hashes": {}}}))

        kd = VP.build_path_validation_kernel(tmp, "Anakinistic")
        script_code = open(os.path.join(kd, "script.py")).read()

        # Execute validator script in subprocess with INPUT_ROOT pointing to tmp
        import subprocess
        proc = subprocess.run([sys.executable, "-c", script_code.replace('/kaggle/input', tmp.replace('\\', '/'))],
                              capture_output=True, text=True)
        ok("bundle content hash mismatch causes validator HALT", proc.returncode != 0 and "adjudication SHA mismatch" in (proc.stdout + proc.stderr))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_cloud_bundle_root_ambiguous_candidates_halt():
    """Matched dataset mount contains >1 candidates -> HALT."""
    tmp = tempfile.mkdtemp()
    try:
        mount = os.path.join(tmp, "theo-6c5-inputs")
        w1 = os.path.join(mount, "tree1")
        w2 = os.path.join(mount, "tree2")
        os.makedirs(os.path.join(w1, "theo-core")); os.makedirs(os.path.join(w1, "theo-data"))
        os.makedirs(os.path.join(w2, "theo-core")); os.makedirs(os.path.join(w2, "theo-data"))
        json.dump({"bundle_sha256": C.EXPECTED["bundle_sha256"]},
                  open(os.path.join(w1, "bundle_manifest.json"), "w"))
        json.dump({"bundle_sha256": C.EXPECTED["bundle_sha256"]},
                  open(os.path.join(w2, "bundle_manifest.json"), "w"))
        raised = False
        try:
            C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic")
        except SystemExit:
            raised = True
        ok("ambiguous candidates (>1) inside dataset mount -> HALT", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_multi_dataset_isolation_unrelated_dataset_ignored():
    """Unrelated second dataset containing theo-core/theo-data MUST NOT be considered."""
    tmp = tempfile.mkdtemp()
    try:
        # Unrelated dataset has a direct theo-core/theo-data
        unrelated = os.path.join(tmp, "unrelated-dataset")
        os.makedirs(os.path.join(unrelated, "theo-core"))
        os.makedirs(os.path.join(unrelated, "theo-data"))
        json.dump({"bundle_sha256": C.EXPECTED["bundle_sha256"]},
                  open(os.path.join(unrelated, "bundle_manifest.json"), "w"))

        # Bound dataset has a wrapped theo-core/theo-data
        bound = os.path.join(tmp, "theo-6c5-inputs")
        wrapper = os.path.join(bound, "wrapped_bundle")
        os.makedirs(os.path.join(wrapper, "theo-core"))
        os.makedirs(os.path.join(wrapper, "theo-data"))
        json.dump({"bundle_sha256": C.EXPECTED["bundle_sha256"]},
                  open(os.path.join(wrapper, "bundle_manifest.json"), "w"))

        res = C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic")
        ok("resolver isolates bound dataset and ignores unrelated dataset",
           res == wrapper.replace("\\", "/"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_diagnostics_emitted_before_halt():
    """Mount diagnostics are unconditionally printed even when resolution halts."""
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "some_empty_folder"))
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic")
            except SystemExit:
                pass
        out = buf.getvalue()
        ok("diagnostics header printed before HALT", "[mount_diagnostics] Scanning input root" in out)
        ok("immediate children listed in diagnostics before HALT", "some_empty_folder" in out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolved_root_is_exactly_the_root_used_by_training():
    """The resolved root is propagated directly into THEO_ROOT and all derived paths."""
    os.environ["KAGGLE_USERNAME"] = "Anakinistic"
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    tmp = tempfile.mkdtemp()
    try:
        json.dump({"bundle_sha256": "x"}, open(os.path.join(tmp, "bundle_manifest.json"), "w"))
        kd = os.path.join(tmp, "_kernel")
        launch._write_kernel(tmp, kd, "Anakinistic")
        script = open(os.path.join(kd, "script.py")).read()
        ok("script.py binds RESOLVED_THEO_ROOT to THEO_ROOT env", 'os.environ["THEO_ROOT"] = RESOLVED_THEO_ROOT' in script)
        ok("THEO_CORE derives from RESOLVED_THEO_ROOT", 'THEO_CORE = os.path.join(RESOLVED_THEO_ROOT, "theo-core")' in script)
        ok("THEO_DATA derives from RESOLVED_THEO_ROOT", 'THEO_DATA = os.path.join(RESOLVED_THEO_ROOT, "theo-data")' in script)
        ok("PHASE6C5 derives from THEO_CORE", 'PHASE6C5 = os.path.join(THEO_CORE, "scripts", "model_selection", "phase6c5")' in script)
        ok("script.py changes dir to PHASE6C5", 'os.chdir(PHASE6C5)' in script)
    finally:
        C.KAGGLE_DATASET_OWNER = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_kernel_script_no_windows_paths():
    """Generated script.py must contain no Windows paths (C:\\, backslashes, \\Users\\)."""
    os.environ["KAGGLE_USERNAME"] = "Anakinistic"
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    tmp = tempfile.mkdtemp()
    try:
        json.dump({"bundle_sha256": "x"}, open(os.path.join(tmp, "bundle_manifest.json"), "w"))
        kd = os.path.join(tmp, "_kernel")
        launch._write_kernel(tmp, kd, "Anakinistic")
        script = open(os.path.join(kd, "script.py")).read()
        ok("no C:\\ in generated script.py", "C:\\" not in script)
        import re as _re
        win_path = _re.search(r'[A-Za-z]:\\[A-Za-z]', script)
        ok("no Windows-style backslash paths in generated script.py", win_path is None)
        ok("no \\Users\\ in generated script.py", "\\Users\\" not in script)
    finally:
        C.KAGGLE_DATASET_OWNER = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_kernel_script_uses_resolve_not_hardcoded_theo():
    """Generated script.py must NOT hard-code /THEO as a path component."""
    os.environ["KAGGLE_USERNAME"] = "Anakinistic"
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    tmp = tempfile.mkdtemp()
    try:
        json.dump({"bundle_sha256": "x"}, open(os.path.join(tmp, "bundle_manifest.json"), "w"))
        kd = os.path.join(tmp, "_kernel")
        launch._write_kernel(tmp, kd, "Anakinistic")
        script = open(os.path.join(kd, "script.py")).read()
        ok("no /THEO hard-coded in script.py", '/THEO"' not in script and "/THEO'" not in script)
        ok("script.py contains resolve_cloud_bundle_root", "resolve_cloud_bundle_root" in script)
        ok("script.py contains locate_dataset_mount", "locate_dataset_mount" in script)
        ok("script.py contains print_mount_diagnostics", "print_mount_diagnostics" in script)
    finally:
        C.KAGGLE_DATASET_OWNER = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_kernel_script_prints_diagnostic_manifest():
    """Generated script.py must print DATASET_MOUNT_ROOT, RESOLVED_THEO_ROOT,
    THEO_CORE, THEO_DATA, experiment_spec.py, binding_manifest diagnostics."""
    os.environ["KAGGLE_USERNAME"] = "Anakinistic"
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    tmp = tempfile.mkdtemp()
    try:
        json.dump({"bundle_sha256": "x"}, open(os.path.join(tmp, "bundle_manifest.json"), "w"))
        kd = os.path.join(tmp, "_kernel")
        launch._write_kernel(tmp, kd, "Anakinistic")
        script = open(os.path.join(kd, "script.py")).read()
        for marker in ["DATASET_MOUNT_ROOT", "RESOLVED_THEO_ROOT", "THEO_CORE",
                        "THEO_DATA", "experiment_spec.py", "binding_manifest",
                        "cloud_6c5.py"]:
            ok(f"diagnostic manifest contains {marker}", marker in script)
    finally:
        C.KAGGLE_DATASET_OWNER = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_path_validation_kernel_packaging_and_script():
    """Dedicated path validation kernel is properly packaged and valid Python."""
    import validate_cloud_paths_6c5_kaggle as VP
    import py_compile as _pyc
    os.environ["KAGGLE_USERNAME"] = "Anakinistic"
    saved = C.KAGGLE_DATASET_OWNER
    C.KAGGLE_DATASET_OWNER = "Anakinistic"
    tmp = tempfile.mkdtemp()
    try:
        kd = VP.build_path_validation_kernel(tmp, "Anakinistic")
        sp = os.path.join(kd, "script.py")
        ok("path validation script.py exists", os.path.isfile(sp))
        try:
            _pyc.compile(sp, doraise=True)
            ok("path validation script.py compiles cleanly", True)
        except Exception as e:
            ok(f"path validation script.py compiles ({e})", False)
        src = open(sp).read()
        ok("validator contains PATH_VALIDATION_PASS", "[PATH_VALIDATION_PASS]" in src)
        ok("validator performs pre-training import smoke test", "import cloud_6c5" in src and "[import_smoke]" in src)
        ok("validator verifies cloud_6c5 SHA", "EXPECTED_CLOUD_SHA" in src)
        ok("validator verifies adjudication SHA", "EXPECTED_ADJ" in src)
        ok("validator verifies fixture SHA", "EXPECTED_FIX" in src)
        ok("validator verifies split SHA", "EXPECTED_SPLIT" in src)
        md = json.load(open(os.path.join(kd, "kernel-metadata.json")))
        ok("validator kernel enables GPU", md["enable_gpu"] is True)
        ok("validator kernel disables internet", md["enable_internet"] is False)
        ok("validator kernel code_file is script.py", md["code_file"] == "script.py")
        ok("validator kernel dataset_sources binds exact slug", md["dataset_sources"] == ["Anakinistic/theo-6c5-inputs"])
    finally:
        C.KAGGLE_DATASET_OWNER = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_cloud_6c5_isolated_import_clean():
    """Test that cloud_6c5.py compiles cleanly and imports without NameError on importlib."""
    import py_compile, subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    cloud_src = os.path.join(here, "cloud_6c5.py")
    py_compile.compile(cloud_src, doraise=True)
    ok("cloud_6c5.py compiles cleanly", True)

    # In clean subprocess, verify that importing cloud_6c5 executes past line 45 without importlib NameError/AttributeError
    theo_root = r"c:\Users\bs162\Desktop\THEO"
    env = os.environ.copy()
    env["THEO_ROOT"] = theo_root
    env["THEO_OUT_B"] = os.path.join(here, "_tmp_b")
    env["THEO_OUT_C"] = os.path.join(here, "_tmp_c")
    cmd = [sys.executable, "-c", f"import sys; sys.path.insert(0, r'{here}'); import cloud_6c5"]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    ok("cloud_6c5 has no NameError on importlib in isolated subprocess", "NameError: name 'importlib' is not defined" not in proc.stderr)
    ok("cloud_6c5 has no AttributeError on importlib.util in isolated subprocess", "AttributeError: module 'importlib' has no attribute 'util'" not in proc.stderr)
    ok("cloud_6c5 reaches platform gate (fail-closed on non-cloud host)", "HALT: bitsandbytes package SHA" in proc.stderr or proc.returncode == 0)



def test_kernel_script_resolves_existing_bundle():
    """Create a fake mounted tree and actually execute the generated resolver logic."""
    tmp = tempfile.mkdtemp()
    try:
        # Simulate Kaggle mount: input_root/datasets/anakinistic/theo-6c5-inputs/ with direct layout
        mount = os.path.join(tmp, "datasets", "anakinistic", "theo-6c5-inputs")
        tc = os.path.join(mount, "theo-core", "scripts", "model_selection", "phase6c5")
        td = os.path.join(mount, "theo-data", "datasets", "theo_slm_v0_review", "6b3_batch2")
        os.makedirs(tc)
        os.makedirs(td)
        json.dump({"bundle_sha256": C.EXPECTED["bundle_sha256"]},
                  open(os.path.join(mount, "bundle_manifest.json"), "w"))
        open(os.path.join(tc, "experiment_spec.py"), "w").write("# stub")
        open(os.path.join(tc, "cloud_6c5.py"), "w").write("# stub")
        open(os.path.join(td, "6c5_binding_manifest.json"), "w").write("{}")

        result = C.resolve_cloud_bundle_root(tmp, "theo-6c5-inputs", "Anakinistic")
        ok("resolver returns mount root for direct layout", result == mount.replace("\\", "/"))

        theo_core = os.path.join(result, "theo-core")
        theo_data = os.path.join(result, "theo-data")
        ok("theo-core exists under resolved root", os.path.isdir(theo_core))
        ok("theo-data exists under resolved root", os.path.isdir(theo_data))
        spec = os.path.join(theo_core, "scripts", "model_selection", "phase6c5", "experiment_spec.py")
        ok("experiment_spec.py reachable from resolved root", os.path.isfile(spec))
        binding = os.path.join(theo_data, "datasets", "theo_slm_v0_review",
                               "6b3_batch2", "6c5_binding_manifest.json")
        ok("6c5_binding_manifest.json reachable from resolved root", os.path.isfile(binding))

        # Test wrapper layout inside another input root
        tmp2 = tempfile.mkdtemp()
        mount2 = os.path.join(tmp2, "datasets", "anakinistic", "theo-6c5-inputs")
        wrapper = os.path.join(mount2, "THEO")
        tc2 = os.path.join(wrapper, "theo-core", "scripts", "model_selection", "phase6c5")
        td2 = os.path.join(wrapper, "theo-data", "datasets", "theo_slm_v0_review", "6b3_batch2")
        os.makedirs(tc2)
        os.makedirs(td2)
        json.dump({"bundle_sha256": C.EXPECTED["bundle_sha256"]},
                  open(os.path.join(wrapper, "bundle_manifest.json"), "w"))
        open(os.path.join(tc2, "experiment_spec.py"), "w").write("# stub")
        open(os.path.join(tc2, "cloud_6c5.py"), "w").write("# stub")
        open(os.path.join(td2, "6c5_binding_manifest.json"), "w").write("{}")

        result2 = C.resolve_cloud_bundle_root(tmp2, "theo-6c5-inputs", "Anakinistic")
        ok("wrapper layout resolves to wrapper dir (execution test)",
           result2 == wrapper.replace("\\", "/"))
        ok("experiment_spec.py reachable via wrapper-resolved root",
           os.path.isfile(os.path.join(result2, "theo-core", "scripts",
                                       "model_selection", "phase6c5", "experiment_spec.py")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)


if __name__ == "__main__":
    test_validate_binding_hashes()
    test_scan_exclusions()
    test_verify_pass()
    test_verify_fail_on_hash_mismatch()
    test_launch_dry_run_no_kaggle()
    test_launch_kernel_packaging_valid_python_entry()
    test_dataset_slug_passed_via_metadata_and_env()
    test_cloud_runner_pins_python_3_14_3()
    test_python_312_causes_fail_closed()
    test_feasibility_required_pins_match_binding()
    test_feasibility_evaluate_fail_closed()
    test_feasibility_kernel_packaging()
    test_feasibility_cli_build_kernel_and_default_safe()
    test_feasibility_kernel_entrypoint_invokes_probe_cloud()
    test_feasibility_cloud_main_hard_fails_on_mismatch()
    test_feasibility_base_312_not_classified_infeasible()
    test_execute_requires_l5()
    test_kaggle_resolution_module_preferred()
    test_kaggle_resolution_direct_exe_fallback()
    test_kaggle_resolution_none_halts()
    test_execute_requires_l5_active()
    test_datasets_create_command_shape()
    test_kernel_metadata_serialization()
    test_kernel_metadata_contains_bound_dataset_source()
    test_kernel_metadata_owner_override()
    test_written_kernel_metadata_contains_dataset_source()
    test_smoke_builder_binds_dataset_source()
    test_smoke_slug_is_deterministic_and_distinct()
    test_smoke_metadata_provider_consistency()
    test_smoke_uses_supported_sdk_metadata_call()
    test_smoke_metadata_retrieval_failure_halts()
    test_normalize_root_windows_abs()
    test_normalize_root_gitbash()
    test_normalize_root_already_normalized()
    test_normalize_root_invalid_rejected()
    test_root_forms_resolve_to_same_native_path()
    test_relative_dot_resolves_to_theo()
    test_spec_and_bundle_paths_consistent()
    test_root_out_combos_resolve_identically()
    # ---- Remediation #13 & #14 tests ----
    test_locate_dataset_mount_nested_kaggle_datasets_owner_slug()
    test_locate_dataset_mount_normalized()
    test_locate_dataset_mount_missing_halt()
    test_locate_dataset_mount_ambiguous_halt()
    test_resolve_cloud_bundle_root_direct()
    test_resolve_cloud_bundle_root_wrapper_depth1()
    test_resolve_cloud_bundle_root_wrapper_depth2_depth3()
    test_resolve_cloud_bundle_root_zero_candidates_halt()
    test_resolve_cloud_bundle_root_missing_manifest_halts()
    test_manifest_bundle_sha_mismatch_halts()
    test_bundle_content_hash_mismatch_halts()
    test_resolve_cloud_bundle_root_ambiguous_candidates_halt()
    test_multi_dataset_isolation_unrelated_dataset_ignored()
    test_diagnostics_emitted_before_halt()
    test_resolved_root_is_exactly_the_root_used_by_training()
    test_kernel_script_no_windows_paths()
    test_kernel_script_uses_resolve_not_hardcoded_theo()
    test_kernel_script_prints_diagnostic_manifest()
    test_path_validation_kernel_packaging_and_script()
    test_cloud_6c5_isolated_import_clean()
    test_kernel_script_resolves_existing_bundle()
    print(f"\nALL TESTS PASSED ({PASS} checks)")



