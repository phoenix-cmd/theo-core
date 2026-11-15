# 6C.5 Cloud Execution Orchestrator — Independent Source Audit

**Scope:** The four orchestrator scripts plus `verify_6c5_cloud_results.py` and the offline
test suite, as produced under the frozen L5 cloud execution authorization
(`theo-slm-phase-6c5-l5-cloud-execution-authorization.md`, APPROVED).

**Auditor stance:** Read the code as if written by a third party. Confirm it (a) performs only
orchestration, (b) enforces every invariant from the L5 authorization and the cloud amendment,
and (c) cannot silently mutate a frozen input or produce an unverified result.

---

## 1. Modules under audit

| File | Role |
|------|------|
| `kaggle_6c5_common.py` | Shared allowlist, expected hashes, L5 gate, HALT, hash helpers |
| `prepare_6c5_cloud_bundle.py` | Build allowlisted bundle + `bundle_manifest.json` + bundle SHA |
| `launch_6c5_kaggle.py` | Submit dataset/kernel to Kaggle (dry-run default, `--execute`) |
| `retrieve_6c5_kaggle_results.py` | Pull B/C adapters, logs, manifest; preserve unchanged + record hashes |
| `verify_6c5_cloud_results.py` | Verify returned artifacts vs cloud manifest; HALT on mismatch |
| `test_6c5_orchestrator.py` | Offline tests (no Kaggle contact) |

---

## 2. Invariant enforcement (per L5 authorization)

| # | Invariant | Where enforced | Verdict |
|---|-----------|----------------|---------|
| 1 | Run **exactly one** B and **exactly one** C | `cloud_6c5.py` trains B once + C once; orchestrator never re-submits or auto-retries | OK (see §4) |
| 2 | QLoRA-only, 4-bit + LoRA, architecture `.py` unmodified | `cloud_6c5.py` hard-gates deps + asserts architecture import; orchestrator never opens/writes arch files | OK |
| 3 | `adjudication.json` immutable | `experiment_spec.py` asserts `ADJUDICATION_SHA256 == 97bc6bc6…`; `common.validate_binding_hashes` re-checks; prepare allowlist is read-only copy | OK |
| 4 | 66-case fixture (`f99ca025…`), frozen split (`eda4cb66…`), prompt (`a29e12a2…`), M1 (`fdf756fa…` / `18e18af…`) unchanged | `validate_binding_hashes` re-validated before prepare (test: PASS) and again in `verify_results` | OK |
| 5 | Platform-specific deps: Windows `bitsandbytes==0.50.1` hash is **historical only**; cloud binds Linux wheel SHA at runtime | `cloud_6c5.py` computes bnb wheel SHA on the cloud runtime and records it in provenance; `verify_results` requires a non-empty `bitsandbytes_wheel_sha256` in cloud manifest | OK |
| 6 | Latency gates measured locally on GTX 1650 | Orchestrator has no role in latency; `cloud_6c5.py` only records wallclock as provenance, not as a gate | OK |
| 7 | No modification of 300-GOLD, 6C.4 artifacts, dataset, labels, gates, thresholds, seed | Allowlist excludes `6e20`/`6E20`/`checkpoint`; prepare copies only the enumerated files; scan_exclusions test confirms no excluded path overlaps allowlist | OK |
| 8 | Dry-run is default; execution requires `--execute` **and** active L5 | `launch`/`retrieve` compute `dry_run = not execute`; `__main__` shows `--execute` optional; `check_l5_authorization` gates `execute` | OK |
| 9 | No auto-retry on Kaggle failure | `run_kaggle_cmd` and the single status poll all pass `halt_on_fail=True`; no retry loops anywhere | OK |
| 10 | HALT (non-zero exit) on any auth/hash/runtime failure | `common.halt` raises `SystemExit`; used on L5-inactive, missing username, hash mismatch, missing manifest | OK |

---

## 3. Independent checks performed by the auditor

- **Allowlist is closed.** `ALLOWLIST` enumerates only: `6c5_split_manifest.json`,
  `6c5_binding_manifest.json`, `adjudication.json`, `populated-evidence-corpus.json`,
  `b_architecture.py`, `c_architecture.py`, `experiment_spec.py`, `train_6c5.py`,
  `eval_6c5.py`, `phase2_audit.py`, `amendment_audit.py`, `cloud_6c5.py`, `requirements.txt`,
  `m1_architecture_outputs/Qwen2.5-0.5B-Instruct/` (`model.safetensors`, `config.json`).
  No training script, no orchestrator script, no 6E.20.1 artifact is in scope.
- **Hashes are bound, not recomputed magic.** `EXPECTED_SHA256` is a literal constant;
  `validate_binding_hashes` only compares. Nothing writes these values.
- **`experimental_spec` import is read-only.** `common` imports it solely to call
  `validate_binding_hashes` and `evidence_token_from_prompt` (deterministic). No spec mutation.
- **`prepare` is copy + hash, never transform.** `shutil.copy2` + `sha256_file`/`dir_sha256`.
  The bundle manifest is emitted **after** hashing; bundle SHA-256 is computed over the manifest.
- **`launch` never trains.** It only writes `dataset-metadata.json`/`kernel-metadata.json`
  (Kaggle control files) and calls `run_kaggle_cmd`. Training happens only inside the kernel
  via `cloud_6c5.py`.
- **`retrieve` preserves verbatim.** Downloads into `out_dir`, then records
  `dir_sha256`/`sha256_file` without modifying content. Output is handed to `verify`.
- **`verify` is the only consumer of the cloud manifest.** It fails closed (HALT) unless every
  adapter dir SHA matches the recorded cloud SHA **and** B != C **and** local binding hashes
  are still intact.

---

## 4. Residual risks & required human controls

1. **Cloud runtime substitution.** `cloud_6c5.py` HALTs if it cannot `import torch/transformers/
   peft/bitsandbytes` at the pinned versions or if its 16-check `phase2_audit` fails. The
   orchestrator trusts that HALT. *Human control:* confirm the kernel log shows
   `phase2 AUDIT: 16/16 PASS` and a real Linux bnb wheel SHA before accepting results.
2. **Kaggle secret hygiene.** Credentials come only from the environment
   (`KAGGLE_USERNAME`/`KAGGLE_DATASET_SLUG`/`KAGGLE_KERNEL_SLUG`). The scripts never read or
   write `kaggle.json`. The bundle is uploaded as a **private** dataset/kernel (`is_private: true`).
   *Human control:* rotate Kaggle token after the run; confirm dataset/kernel are private.
3. **No third-party code.** Only `kaggle` CLI and standard library are used. No `pip install`
   inside the kernel beyond the pinned `requirements.txt`. *Human control:* review
   `requirements.txt` before upload.
4. **Single-B/single-C guarantee is logical, not time-locked.** A user could re-run the kernel.
   *Human control:* retrieve exactly once; the `verify` step requires the cloud manifest's
   recorded adapter SHAs — a re-run produces a different manifest and must be re-verified.

---

## 5. Audit conclusion

The orchestrator is **orchestration-only** and enforces every L5 invariant through positive
gates and fail-closed HALTs. The offline test suite (`test_6c5_orchestrator.py`) passes 7/7,
covering binding-hash integrity, exclusion safety, verify pass, verify HALT-on-mismatch,
dry-run planning, and `--execute` gating. The orchestrator is **not executed** as part of this
audit (dry-run only). 

**Status: APPROVED FOR HUMAN EXECUTION ON KAGGLE** — subject to the human controls in §4.
No local training occurred.
