# Phase 6A.1 Exit-Gate Audit — 20-Check Checklist

**Status:** DRAFT FOR REVIEW (verdicts are provisional pending human approval)
**Date:** 2026-08-10
**Scope:** Qwen3 reference SLM hypothesis-proposal provider (Phase 6A.1) vs the
official 6A interface-validation gate in `v0.5-neural-interface-plan.md` and the
phase task checklist.
**Evidence base:** tests run fresh on 2026-08-10 in `theo-core/.venv` (Python 3.13,
torch 2.13.0+cu126, transformers 5.14.1, GTX 1650 CUDA fp16).

---

## Checklist

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Model loads with real weights | **PASS** | `Qwen3Model()` loads `Qwen/Qwen3-0.6B` on CUDA fp16; `scripts/qwen3_smoke_test.py` checks 1–2 pass on real model. |
| 2 | Exact revision pinned | **PASS** | `_REVISION = c1899de289a04d12100db370d81485cdf75e47ca`; `resolved_revision()` returns the same SHA. |
| 3 | Model/tokenizer hashes recorded and reproducible | **PASS** | `model_hash a80410e0…`, `tokenizer_hash 92ba2f61…` (SHA-256 over Hub content manifest: `lfs.sha256`/`blob_id`). Reproducibility unit-tested (`test_manifest_sorting_is_deterministic`, `test_files_without_hash_are_omitted`); recorded in `model-info.json` and each `ProviderExecution`. |
| 4 | Provider version + capability discovery | **PASS** | v0.1.0, name `qwen3_hypothesis`; `capabilities() == {HYPOTHESIS_PROPOSAL}` only (`TestCapabilityDiscovery`). |
| 5 | Generation config — strict parameter record | **PASS** | seed 0, temperature 0.0, `do_sample=False`, `max_new_tokens=512` in `ModelMetadata` + `model-info.json`. Caveat recorded: transformers 5.14.1 logs `temperature`/`top_k` as ignored under unified generation API; determinism rests on greedy decoding and was verified empirically. |
| 6 | Grounding verification (pass/fail counts) | **PASS** | 51-case blind eval: 20 proposals generated, **20 grounded**, **0 rejected** (no ungrounded output). Parser/`verify_grounding` reject unknown ids, malformed JSON, over-cap, duplicates; no `grounded=False` path. |
| 7 | Determinism — run A/B identical | **PASS** | Real model: two `generate()` calls returned byte-identical 512-token completions (`qwen3_timing_probe.py`); provider path determinism test (`test_prompt_is_deterministic_and_contains_grounding_ids`). |
| 8 | Isolation invariant | **PASS** | qwen3 modules import only `theo_core.models.ports` (never `theo_core.symbolic`); no `proposal://` provenance in any belief/decision (isolation tests + real run); `TestIsolation` green. |
| 9 | Replay invariant | **PASS** | Eval replay with a fresh `ProviderCoordinator` produced identical decisions (replay deterministic = True). |
| 10 | Provider protocol compliance (ADR-0028) | **PASS** | `propose_hypotheses(percept, concepts, beliefs, rules, grounding) -> ProviderExecution[tuple[HypothesisProposal, ...]]`; signature asserted in `TestProposalContract`. |
| 11 | Provider provenance (execution + model) | **PASS** | `ProviderExecution` carries `provider_name`, `provider_version`, model hash, tokenizer hash; model provenance in `model-info.json`. |
| 12 | Cross-process determinism where supported | **PASS (scoped)** | Pipeline-level cross-process equivalence verified for provider-pipeline via `test_cross_process_equivalence.py` (baseline vs NullProvider digest, 3-run stability). Real-model qwen3 cross-process is not exercised (model load cost); same-process determinism + fresh-coordinator replay verified instead — documented as the supported scope. |
| 13 | 51-case benchmark + no-provider equivalence | **PASS** | Baseline pass **51/51**, Qwen-config pass **51/51**, decision changes **0**, mean confidence delta **+0.0000**. Original 35-case benchmark set is a subset of the 51; no regression. |
| 14 | No newly-failing v0.4.1 tests | **PASS** | Full default suite: **144 passed, 3 skipped (THEO_SLM gate)**; with `THEO_SLM=1`: **146 passed**. Coverage 89.71% (default) / 97.35% (SLM). |
| 15 | mypy --strict green (phase files) | **PASS (scoped)** | qwen3 `src` (6 files) and qwen3 tests: no issues. Package-wide: **24 pre-existing errors** in heuristic files (`heuristic/salience.py`, `test_heuristic_salience.py`, `test_heuristic_calibration.py`) — present before this phase, untouched, not part of 6A.1. |
| 16 | ruff green (phase files) | **PASS (scoped)** | qwen3 `src` + qwen3 tests + eval script + scripts: clean. Package-wide: 15 pre-existing issues in heuristic files (`salience.py`/`test_heuristic_salience.py`), untouched. |
| 17 | No theo-core → theo-providers dependency | **PASS** | `theo_core` imports no `torch`/`transformers`/`accelerate`/`qwen`/`huggingface_hub`; direction is `theo-core ← theo-providers ← qwen3`. Verified by import scan + `TestIsolation`. |
| 18 | Verified installability of package extras | **PASS** | **Real bug found & fixed this phase:** the `theo-core @ file://../theo-core` relative direct reference failed to build/install on Windows under both pip (`\\..\theo-core` invalid) and uv. Fixed by replacing the direct reference with a plain `theo-core>=0.5` dependency + `[tool.uv.sources] theo-core = { path = "../theo-core" }`. Verified: `uv pip install --dry-run ".[providers-slm]"` resolves theo-core + torch 2.13 + transformers + accelerate + safetensors cleanly; `uv build --wheel` succeeds; wheel metadata declares all four extras. Caveat: pip-only installs require building/installing theo-core from source first (theo-core is not on PyPI). |
| 19 | Report: decision impact + reasoning impact + latencies as data | **PASS** | `phase-6a1-report.md` §3/§6 (decision impact 0, conf delta +0.0000), §5/§6 (premise-echo valuation A–E, rule engagement 2/20), §8 (latency measured: 50–51 s/call, ~10 tok/s, 512-cap hit; determinism). |
| 20 | Test hygiene / suite regression | **PASS** | New tests: parser, hashing, provider contract, prompt-archive sync, integration (gated `THEO_SLM=1`), eval script (ruff+mypy clean). All green; no frozen corpus/baseline modified. |

---

## Two independent success questions (plan §6A)

### Architecture success — binary, all-or-nothing

| Contract | Status |
|---|---|
| Provider contract | **PASS** |
| Grounding | **PASS** |
| Replay | **PASS** |
| Provenance | **PASS** |
| Isolation | **PASS** |
| Cross-process | **PASS (scoped — see #12)** |

**Architecture success: YES.** A real 0.6B SLM fits cleanly into the ADR-0028
provider architecture: it loads, produces grounded proposals, stays deterministic,
records provenance, and never leaks into or depends on symbolic state.

### Cognitive success — measured, scalar

| Metric | Value |
|---|---|
| Decision delta | 0 changes |
| Correct decision changes | 0 |
| Incorrect decision changes | 0 |
| Grounded proposal rate | 20/20 (100%) |
| Novel proposal rate | 20/20 (formal only — see report §6) |
| **Useful proposal rate** | **0/20** |
| Unsupported (rejected) proposal rate | 0/20 |
| Rule-conclusion proposals (fired) | 2/20 |
| Mean symbolic hypotheses/case | 2.25 |
| Mean fired rules/case | 0.47 |
| Mean confidence delta | +0.0000 |
| Per-domain proposals | ambiguity 5, causal 4, taxonomy 5, uncertainty 3, commonsense 2, contradiction 1 |

**Cognitive success: NO (null result).** The reference SLM changed nothing and
produced no decision-useful proposals. Per the plan's own framing, this is a valid
outcome: *an SLM can be architecturally successful but cognitively useless* — which
is exactly what Phase 6A.1 measured.

---

## Verdict

- **Exit gate: PASS for the architecture-validation half.** All 20 checks pass for
  the phase's scope, with the two scoping caveats recorded (cross-process real-model
  determinism not exercised; mypy/ruff clean restricted to phase files because
  heuristic-file findings pre-date this phase).
- **One packaging defect was found and fixed as a direct consequence of the audit**
  (check #18): the relative `file://` dependency made theo-providers uninstallable on
  Windows. Fix: `[tool.uv.sources]` mapping; verified with a real resolver.
- **Cognitive conclusion stands:** 0/20 useful proposals; no basis to proceed to 6A.2
  (calibration) on this evidence. Recommendation stands at the human gate.

## Artifacts referenced

- `theo-core/docs/research/reference-slm/qwen3-0.6b/phase-6a1-report.md`
- `theo-core/docs/research/reference-slm/qwen3-0.6b/prompt-v1.txt`
- `theo-providers/model-info.json`
- `theo-providers/eval_phase6a1_qwen.py` / `.json` / `.log`
- `theo-providers/qwen_proposals_sample.json`
- `theo-providers/scripts/qwen3_smoke_test.py`, `scripts/qwen3_timing_probe.py`
- `theo-providers/tests/test_qwen3_{provider,parser,hashing,integration}.py`
