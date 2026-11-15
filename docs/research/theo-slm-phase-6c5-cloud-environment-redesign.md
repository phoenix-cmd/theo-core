# 6C.5 Cloud Environment Redesign (to satisfy the Python 3.14.3 + pinned-stack binding)

**Trigger:** pre–L5 v10 read-only runtime compatibility check FAILED — Kaggle base image is **Python 3.12**, but
the binding hard-requires **Python 3.14.3** (with torch 2.13.0+cu126, CUDA 12.6, transformers 5.8.0, peft 0.20.0,
bitsandbytes 0.50.1). The pin is **not** relaxed. No scientific/training specification is modified.
**Status:** DESIGN ONLY. Requires human decision + a fresh authorization before any execution.

---

## 1. Invariant (non-negotiable)
- The cloud runner (`cloud_6c5.py`) hard-gates `Python == 3.14.3` and HALTS otherwise. This is a bound execution
  variable; it is **not** changed to 3.12 (or any other value) to "fit" the available image.
- Scientific files, training hyperparameters, architecture, dataset, fixture, gates, thresholds, seed: **unchanged**.

## 2. Options (evaluate before authorizing any run)

### Option A — Build Python 3.14.3 inside the kernel (conda/pyenv), then run
At kernel startup (`run.sh` / `script.py` preamble), create a 3.14.3 environment and re-exec under it:
```
conda create -y -n py314 python=3.14.3
source activate py314
pip install torch==2.13.0+cu126 ... transformers==5.8.0 peft==0.20.0 bitsandbytes==0.50.1 ...
python script.py        # re-run the real entry under 3.14.3
```
- Pros: works on the existing Kaggle GPU image; no new provider.
- Cons: kernel build time; must verify the `torch 2.13.0+cu126` wheel exists for CPython 3.14.3 (it may not yet
  be published — that itself is a gate); CUDA driver on the image must support CUDA 12.6 toolkit.

### Option B — Custom base image / Docker with Python 3.14.3 preinstalled
Push a Kaggle-compatible base image (or use a notebook/image that ships 3.14.3) with the full pinned stack baked in.
- Pros: reproducible, fast startup.
- Cons: image build/distribution overhead; must still confirm wheel availability for 3.14.3.

### Option C — Alternative GPU cloud offering 3.14.3
Run the identical `cloud_6c5.py` on a GPU provider whose image provides Python 3.14.3 + the pinned stack.
- Pros: cleanest match to the binding.
- Cons: leaves Kaggle; needs a separate execution authorization (different provider boundary).

### Option D — Re-bind only via explicit human authorization (NOT a silent relax)
If a 3.14.3 environment is genuinely unobtainable in a reasonable timeframe, the *only* legitimate path to a
different Python is a NEW L5 authorization that **explicitly re-binds** Python (and the dependent torch/CUDA stack)
to the achievable version, with the change reviewed and recorded. This is a conscious, reviewed decision — never a
quiet edit of `PINNED["python"]` in `cloud_6c5.py`.

## 3. Recommended sequence
1. Validate Option A feasibility first: confirm a `torch 2.13.0+cu126` wheel for CPython 3.14.3 exists; if yes,
   adopt A and bake the env into the kernel wrapper.
2. Add a **read-only diagnostic kernel** (no training) that prints the full stack
   (`python -V`, `torch.__version__`, `torch.version.cuda`, `transformers/peft/bitsandbytes` versions, Linux
   release, glibc) so the runtime compatibility check can confirm ALL components before any training authorization.
3. Re-run the provider smoke gate with the diagnostic kernel under a fresh authorization; confirm every pin matches.
4. Only after a clean, full-stack match: issue L5 v10 for the authorized B×1/C×1 training run.

## 4. What is preserved
- v9 smoke-gate PASS result (metadata: python/script, GPU on, internet off, dataset bound, title==slug).
- All prior failed attempts (409 title/slug; `kernel_metadata_get` AttributeError; 0.8 s shell-in-script.py
  SyntaxError) as historical evidence.
- All scientific artifacts and the frozen binding (`adjudication`, fixture, split, prompt, dataset, M1, bundle SHA
  `ace20fc0…`).

## 5. Authorization state
- L5 v9 remains the active authorization (smoke-only). **L5 v10 is NOT issued** until the environment provides
  Python 3.14.3 (and the rest of the pinned stack) and a fresh authorization binds the verified runtime.
