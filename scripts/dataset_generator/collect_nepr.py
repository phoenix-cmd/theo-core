#!/usr/bin/env python3
"""Numerical Environment Provenance Record (NEPR) — Canonical Implementation.

Single source of truth for all NEPR logic:
  - Collection (hardware, software, numerical, reproducibility, artifact binding)
  - Canonical hashing (SHA-256 with sorted-key JSON)
  - Schema validation (MAJOR.MINOR.PATCH with required-field registry)
  - Deterministic environment comparison (SAME / CROSS / UNVERIFIABLE)
  - Runtime detection (execution topology, LoRA dtype, attention backend)

This module is imported by:
  - run_phase_6e20_reevaluation.py (pipeline integration)
  - test_phase_6e20_reevaluation.py (unit tests)

Revision: 6E.20.3 Implementation Plan Revision 3 (2026-08-18).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NEPR_SCHEMA_VERSION = "1.0.0"
NEPR_FILE_NAME = "numerical_environment_provenance.json"
BINDING_FILE_NAME = "execution_environment_binding.json"

# Required fields per schema version.  Key = schema version string,
# Value = set of dotted-path field names that are REQUIRED in that version.
# Fields not listed here for a given version are treated as not-yet-defined
# when comparing against an older compatible MINOR version.
REQUIRED_FIELDS_BY_VERSION: Dict[str, set] = {
    "1.0.0": {
        "nepr_schema_version",
        "category_A_software.python_version",
        "category_A_software.torch_version",
        "category_A_software.cuda_version",
        "category_A_software.transformers_version",
        "category_A_software.peft_version",
        "category_A_software.numpy_version",
        "category_A_software.platform",
        "category_A_software.processor",
        "category_B_hardware.gpu_count",
        "category_B_hardware.gpu_names",
        "category_B_hardware.gpu_memory_bytes",
        "category_B_hardware.cuda_device_count",
        "category_B_hardware.cuda_device_names",
        "category_B_hardware.cuda_compute_capability",
        "category_B_hardware.cuda_supports_tf32",
        "category_B_hardware.model_parallel",
        "category_B_hardware.device_map",
        "category_B_hardware.world_size",
        "category_B_hardware.per_device_batch_size",
        "category_B_hardware.cpu_count",
        "category_B_hardware.ram_bytes",
        "category_B_hardware.hostname_hash",
        "category_C_numerical.base_model_dtype",
        "category_C_numerical.lora_dtype",
        "category_C_numerical.lora_dtype_detail",
        "category_C_numerical.attention_backend",
        "category_C_numerical.gradient_checkpointing",
        "category_C_numerical.use_deterministic_algorithms",
        "category_C_numerical.cudnn_deterministic",
        "category_C_numerical.cudnn_benchmark",
        "category_C_numerical.matmul_allow_tf32",
        "category_C_numerical.cudnn_allow_tf32",
        "category_D_reproducibility_state.random_seed_set",
        "category_D_reproducibility_state.torch_manual_seed",
        "category_D_reproducibility_state.numpy_manual_seed",
        "category_D_reproducibility_state.cuda_manual_seed_all",
        "category_D_reproducibility_state.torch_deterministic_algorithms",
        "category_D_reproducibility_state.env_timestamp_utc",
        "category_E_artifact_binding.runner_sha256",
        "category_E_artifact_binding.test_sha256",
        "category_E_artifact_binding.base_model_id",
        "category_E_artifact_binding.base_model_revision",
        "category_E_artifact_binding.phase_id",
        "category_F_metadata.nepr_timestamp_utc",
        "category_F_metadata.nepr_hash",
    },
}


# ---------------------------------------------------------------------------
# Canonical Hashing
# ---------------------------------------------------------------------------
def compute_nepr_hash(nepr_dict: dict) -> str:
    """Compute SHA-256 of NEPR, excluding ONLY the self-referential nepr_hash field.

    Per Revision 3 §4.1 / §4.3: the canonical hash excludes nepr_hash (self-
    referential exclusion) and NOTHING ELSE. Timestamp fields (nepr_timestamp_utc,
    env_timestamp_utc) are intentionally INCLUDED — they are informational for
    comparison gating (category D/F) but are part of the tamper-evident hash
    contract. Excluding them would be a specification drift from Revision 3.

    Uses canonical JSON: sorted keys, compact separators, ensure_ascii=True.
    """
    import copy as _copy
    d = _copy.deepcopy(nepr_dict)
    # Remove nepr_hash wherever it appears (top-level or nested in category_F_metadata)
    d.pop("nepr_hash", None)
    if "category_F_metadata" in d:
        fm = dict(d["category_F_metadata"])
        fm.pop("nepr_hash", None)
        d["category_F_metadata"] = fm
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Atomic Write
# ---------------------------------------------------------------------------
def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically using os.replace (no unlink+rename window)."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Hardware Topology Collection
# ---------------------------------------------------------------------------
def _get_ram_bytes() -> int:
    """Return total system RAM in bytes."""
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except ImportError:
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 0


def _hash_hostname() -> str:
    """Return SHA-256 hash of the hostname."""
    node = platform.node()
    return "sha256:" + hashlib.sha256(node.encode("utf-8")).hexdigest()


def collect_category_B_hardware() -> dict:
    """Collect hardware topology from the current environment.

    model_parallel and device_map are placeholders; they are updated after
    model load by detect_execution_topology().
    """
    import torch

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_names: list = []
    gpu_memory_bytes: list = []
    cuda_device_names: list = []
    cuda_compute_capability: list = []

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        gpu_names.append(props.name)
        gpu_memory_bytes.append(props.total_memory)  # raw bytes
        cuda_device_names.append(props.name)
        cuda_compute_capability.append(f"{props.major}.{props.minor}")

    cuda_supports_tf32 = False
    if gpu_count > 0:
        cc = torch.cuda.get_device_capability(0)
        cuda_supports_tf32 = (cc[0] >= 8)  # Ampere+

    return {
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "gpu_memory_bytes": gpu_memory_bytes,
        "cuda_device_count": gpu_count,
        "cuda_device_names": cuda_device_names,
        "cuda_compute_capability": cuda_compute_capability,
        "cuda_supports_tf32": cuda_supports_tf32,
        "model_parallel": False,   # placeholder — updated after model load
        "device_map": "unknown",   # placeholder — updated after model load
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "per_device_batch_size": 1,
        "cpu_count": os.cpu_count() or 0,
        "ram_bytes": _get_ram_bytes(),
        "hostname_hash": _hash_hostname(),
    }


# ---------------------------------------------------------------------------
# Execution Topology Detection (NOT derived from gpu_count)
# ---------------------------------------------------------------------------
def _canonicalize_device_map(hf_device_map) -> str:
    """Canonicalize device_map to a deterministic string representation.

    Dict device maps are serialized as canonical JSON with sorted keys.
    This prevents false CROSS_ENVIRONMENT from insertion-order differences.
    """
    if isinstance(hf_device_map, dict):
        return json.dumps(hf_device_map, sort_keys=True, separators=(",", ":"))
    return str(hf_device_map)


def detect_execution_topology(model, gpu_count: int) -> Tuple[bool, str]:
    """Detect actual model parallelism and device map from loaded model.

    Returns (model_parallel, device_map).
    """
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map is not None and hf_device_map:
        device_map = _canonicalize_device_map(hf_device_map)
        if isinstance(hf_device_map, dict):
            devices = set(v for v in hf_device_map.values() if isinstance(v, str))
            model_parallel = len(devices) > 1
        elif isinstance(hf_device_map, str) and hf_device_map != "cpu":
            model_parallel = gpu_count > 1
        else:
            model_parallel = False
        return model_parallel, device_map

    if hasattr(model, "module"):  # DataParallel / DDP wrapper
        return True, "distributed"

    try:
        device = next(model.parameters()).device
        device_map = str(device)
    except StopIteration:
        device_map = "cpu"

    return False, device_map


# ---------------------------------------------------------------------------
# LoRA Dtype Measurement (NEVER inferred from base model dtype)
# ---------------------------------------------------------------------------
def measure_lora_dtype(peft_model) -> Tuple[str, dict]:
    """Measure actual LoRA parameter dtypes from the loaded PEFT model.

    Returns (lora_dtype, lora_dtype_detail) where:
      lora_dtype: "bfloat16" | "float32" | "float16" | "mixed" | "unknown"
      lora_dtype_detail: {dtype_name: param_count}
    """
    dtype_counter: Counter = Counter()
    total_params = 0

    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            dtype_name = str(param.dtype).replace("torch.", "")
            dtype_counter[dtype_name] += 1
            total_params += 1

    if total_params == 0:
        return "unknown", {}

    detail = dict(dtype_counter)
    unique_dtypes = list(dtype_counter.keys())
    lora_dtype = unique_dtypes[0] if len(unique_dtypes) == 1 else "mixed"
    return lora_dtype, detail


# ---------------------------------------------------------------------------
# Attention Backend Detection (actual selected backend, NOT library availability)
# ---------------------------------------------------------------------------
def detect_attention_backend(model) -> str:
    """Detect the actual attention backend selected for this model.

    Hierarchy:
      1. Model config attn_implementation (explicitly configured)
      2. "unknown" if resolution impossible

    Package availability (import flash_attn, import xformers) is NOT used.
    """
    config = getattr(model, "config", None)
    if config is not None:
        attn_impl = getattr(config, "attn_implementation", None)
        if attn_impl is not None:
            mapping = {
                "flash_attention_2": "flash",
                "sdpa": "sdpa",
                "eager": "eager",
                "xformers": "xformers",
            }
            return mapping.get(attn_impl, str(attn_impl))
    return "unknown"


# ---------------------------------------------------------------------------
# NEPR Collection (main entry point)
# ---------------------------------------------------------------------------
def collect_nepr(
    model=None,
    tokenizer=None,
    peft_model=None,
    phase_id: str = "6E.20.3",
    runner_sha256: Optional[str] = None,
    test_sha256: Optional[str] = None,
    batch_size_override: int = 1,
) -> dict:
    """Collect the full NEPR from the current environment.

    If model/peft_model are provided, measures runtime-dependent fields.
    If not provided, runtime-dependent fields are set to "unknown".
    """
    import torch

    # --- Category A: Software ---
    category_a = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "transformers_version": _import_version("transformers"),
        "peft_version": _import_version("peft"),
        "numpy_version": _import_version("numpy"),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    # --- Category B: Hardware ---
    category_b = collect_category_B_hardware()
    category_b["per_device_batch_size"] = batch_size_override

    # Update model_parallel and device_map if model is provided
    if model is not None:
        model_parallel, device_map = detect_execution_topology(
            model, category_b["gpu_count"]
        )
        category_b["model_parallel"] = model_parallel
        category_b["device_map"] = device_map

    # --- Category C: Numerical ---
    category_c: dict = {}
    if model is not None:
        dtype = getattr(getattr(model, "config", None), "torch_dtype", None)
        category_c["base_model_dtype"] = str(dtype) if dtype is not None else "unknown"
    else:
        category_c["base_model_dtype"] = "unknown"

    if peft_model is not None:
        lora_dtype, lora_dtype_detail = measure_lora_dtype(peft_model)
    else:
        lora_dtype, lora_dtype_detail = "unknown", {}

    category_c["lora_dtype"] = lora_dtype
    category_c["lora_dtype_detail"] = lora_dtype_detail

    if model is not None:
        category_c["attention_backend"] = detect_attention_backend(model)
    else:
        category_c["attention_backend"] = "unknown"

    category_c["gradient_checkpointing"] = bool(
        getattr(model, "gradient_checkpointing", False) if model is not None else False
    )
    _det_fn = getattr(torch, "is_deterministic_algorithms_mode", None)
    if _det_fn is None:
        _det_fn = getattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    category_c["use_deterministic_algorithms"] = bool(_det_fn())
    category_c["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    category_c["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    category_c["matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    category_c["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)

    # --- Category D: Reproducibility State ---
    category_d = {
        "random_seed_set": False,
        "torch_manual_seed": None,
        "numpy_manual_seed": None,
        "cuda_manual_seed_all": None,
        "torch_deterministic_algorithms": bool(_det_fn()),
        "env_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # --- Category E: Artifact Binding ---
    category_e = {
        "runner_sha256": runner_sha256 or "unknown",
        "test_sha256": test_sha256 or "unknown",
        "base_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "base_model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "phase_id": phase_id,
    }

    # --- Category F: Metadata (hash computed last) ---
    nepr = {
        "nepr_schema_version": NEPR_SCHEMA_VERSION,
        "category_A_software": category_a,
        "category_B_hardware": category_b,
        "category_C_numerical": category_c,
        "category_D_reproducibility_state": category_d,
        "category_E_artifact_binding": category_e,
        "category_F_metadata": {
            "nepr_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "nepr_hash": "",  # placeholder — computed below
        },
    }

    nepr["category_F_metadata"]["nepr_hash"] = compute_nepr_hash(nepr)
    return nepr


def _import_version(module_name: str) -> str:
    """Safely import and return a module's __version__."""
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "unknown"


# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------
def validate_nepr_schema(nepr: dict) -> Tuple[bool, str]:
    """Validate NEPR schema version and format.

    Returns (valid, reason).
    """
    version = nepr.get("nepr_schema_version")
    if version is None:
        return False, "Legacy NEPR: missing nepr_schema_version"

    if not re.match(r"^\d+\.\d+\.\d+$", version):
        return False, f"Invalid nepr_schema_version format: {version}"

    major = int(version.split(".")[0])
    if major != 1:
        return False, f"NEPR schema MAJOR version {major} != expected 1"

    return True, "ok"


# ---------------------------------------------------------------------------
# Deterministic Environment Comparison
# ---------------------------------------------------------------------------
def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict to dotted-path keys."""
    flat = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, full_key))
        else:
            flat[full_key] = v
    return flat


def compare_nepr(current: dict, reference: dict) -> dict:
    """Compare two NEPR records deterministically.

    Returns a comparison report dict with:
      classification: "SAME_ENVIRONMENT" | "CROSS_ENVIRONMENT" | "ENVIRONMENT_UNVERIFIABLE"
      category_results: {A: bool, B: bool, C: bool, D: informational, E: bool, F: informational}
      field_details: {field_name: {current, reference, equal, category, informational}}
      disclaimers: list[str]
      comparison_timestamp_utc: str
      current_nepr_hash: str
      reference_nepr_hash: str
    """
    current_flat = _flatten_dict(current)
    reference_flat = _flatten_dict(reference)

    # Determine which fields to compare based on required-field registry
    curr_version = current.get("nepr_schema_version", "0.0.0")
    ref_version = reference.get("nepr_schema_version", "0.0.0")

    curr_major = int(curr_version.split(".")[0]) if curr_version else 0
    ref_major = int(ref_version.split(".")[0]) if ref_version else 0

    # Step 1: Schema compatibility
    if curr_major != ref_major:
        return _make_report(
            "ENVIRONMENT_UNVERIFIABLE",
            reason=f"MAJOR schema version mismatch: {curr_version} vs {ref_version}",
            current_hash=compute_nepr_hash(current),
            reference_hash=compute_nepr_hash(reference),
        )

    if curr_major == 0:
        return _make_report(
            "ENVIRONMENT_UNVERIFIABLE",
            reason="Legacy NEPR: missing or invalid nepr_schema_version",
            current_hash=compute_nepr_hash(current),
            reference_hash=compute_nepr_hash(reference),
        )

    # Step 2: Determine required fields for comparison
    # Use the INTERSECTION of required fields from both schema versions.
    # Fields that exist only in one version are excluded (not-yet-defined).
    curr_required = REQUIRED_FIELDS_BY_VERSION.get(curr_version, set())
    ref_required = REQUIRED_FIELDS_BY_VERSION.get(ref_version, set())

    # For same-version comparison, use that version's required fields
    if curr_version == ref_version:
        required_for_comparison = curr_required
    else:
        # Cross-MINOR: use intersection of required fields.
        # A field required in v1.0.0 but not defined in v1.1.0 is excluded.
        required_for_comparison = curr_required & ref_required

    # All fields that exist in either NEPR (for informational reporting)
    all_fields = set(current_flat.keys()) | set(reference_flat.keys())

    # Fields to compare: all fields present in both NEPRs
    comparable_fields = set(current_flat.keys()) & set(reference_flat.keys())

    field_details = {}
    category_results = {
        "A_software": True,
        "B_hardware": True,
        "C_numerical": True,
        "D_reproducibility": "informational",
        "E_artifact_binding": True,
        "F_metadata": "informational",
    }
    disclaimers = []
    first_mismatch_category = None

    for field in sorted(all_fields):
        curr_val = current_flat.get(field)
        ref_val = reference_flat.get(field)

        # Determine category
        category = _field_category(field)
        informational = category in ("D", "F")

        # Check for missing required field
        if field in required_for_comparison:
            if field not in current_flat or field not in reference_flat:
                # Missing required field -> ENVIRONMENT_UNVERIFIABLE
                return _make_report(
                    "ENVIRONMENT_UNVERIFIABLE",
                    reason=f"Required field missing: {field}",
                    current_hash=compute_nepr_hash(current),
                    reference_hash=compute_nepr_hash(reference),
                )

        if field not in comparable_fields:
            # Field exists in only one NEPR — not comparable (not-yet-defined)
            field_details[field] = {
                "current": curr_val,
                "reference": ref_val,
                "equal": False,
                "category": category,
                "informational": True,
                "note": "not defined in both schema versions",
            }
            continue

        # Compare
        equal = (curr_val == ref_val)
        field_details[field] = {
            "current": curr_val,
            "reference": ref_val,
            "equal": equal,
            "category": category,
            "informational": informational,
        }

        if not equal and not informational:
            cat_key = f"{category}_software" if category == "A" else \
                      f"{category}_hardware" if category == "B" else \
                      f"{category}_numerical" if category == "C" else \
                      f"{category}_artifact_binding" if category == "E" else None
            if cat_key and cat_key in category_results:
                category_results[cat_key] = False
                if first_mismatch_category is None:
                    first_mismatch_category = category

    # Step 5: Classification
    all_gating_equal = all(
        v is True for k, v in category_results.items()
        if k not in ("D_reproducibility", "F_metadata")
    )

    if all_gating_equal:
        classification = "SAME_ENVIRONMENT"
        disclaimers.append(
            "SAME_ENVIRONMENT does not imply SAME_EXECUTION_STATE. "
            "Category D (reproducibility state) differences may affect "
            "numerical reproducibility."
        )
    else:
        classification = "CROSS_ENVIRONMENT"

    return {
        "classification": classification,
        "category_results": category_results,
        "field_details": field_details,
        "disclaimers": disclaimers,
        "comparison_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "current_nepr_hash": compute_nepr_hash(current),
        "reference_nepr_hash": compute_nepr_hash(reference),
    }


def _field_category(field: str) -> str:
    """Extract category letter from a dotted field path."""
    if field.startswith("category_A"): return "A"
    if field.startswith("category_B"): return "B"
    if field.startswith("category_C"): return "C"
    if field.startswith("category_D"): return "D"
    if field.startswith("category_E"): return "E"
    if field.startswith("category_F"): return "F"
    return "?"


def _make_report(
    classification: str,
    reason: str = "",
    current_hash: str = "",
    reference_hash: str = "",
) -> dict:
    """Build a minimal comparison report."""
    report = {
        "classification": classification,
        "category_results": {},
        "field_details": {},
        "disclaimers": [],
        "comparison_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "current_nepr_hash": current_hash,
        "reference_nepr_hash": reference_hash,
    }
    if reason:
        report["reason"] = reason
    return report


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    """CLI entry point. Collects and prints NEPR as JSON."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect Numerical Environment Provenance Record")
    parser.add_argument("--runner-sha256", type=str, default=None)
    parser.add_argument("--test-sha256", type=str, default=None)
    parser.add_argument("--phase-id", type=str, default="6E.20.3")
    args = parser.parse_args()

    nepr = collect_nepr(
        phase_id=args.phase_id,
        runner_sha256=args.runner_sha256,
        test_sha256=args.test_sha256,
    )
    print(json.dumps(nepr, indent=2))


if __name__ == "__main__":
    main()
