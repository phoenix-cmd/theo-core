"""Determinism — cross-process fingerprint equality.

The determinism guarantee of the canonical symbolic pipeline is verified
across separate OS processes: two independent processes running identical
input MUST produce identical fingerprints (sha256 of the golden trace).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"

_PROBE = r"""
import hashlib
import json
import sys

sys.path.insert(0, r"@SRC@")

from decimal import Decimal

from theo_core.symbolic.inference.models import (
    InferenceRule,
    RuleCondition,
    RuleId,
)
from theo_core.symbolic.pipeline import SymbolicCognitivePipeline


def main() -> None:
    pipeline = SymbolicCognitivePipeline(
        rules=(
            InferenceRule(
                id=RuleId.of("rule://probe/rain_wet"),
                name="rain wets the ground",
                conditions=(RuleCondition(premise_predicate="rain"),),
                conclusion_template="The ground is wet",
                confidence_multiplier=Decimal("0.9"),
            ),
        ),
    )
    decision, _, trace = pipeline.execute_cycle("rain is falling")
    fingerprint = {
        "decision_id": trace.decision_id.value if trace.decision_id else None,
        "response_text": decision.action_text,
        "fired_rule_ids": [str(s) for s in trace.fired_rule_ids],
        "derived_belief_ids": [str(s) for s in trace.derived_belief_ids],
        "generated_hypothesis_ids": [str(s) for s in trace.generated_hypothesis_ids],
        "thought_dag_node_count": trace.thought_dag_node_count,
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    ).hexdigest()
    print(digest)


if __name__ == "__main__":
    main()
""".replace("@SRC@", str(_SRC))


class TestCrossProcessDeterminism:
    def _run_probe(self) -> str:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(  # noqa: S603 - trusted module-constant probe script
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_SRC.parent),
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    def test_two_processes_produce_identical_fingerprint(self) -> None:
        first = self._run_probe()
        second = self._run_probe()

        assert len(first) == 64
        assert first == second

    def test_fingerprint_stable_across_repeated_process_runs(self) -> None:
        digests = {self._run_probe() for _ in range(3)}
        assert len(digests) == 1
