"""Corpus-wide benchmark battery — every registered case must pass the harness."""

from __future__ import annotations

import pytest

from theo_core.domain.runtime.entities.goal import _slugify
from theo_core.evaluation.benchmark_schema import FailureMode
from theo_core.evaluation.benchmarks import (
    ALL_CASES,
    DOMAIN_CASES,
    case_by_id,
    cases_for_domain,
)
from theo_core.evaluation.harness import BenchmarkHarness
from theo_core.symbolic.decisions.engine import _INTENT_BY_GOAL_SLUG
from theo_core.symbolic.decisions.models import Intent

_BASELINE_MARKER = "v0.4.1 (ADR-0028 Phase 2, pre-provider)"


@pytest.mark.parametrize("domain", sorted(DOMAIN_CASES))
def test_all_cases_in_domain_pass(domain: str) -> None:
    """Every benchmark case in a domain must pass the deterministic harness."""
    results = BenchmarkHarness.run_all(DOMAIN_CASES[domain])
    failures = [(result.case_id, result.failures) for result in results if not result.passed]
    assert not failures, f"{len(failures)} case(s) failed: {failures}"


@pytest.mark.parametrize("domain", sorted(DOMAIN_CASES))
def test_domain_has_at_least_five_cases(domain: str) -> None:
    """Phase E requires at least five cases per domain."""
    assert len(DOMAIN_CASES[domain]) >= 5


def test_case_ids_are_unique() -> None:
    """Every benchmark case must carry a unique bm:// identifier."""
    ids = [case.id.value for case in ALL_CASES]
    assert len(ids) == len(set(ids))


def test_corpus_has_at_least_fifty_cases() -> None:
    """ADR-0028 Phase 2 expanded the corpus to at least fifty blind cases."""
    assert len(ALL_CASES) >= 50


def test_every_failure_mode_is_covered() -> None:
    """Each declared FailureMode must be exercised by at least one case."""
    declared = {case.failure_mode for case in ALL_CASES if case.failure_mode is not None}
    assert declared == set(FailureMode)


def test_failure_mode_cases_have_frozen_baseline() -> None:
    """Cases declaring a failure mode must carry a frozen v0.4.1 baseline."""
    for case in ALL_CASES:
        if case.failure_mode is None:
            continue
        assert case.baseline, f"{case.id}: failure mode without frozen baseline"
        assert case.baseline["captured_at"] == _BASELINE_MARKER, case.id
        assert case.baseline["decision_id"], case.id
        assert case.baseline["state_checksum"], case.id
        fingerprint = case.baseline["fingerprint"]
        assert isinstance(fingerprint, dict), case.id
        assert fingerprint["decision_id"], case.id


def test_expected_intent_matches_seeded_goal() -> None:
    """A seeded goal must slugify to the intent the decision engine will infer."""
    for case in ALL_CASES:
        if not case.initial_goals:
            assert case.expected_intent in (None, Intent.MAINTAIN_CONVERSATION), (
                f"{case.id}: unseeded case declares non-default intent"
            )
            continue
        assert case.expected_intent is not None, f"{case.id}: goals seeded but no intent declared"
        slug = _slugify(case.initial_goals[0])
        assert _INTENT_BY_GOAL_SLUG.get(slug) == case.expected_intent, (
            f"{case.id}: goal slug {slug!r} does not map to {case.expected_intent}"
        )


def test_frozen_baselines_match_fresh_runs() -> None:
    """Frozen baselines must still match the deterministic v0.4.1 pipeline."""
    harness = BenchmarkHarness()
    for case in ALL_CASES:
        if not case.baseline:
            continue
        decision = harness.run(case).decision
        baseline = case.baseline
        assert decision.id.value == baseline["decision_id"], case.id
        assert format(decision.confidence, ".4f") == baseline["confidence"], case.id
        assert decision.intent.value == baseline["intent"], case.id
        assert decision.referenced_goal.value == baseline["referenced_goal"], case.id


def test_case_by_id_lookup() -> None:
    """case_by_id must resolve a known case and return None for an unknown one."""
    case = case_by_id("bm://causal_reasoning/001")
    assert case is not None
    assert case.name == "CAUSAL-001: Rain and Wet Ground"
    assert case_by_id("bm://missing/999") is None


def test_cases_for_domain_rejects_unknown() -> None:
    """cases_for_domain must reject unknown domains with a ValueError."""
    with pytest.raises(ValueError, match="Unknown benchmark domain"):
        cases_for_domain("nonsense")
