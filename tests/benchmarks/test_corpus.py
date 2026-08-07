"""Corpus-wide benchmark battery — every registered case must pass the harness."""

from __future__ import annotations

import pytest

from theo_core.evaluation.benchmarks import (
    ALL_CASES,
    DOMAIN_CASES,
    case_by_id,
    cases_for_domain,
)
from theo_core.evaluation.harness import BenchmarkHarness


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
