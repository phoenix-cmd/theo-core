"""Cognitive Benchmark Corpus — the standardized v0.4.1 evaluation battery.

Each domain module exports a ``CASES`` tuple of ``BenchmarkCase`` instances.
This registry aggregates them for the ``theo benchmark run`` CLI and tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import causal_reasoning, commonsense, contradiction, taxonomy, uncertainty

if TYPE_CHECKING:
    from theo_core.evaluation.benchmark_schema import BenchmarkCase

DOMAIN_CASES: dict[str, tuple[BenchmarkCase, ...]] = {
    "causal_reasoning": causal_reasoning.CASES,
    "commonsense": commonsense.CASES,
    "contradiction": contradiction.CASES,
    "taxonomy": taxonomy.CASES,
    "uncertainty": uncertainty.CASES,
}

ALL_CASES: tuple[BenchmarkCase, ...] = tuple(
    case for cases in DOMAIN_CASES.values() for case in cases
)


def cases_for_domain(domain: str | None) -> tuple[BenchmarkCase, ...]:
    """Return the cases for a single domain, or the full corpus when None.

    Args:
        domain: Optional domain name filter (e.g. ``"causal_reasoning"``).

    Returns:
        The matching benchmark cases in deterministic order.

    Raises:
        ValueError: If the requested domain is unknown.

    """
    if domain is None:
        return ALL_CASES
    if domain not in DOMAIN_CASES:
        known = ", ".join(sorted(DOMAIN_CASES))
        msg = f"Unknown benchmark domain {domain!r}; known domains: {known}"
        raise ValueError(msg)
    return DOMAIN_CASES[domain]


def case_by_id(benchmark_id: str) -> BenchmarkCase | None:
    """Look up a single benchmark case by its ``bm://`` identifier.

    Args:
        benchmark_id: The benchmark case URI (e.g. ``"bm://causal_reasoning/001"``).

    Returns:
        The matching case, or None if not found.

    """
    for case in ALL_CASES:
        if case.id.value == benchmark_id:
            return case
    return None
