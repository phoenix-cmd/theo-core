"""Deterministic ordering utilities.

Provides utility functions for deterministic iteration over dictionaries
and collections. No ``dict`` subclassing — only pure functions.
"""

from __future__ import annotations

from typing import TypeVar

K = TypeVar("K")
V = TypeVar("V")


def sorted_items[K, V](d: dict[K, V]) -> list[tuple[K, V]]:
    """Return dict items sorted by key.

    Args:
        d: A dictionary with comparable keys.

    Returns:
        A list of ``(key, value)`` tuples sorted by key.

    """
    return sorted(d.items(), key=lambda item: str(item[0]))


def sorted_keys[K, V](d: dict[K, V]) -> list[K]:
    """Return dict keys in sorted order.

    Args:
        d: A dictionary with comparable keys.

    Returns:
        A sorted list of keys.

    """
    return sorted(d.keys(), key=str)


def sorted_values[K, V](d: dict[K, V]) -> list[V]:
    """Return dict values sorted by their corresponding key.

    Args:
        d: A dictionary with comparable keys.

    Returns:
        A list of values in key-sorted order.

    """
    return [v for _, v in sorted_items(d)]
