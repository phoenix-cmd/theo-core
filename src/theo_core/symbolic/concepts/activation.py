"""Deterministic spreading activation engine using Decimal arithmetic.

Transient activation state is returned in an ActivationResult object and is NOT stored
on the immutable Concept nodes themselves.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from theo_core.symbolic.concepts.models import ConceptId

if TYPE_CHECKING:
    from theo_core.symbolic.concepts.graph import ConceptGraph


@dataclass(frozen=True, slots=True)
class ActivationResult:
    """Transient runtime activation state snapshot.

    Attributes:
        activations: Mapping of ConceptId to Decimal activation weight.
        seed_ids: Tuple of initial seed ConceptIds activated.
        decay_factor: Spreading activation decay factor (0.0 to 1.0).
        max_depth: Maximum propagation depth steps.

    """

    activations: dict[ConceptId, Decimal]
    seed_ids: tuple[ConceptId, ...]
    decay_factor: Decimal
    max_depth: int


class ActivationEngine:
    """Deterministic spreading activation engine.

    Complexity Contract:
        Time: O(V + E)
        Memory: O(V)
        Deterministic: YES
    """

    @staticmethod
    def activate(
        graph: ConceptGraph,
        seeds: dict[ConceptId, Decimal],
        decay_factor: Decimal = Decimal("0.5"),
        max_depth: int = 3,
    ) -> ActivationResult:
        """Propagate activation values from seed concepts through the graph.

        Args:
            graph: The ConceptGraph to activate over.
            seeds: Initial dictionary mapping ConceptId to seed activation value (e.g. 1.0).
            decay_factor: Decay multiplier applied per hop step (default 0.5).
            max_depth: Maximum number of edge hops to propagate.

        Returns:
            An immutable ActivationResult containing computed activation weights.

        """
        activations: dict[ConceptId, Decimal] = {}
        queue: deque[tuple[ConceptId, Decimal, int]] = deque()

        # Initialize seeds
        for seed_id, val in seeds.items():
            if graph.has_concept(seed_id):
                activations[seed_id] = val
                queue.append((seed_id, val, 0))

        while queue:
            current_id, current_val, depth = queue.popleft()

            if depth >= max_depth:
                continue

            # Spread to outgoing edges in deterministic sorted order
            edges = graph.raw_graph.get_edges_from(current_id.to_symbolic_id())
            for ek, edge in edges:
                target_id = ConceptId(value=ek.target.value)
                # Spread value = current_val * edge.weight * decay_factor
                propagated = current_val * edge.weight * decay_factor

                if propagated <= Decimal("0.001"):  # Threshold cutoff
                    continue

                existing = activations.get(target_id, Decimal("0.0"))
                new_val = existing + propagated

                if target_id not in activations or new_val > existing:
                    activations[target_id] = new_val
                    queue.append((target_id, propagated, depth + 1))

        return ActivationResult(
            activations=activations,
            seed_ids=tuple(seeds.keys()),
            decay_factor=decay_factor,
            max_depth=max_depth,
        )
