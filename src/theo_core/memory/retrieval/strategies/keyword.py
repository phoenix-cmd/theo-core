"""KeywordRetrievalStrategy — deterministic keyword & exact-match memory retrieval strategy."""

from __future__ import annotations

from theo_core.domain.runtime.entities.memory_entry import MemoryEntry, MemoryImportance
from theo_core.domain.runtime.entities.retrieved_memory import RetrievedMemory
from theo_core.memory.retrieval.strategies.base import RetrievalStrategy


class KeywordRetrievalStrategy(RetrievalStrategy):
    """Deterministic keyword & exact key match memory retrieval strategy.

    Sorting Invariant:
        Results are sorted deterministically by:
        1. Score (DESC)
        2. Importance (PERMANENT > HIGH > MEDIUM > LOW)
        3. Creation timestamp (DESC)
        4. Memory ID (ASC)
    """

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "KeywordRetrievalStrategy"

    def search(
        self,
        query: str,
        entries: list[MemoryEntry],
        top_k: int = 5,
    ) -> list[RetrievedMemory]:
        """Search memory entries with deterministic sorting.

        Args:
            query: User query text string.
            entries: List of active MemoryEntry items.
            top_k: Maximum number of results to return.

        Returns:
            List of scored RetrievedMemory objects.

        """
        query_lower = query.lower()
        words = set(query_lower.split())
        scored_results: list[RetrievedMemory] = []

        importance_order = {
            MemoryImportance.PERMANENT.value: 4,
            MemoryImportance.HIGH.value: 3,
            MemoryImportance.MEDIUM.value: 2,
            MemoryImportance.LOW.value: 1,
            "permanent": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
        }

        for entry in entries:
            k_lower = entry.key.lower()
            v_lower = str(entry.value).lower()
            score = 0.0
            reason = "No match"

            if query_lower in k_lower or query_lower in v_lower:
                score = 1.0
                reason = f"Exact substring match for key '{entry.key}'"
            elif any(w in k_lower or w in v_lower for w in words):
                score = 0.8
                reason = f"Keyword match for key '{entry.key}'"
            elif entry.memory_type in ("identity", "preference"):
                # Core identity/preference baseline retrieval
                score = 0.5
                reason = f"Core {entry.memory_type} memory recall"

            if score > 0.0:
                scored_results.append(
                    RetrievedMemory(
                        memory_id=entry.id,
                        score=score,
                        retrieval_reason=reason,
                        entry=entry,
                    )
                )

        # Deterministic sorting: (score DESC, importance DESC, created_at DESC, memory_id ASC)
        scored_results.sort(
            key=lambda rm: (
                -rm.score,
                -importance_order.get(
                    rm.entry.importance.value
                    if hasattr(rm.entry.importance, "value")
                    else str(rm.entry.importance),
                    2,
                ),
                -rm.entry.created_at.timestamp(),
                rm.entry.id,
            )
        )

        return scored_results[:top_k]
