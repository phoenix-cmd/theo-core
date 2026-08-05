"""DeterministicMemoryEngine — 4-layer decoupled memory engine with append-only persistence.

Decouples:
  - Storage I/O: JSONMemoryRepository
  - Retrieval & Search: MemoryRetrievalEngine & KeywordRetrievalStrategy
  - Classification: MemoryClassifier & MemoryImportance

Assigns unique IDs ('mem-000001'), enforces append-only superseding history,
and persists memory entries across CLI sessions via JSON repository.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from theo_core.domain.runtime.entities.memory_entry import MemoryEntry, MemoryImportance
from theo_core.domain.runtime.ports.memory import MemoryStorePort
from theo_core.memory.classifier.memory_classifier import MemoryClassifier
from theo_core.memory.retrieval.engine import MemoryRetrievalEngine
from theo_core.memory.storage.json_repository import JSONMemoryRepository

if TYPE_CHECKING:
    from theo_core.domain.runtime.entities.retrieved_memory import RetrievedMemory


class DeterministicMemoryEngine(MemoryStorePort):
    """4-layer memory engine with append-only history and JSON persistence.

    Layers:
        - Working Memory: Active short-term capacity buffer.
        - Semantic Memory: Key-value facts (identity & preferences).
        - Episodic Memory: Sequence of past conversation turns.
        - Long-Term Memory: Persistent disk repository.
    """

    def __init__(
        self,
        repository: JSONMemoryRepository | None = None,
        retrieval_engine: MemoryRetrievalEngine | None = None,
        classifier: MemoryClassifier | None = None,
        working_capacity: int = 100,
    ) -> None:
        """Initialize memory engine and load persisted memories.

        Args:
            repository: JSON repository for file persistence.
            retrieval_engine: Memory retrieval search engine.
            classifier: Memory category classifier.
            working_capacity: Working memory capacity bound.

        """
        self._repository = repository or JSONMemoryRepository()
        self._retrieval_engine = retrieval_engine or MemoryRetrievalEngine()
        self._classifier = classifier or MemoryClassifier()
        self._working_capacity = working_capacity

        # Load persisted long-term entries
        self._entries: list[MemoryEntry] = self._repository.load_all()
        self._counter = len(self._entries)

        # In-memory working buffer
        self._working_buffer: list[dict[str, Any]] = []

    def _next_id(self) -> str:
        """Generate next sequential memory ID (e.g. 'mem-000001')."""
        self._counter += 1
        return f"mem-{self._counter:06d}"

    def store_fact(
        self,
        key: str,
        value: Any,
        category: str = "semantic",
        source: str = "user_statement",
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        provenance: dict[str, str] | None = None,
    ) -> MemoryEntry:
        """Store a fact entry with append-only superseding for existing keys.

        Args:
            key: The memory key (e.g. 'user.name').
            value: The memory payload value (e.g. 'Falcon').
            category: Category string ('identity', 'preference', 'experience', 'semantic').
            source: Provenance source description.
            importance: MemoryImportance level enum.
            provenance: Mapping of message_id, percept_id, decision_id provenance.

        Returns:
            The created MemoryEntry instance.

        """
        mem_id = self._next_id()
        prov_dict = provenance or {}

        # Set higher importance for identity
        if category == "identity" or "name" in key:
            importance = MemoryImportance.HIGH

        # Supersede existing active entry for the same key if value differs
        for entry in self._entries:
            if entry.key == key and entry.status == "active":
                if entry.value == value:
                    entry.confirm()
                    self._repository.save_all(self._entries)
                    return entry
                entry.mark_superseded(mem_id)

        new_entry = MemoryEntry(
            id=mem_id,
            memory_type=category,
            key=key,
            value=value,
            importance=importance,
            confidence=1.0,
            source=source,
            status="active",
            provenance=prov_dict,
        )
        self._entries.append(new_entry)
        self._repository.save_all(self._entries)
        return new_entry

    def get_fact(self, key: str) -> MemoryEntry | None:
        """Retrieve the active MemoryEntry for a key.

        Args:
            key: The memory key to lookup.

        Returns:
            The active MemoryEntry, or None if not found.

        """
        for entry in self._entries:
            if entry.key == key and entry.status == "active":
                return entry
        return None

    def get_all_active(self) -> list[MemoryEntry]:
        """Return all active MemoryEntry items.

        Returns:
            List of active MemoryEntry items.

        """
        return [e for e in self._entries if e.status == "active"]

    def retrieve_scored(self, query: str, top_k: int = 5) -> list[RetrievedMemory]:
        """Retrieve scored active memory results using MemoryRetrievalEngine.

        Args:
            query: User search query string.
            top_k: Max items to return.

        Returns:
            List of scored RetrievedMemory objects.

        """
        active_candidates = self.get_all_active()
        return self._retrieval_engine.search(query, active_candidates, top_k=top_k)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve active memory items matching query keywords.

        Args:
            query: Search query string.
            top_k: Max items to return.

        Returns:
            List of matching memory entry dictionaries.

        """
        scored = self.retrieve_scored(query, top_k=top_k)
        return [rm.entry.model_dump(mode="json") for rm in scored]

    def add_working(self, item: dict[str, Any]) -> None:
        """Add an item to working memory.

        Args:
            item: Item dictionary.

        """
        self._working_buffer.append(item)
        if len(self._working_buffer) > self._working_capacity:
            self._working_buffer.pop(0)

    def get_working(self) -> list[dict[str, Any]]:
        """Return all items in working memory buffer.

        Returns:
            List of working memory items.

        """
        return list(self._working_buffer)

    @property
    def total_memory_count(self) -> int:
        """Return the total number of memory entries ever created."""
        return len(self._entries)
