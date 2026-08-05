"""RetrievedMemory — scored memory retrieval result."""

from __future__ import annotations

from pydantic import BaseModel, Field

from theo_core.domain.runtime.entities.memory_entry import MemoryEntry  # noqa: TC001


class RetrievedMemory(BaseModel):
    """A scored memory retrieval result wrapper.

    Provides a clean, uniform interface for retrieval scoring regardless of backend
    (regex matching today, vector similarity or graph traversal tomorrow).

    Attributes:
        memory_id: The ID of the retrieved memory entry (e.g. 'mem-000001').
        score: Relevance score between 0.0 and 1.0.
        retrieval_reason: Explanation of why this memory was retrieved.
        entry: The underlying MemoryEntry model.

    """

    memory_id: str
    score: float = Field(ge=0.0, le=1.0, default=1.0)
    retrieval_reason: str = "Exact key match"
    entry: MemoryEntry


RetrievedMemory.model_rebuild()
