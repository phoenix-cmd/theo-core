"""EmbeddingModelPort — interface for text-to-vector embedding models."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingModelPort(ABC):
    """Abstract interface for embedding model operations.

    Implementations may wrap sentence-transformers, OpenAI embeddings,
    or custom trained embedding models.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text into a vector.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.

        """

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors.

        Args:
            texts: A list of input texts.

        Returns:
            A list of embedding vectors.

        """

    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors.

        Returns:
            The integer dimension of output vectors.

        """
