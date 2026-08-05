"""ModelConfig — immutable configuration for a language model architecture."""

from __future__ import annotations

from pydantic import BaseModel


class ModelConfig(BaseModel, frozen=True):
    """Immutable configuration describing a model architecture.

    Attributes:
        name: Human-readable model name.
        vocab_size: Size of the token vocabulary.
        embedding_dim: Dimension of embedding vectors.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        max_sequence_length: Maximum input sequence length.
        dtype: Data type for model weights (e.g. "float32", "bfloat16").

    """

    name: str = "unnamed"
    vocab_size: int = 32000
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_sequence_length: int = 2048
    dtype: str = "float32"
