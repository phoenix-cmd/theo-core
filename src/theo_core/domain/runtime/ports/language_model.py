"""LanguageModelPort — interface for text generation models.

This port abstracts all language model operations. Implementations may wrap
local transformers, remote APIs, or any text generation backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


class LanguageModelPort(ABC):
    """Abstract interface for language model operations.

    All language model implementations must conform to this contract.
    The port is transport-agnostic: local and remote backends both implement it.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs: object) -> str:
        """Generate a text completion for the given prompt.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional generation parameters.

        Returns:
            The generated text completion.

        """

    @abstractmethod
    def stream(self, prompt: str, **kwargs: object) -> Iterator[str]:
        """Stream a text completion token by token.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional generation parameters.

        Yields:
            Individual text tokens as they are generated.

        """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs.

        Args:
            text: The input text to encode.

        Returns:
            A list of integer token IDs.

        """

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back into text.

        Args:
            token_ids: A list of integer token IDs.

        Returns:
            The decoded text string.

        """
