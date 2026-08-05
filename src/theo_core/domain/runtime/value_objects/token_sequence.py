"""TokenSequence — an immutable sequence of token IDs with metadata."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TokenSequence(BaseModel, frozen=True):
    """An immutable sequence of integer token IDs.

    Attributes:
        ids: The ordered sequence of token IDs.
        text: The original text that produced these tokens.
        tokenizer_name: Name of the tokenizer that produced this sequence.

    """

    ids: tuple[int, ...] = Field(default_factory=tuple)
    text: str = ""
    tokenizer_name: str = "unknown"

    @property
    def length(self) -> int:
        """Return the number of tokens."""
        return len(self.ids)
