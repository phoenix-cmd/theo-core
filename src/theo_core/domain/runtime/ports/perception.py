"""PerceptionPort — interface for normalizing raw inputs into Percepts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theo_core.domain.runtime.entities.percept import Percept


class PerceptionPort(ABC):
    """Abstract interface for perceptual processing.

    All raw inputs (text, images, audio, documents) pass through
    a perception processor to become normalized Percept objects.
    """

    @abstractmethod
    def perceive(self, raw_input: str | bytes, modality: str = "text") -> Percept:
        """Process raw input into a normalized Percept.

        Args:
            raw_input: The raw input data.
            modality: The input modality (text, image, audio, document).

        Returns:
            A normalized Percept object.

        """
