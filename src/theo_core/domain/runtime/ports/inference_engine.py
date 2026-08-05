"""InferenceEnginePort — interface for model inference operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class InferenceEnginePort(ABC):
    """Abstract interface for inference execution.

    Wraps the model inference pipeline, including pre/post-processing.
    """

    @abstractmethod
    def infer(self, input_data: Any, **kwargs: Any) -> Any:
        """Run inference on the given input.

        Args:
            input_data: The input to process.
            **kwargs: Additional inference parameters.

        Returns:
            The inference result.

        """
