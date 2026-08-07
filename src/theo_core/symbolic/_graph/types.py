"""Generic graph types and type variables for the generic graph library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from theo_core.symbolic._primitives.identifiers import SymbolicId

NodeId = SymbolicId

N = TypeVar("N")  # Node payload type
E = TypeVar("E")  # Edge payload type


@dataclass(frozen=True, slots=True)
class EdgeKey:
    """Identifies a directed edge between a source node and a target node.

    Attributes:
        source: SymbolicId of the source node.
        target: SymbolicId of the target node.
        relation: Relationship discriminator string (default: "default").

    """

    source: NodeId
    target: NodeId
    relation: str = "default"

    def __str__(self) -> str:
        """Return human readable representation."""
        return f"({self.source.value} --[{self.relation}]--> {self.target.value})"
