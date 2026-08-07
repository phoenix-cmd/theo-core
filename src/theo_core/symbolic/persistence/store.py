"""SymbolicStateStore — versioned, checksummed persistence for committed symbolic state.

Persists the committed ``CycleState`` (concept graph, belief graph, thought graph,
and percept) as a canonical JSON envelope with an outer SHA-256 checksum. Corruption
is detected at load time and raised as ``ChecksumMismatchError`` — it is never
silently swallowed (Canon Invariant 8, deterministic integrity).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from theo_core.symbolic._graph.serialization import (
    GraphLoader,
    GraphSerializer,
    compute_canonical_checksum,
)
from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.beliefs.models import Belief, BeliefEdge
from theo_core.symbolic.concepts.graph import ConceptGraph
from theo_core.symbolic.concepts.models import Concept, ConceptEdge
from theo_core.symbolic.perception.models import Percept, PerceptId
from theo_core.symbolic.thoughts.graph import ThoughtGraph
from theo_core.symbolic.thoughts.models import Thought, ThoughtEdge

if TYPE_CHECKING:
    from pydantic import BaseModel

    from theo_core.symbolic.pipeline import CycleState


def _model_to_json_dict(model: BaseModel) -> dict[str, Any]:
    """Dump a pydantic model to a JSON-serializable dict."""
    return model.model_dump(mode="json")


def _dict_to_concept(data: dict[str, Any]) -> Concept:
    """Validate a Concept from a JSON dict."""
    return Concept.model_validate(data)


def _dict_to_concept_edge(data: dict[str, Any]) -> ConceptEdge:
    """Validate a ConceptEdge from a JSON dict."""
    return ConceptEdge.model_validate(data)


def _dict_to_belief(data: dict[str, Any]) -> Belief:
    """Validate a Belief from a JSON dict."""
    return Belief.model_validate(data)


def _dict_to_belief_edge(data: dict[str, Any]) -> BeliefEdge:
    """Validate a BeliefEdge from a JSON dict."""
    return BeliefEdge.model_validate(data)


def _dict_to_thought(data: dict[str, Any]) -> Thought:
    """Validate a Thought from a JSON dict."""
    return Thought.model_validate(data)


def _dict_to_thought_edge(data: dict[str, Any]) -> ThoughtEdge:
    """Validate a ThoughtEdge from a JSON dict."""
    return ThoughtEdge.model_validate(data)


def _percept_to_dict(percept: Percept) -> dict[str, Any]:
    """Serialize a Percept value object to a JSON dict."""
    return {
        "id": {"value": percept.id.value},
        "content": percept.content,
        "modality": percept.modality,
        "metadata": percept.metadata,
    }


def _dict_to_percept(data: dict[str, Any]) -> Percept:
    """Rebuild a Percept value object from a JSON dict."""
    return Percept(
        id=PerceptId.of(str(data["id"]["value"])),
        content=str(data["content"]),
        modality=str(data["modality"]),
        metadata=dict(data.get("metadata", {})),
    )


class SymbolicStateStore:
    """Persists committed CycleState snapshots as checksummed JSON envelopes."""

    _SCHEMA_VERSION = "1.0"
    _STATE_TYPE = "cycle_state"

    def __init__(self, path: str | Path) -> None:
        """Initialize the store bound to a single state file.

        Args:
            path: Filesystem path of the state file.

        """
        self._path = Path(path)
        self._concept_serializer: GraphSerializer[Concept, ConceptEdge] = GraphSerializer(
            "concept",
            _model_to_json_dict,
            _model_to_json_dict,
        )
        self._belief_serializer: GraphSerializer[Belief, BeliefEdge] = GraphSerializer(
            "belief",
            _model_to_json_dict,
            _model_to_json_dict,
        )
        self._thought_serializer: GraphSerializer[Thought, ThoughtEdge] = GraphSerializer(
            "thought",
            _model_to_json_dict,
            _model_to_json_dict,
        )
        self._concept_loader = GraphLoader(
            "concept",
            _dict_to_concept,
            _dict_to_concept_edge,
        )
        self._belief_loader = GraphLoader(
            "belief",
            _dict_to_belief,
            _dict_to_belief_edge,
        )
        self._thought_loader = GraphLoader(
            "thought",
            _dict_to_thought,
            _dict_to_thought_edge,
        )

    @property
    def path(self) -> Path:
        """Return the state file path."""
        return self._path

    def save(self, state: CycleState) -> None:
        """Persist a committed CycleState to the state file.

        Args:
            state: The committed CycleState to persist.

        """
        payload = {
            "concepts": json.loads(self._concept_serializer.serialize(state.concepts.raw_graph)),
            "beliefs": json.loads(self._belief_serializer.serialize(state.beliefs.raw_graph)),
            "thoughts": json.loads(self._thought_serializer.serialize(state.thoughts.raw_graph)),
            "percept": None if state.percept is None else _percept_to_dict(state.percept),
        }
        raw_payload = json.dumps(payload, sort_keys=True)
        envelope = {
            "schema_version": self._SCHEMA_VERSION,
            "state_type": self._STATE_TYPE,
            "checksum": compute_canonical_checksum(raw_payload),
            "state": payload,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")

    def load(self) -> CycleState | None:
        """Load a committed CycleState from the state file.

        Returns:
            The committed CycleState, or None if the file does not exist.

        Raises:
            ChecksumMismatchError: If the payload checksum does not match.
            DeserializationError: If the envelope is malformed.

        """
        if not self._path.exists():
            return None

        data = json.loads(self._path.read_text(encoding="utf-8"))
        if data.get("state_type") != self._STATE_TYPE:
            from theo_core.symbolic._primitives.errors import DeserializationError

            raise DeserializationError(
                f"Unexpected state_type {data.get('state_type')!r}; expected {self._STATE_TYPE!r}"
            )

        payload = data["state"]
        expected_checksum = str(data.get("checksum", ""))
        actual_checksum = compute_canonical_checksum(json.dumps(payload, sort_keys=True))
        if actual_checksum != expected_checksum:
            from theo_core.symbolic._primitives.errors import ChecksumMismatchError

            raise ChecksumMismatchError(
                "Symbolic state file checksum mismatch; artifact may be corrupt."
            )

        concepts = ConceptGraph.from_raw_graph(
            self._concept_loader.deserialize(json.dumps(payload["concepts"], sort_keys=True))
        )
        beliefs = BeliefGraph.from_raw_graph(
            self._belief_loader.deserialize(json.dumps(payload["beliefs"], sort_keys=True))
        )
        thoughts = ThoughtGraph.from_raw_graph(
            self._thought_loader.deserialize(json.dumps(payload["thoughts"], sort_keys=True))
        )
        percept_data = payload.get("percept")
        percept = None if percept_data is None else _dict_to_percept(percept_data)

        from theo_core.symbolic.pipeline import CycleState

        return CycleState(concepts=concepts, beliefs=beliefs, thoughts=thoughts, percept=percept)
