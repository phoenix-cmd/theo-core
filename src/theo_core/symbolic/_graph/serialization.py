"""Versioned graph serialization with canonical SHA-256 checksums."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from theo_core.symbolic._graph.graph import Graph
from theo_core.symbolic._graph.types import E, EdgeKey, N
from theo_core.symbolic._primitives.errors import (
    ChecksumMismatchError,
    DeserializationError,
    SchemaVersionError,
)
from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic._primitives.versioning import SchemaVersion

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class SerializedGraphEnvelope:
    """Envelope wrapper for serialized graph artifacts."""

    schema_version: str
    graph_type: str
    graph_version: int
    created_at: str
    canonical_order: str
    checksum: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


def compute_canonical_checksum(payload: str) -> str:
    """Compute SHA-256 hex digest of a canonical JSON payload string."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GraphSerializer[N, E]:
    """Serializer converting Graph[N, E] to versioned JSON with SHA-256 checksum."""

    def __init__(
        self,
        graph_type: str,
        node_to_dict: Callable[[N], dict[str, Any]],
        edge_to_dict: Callable[[E], dict[str, Any]],
        schema_version: SchemaVersion | None = None,
        graph_version: int = 1,
    ) -> None:
        self.graph_type = graph_type
        self.node_to_dict = node_to_dict
        self.edge_to_dict = edge_to_dict
        self.schema_version = schema_version or SchemaVersion(1, 0)
        self.graph_version = graph_version

    def serialize(self, graph: Graph[N, E]) -> str:
        """Serialize Graph instance to canonical versioned JSON string.

        Complexity Contract:
            Time: O(V + E)
            Memory: O(V + E)
            Deterministic: YES

        """
        nodes_payload = [
            {"id": nid.value, "data": self.node_to_dict(node)}
            for nid, node in graph.get_nodes()
        ]

        edges_payload = [
            {
                "source": ek.source.value,
                "target": ek.target.value,
                "relation": ek.relation,
                "data": self.edge_to_dict(edge),
            }
            for ek, edge in graph.get_edges()
        ]

        # Compute payload checksum over deterministically ordered node/edge structure
        raw_payload_str = json.dumps(
            {"nodes": nodes_payload, "edges": edges_payload},
            sort_keys=True,
            indent=None,
        )
        checksum = compute_canonical_checksum(raw_payload_str)

        envelope_dict = {
            "schema_version": str(self.schema_version),
            "graph_type": self.graph_type,
            "graph_version": self.graph_version,
            "created_at": "2026-08-07T00:00:00Z",
            "canonical_order": "node_id_ascending",
            "checksum": checksum,
            "nodes": nodes_payload,
            "edges": edges_payload,
        }

        return json.dumps(envelope_dict, indent=2, sort_keys=True)


class GraphLoader[N, E]:
    """Loader converting versioned JSON strings back to Graph[N, E]."""

    def __init__(
        self,
        expected_graph_type: str,
        dict_to_node: Callable[[dict[str, Any]], N],
        dict_to_edge: Callable[[dict[str, Any]], E],
        expected_schema_version: SchemaVersion | None = None,
    ) -> None:
        self.expected_graph_type = expected_graph_type
        self.dict_to_node = dict_to_node
        self.dict_to_edge = dict_to_edge
        self.expected_schema_version = expected_schema_version or SchemaVersion(1, 0)

    def deserialize(self, json_str: str) -> Graph[N, E]:
        """Deserialize versioned JSON string to Graph instance.

        Complexity Contract:
            Time: O(V + E)
            Memory: O(V + E)
            Deterministic: YES

        Raises:
            DeserializationError: On JSON parse failure.
            SchemaVersionError: On version mismatch.
            ChecksumMismatchError: On payload checksum mismatch.

        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as err:
            msg = f"Invalid JSON string: {err}"
            raise DeserializationError(msg) from err

        # Verify schema version
        ver_str = data.get("schema_version", "0.0")
        parts = ver_str.split(".")
        try:
            major, minor = int(parts[0]), int(parts[1])
            version = SchemaVersion(major, minor)
        except (IndexError, ValueError) as err:
            msg = f"Invalid schema_version string {ver_str!r}"
            raise SchemaVersionError(msg) from err

        if not self.expected_schema_version.is_compatible_with(version):
            msg = (
                f"Incompatible schema version {version!r}, "
                f"expected major version {self.expected_schema_version.major}"
            )
            raise SchemaVersionError(msg)

        # Verify graph type
        gtype = data.get("graph_type", "")
        if gtype != self.expected_graph_type:
            msg = f"Graph type mismatch: expected {self.expected_graph_type!r}, got {gtype!r}"
            raise DeserializationError(msg)

        nodes_payload = data.get("nodes", [])
        edges_payload = data.get("edges", [])

        # Verify checksum
        raw_payload_str = json.dumps(
            {"nodes": nodes_payload, "edges": edges_payload},
            sort_keys=True,
            indent=None,
        )
        expected_checksum = compute_canonical_checksum(raw_payload_str)
        actual_checksum = data.get("checksum", "")

        if actual_checksum != expected_checksum:
            msg = (
                f"Checksum mismatch! Artifact may be corrupt. "
                f"Got {actual_checksum!r}, calculated {expected_checksum!r}"
            )
            raise ChecksumMismatchError(msg)

        graph: Graph[N, E] = Graph()

        for item in nodes_payload:
            nid = SymbolicId.of(item["id"])
            node = self.dict_to_node(item["data"])
            graph.add_node(nid, node)

        for item in edges_payload:
            src = SymbolicId.of(item["source"])
            tgt = SymbolicId.of(item["target"])
            rel = item.get("relation", "default")
            ek = EdgeKey(source=src, target=tgt, relation=rel)
            edge = self.dict_to_edge(item["data"])
            graph.add_edge(ek, edge)

        return graph
