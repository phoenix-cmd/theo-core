"""KnowledgeGraphEngine — deterministic graph knowledge store and relationship traversal engine.

Stores Entity -> Predicate -> Entity triples and performs multi-hop relationship traversal.
"""

from __future__ import annotations

import json
import os
from typing import Any

from theo_core.domain.runtime.entities.fact_triple import FactTriple


class KnowledgeGraphEngine:
    """Deterministic knowledge graph store and multi-hop concept traversal engine.

    Persists knowledge graph triples to disk at `data/knowledge_graph.json`.
    """

    def __init__(self, file_path: str = "data/knowledge_graph.json") -> None:
        """Initialize KnowledgeGraphEngine and seed core domain knowledge.

        Args:
            file_path: Relative or absolute path to JSON storage file.

        """
        self._file_path = file_path
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
        self._triples: list[FactTriple] = self._load()
        if not self._triples:
            self._seed_default_knowledge()

    def _seed_default_knowledge(self) -> None:
        """Seed baseline domain relationships."""
        seed = [
            FactTriple(subject="Astronomy", predicate="related_to", object="Astrophotography"),
            FactTriple(subject="Astrophotography", predicate="requires", object="Telescope"),
            FactTriple(subject="Telescope", predicate="utilizes", object="Optics"),
            FactTriple(subject="Python", predicate="related_to", object="Data Science"),
            FactTriple(subject="Data Science", predicate="utilizes", object="Machine Learning"),
        ]
        self._triples.extend(seed)
        self._save()

    def _load(self) -> list[FactTriple]:
        """Load triples from JSON file."""
        if not os.path.exists(self._file_path):
            return []
        try:
            with open(self._file_path, encoding="utf-8") as f:
                data: list[dict[str, Any]] = json.load(f)
                return [FactTriple(**item) for item in data]
        except Exception:
            return []

    def _save(self) -> None:
        """Save triples to JSON file."""
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)
        data = [t.model_dump(mode="json") for t in self._triples]
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_triple(self, subject: str, predicate: str, object_entity: str) -> FactTriple:
        """Add a new fact triple to the Knowledge Graph.

        Args:
            subject: Subject entity string.
            predicate: Predicate relationship string.
            object_entity: Object entity string.

        Returns:
            The created FactTriple object.

        """
        for t in self._triples:
            if (
                t.subject.lower() == subject.lower()
                and t.predicate.lower() == predicate.lower()
                and t.object.lower() == object_entity.lower()
            ):
                return t

        triple = FactTriple(subject=subject, predicate=predicate, object=object_entity)
        self._triples.append(triple)
        self._save()
        return triple

    def traverse(self, start_entity: str, max_depth: int = 2) -> list[FactTriple]:
        """Perform multi-hop deterministic graph traversal starting from an entity.

        Args:
            start_entity: Entity string to start traversal from.
            max_depth: Maximum hop depth (default: 2).

        Returns:
            List of connected FactTriple objects.

        """
        visited_entities: set[str] = {start_entity.lower()}
        result_triples: list[FactTriple] = []

        current_frontier = [start_entity.lower()]

        for _ in range(max_depth):
            next_frontier = []
            for entity in current_frontier:
                for t in self._triples:
                    if t.subject.lower() == entity:
                        if t not in result_triples:
                            result_triples.append(t)
                        obj_lower = t.object.lower()
                        if obj_lower not in visited_entities:
                            visited_entities.add(obj_lower)
                            next_frontier.append(obj_lower)
            current_frontier = next_frontier

        return result_triples

    def search_concepts(self, concept: str) -> list[dict[str, Any]]:
        """Search concepts matching query string.

        Args:
            concept: Concept keyword string.

        Returns:
            List of fact triple dictionaries.

        """
        scored = self.traverse(concept, max_depth=2)
        return [t.model_dump(mode="json") for t in scored]

    @property
    def total_triple_count(self) -> int:
        """Return total number of triples in Knowledge Graph."""
        return len(self._triples)
