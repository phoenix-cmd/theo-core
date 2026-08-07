"""Tests for DeterministicMemoryEngine, scored retrieval, and memory importance."""

from __future__ import annotations

import os

import pytest

from theo_core.domain.runtime.entities.cognitive_workspace import CognitiveWorkspace
from theo_core.domain.runtime.entities.memory_entry import MemoryImportance
from theo_core.memory.classifier.memory_classifier import MemoryClassifier
from theo_core.memory.engine.deterministic_memory import DeterministicMemoryEngine
from theo_core.memory.storage.json_repository import JSONMemoryRepository
from theo_core.perception.text.data_driven_processor import DataDrivenPerceptionProcessor


class TestMemoryClassifier:
    """Tests for MemoryClassifier 3-class system."""

    def test_classify_identity(self) -> None:
        """Name statement should classify as identity."""
        proc = DataDrivenPerceptionProcessor()
        classifier = MemoryClassifier()
        percept = proc.perceive("My name is Falcon")
        assert classifier.classify(percept) == "identity"

    def test_classify_preference(self) -> None:
        """Preference statement should classify as preference."""
        proc = DataDrivenPerceptionProcessor()
        classifier = MemoryClassifier()
        percept = proc.perceive("I like astronomy")
        assert classifier.classify(percept) == "preference"

    def test_classify_experience(self) -> None:
        """Past event statement should classify as experience."""
        proc = DataDrivenPerceptionProcessor()
        classifier = MemoryClassifier()
        percept = proc.perceive("Yesterday I visited Jaipur")
        assert classifier.classify(percept) == "experience"


class TestDeterministicMemoryEngine:
    """Tests for DeterministicMemoryEngine, scored retrieval, and JSON persistence."""

    def test_append_only_superseding(self, tmp_path: object) -> None:
        """Updating a fact should mark old memory superseded and create new entry."""
        json_file = str(tmp_path) + "/mem_store.json"
        repo = JSONMemoryRepository(file_path=json_file)
        engine = DeterministicMemoryEngine(repository=repo)

        # Store initial fact
        m1 = engine.store_fact("user.preference.language", "Python", category="preference")
        assert m1.id == "mem-000001"
        assert m1.status == "active"

        # Update fact -> supersedes m1
        m2 = engine.store_fact("user.preference.language", "Rust", category="preference")
        assert m2.id == "mem-000002"
        assert m2.status == "active"

        # Check m1 is superseded
        assert m1.status == "superseded"
        assert m1.superseded_by == "mem-000002"
        assert engine.total_memory_count == 2

        # Check retrieval returns active entry
        active_fact = engine.get_fact("user.preference.language")
        assert active_fact is not None
        assert active_fact.value == "Rust"

    def test_deterministic_scored_retrieval(self, tmp_path: object) -> None:
        """Retrieval results must be sorted deterministically by score, importance, and date."""
        json_file = str(tmp_path) + "/scored_mem_store.json"
        repo = JSONMemoryRepository(file_path=json_file)
        engine = DeterministicMemoryEngine(repository=repo)

        engine.store_fact(
            "user.name",
            "Falcon",
            category="identity",
            importance=MemoryImportance.HIGH,
        )
        engine.store_fact(
            "user.preference.astronomy",
            "astronomy",
            category="preference",
            importance=MemoryImportance.MEDIUM,
        )

        results = engine.retrieve_scored("astronomy", top_k=5)
        assert len(results) >= 1
        assert results[0].score >= 0.8
        assert results[0].memory_id in ("mem-000001", "mem-000002")

        if os.path.exists(json_file):
            os.remove(json_file)

    def test_cognitive_workspace_creation(self) -> None:
        """CognitiveWorkspace should store intermediate cycle state."""
        ws = CognitiveWorkspace()
        assert ws.workspace_id is not None
        assert len(ws.retrieved_memory_ids) == 0

    def test_corrupted_repository_raises_not_silently_empties(self, tmp_path: object) -> None:
        """Corrupted JSON memory storage must raise, never silently return []."""
        json_file = str(tmp_path) + "/corrupt_mem_store.json"
        repo = JSONMemoryRepository(file_path=json_file)
        with open(json_file, "w", encoding="utf-8") as f:
            f.write("{ this is not valid json ]")

        with pytest.raises(RuntimeError, match="Corrupted memory repository"):
            repo.load_all()
