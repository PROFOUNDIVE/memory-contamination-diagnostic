from __future__ import annotations

import pytest

from memcontam.baselines.bot_write import BoTTemplatePayload
from memcontam.memory.bot_buffer import evaluate_native_novelty
from memcontam.memory.stores import MemoryEntry


def _candidate() -> BoTTemplatePayload:
    return BoTTemplatePayload(
        description="Create a denominator before the final division.",
        template="Use complements to build a useful denominator.",
        category="procedure-based",
        explicitly_used_memory_ids=(),
    )


def _entry(description: str = "Build factor pairs before combining values.") -> MemoryEntry:
    return MemoryEntry(
        entry_id="existing-template",
        content="Do not use this content for novelty.",
        memory_type="thought_template",
        clean_or_contaminated="clean",
        metadata={"description": description},
    )


def test_empty_bot_buffer_admits_template_without_model_decision() -> None:
    decision = evaluate_native_novelty(_candidate(), [])

    assert decision.admitted is True
    assert decision.compared_entry_id is None
    assert decision.top_similarity is None


def test_bot_buffer_uses_description_similarity_below_threshold() -> None:
    encoded_documents: list[str] = []

    class BelowThresholdProvider:
        metadata = {}

        def encode_query(self, text: str) -> list[float]:
            del text
            return [1.0, 0.0]

        def encode_document(self, text: str) -> list[float]:
            encoded_documents.append(text)
            return [0.699, (1 - 0.699**2) ** 0.5]

    decision = evaluate_native_novelty(_candidate(), [_entry()], BelowThresholdProvider())

    assert decision.admitted is True
    assert decision.compared_entry_id == "existing-template"
    assert decision.top_similarity == pytest.approx(0.699)
    assert encoded_documents == ["Build factor pairs before combining values."]


def test_bot_buffer_rejects_description_similarity_at_threshold() -> None:
    class EqualityProvider:
        metadata = {}

        def encode_query(self, text: str) -> list[float]:
            del text
            return [1.0, 0.0]

        def encode_document(self, text: str) -> list[float]:
            del text
            return [0.7, (1 - 0.7**2) ** 0.5]

    decision = evaluate_native_novelty(_candidate(), [_entry()], EqualityProvider())

    assert decision.admitted is False
    assert decision.top_similarity == pytest.approx(0.7)
