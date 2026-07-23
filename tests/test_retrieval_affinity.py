from __future__ import annotations

import pytest

from memcontam.rag.affinity import AffinityError, calibrate_affinity
from memcontam.rag.relevance import recall_at_k


class VectorEmbedder:
    def encode_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def encode_document(self, text: str) -> list[float]:
        return {
            "low": [0.2, 0.9797958971],
            "mid": [0.6, 0.8],
            "high": [0.9, 0.4358898944],
        }[text]


def test_calibrates_disjoint_outcome_free_affinity_bands_and_recall() -> None:
    result = calibrate_affinity(
        [{"id": "query", "text": "query"}],
        [{"id": "low", "text": "low"}, {"id": "mid", "text": "mid"}, {"id": "high", "text": "high"}],
        VectorEmbedder(),
        {"low": (0.0, 0.4), "mid": (0.4, 0.8), "high": (0.8, 1.0)},
    )

    assert {assignment.candidate_id: assignment.band for assignment in result.assignments} == {
        "low": "low",
        "mid": "mid",
        "high": "high",
    }
    assert recall_at_k(("doc-a", "doc-b"), {"doc-b", "doc-c"}, 2) == 0.5
    with pytest.raises(AffinityError, match="AFFINITY_BANDS_OVERLAP"):
        calibrate_affinity([], [], VectorEmbedder(), {"low": (0.0, 0.5), "mid": (0.4, 1.0)})
    with pytest.raises(AffinityError, match="OUTCOME_TUNED_AFFINITY"):
        calibrate_affinity(
            [{"id": "query", "text": "query", "main_outcome": True}],
            [],
            VectorEmbedder(),
            {"low": (0.0, 1.0)},
        )
