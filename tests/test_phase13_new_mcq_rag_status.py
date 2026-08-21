from __future__ import annotations

import json
from pathlib import Path


STATUS_PATH = Path("data/phase13/rag/new_mcq_rag_status_v1.json")
TASKS = ("mmlu_pro_engineering", "mmlu_pro_physics")


def test_retained_new_mcq_rag_cells_are_blocked_before_cutoff() -> None:
    registry = json.loads(STATUS_PATH.read_text(encoding="utf-8"))

    assert registry["cutoff"] == "2026-08-22T18:00:00+09:00"
    assert registry["cutoff_applied"] is False
    assert registry["cutoff_status"] == "PENDING_REGISTERED_CUTOFF"
    assert registry["candidate_package"]["status"] == "NOT_READY"
    assert registry["retrieval_contract"] == {
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "similarity": "cosine",
        "top_k": 3,
        "reranker": None,
        "score_threshold": None,
        "tie_break": "lexical_document_id",
        "corpus_scope": "same_task_only",
        "update_mode": "frozen_read_only",
    }
    assert tuple(registry["cells"]) == TASKS
    for task in TASKS:
        cell = registry["cells"][task]
        assert cell["status"] == "NOT_READY"
        assert cell["reason"] == "NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN"
        assert cell["entry_condition_met"] is False
        assert cell["missing_objects"] == [
            "clean_document_applicability_predicates_and_relevance_universe"
        ]
        assert set(cell["index_hashes"]) == {"clean", "correct", "irrelevant", "contam"}
