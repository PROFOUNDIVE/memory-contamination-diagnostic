from __future__ import annotations

import json
from pathlib import Path


STATUS_PATH = Path("data/phase13/rag/new_mcq_rag_status_v1.json")
TASKS = ("mmlu_pro_engineering", "mmlu_pro_physics", "gpqa_diamond")


def test_new_mcq_rag_cells_fail_closed_before_registered_cutoff() -> None:
    registry = json.loads(STATUS_PATH.read_text(encoding="utf-8"))

    assert registry["cutoff"] == "2026-08-22T18:00:00+09:00"
    assert registry["cutoff_applied"] is False
    assert registry["candidate_package"]["status"] == (
        "CLEAN_CANDIDATES_ACCEPTED_PROMOTION_BLOCKED"
    )
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
            "relevance_universe",
            "task_local_intervention_triplets",
            "clean_correct_irrelevant_contam_branch_indices",
        ]
        assert "clean_index_identity_and_hash" in cell["materialized_objects"]
