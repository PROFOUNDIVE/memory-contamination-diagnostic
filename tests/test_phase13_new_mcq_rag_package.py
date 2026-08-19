from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from memcontam.readiness import phase13_new_mcq_rag


PACKAGE_ROOT = Path("data/phase13/rag/new_mcq")
EVALUATION_ROOT = Path("data/phase13/core/materialized")


def test_reviewed_clean_corpora_remain_blocked_without_intervention_triplets() -> None:
    report = phase13_new_mcq_rag.validate_new_mcq_rag_package(PACKAGE_ROOT, EVALUATION_ROOT)

    assert report.status == "NOT_READY"
    assert report.reason == "NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN"
    assert report.accepted_documents == {
        "mmlu_pro_engineering": 24,
        "mmlu_pro_physics": 24,
        "gpqa_diamond": 24,
    }
    assert report.source_classes == {
        "mmlu_pro_engineering": "public_task_specification",
        "mmlu_pro_physics": "public_task_specification",
        "gpqa_diamond": "public_task_specification",
    }
    assert report.clean_corpus_hashes.keys() == report.accepted_documents.keys()
    assert report.remaining_objects == (
        "relevance_universe",
        "task_local_intervention_triplets",
        "clean_correct_irrelevant_contam_branch_indices",
    )
    assert report.promotion_ready is False


def test_candidate_corpus_rejects_exact_evaluation_text(tmp_path: Path) -> None:
    package = tmp_path / "new_mcq"
    shutil.copytree(PACKAGE_ROOT, package)
    evaluation = json.loads(
        Path("data/phase13/core/materialized/mmlu_pro_engineering.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    candidate_path = package / "candidates" / "mmlu_pro_engineering.jsonl"
    candidates = candidate_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(candidates[0])
    first["text"] = evaluation["input"]["question"]
    candidates[0] = json.dumps(first, separators=(",", ":"))
    candidate_path.write_text("\n".join(candidates) + "\n", encoding="utf-8")

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_EVALUATION_OVERLAP",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)
