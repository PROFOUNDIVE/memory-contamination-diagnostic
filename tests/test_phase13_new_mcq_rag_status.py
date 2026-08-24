from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from pathlib import Path

from memcontam.readiness.phase13_execution_contract import CORE_MAIN_REGISTRY

STATUS_PATH = Path("data/phase13/rag/new_mcq_rag_status_v1.json")
POST_CUTOFF_PATH = Path("data/phase13/main/post_cutoff_package_selection_v1.json")
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


def test_post_cutoff_package_records_fired_contingency_without_rewriting_pre_cutoff() -> None:
    assert POST_CUTOFF_PATH.exists()
    package = json.loads(POST_CUTOFF_PATH.read_text(encoding="utf-8"))

    assert package["schema_version"] == "phase13_post_cutoff_package_selection_v1"
    assert package["authority"]["addendum_sha256"] == (
        "d971c24439cc551655e9e1f5dbba6efa5a27242802f1db66a32749ec61350edc"
    )
    assert package["cutoff"]["status"] == "CONTINGENCY_FIRED"
    assert package["pre_cutoff_package"]["status"] == "NOT_READY"
    assert package["pre_cutoff_package"]["status_path"] == str(STATUS_PATH)
    assert package["resolution"]["excluded_current_main_cells"] == [
        ["mmlu_pro_engineering", "rag_frozen"],
        ["mmlu_pro_physics", "rag_frozen"],
    ]
    assert package["resolution"]["prospective_extension_id"] == (
        "new_mcq_rag_prospective_extension_v1"
    )
    assert package["selected_current_main"]["attempted_seed_count_per_task"] == 10
    assert package["selected_current_main"]["seed_replacement"] == "prohibited"
    assert package["selected_current_main"]["adaptive_seed_augmentation"] == "prohibited"
    unsigned = dict(package)
    package_hash = unsigned.pop("package_hash")
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    assert package_hash == hashlib.sha256(canonical).hexdigest()


def test_post_cutoff_package_is_bound_to_runtime_registry_and_checkpoint() -> None:
    package = json.loads(POST_CUTOFF_PATH.read_text(encoding="utf-8"))
    checkpoint = json.loads(
        Path("data/phase13/main/track1_authority_state_sync_checkpoint_v1.json").read_text(
            encoding="utf-8"
        )
    )
    registry = asdict(CORE_MAIN_REGISTRY)

    exclusions = [list(cell) for cell in registry["current_main_excluded_cells"]]
    assert package["resolution"]["excluded_current_main_cells"] == exclusions
    assert checkpoint["completed_repository_sync"]["new_mcq_rag_current_main_exclusions"] == (
        exclusions
    )
    assert package["selected_current_main"]["attempted_seed_count_per_task"] == registry[
        "attempted_seed_count"
    ]
    assert package["resolution"]["prospective_extension_id"] == registry[
        "prospective_rag_extension_id"
    ]
    for key in ("manifest", "status"):
        path = Path(package["pre_cutoff_package"][f"{key}_path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == package["pre_cutoff_package"][
            f"{key}_sha256"
        ]
