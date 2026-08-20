from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from memcontam.readiness import phase13_new_mcq_rag
from memcontam.readiness import phase13_new_mcq_rag_frozen
from memcontam.contamination.phase12.models import canonical_json_hash


PACKAGE_ROOT = Path("data/phase13/rag/new_mcq")
EVALUATION_ROOT = Path("data/phase13/core/materialized")
STATUS_PATH = PACKAGE_ROOT.parent / "new_mcq_rag_status_v1.json"


def _copy_package(tmp_path: Path) -> Path:
    package = tmp_path / "new_mcq"
    shutil.copytree(PACKAGE_ROOT, package)
    shutil.copy2(STATUS_PATH, tmp_path / STATUS_PATH.name)
    return package


def _update_status_manifest_hash(package: Path) -> None:
    status_path = package.parent / STATUS_PATH.name
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["candidate_package"]["sha256"] = hashlib.sha256(
        (package / "package_manifest_v1.json").read_bytes()
    ).hexdigest()
    status_path.write_text(json.dumps(status), encoding="utf-8")


def test_clean_package_remains_blocked_by_unfrozen_required_artifacts() -> None:
    report = phase13_new_mcq_rag.validate_new_mcq_rag_package(PACKAGE_ROOT, EVALUATION_ROOT)
    payload = asdict(report)

    assert report.status == "NOT_READY"
    assert report.reason == "NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN"
    assert payload["reviewed_candidates"] == {
        "mmlu_pro_engineering": 24,
        "mmlu_pro_physics": 24,
        "gpqa_diamond": 24,
    }
    assert report.source_classes == {
        "mmlu_pro_engineering": "public_task_specification",
        "mmlu_pro_physics": "public_task_specification",
        "gpqa_diamond": "public_task_specification",
    }
    assert payload["candidate_corpus_hashes"].keys() == payload["reviewed_candidates"].keys()
    assert payload["clean_index_hashes"].keys() == payload["reviewed_candidates"].keys()
    assert report.remaining_objects == (
        "authority_required_leakage_gate_artifacts",
        "task_local_candidate_selection_and_certification",
        "task_local_intervention_relevance",
        "clean_correct_irrelevant_contam_branch_indices",
    )
    assert report.promotion_ready is False


def test_candidate_corpus_rejects_exact_evaluation_text(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
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


def test_candidate_package_rejects_stale_manifest_hash(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    candidate_path = package / "candidates" / "mmlu_pro_engineering.jsonl"
    candidates = candidate_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(candidates[0])
    first["text"] = f"{first['text']} Recheck the extracted constraints."
    candidates[0] = json.dumps(first, separators=(",", ":"))
    candidate_path.write_text("\n".join(candidates) + "\n", encoding="utf-8")

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


def test_candidate_package_rejects_empty_task_source_binding(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    registry_path = package / "source_registry_v1.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["task_sources"]["mmlu_pro_engineering"] = []
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    candidate_path = package / "candidates" / "mmlu_pro_engineering.jsonl"
    candidates = [json.loads(line) for line in candidate_path.read_text().splitlines()]
    for candidate in candidates:
        candidate["source_registry_ids"] = []
    candidate_path.write_text("".join(json.dumps(row) + "\n" for row in candidates))

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_DOCUMENT_REGISTRY_INVALID",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


def test_candidate_package_rejects_incomplete_remaining_objects(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    manifest_path = package / "package_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["promotion"]["remaining_objects"].pop()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


@pytest.mark.parametrize(
    "relative_path",
    (
        "source_eligibility_registry_v1.json",
        "accepted/mmlu_pro_engineering.jsonl",
        "embedding_runtime_v1.json",
        "indices/mmlu_pro_physics.json",
        "leakage_report_v1.json",
        "candidate_evidence_v1.json",
        "sources/mmlu_pro_validation_475d58ba.parquet",
        "sources/gpqa_tree_633f5ee8.json",
    ),
)
def test_package_rejects_tampering_in_every_required_object_class(
    tmp_path: Path,
    relative_path: str,
) -> None:
    package = _copy_package(tmp_path)
    artifact = package / relative_path
    artifact.write_bytes(artifact.read_bytes() + b"\n")

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


def test_clean_index_rejects_text_not_bound_to_accepted_registry(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    index_path = package / "indices" / "gpqa_diamond.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["documents"][0]["text"] += " altered"
    index["corpus_content_hash"] = canonical_json_hash(index["documents"])
    index["index_artifact_hash"] = canonical_json_hash(
        {
            "documents": index["documents"],
            "embedding_contract": index["embedding_contract"],
            "vectors": index["vectors"],
        }
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")
    report = phase13_new_mcq_rag.validate_new_mcq_rag_package(
        PACKAGE_ROOT,
        EVALUATION_ROOT,
    )

    with pytest.raises(
        phase13_new_mcq_rag_frozen.FrozenArtifactError,
        match="NEW_MCQ_RAG_SERIALIZED_INDEX_INVALID",
    ):
        phase13_new_mcq_rag_frozen.validate_frozen_artifacts(
            package,
            EVALUATION_ROOT,
            report.candidate_corpus_hashes,
        )


def test_reconstruction_identity_binds_task_review(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    review_path = package / "reviews" / "gpqa_diamond.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    manifest_path = package / "package_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tasks"]["gpqa_diamond"]["review"]["sha256"] = hashlib.sha256(
        review_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _update_status_manifest_hash(package)

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


def test_package_rejects_status_manifest_hash_mismatch(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    status_path = package.parent / STATUS_PATH.name
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["candidate_package"]["sha256"] = "0" * 64
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


@pytest.mark.parametrize(
    "relative_path",
    (
        "sources/mmlu_pro_validation_475d58ba.parquet",
        "sources/gpqa_tree_633f5ee8.json",
    ),
)
def test_package_maps_missing_bound_source_to_domain_error(
    tmp_path: Path,
    relative_path: str,
) -> None:
    package = _copy_package(tmp_path)
    (package / relative_path).unlink()

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


def test_package_rejects_unrecognized_status_fields(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    status_path = package.parent / STATUS_PATH.name
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["unregistered_override"] = True
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


def test_package_validation_is_independent_of_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    report = phase13_new_mcq_rag.validate_new_mcq_rag_package(
        repository_root / PACKAGE_ROOT,
        repository_root / EVALUATION_ROOT,
    )

    assert report.reason == "NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN"
