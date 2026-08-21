from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest
from pydantic import JsonValue

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


def _reseal_index(package: Path, task: str, index: dict[str, JsonValue]) -> None:
    index_path = package / "indices" / f"{task}.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    index_sha256 = hashlib.sha256(index_path.read_bytes()).hexdigest()
    manifest_path = package / "package_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tasks"][task]["index"]["sha256"] = index_sha256
    for artifact in manifest["required_artifacts"]["serialized_branch_index_artifacts"]:
        if artifact["path"] == f"indices/{task}.json":
            artifact["sha256"] = index_sha256
    branches_value: JsonValue = index["branches"]
    assert isinstance(branches_value, dict)
    branches: dict[str, JsonValue] = branches_value
    index_hashes: dict[str, str] = {}
    for branch, payload in branches.items():
        assert isinstance(payload, dict)
        artifact_hash = payload["index_artifact_hash"]
        assert isinstance(artifact_hash, str)
        index_hashes[branch] = artifact_hash
    manifest["tasks"][task]["index_hashes"] = index_hashes
    from memcontam.readiness.phase13_new_mcq_rag_manifest import (
        Artifact,
        PackageManifest,
        package_reconstruction_identity,
    )

    parsed = PackageManifest.model_validate(manifest)
    artifacts = (parsed.source_registry, parsed.authoring_contract)
    artifacts += tuple(
        artifact
        for artifact_class in parsed.required_artifacts.values()
        for artifact in artifact_class
    )
    artifacts += tuple(
        artifact
        for task_artifacts in parsed.tasks.values()
        for artifact in (
            task_artifacts.candidate,
            task_artifacts.review,
            task_artifacts.accepted,
            task_artifacts.index,
        )
    )
    manifest["package_reconstruction_identity"] = package_reconstruction_identity(
        tuple(Artifact.model_validate(artifact) for artifact in artifacts)
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    status_path = package.parent / STATUS_PATH.name
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["candidate_package"]["reconstruction_identity"] = manifest[
        "package_reconstruction_identity"
    ]
    status["candidate_package"]["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    status["cells"][task]["index_hashes"] = manifest["tasks"][task]["index_hashes"]
    status_path.write_text(json.dumps(status), encoding="utf-8")


def test_retained_package_is_blocked_only_on_clean_relevance_universe() -> None:
    report = phase13_new_mcq_rag.validate_new_mcq_rag_package(PACKAGE_ROOT, EVALUATION_ROOT)
    payload = asdict(report)

    assert report.status == "NOT_READY"
    assert report.reason == "NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN"
    assert payload["reviewed_candidates"] == {
        "mmlu_pro_engineering": 24,
        "mmlu_pro_physics": 24,
    }
    assert report.source_classes == {
        "mmlu_pro_engineering": "public_task_specification",
        "mmlu_pro_physics": "public_task_specification",
    }
    assert payload["candidate_corpus_hashes"].keys() == payload["reviewed_candidates"].keys()
    assert payload["clean_index_hashes"].keys() == payload["reviewed_candidates"].keys()
    assert report.remaining_objects == (
        "clean_document_applicability_predicates_and_relevance_universe",
    )
    assert report.promotion_ready is False


def test_source_registry_preserves_gpqa_as_dormant_provenance() -> None:
    registry = json.loads((PACKAGE_ROOT / "source_registry_v1.json").read_text(encoding="utf-8"))
    sources = {source["source_registry_id"]: source for source in registry["sources"]}
    manifest = json.loads((PACKAGE_ROOT / "package_manifest_v1.json").read_text(encoding="utf-8"))
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    leakage = json.loads((PACKAGE_ROOT / "leakage_report_v1.json").read_text(encoding="utf-8"))

    assert sources["gpqa_spec_v1"] == {
        "source_registry_id": "gpqa_spec_v1",
        "source_class": "public_task_specification",
        "repo": "Idavidrein/gpqa",
        "revision": "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
        "path": "README.md",
        "sha256": "c703e934fa58d9ab66e470e5283de7d4c6ef02fe4ccb3e2d7fe97e5399e6eb36",
        "allowed_sections": [
            "dataset description",
            "task categories and scientific domains",
            "intended scalable-oversight and capabilities uses",
        ],
        "evaluation_rows_eligible": False,
        "gold_fields_eligible": False,
    }
    assert tuple(registry["task_sources"]) == (
        "mmlu_pro_engineering",
        "mmlu_pro_physics",
    )
    assert "gpqa_diamond" not in manifest["tasks"]
    assert "gpqa_diamond" not in status["cells"]
    assert all("gpqa" not in row["document_id"] for row in leakage["document_evidence"])


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
    manifest["promotion"]["remaining_objects"].append("stale_blocker")
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
        "intervention_registry_v1.json",
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
    index_path = package / "indices" / "mmlu_pro_physics.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    clean = index["branches"]["clean"]
    clean["documents"][0]["text"] += " altered"
    clean["corpus_content_hash"] = canonical_json_hash(clean["documents"])
    clean["index_artifact_hash"] = canonical_json_hash(
        {
            "documents": clean["documents"],
            "embedding_contract": clean["embedding_contract"],
            "vectors": clean["vectors"],
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
    review_path = package / "reviews" / "mmlu_pro_physics.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    manifest_path = package / "package_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tasks"]["mmlu_pro_physics"]["review"]["sha256"] = hashlib.sha256(
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
        "source_registry_v1.json",
        "intervention_registry_v1.json",
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


def test_package_rejects_leakage_evidence_bound_to_stale_evaluation(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "materialized"
    shutil.copytree(EVALUATION_ROOT, evaluation_root)
    evaluation_path = evaluation_root / "mmlu_pro_engineering.jsonl"
    rows = evaluation_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["input"]["question"] += " changed after leakage certification"
    rows[0] = json.dumps(first, separators=(",", ":"))
    evaluation_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_LEAKAGE_EVALUATION_INVALID",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(PACKAGE_ROOT, evaluation_root)


def test_package_rejects_resealed_branch_key_swap(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    task = "mmlu_pro_engineering"
    index = json.loads((package / "indices" / f"{task}.json").read_text(encoding="utf-8"))
    index["branches"]["clean"], index["branches"]["contam"] = (
        index["branches"]["contam"],
        index["branches"]["clean"],
    )
    _reseal_index(package, task, index)

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_SERIALIZED_INDEX_INVALID",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


@pytest.mark.parametrize("field", ("corpus_serialization_id", "index_serialization_id"))
def test_package_rejects_resealed_serialization_identity_forgery(
    tmp_path: Path,
    field: str,
) -> None:
    package = _copy_package(tmp_path)
    task = "mmlu_pro_physics"
    index = json.loads((package / "indices" / f"{task}.json").read_text(encoding="utf-8"))
    index["branches"]["correct"][field] = "forged"
    _reseal_index(package, task, index)

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_SERIALIZED_INDEX_INVALID",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)
