import hashlib
import json
import shutil
from pathlib import Path

import pytest
from pydantic import JsonValue

from memcontam.readiness import phase13_new_mcq_rag
from memcontam.readiness.phase13_new_mcq_rag_manifest import (
    PackageManifest,
    package_reconstruction_identity,
)

PACKAGE_ROOT = Path("data/phase13/rag/new_mcq")
EVALUATION_ROOT = Path("data/phase13/core/materialized")
STATUS_PATH = PACKAGE_ROOT.parent / "new_mcq_rag_status_v1.json"


def _copy_package(tmp_path: Path) -> Path:
    package = tmp_path / "new_mcq"
    shutil.copytree(PACKAGE_ROOT, package)
    shutil.copy2(STATUS_PATH, tmp_path / STATUS_PATH.name)
    return package


def _reconstruction_identity(payload: dict[str, JsonValue]) -> str:
    manifest = PackageManifest.model_validate(payload)
    artifacts = [manifest.source_registry, manifest.authoring_contract]
    artifacts.extend(
        artifact
        for artifact_class in manifest.required_artifacts.values()
        for artifact in artifact_class
    )
    artifacts.extend(
        artifact
        for task in manifest.tasks.values()
        for artifact in (task.candidate, task.review, task.accepted, task.index)
    )
    return package_reconstruction_identity(tuple(artifacts))


def _reseal_candidate_evidence(package: Path, evidence: dict[str, JsonValue]) -> None:
    evidence_path = package / "candidate_evidence_v1.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    manifest_path = package / "package_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["required_artifacts"]["partial_task_local_candidate_evidence"][0][
        "sha256"
    ] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    manifest["package_reconstruction_identity"] = _reconstruction_identity(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    status_path = package.parent / STATUS_PATH.name
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["candidate_package"]["reconstruction_identity"] = manifest[
        "package_reconstruction_identity"
    ]
    status["candidate_package"]["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    status_path.write_text(json.dumps(status), encoding="utf-8")


def test_package_rejects_resealed_semantic_candidate_evidence_forgery(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    evidence = json.loads((package / "candidate_evidence_v1.json").read_text(encoding="utf-8"))
    evidence["tasks"]["gpqa_diamond"]["source_role"] = "eligible_source_exists"
    _reseal_candidate_evidence(package, evidence)

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_CANDIDATE_EVIDENCE_INVALID",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


@pytest.mark.parametrize("field", ("certification_status", "mechanical_candidate_id"))
def test_package_rejects_resealed_candidate_evidence_with_missing_audit_field(
    tmp_path: Path,
    field: str,
) -> None:
    package = _copy_package(tmp_path)
    evidence = json.loads((package / "candidate_evidence_v1.json").read_text(encoding="utf-8"))
    del evidence["tasks"]["mmlu_pro_engineering"][field]
    _reseal_candidate_evidence(package, evidence)

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_CANDIDATE_EVIDENCE_INVALID",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


def test_package_contains_partial_candidate_evidence_but_no_dependent_branches() -> None:
    manifest = json.loads((PACKAGE_ROOT / "package_manifest_v1.json").read_text(encoding="utf-8"))
    evidence = json.loads((PACKAGE_ROOT / "candidate_evidence_v1.json").read_text(encoding="utf-8"))
    leakage = json.loads((PACKAGE_ROOT / "leakage_report_v1.json").read_text(encoding="utf-8"))
    runtime = json.loads((PACKAGE_ROOT / "embedding_runtime_v1.json").read_text(encoding="utf-8"))

    assert not (PACKAGE_ROOT / "intervention_triplets_v1.json").exists()
    assert not (PACKAGE_ROOT / "relevance_universe_v1.json").exists()
    assert "task_local_intervention_triplets" not in manifest["required_artifacts"]
    assert {
        artifact["path"]
        for artifact in manifest["required_artifacts"]["partial_task_local_candidate_evidence"]
    } == {
        "candidate_evidence_v1.json",
        "sources/mmlu_pro_validation_475d58ba.parquet",
        "sources/gpqa_tree_633f5ee8.json",
    }
    assert evidence["tasks"]["gpqa_diamond"]["status"] == "NOT_READY_NO_ELIGIBLE_SOURCE"
    assert (
        evidence["tasks"]["mmlu_pro_engineering"]["status"]
        == "NOT_READY_SPLIT_REGISTRY_UNFROZEN"
    )
    assert leakage["status"] == "NOT_READY_REQUIRED_LEAKAGE_GATE_UNFROZEN"
    assert runtime["status"] == "COMPLETE"
    for task in manifest["tasks"].values():
        assert set(task["index_hashes"]) == {"clean"}
