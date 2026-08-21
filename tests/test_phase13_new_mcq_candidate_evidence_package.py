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


def _reseal_interventions(package: Path, registry: dict[str, JsonValue]) -> None:
    registry_path = package / "intervention_registry_v1.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    manifest_path = package / "package_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["required_artifacts"]["retained_h2_intervention_registry"][0][
        "sha256"
    ] = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    manifest["package_reconstruction_identity"] = _reconstruction_identity(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    status_path = package.parent / STATUS_PATH.name
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["candidate_package"]["reconstruction_identity"] = manifest[
        "package_reconstruction_identity"
    ]
    status["candidate_package"]["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    status_path.write_text(json.dumps(status), encoding="utf-8")


def test_package_rejects_resealed_semantic_intervention_forgery(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    registry = json.loads((package / "intervention_registry_v1.json").read_text(encoding="utf-8"))
    registry["tasks"]["mmlu_pro_engineering"]["documents"]["contam"][
        "task_id"
    ] = "mmlu_pro_physics"
    _reseal_interventions(package, registry)

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_INTERVENTION_REGISTRY_INVALID",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


@pytest.mark.parametrize("field", ("selected_candidate_id", "candidate_family_status"))
def test_package_rejects_intervention_registry_with_missing_freeze_field(
    tmp_path: Path,
    field: str,
) -> None:
    package = _copy_package(tmp_path)
    registry = json.loads((package / "intervention_registry_v1.json").read_text(encoding="utf-8"))
    del registry["tasks"]["mmlu_pro_engineering"][field]
    _reseal_interventions(package, registry)

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


def test_package_contains_complete_h2_interventions_and_branch_indices() -> None:
    manifest = json.loads((PACKAGE_ROOT / "package_manifest_v1.json").read_text(encoding="utf-8"))
    registry = json.loads((PACKAGE_ROOT / "intervention_registry_v1.json").read_text(encoding="utf-8"))
    leakage = json.loads((PACKAGE_ROOT / "leakage_report_v1.json").read_text(encoding="utf-8"))
    runtime = json.loads((PACKAGE_ROOT / "embedding_runtime_v1.json").read_text(encoding="utf-8"))

    assert {
        artifact["path"]
        for artifact in manifest["required_artifacts"]["retained_h2_intervention_registry"]
    } == {"intervention_registry_v1.json"}
    assert tuple(registry["tasks"]) == ("mmlu_pro_engineering", "mmlu_pro_physics")
    assert all(
        task["selected_candidate_id"] == "MCQ-H2-DETAIL-LENGTH-v1"
        and tuple(task["documents"]) == ("contam", "correct", "irrelevant")
        and "relevance" not in task
        for task in registry["tasks"].values()
    )
    assert leakage["status"] == "PASS"
    assert runtime["status"] == "COMPLETE"
    for task in manifest["tasks"].values():
        assert set(task["index_hashes"]) == {"clean", "correct", "irrelevant", "contam"}


def test_package_binds_protocol_selection_record() -> None:
    manifest = json.loads((PACKAGE_ROOT / "package_manifest_v1.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (PACKAGE_ROOT / "authority_selection_v1.json").read_text(encoding="utf-8")
    )
    interventions = json.loads(
        (PACKAGE_ROOT / "intervention_registry_v1.json").read_text(encoding="utf-8")
    )

    assert {
        artifact["path"]
        for artifact in manifest["required_artifacts"]["accepted_h2_selection_record"]
    } == {"authority_selection_v1.json"}
    assert selection == {
        "schema_version": "phase13_new_mcq_rag_authority_selection_v1",
        "protocol_authority_sha256": (
            "022879f559b145e30e645b6ccbd139e9927899d370f1956d27a0562580acf85f"
        ),
        "selection_law": "accepted_continuation_state_2026_08_21",
        "task_selections": {
            "mmlu_pro_engineering": "MCQ-H2-DETAIL-LENGTH-v1",
            "mmlu_pro_physics": "MCQ-H2-DETAIL-LENGTH-v1",
        },
        "gpqa_extension_status": "H2_BLINDED_PLAUSIBILITY_GATE_FAILED_EXTENSION_ONLY",
    }
    assert interventions["authority_selection_sha256"] == hashlib.sha256(
        (PACKAGE_ROOT / "authority_selection_v1.json").read_bytes()
    ).hexdigest()


def test_package_excludes_unfrozen_clean_relevance_universe() -> None:
    manifest = json.loads((PACKAGE_ROOT / "package_manifest_v1.json").read_text(encoding="utf-8"))
    leakage = json.loads((PACKAGE_ROOT / "leakage_report_v1.json").read_text(encoding="utf-8"))

    assert "task_local_relevance_universe" not in manifest["required_artifacts"]
    assert not (PACKAGE_ROOT / "relevance_universe_v1.json").exists()
    assert "relevance_universe" not in dict(leakage["input_hashes"])
