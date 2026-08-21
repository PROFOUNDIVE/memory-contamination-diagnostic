from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import JsonValue, TypeAdapter

from memcontam.contamination.phase12.models import canonical_json_hash
from memcontam.readiness.phase13_new_mcq_rag import Candidate, TASKS
from memcontam.readiness.phase13_new_mcq_rag_manifest import (
    Artifact,
    package_reconstruction_identity,
)
from memcontam.readiness.phase13_new_mcq_rag_models import (
    BRANCHES,
    EXPECTED_CLASSES,
    SerializedIndexBundle,
)

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


def write_manifest(
    root: Path,
    candidates: dict[str, tuple[Candidate, ...]],
    indices: dict[str, dict[str, JsonValue]],
) -> None:
    artifacts = {
        "complete_source_eligibility_registry": [artifact(root, "source_eligibility_registry_v1.json")],
        "accepted_document_registry": [artifact(root, f"accepted/{task}.jsonl") for task in TASKS],
        "verified_embedding_runtime_artifact": [artifact(root, "embedding_runtime_v1.json")],
        "serialized_branch_index_artifacts": [artifact(root, f"indices/{task}.json") for task in TASKS],
        "complete_leakage_evidence": [artifact(root, "leakage_report_v1.json")],
        "retained_h2_intervention_registry": [artifact(root, "intervention_registry_v1.json")],
        "accepted_h2_selection_record": [artifact(root, "authority_selection_v1.json")],
    }
    assert set(artifacts) == set(EXPECTED_CLASSES)
    tasks: dict[str, JsonValue] = {}
    task_artifacts: list[dict[str, str]] = []
    for task, rows in candidates.items():
        serialized = SerializedIndexBundle.model_validate(indices[task])
        task_files = [
            artifact(root, f"candidates/{task}.jsonl"),
            artifact(root, f"reviews/{task}.json"),
            artifact(root, f"accepted/{task}.jsonl"),
            artifact(root, f"indices/{task}.json"),
        ]
        task_artifacts.extend(task_files)
        tasks[task] = _JSON_OBJECT.validate_python(
            {
                "documents": 24,
                "candidate": task_files[0],
                "review": task_files[1],
                "accepted": task_files[2],
                "index": task_files[3],
                "corpus_hash": canonical_json_hash(
                    [{"id": row.document_id, "text": row.text} for row in rows]
                ),
                "index_hashes": {
                    branch: serialized.branches[branch].index_artifact_hash
                    for branch in BRANCHES
                },
            }
        )
    source_registry = artifact(root, "source_registry_v1.json")
    authoring_contract = artifact(root, "authoring_contract_v1.json")
    reconstruction = package_reconstruction_identity(
        tuple(
            Artifact.model_validate(value)
            for value in (
                source_registry,
                authoring_contract,
                *(item for values in artifacts.values() for item in values),
                *task_artifacts,
            )
        )
    )
    write_json(
        root / "package_manifest_v1.json",
        _JSON_OBJECT.validate_python(
            {
                "schema_version": "new_mcq_rag_package_manifest_v1",
                "source_registry": source_registry,
                "authoring_contract": authoring_contract,
                "required_artifacts": artifacts,
                "tasks": tasks,
                "package_reconstruction_identity": reconstruction,
                "promotion": {
                    "status": "NOT_READY",
                    "reason": "NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN",
                    "remaining_objects": [
                        "clean_document_applicability_predicates_and_relevance_universe"
                    ],
                },
            }
        ),
        pretty=True,
    )


def write_status(root: Path) -> None:
    manifest = json.loads((root / "package_manifest_v1.json").read_text(encoding="utf-8"))
    status_path = root.parent / "new_mcq_rag_status_v1.json"
    status = {
        "schema_version": "phase13_new_mcq_rag_status_v1",
        "authority_sha256": (
            "4b1db4e55e68ec8e00fe022b9bea1685bebb340138df0e39fddc7823aafdc374"
        ),
        "candidate_package": {
            "path": "data/phase13/rag/new_mcq/package_manifest_v1.json",
            "sha256": sha256(root / "package_manifest_v1.json"),
            "status": "NOT_READY",
            "reconstruction_identity": manifest["package_reconstruction_identity"],
        },
        "cells": {
            task: {
                "status": "NOT_READY",
                "reason": "NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN",
                "entry_condition_met": False,
                "missing_objects": [
                    "clean_document_applicability_predicates_and_relevance_universe"
                ],
                "index_hashes": manifest["tasks"][task]["index_hashes"],
            }
            for task in TASKS
        },
        "cutoff": "2026-08-22T18:00:00+09:00",
        "cutoff_applied": False,
        "cutoff_status": "PENDING_REGISTERED_CUTOFF",
        "retrieval_contract": {
            "embedding_model": "BAAI/bge-m3",
            "embedding_revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "similarity": "cosine",
            "top_k": 3,
            "reranker": None,
            "score_threshold": None,
            "tie_break": "lexical_document_id",
            "corpus_scope": "same_task_only",
            "update_mode": "frozen_read_only",
        },
        "scientific_contract": {
            "answer_free": True,
            "atomic_documents": True,
            "documents_per_stratum": 6,
            "documents_per_task": 24,
            "procedural_only": True,
            "semantic_strata": [
                "requirement_quantifier_constraint_interpretation",
                "option_wise_evidence_comparison_elimination",
                "contradiction_counterexample_consistency_checking",
                "uncertainty_management_final_answer_verification",
            ],
            "task_specific": True,
        }
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def artifact(root: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": sha256(root / relative)}


def write_json(path: Path, value: JsonValue, *, pretty: bool = False) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["sha256", "write_json", "write_manifest", "write_status"]
