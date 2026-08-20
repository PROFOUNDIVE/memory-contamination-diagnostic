from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from memcontam.contamination.phase12.models import canonical_json_hash
from memcontam.readiness.phase13_core_bundle import CoreTask, load_bundle_manifest
from memcontam.readiness.phase13_core_datasets import CANONICAL_CORE_ARTIFACT_SHA256
from memcontam.readiness.phase13_new_mcq_candidate import (
    DisplayedMcq,
    H1_ID,
    H2_ID,
    h1_selection,
    h2_selection,
)

_MMLU_REVISION: Final = "475d58ba0cc18a15fd5d4221f41919199e692331"
_MMLU_SOURCE_SHA256: Final = "a6db33e44c7a8d6a0a9665aabe6596a5e7436bebb62412d1219821283835e457"
_GPQA_REVISION: Final = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
_GPQA_TREE_SHA256: Final = "3a722b406849c230a76cf797f0e5481a2dd17fe403be650b5798703ecfa54526"
_COMMON_BLOCKERS: Final = (
    "prospectively_frozen_build_calibration_split",
    "candidate_coverage_contract",
    "baseline_native_render_packets",
    "three_evaluator_blinded_plausibility_panel",
    "leakage_metric_threshold_registry",
    "full_leakage_conformance",
    "correct_i1_constructibility_and_validity",
    "unicode_source_test_vector_provenance",
    "candidate_freeze_identity",
    "deterministic_relevance_affinity_constructibility",
)
_EVALUATION_PATHS: Final[dict[CoreTask, str]] = {
    "gpqa_diamond": "gpqa_diamond.jsonl",
    "mmlu_pro_engineering": "mmlu_pro_engineering.jsonl",
    "mmlu_pro_physics": "mmlu_pro_physics.jsonl",
}
_GPQA_EVALUATION_FILES: Final = ("gpqa_diamond.csv", "gpqa_extended.csv", "gpqa_main.csv")

class EvidenceError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MmluRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    question_id: int
    question: str
    options: tuple[str, ...]
    answer_index: int
    category: str


class GpqaTreeEntry(_FrozenModel):
    type: Literal["file"]
    oid: str
    size: int
    path: str


class MechanicalObservation(_FrozenModel):
    query_ids: tuple[str, ...]
    applicable_query_ids: tuple[str, ...]
    counterexample_query_ids: tuple[str, ...]
    displayed_option_counts: tuple[int, ...]
    selected_option_indices_audit_only: tuple[int | None, ...]
    official_gold_indices_audit_only: tuple[int, ...]


class SourceInventoryEntry(_FrozenModel):
    path: str
    git_oid: str
    size: int
    question_bearing: bool
    eligibility: Literal[
        "INELIGIBLE_EVALUATION_SET",
        "INELIGIBLE_METADATA_ONLY",
        "INELIGIBLE_NON_QUESTION_FILE",
    ]


class TaskCandidateEvidence(_FrozenModel):
    status: Literal["NOT_READY_SPLIT_REGISTRY_UNFROZEN", "NOT_READY_NO_ELIGIBLE_SOURCE"]
    certification_status: Literal["NOT_ESTABLISHED"]
    upstream_revision: str
    source_role: str
    source_sha256: str
    eligible_question_rows: int
    mechanical_candidate_id: None
    mechanical_observations: dict[str, MechanicalObservation]
    source_inventory: tuple[SourceInventoryEntry, ...]
    prohibited_source_configs: tuple[str, ...]
    remaining_objects: tuple[str, ...]


class CandidateEvidence(_FrozenModel):
    schema_version: Literal["phase13_new_mcq_candidate_evidence_v1"]
    authority_stack: tuple[str, ...]
    split_registry_status: Literal["NOT_ESTABLISHED"]
    source_role_evidence: tuple[str, ...]
    evaluation_artifact_hashes: dict[str, str]
    tasks: dict[str, TaskCandidateEvidence]
    evidence_hash: str


def build_candidate_evidence(root: Path, evaluation_root: Path) -> CandidateEvidence:
    rows = _mmlu_rows(root / "sources" / "mmlu_pro_validation_475d58ba.parquet")
    gpqa_inventory = _gpqa_inventory(root / "sources" / "gpqa_tree_633f5ee8.json")
    tasks = {
        "mmlu_pro_engineering": _mmlu_task(rows, "engineering"),
        "mmlu_pro_physics": _mmlu_task(rows, "physics"),
        "gpqa_diamond": TaskCandidateEvidence(
            status="NOT_READY_NO_ELIGIBLE_SOURCE",
            certification_status="NOT_ESTABLISHED",
            upstream_revision=_GPQA_REVISION,
            source_role="pinned_tree_has_no_non_evaluation_question_partition",
            source_sha256=_GPQA_TREE_SHA256,
            eligible_question_rows=0,
            mechanical_candidate_id=None,
            mechanical_observations={},
            source_inventory=gpqa_inventory,
            prohibited_source_configs=("gpqa_diamond", "gpqa_main", "gpqa_extended"),
            remaining_objects=(
                "eligible_question_bearing_construction_source",
                "prospective_displayed_permutation_identity",
                *_COMMON_BLOCKERS,
            ),
        ),
    }
    evidence = CandidateEvidence(
        schema_version="phase13_new_mcq_candidate_evidence_v1",
        authority_stack=(
            "phase13_theory_revised_v1",
            "phase13_baseline_revised_v5",
            "phase13_protocol_revised_v7",
            "phase13_experiment_revised_v7",
        ),
        split_registry_status="NOT_ESTABLISHED",
        source_role_evidence=(
            "MMLU-Pro/evaluate_from_local.py@996df131:validation_few_shot_prompt_source",
            "MMLU-Pro/evaluate_from_local.py@996df131:test_scored_evaluation_source",
            "GPQA@633f5ee8:pinned_source_tree_inventory",
        ),
        evaluation_artifact_hashes=_evaluation_hashes(evaluation_root),
        tasks=tasks,
        evidence_hash="0" * 64,
    )
    return evidence.model_copy(update={"evidence_hash": canonical_json_hash(evidence.model_dump(exclude={"evidence_hash"}, mode="json"))})


def materialize_candidate_evidence(root: Path, evaluation_root: Path) -> Path:
    output = root / "candidate_evidence_v1.json"
    output.write_text(build_candidate_evidence(root, evaluation_root).model_dump_json(indent=2) + "\n", encoding="utf-8")
    return output


def validate_candidate_evidence(root: Path, evaluation_root: Path) -> CandidateEvidence:
    expected = build_candidate_evidence(root, evaluation_root)
    try:
        actual = CandidateEvidence.model_validate_json((root / "candidate_evidence_v1.json").read_bytes())
    except (OSError, ValidationError) as error:
        raise EvidenceError("NEW_MCQ_CANDIDATE_EVIDENCE_INVALID") from error
    if actual != expected:
        raise EvidenceError("NEW_MCQ_CANDIDATE_EVIDENCE_INVALID")
    return actual


def _mmlu_rows(path: Path) -> tuple[MmluRow, ...]:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise EvidenceError("NEW_MCQ_CANDIDATE_SOURCE_INVALID") from error
    if digest != _MMLU_SOURCE_SHA256:
        raise EvidenceError("NEW_MCQ_BUILD_SOURCE_MISMATCH")
    try:
        load_dataset = importlib.import_module("datasets").load_dataset
        rows = tuple(
            MmluRow.model_validate(row)
            for row in load_dataset("parquet", data_files=str(path), split="train")
        )
    except (AttributeError, ImportError, TypeError, ValidationError, ValueError) as error:
        raise EvidenceError("NEW_MCQ_CANDIDATE_SOURCE_INVALID") from error
    return rows


def _gpqa_inventory(path: Path) -> tuple[SourceInventoryEntry, ...]:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise EvidenceError("NEW_MCQ_CANDIDATE_SOURCE_INVALID") from error
    if digest != _GPQA_TREE_SHA256:
        raise EvidenceError("NEW_MCQ_CANDIDATE_SOURCE_INVALID")
    try:
        raw = TypeAdapter(tuple[GpqaTreeEntry, ...]).validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise EvidenceError("NEW_MCQ_CANDIDATE_SOURCE_INVALID") from error
    return tuple(
        SourceInventoryEntry(
            path=entry.path,
            git_oid=entry.oid,
            size=entry.size,
            question_bearing=entry.path in _GPQA_EVALUATION_FILES,
            eligibility=(
                "INELIGIBLE_EVALUATION_SET"
                if entry.path in _GPQA_EVALUATION_FILES
                else "INELIGIBLE_METADATA_ONLY"
                if entry.path == "gpqa_experts.csv"
                else "INELIGIBLE_NON_QUESTION_FILE"
            ),
        )
        for entry in raw
    )


def _evaluation_hashes(root: Path) -> dict[str, str]:
    try:
        manifest, _ = load_bundle_manifest(root)
    except (OSError, ValueError) as error:
        raise EvidenceError("NEW_MCQ_EVALUATION_ARTIFACT_MISMATCH") from error
    if set(manifest.artifacts) != set(_EVALUATION_PATHS):
        raise EvidenceError("NEW_MCQ_EVALUATION_ARTIFACT_MISMATCH")
    hashes: dict[str, str] = {}
    try:
        for task, expected_path in _EVALUATION_PATHS.items():
            artifact = manifest.artifacts[task]
            actual = hashlib.sha256((root / expected_path).read_bytes()).hexdigest()
            if (
                artifact.path != expected_path
                or artifact.sha256 != actual
                or actual != CANONICAL_CORE_ARTIFACT_SHA256[task]
            ):
                raise EvidenceError("NEW_MCQ_EVALUATION_ARTIFACT_MISMATCH")
            hashes[task] = actual
    except OSError as error:
        raise EvidenceError("NEW_MCQ_EVALUATION_ARTIFACT_MISMATCH") from error
    return hashes


def _mmlu_task(rows: tuple[MmluRow, ...], category: str) -> TaskCandidateEvidence:
    selected = tuple(row for row in rows if row.category == category)
    if len(selected) != 5:
        raise EvidenceError("NEW_MCQ_BUILD_SOURCE_CARDINALITY_INVALID")
    return TaskCandidateEvidence(
        status="NOT_READY_SPLIT_REGISTRY_UNFROZEN",
        certification_status="NOT_ESTABLISHED",
        upstream_revision=_MMLU_REVISION,
        source_role="upstream_validation_few_shot_development_unpartitioned",
        source_sha256=_MMLU_SOURCE_SHA256,
        eligible_question_rows=5,
        mechanical_candidate_id=None,
        mechanical_observations={
            H1_ID: _observation(selected, h1_selection),
            H2_ID: _observation(selected, h2_selection),
        },
        source_inventory=(),
        prohibited_source_configs=(),
        remaining_objects=_COMMON_BLOCKERS,
    )


def _observation(rows: tuple[MmluRow, ...], selector: Callable[[DisplayedMcq], int | None]) -> MechanicalObservation:
    items = tuple(_displayed(row) for row in rows)
    selected = tuple(selector(item) for item in items)
    return MechanicalObservation(
        query_ids=tuple(item.query_id for item in items),
        applicable_query_ids=tuple(item.query_id for item, value in zip(items, selected, strict=True) if value is not None),
        counterexample_query_ids=tuple(item.query_id for item, value in zip(items, selected, strict=True) if value is not None and value != item.gold_index),
        displayed_option_counts=tuple(len(item.options) for item in items),
        selected_option_indices_audit_only=selected,
        official_gold_indices_audit_only=tuple(item.gold_index for item in items),
    )


def _displayed(row: MmluRow) -> DisplayedMcq:
    return DisplayedMcq(query_id=str(row.question_id), stem=row.question, options=row.options, gold_index=row.answer_index, display_identity=f"mmlu_pro_validation_source_order::{row.question_id}")


__all__ = ["CandidateEvidence", "EvidenceError", "TaskCandidateEvidence", "build_candidate_evidence", "materialize_candidate_evidence", "validate_candidate_evidence"]
