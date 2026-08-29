from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path

from pydantic import ValidationError

from memcontam.baselines.bot_read import RETRIEVAL_THRESHOLD
from memcontam.readiness.phase13_legacy_rag_models import IndexBundle
from memcontam.readiness.phase13_readiness0_f1c_build import build_f1c_report
from memcontam.readiness.phase13_readiness0_f1c_contract import (
    ARMS,
    F1CReportError,
    LEGACY_TASKS,
    TASKS,
    QuerySpec,
    canonical_hash,
    file_hash,
    memory_candidates,
    queries,
    text_hash,
)
from memcontam.readiness.phase13_readiness0_f1c_models import F1CReport, F1CRetrievalRow


def validate_f1c_report(raw: bytes, repository_root: Path) -> F1CReport:
    try:
        report = F1CReport.model_validate_json(raw)
    except ValidationError as error:
        raise F1CReportError("READINESS0_F1C_REPORT_INVALID") from error
    expected_keys = {
        (baseline, task, arm)
        for baseline, tasks in (
            ("rag_frozen", LEGACY_TASKS),
            ("bot_style", TASKS),
            ("dc_rs", TASKS),
        )
        for task in tasks
        for arm in ARMS
    }
    runtime_payload = report.runtime.model_dump(mode="json", exclude={"runtime_hash"})
    report_payload = report.model_dump(mode="json", exclude={"report_hash"})
    if (
        len(report.rows) != 52
        or report.row_count != len(report.rows)
        or {(row.baseline, row.task, row.arm) for row in report.rows} != expected_keys
        or report.runtime.runtime_hash != canonical_hash(runtime_payload)
        or report.report_hash != canonical_hash(report_payload)
        or report.runtime.device != "cpu"
        or report.runtime.dtype != "torch.float32"
    ):
        raise F1CReportError("READINESS0_F1C_REPORT_INVALID")
    query_specs = queries(repository_root)
    for row in report.rows:
        _validate_row(row, query_specs[row.task], repository_root)
    return report


def validate_f1c_reproducibility(
    report_raw: bytes,
    repository_root: Path,
    cache_root: Path,
) -> F1CReport:
    report = validate_f1c_report(report_raw, repository_root)
    reproduced = build_f1c_report(repository_root, cache_root)
    if reproduced.model_dump_json(indent=2).encode() + b"\n" != report_raw:
        raise F1CReportError("READINESS0_F1C_REPRODUCTION_MISMATCH")
    return report


def _validate_row(row: F1CRetrievalRow, query: QuerySpec, root: Path) -> None:
    identity = {
        "task": row.task,
        "baseline": row.baseline,
        "arm": row.arm,
        "sample_id": query.sample_id,
    }
    valid_scores = all(math.isfinite(score) and -1.0 <= score <= 1.0 for score in row.scores)
    ordered = all(
        left_score > right_score
        or (left_score == right_score and left_id < right_id)
        for (left_id, left_score), (right_id, right_score) in pairwise(
            zip(row.candidate_ids, row.scores, strict=True)
        )
    )
    if (
        row.row_id != canonical_hash(identity)
        or row.sample_id != query.sample_id
        or row.query_source != query.source
        or row.query_source_sha256 != file_hash(root / query.source)
        or row.query_sha256 != text_hash(query.text)
        or len(set(row.candidate_ids)) != len(row.candidate_ids)
        or not valid_scores
        or not ordered
        or row.source_span_ids != row.selected_ids
        or row.source_span_join_sha256
        != canonical_hash({"selected_ids": row.selected_ids})
    ):
        raise F1CReportError("READINESS0_F1C_ROW_INVALID")
    if row.baseline == "rag_frozen":
        _validate_rag(row, root)
    else:
        _validate_memory(row, root)


def _validate_rag(row: F1CRetrievalRow, root: Path) -> None:
    bundle = IndexBundle.model_validate_json(
        (root / f"data/phase13/rag/legacy/{row.task}/indices.json").read_bytes()
    )
    branch = bundle.branches[row.arm]
    if (
        row.state_identity_sha256 is not None
        or row.corpus_identity_sha256 != branch.corpus_content_hash
        or row.index_identity_sha256 != branch.index_artifact_hash
        or set(row.candidate_ids) != {document.id for document in branch.documents}
        or row.threshold is not None
        or row.top_k != 3
        or row.selected_ids != row.candidate_ids[:3]
    ):
        raise F1CReportError("READINESS0_F1C_RAG_ROW_INVALID")


def _validate_memory(row: F1CRetrievalRow, root: Path) -> None:
    candidates = memory_candidates(root, row.task, row.arm, row.baseline)
    expected_state = canonical_hash(
        {"task": row.task, "baseline": row.baseline, "arm": row.arm, "candidates": candidates}
    )
    expected_index = canonical_hash(
        {"candidate_ids": [item[0] for item in candidates], "scores": list(zip(row.candidate_ids, row.scores, strict=True))}
    )
    threshold = RETRIEVAL_THRESHOLD if row.baseline == "bot_style" else None
    top_k = 1 if row.baseline == "bot_style" else min(3, len(candidates))
    selected = tuple(
        entry_id
        for entry_id, score in zip(row.candidate_ids[:top_k], row.scores[:top_k], strict=True)
        if threshold is None or score >= threshold
    )
    if (
        row.state_identity_sha256 != expected_state
        or row.corpus_identity_sha256 is not None
        or row.index_identity_sha256 != expected_index
        or set(row.candidate_ids) != {item[0] for item in candidates}
        or row.threshold != threshold
        or row.top_k != top_k
        or row.selected_ids != selected
    ):
        raise F1CReportError("READINESS0_F1C_MEMORY_ROW_INVALID")


__all__ = ["validate_f1c_report", "validate_f1c_reproducibility"]
