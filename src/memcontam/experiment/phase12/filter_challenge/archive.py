from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from memcontam.experiment.phase12.filter_challenge.archive_authority import (
    ArchiveRegistryAuthority,
    validate_archive_authority,
)
from memcontam.experiment.phase12.filter_challenge.records import (
    PUBLIC_AUDIT_KEYS,
    AssessmentRecord,
    FilterChallengeArchive,
    FilterChallengeArchiveError,
)


PUBLIC_STREAMS = ("run.json", "assessments.jsonl", "candidate_aggregates.jsonl", "calls.jsonl")
ALL_STREAMS = (*PUBLIC_STREAMS, "audit/audit_labels.jsonl", "public_artifact_manifest.json", "archive_seal.json")


@dataclass(frozen=True, slots=True)
class ArchiveValidationReport:
    archive_valid: bool
    reason_code: str | None = None


def write_archive(
    root: Path, archive: FilterChallengeArchive, authority: ArchiveRegistryAuthority
) -> None:
    if root.exists():
        raise FilterChallengeArchiveError("ARCHIVE_ROOT_EXISTS")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
    try:
        _write_json(staging / "run.json", archive.run.model_dump(mode="json"))
        _write_jsonl(staging / "calls.jsonl", archive.calls)
        (staging / "audit").mkdir()
        _write_jsonl(staging / "audit" / "audit_labels.jsonl", archive.audit_labels)
        _write_jsonl(staging / "assessments.jsonl", archive.assessments)
        _write_jsonl(staging / "candidate_aggregates.jsonl", archive.candidate_aggregates)
        _write_manifest(staging)
        _write_json(staging / "archive_seal.json", _seal_payload(staging))
        report = validate_archive(staging, authority)
        if not report.archive_valid:
            raise FilterChallengeArchiveError(report.reason_code or "ARCHIVE_VALIDATION_FAILED")
        staging.rename(root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def validate_archive(root: Path, authority: ArchiveRegistryAuthority) -> ArchiveValidationReport:
    try:
        _validate_archive(root, authority)
    except FilterChallengeArchiveError as error:
        return ArchiveValidationReport(False, error.code)
    return ArchiveValidationReport(True)


def _validate_archive(root: Path, authority: ArchiveRegistryAuthority) -> None:
    files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if files != set(ALL_STREAMS):
        raise FilterChallengeArchiveError("ARCHIVE_STREAM_SET_INVALID")
    manifest = _read_json(root / "public_artifact_manifest.json")
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        raise FilterChallengeArchiveError("ARCHIVE_MANIFEST_INVALID")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(PUBLIC_STREAMS):
        raise FilterChallengeArchiveError("ARCHIVE_MANIFEST_INVALID")
    for filename in PUBLIC_STREAMS:
        record = artifacts[filename]
        if not isinstance(record, dict) or record.get("sha256") != _sha256(root / filename) or record.get(
            "count"
        ) != _count(root / filename):
            raise FilterChallengeArchiveError("ARCHIVE_HASH_MISMATCH")
    seal = _read_json(root / "archive_seal.json")
    expected_seal = _seal_payload(root)
    if not isinstance(seal, dict) or seal.get("public_artifact_manifest_sha256") != expected_seal[
        "public_artifact_manifest_sha256"
    ]:
        raise FilterChallengeArchiveError("ARCHIVE_SEAL_MISMATCH")
    if seal.get("audit_artifacts") != expected_seal["audit_artifacts"]:
        raise FilterChallengeArchiveError("AUDIT_HASH_MISMATCH")
    run = _read_json(root / "run.json")
    assessments = _read_jsonl(root / "assessments.jsonl")
    aggregates = _read_jsonl(root / "candidate_aggregates.jsonl")
    calls = _read_jsonl(root / "calls.jsonl")
    audit_labels = _read_jsonl(root / "audit" / "audit_labels.jsonl")
    if any(_contains_audit_key(payload) for payload in (run, assessments, aggregates, calls)):
        raise FilterChallengeArchiveError("AUDIT_FIELD_IN_PUBLIC_STREAM")
    try:
        archive = FilterChallengeArchive.model_validate(
            {
                "run": run,
                "assessments": assessments,
                "candidate_aggregates": aggregates,
                "calls": calls,
                "audit_labels": audit_labels,
            }
        )
    except ValidationError as error:
        raise FilterChallengeArchiveError(_validation_code(error)) from error
    _validate_ranges(root, archive.assessments)
    validate_archive_authority(archive, authority)
    _validate_aggregate_rollups(archive, authority)


def _validate_ranges(root: Path, assessments: tuple[AssessmentRecord, ...]) -> None:
    lines = _jsonl_boundaries(root / "calls.jsonl")
    for assessment in assessments:
        declared = (
            assessment.control_answer_call_id,
            assessment.challenge_answer_call_id,
            *assessment.baseline_native_aux_call_ids_control,
            *assessment.baseline_native_aux_call_ids_challenge,
        )
        ranged: list[str] = []
        for raw_range in assessment.raw_record_ranges:
            if raw_range.path != "calls.jsonl":
                raise FilterChallengeArchiveError("RAW_RECORD_PATH_INVALID")
            row = lines.get((raw_range.start, raw_range.end))
            if row is None:
                raise FilterChallengeArchiveError("RAW_RECORD_RANGE_INVALID")
            call_id = row.get("call_id")
            if not isinstance(call_id, str):
                raise FilterChallengeArchiveError("RAW_RECORD_RANGE_INVALID")
            ranged.append(call_id)
        if len(set(ranged)) != len(ranged) or set(ranged) != set(declared) or len(ranged) != len(declared):
            raise FilterChallengeArchiveError("RAW_RECORD_RANGE_INVALID")


def _validate_aggregate_rollups(archive: FilterChallengeArchive, authority: ArchiveRegistryAuthority) -> None:
    for aggregate in archive.candidate_aggregates:
        rows = tuple(row for row in archive.assessments if row.candidate_entry_id == aggregate.candidate_entry_id)
        if any(
            getattr(aggregate, name) != getattr(rows[0], name)
            for name in (
                "filter_policy_version", "calibration_probe_inventory_id",
                "calibration_probe_inventory_manifest_hash", "operational_probe_suite_id",
                "operational_probe_suite_manifest_hash", "decision_rule_id",
            )
        ):
            raise FilterChallengeArchiveError("AGGREGATE_IDENTITY_INVALID")
        evaluable = tuple(row for row in rows if row.probe_disposition != "not_evaluable")
        witnesses = tuple(row for row in rows if row.probe_disposition == "witness")
        reasons: dict[str, int] = {}
        for row in rows:
            if row.probe_disposition == "not_evaluable":
                reasons[row.probe_reason_code] = reasons.get(row.probe_reason_code, 0) + 1
        total_aux_calls = sum(
            len(row.baseline_native_aux_call_ids_control) + len(row.baseline_native_aux_call_ids_challenge)
            for row in rows
        )
        total_retries = sum(row.retry_count_control + row.retry_count_challenge for row in rows)
        expected = {
            "n_nominal_attempted_pairs": len(rows),
            "n_control_strict_primary_eligible": sum(
                row.control_probe_eligibility_state == "strict_primary_eligible" for row in rows
            ),
            "n_control_canonicalization_sensitivity_eligible": sum(
                row.control_probe_eligibility_state == "canonicalization_sensitivity_eligible" for row in rows
            ),
            "n_candidate_exposed": sum(row.candidate_final_context_inclusion for row in rows),
            "n_strictly_evaluable": len(evaluable), "n_witness": len(witnesses),
            "n_no_witness": sum(row.probe_disposition == "evaluated_no_witness" for row in rows),
            "n_not_evaluable": sum(row.probe_disposition == "not_evaluable" for row in rows),
            "n_distinct_evaluable_probes": len({row.probe_id for row in evaluable}),
            "n_distinct_witness_probes": len({row.probe_id for row in witnesses}),
            "witness_probe_ids": tuple(sorted({row.probe_id for row in witnesses})),
            "not_evaluable_reason_counts": reasons,
            "total_answer_calls": 2 * len(rows),
            "total_baseline_native_aux_calls": total_aux_calls,
            "total_retries": total_retries,
            "total_tokens": sum(row.input_tokens + row.output_tokens for row in rows),
            "total_latency_ms": sum(row.total_latency_ms for row in rows),
            "total_calls": 2 * len(rows) + total_aux_calls + total_retries,
        }
        if any(getattr(aggregate, name) != value for name, value in expected.items()) or abs(
            aggregate.total_cost - sum(row.monetary_cost for row in rows)
        ) > 1e-12:
            raise FilterChallengeArchiveError("AGGREGATE_RECONCILIATION_FAILED")


def _write_manifest(root: Path) -> None:
    _write_json(
        root / "public_artifact_manifest.json",
        {
            "status": "completed",
            "artifacts": {
                filename: {"count": _count(root / filename), "sha256": _sha256(root / filename)}
                for filename in PUBLIC_STREAMS
            },
        },
    )


def _seal_payload(root: Path) -> dict[str, object]:
    audit = root / "audit" / "audit_labels.jsonl"
    return {
        "public_artifact_manifest_sha256": _sha256(root / "public_artifact_manifest.json"),
        "audit_artifacts": {"audit/audit_labels.jsonl": {"count": _count(audit), "sha256": _sha256(audit)}},
    }


def _jsonl_boundaries(path: Path) -> dict[tuple[int, int], dict[str, object]]:
    offset = 0
    rows: dict[tuple[int, int], dict[str, object]] = {}
    for line in path.read_bytes().splitlines(keepends=True):
        end = offset + len(line)
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise FilterChallengeArchiveError("ARCHIVE_JSON_INVALID") from error
        if not isinstance(row, dict):
            raise FilterChallengeArchiveError("ARCHIVE_JSON_INVALID")
        rows[(offset, end)] = row
        offset = end
    return rows


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FilterChallengeArchiveError("ARCHIVE_JSON_INVALID") from error


def _read_jsonl(path: Path):
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FilterChallengeArchiveError("ARCHIVE_JSON_INVALID") from error


def _write_json(path: Path, payload) -> None:
    path.write_text(_canonical(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(_canonical(row.model_dump(mode="json")) for row in rows), encoding="utf-8")


def _canonical(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count(path: Path) -> int:
    return 1 if path.suffix == ".json" else len(path.read_text(encoding="utf-8").splitlines())


def _contains_audit_key(value) -> bool:
    if isinstance(value, dict):
        return bool(PUBLIC_AUDIT_KEYS & value.keys()) or any(_contains_audit_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_audit_key(item) for item in value)
    return False


def _validation_code(error: ValidationError) -> str:
    for detail in error.errors():
        context = detail.get("ctx")
        if isinstance(context, dict):
            nested = context.get("error")
            if isinstance(nested, FilterChallengeArchiveError):
                return nested.code
    return "ARCHIVE_SCHEMA_INVALID"


__all__ = (
    "ALL_STREAMS", "PUBLIC_STREAMS", "ArchiveRegistryAuthority", "ArchiveValidationReport",
    "validate_archive", "write_archive",
)
