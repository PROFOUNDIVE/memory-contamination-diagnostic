from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import ValidationError

from memcontam.experiment.phase12.filter_challenge.final_verifier_types import FinalVerifierError
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.experiment.phase12.filter_challenge.validation_summary import (
    Task17CommandRecord,
    Task17ValidationSummary,
)


def command_record(
    repository_root: Path,
    scratch_root: Path,
    command_id: str,
    arguments: tuple[str, ...],
    stdout: str,
    stderr: str,
) -> Task17CommandRecord:
    return Task17CommandRecord(
        command_id=command_id,
        cwd=_normalize_path(str(repository_root), repository_root, scratch_root),
        exit_code=0,
        normalized_argv=tuple(
            _normalize_path(value, repository_root, scratch_root)
            for value in ("phase12", "filter-v5", command_id, *arguments)
        ),
        stdout_sha256=hashlib.sha256(stdout.encode()).hexdigest(),
        stderr_sha256=hashlib.sha256(stderr.encode()).hexdigest(),
    )


def reconcile_summary_records(
    validation_summary: Path, actual_records: tuple[Task17CommandRecord, ...]
) -> None:
    try:
        summary = Task17ValidationSummary.model_validate_json(validation_summary.read_bytes())
    except ValidationError as error:
        raise FinalVerifierError("INTEGRATION_SUMMARY_RECORDS_MISMATCH") from error
    if summary.command_records != actual_records:
        raise FinalVerifierError("INTEGRATION_SUMMARY_RECORDS_MISMATCH")


def record_json(record: Task17CommandRecord) -> dict[str, JsonValue]:
    argv: list[JsonValue] = list(record.normalized_argv)
    return {
        "command_id": record.command_id,
        "cwd": record.cwd,
        "exit_code": record.exit_code,
        "normalized_argv": argv,
        "stderr_sha256": record.stderr_sha256,
        "stdout_sha256": record.stdout_sha256,
    }


def _normalize_path(value: str, repository_root: Path, scratch_root: Path) -> str:
    candidate = Path(value)
    if not candidate.is_absolute():
        return value
    for root, placeholder in ((repository_root, "<repository>"), (scratch_root, "<scratch>")):
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        return placeholder if relative == Path() else f"{placeholder}/{relative.as_posix()}"
    return value
