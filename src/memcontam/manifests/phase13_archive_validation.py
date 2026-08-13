from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Final

from pydantic import ValidationError

from memcontam.manifests.phase13_archive_models import Phase13Archive, SourceAttemptRow
from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow


EXECUTION_OWNER: Final = "phase13-h10-execution-owner-v1"
OFFLINE_OWNER: Final = "phase13-offline-compute-owner-v1"
NATIVE_STATE: Final = "phase13-native-capacity-registry-v1"
FAMILY: Final = "phase13-primary-seven-slot-family-game24-v1"
HISTORICAL_RUN: Final = "phase13-pre-main-calibration-15usd-rerun1"


class Phase13ArchiveError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class Phase13ArchiveReport:
    archive_valid: bool
    resolved_edges: int
    claim_ids: tuple[str, ...]
    errors: tuple[Phase13ArchiveError, ...] = ()

    @property
    def reason_code(self) -> str | None:
        return None if not self.errors else self.errors[0].code

    def to_dict(self) -> dict[str, bool | int | str | None | list[str]]:
        return {
            "archive_valid": self.archive_valid,
            "resolved_edges": self.resolved_edges,
            "claim_ids": list(self.claim_ids),
            "reason_code": self.reason_code,
        }


def validate_phase13_archive(root: Path) -> Phase13ArchiveReport:
    try:
        archive = _read_archive(root)
        edges = _validate(archive)
    except Phase13ArchiveError as error:
        return Phase13ArchiveReport(False, 0, (), (error,))
    return Phase13ArchiveReport(True, edges, tuple(row.claim_id for row in archive.claims))


def _read_archive(root: Path) -> Phase13Archive:
    try:
        return Phase13Archive.model_validate_json((root / "phase13_archive.json").read_bytes())
    except (OSError, ValidationError) as error:
        raise Phase13ArchiveError("PHASE13_ARCHIVE_SCHEMA_INVALID") from error


def _validate(archive: Phase13Archive) -> int:
    _validate_authorities(archive)
    attempts = _validate_sources(archive)
    window_scores = _validate_windows(archive, attempts)
    _validate_ledgers(archive, attempts)
    _validate_aggregates_and_claims(archive, window_scores)
    if (
        archive.historical_reference.run_id != HISTORICAL_RUN
        or archive.historical_reference.availability != "external_reference_unavailable"
        or archive.historical_reference.imported
    ):
        raise Phase13ArchiveError("HISTORICAL_REFERENCE_INVALID")
    return 4 + sum(len(row.events) for row in attempts.values()) + len(archive.provider_ledger) + len(
        archive.offline_ledger
    ) + len(archive.derived_windows) + len(archive.aggregates) + len(archive.claims)


def _validate_authorities(archive: Phase13Archive) -> None:
    for binding in (
        archive.authorities.execution,
        archive.authorities.analysis,
        archive.authorities.historical,
        archive.authorities.checkpoint,
    ):
        try:
            observed = hashlib.sha256(read_regular_nofollow(Path(binding.path))).hexdigest()
        except AuthorityFileError as error:
            raise Phase13ArchiveError("AUTHORITY_HASH_MISMATCH") from error
        if observed != binding.sha256:
            raise Phase13ArchiveError("AUTHORITY_HASH_MISMATCH")


def _validate_sources(archive: Phase13Archive) -> dict[str, SourceAttemptRow]:
    attempts: dict[str, SourceAttemptRow] = {}
    completed: dict[str, SourceAttemptRow] = {}
    for row in archive.source_attempts:
        if row.attempt_id in attempts:
            raise Phase13ArchiveError("DUPLICATE_ATTEMPT_ID")
        if row.rerun_parent_id is not None and row.rerun_parent_id not in attempts:
            raise Phase13ArchiveError("RERUN_PARENT_MISMATCH")
        if row.status == "invalidated":
            if not row.invalidated_reason or row.rerun_parent_id is not None:
                raise Phase13ArchiveError("INVALIDATED_ATTEMPT_EVIDENCE_REQUIRED")
        elif row.invalidated_reason is not None:
            raise Phase13ArchiveError("INVALIDATED_ATTEMPT_EVIDENCE_REQUIRED")
        _validate_events(row)
        _validate_raw(row)
        attempts[row.attempt_id] = row
        if row.status == "completed":
            completed[row.source_run_id] = row
    return completed


def _validate_raw(row: SourceAttemptRow) -> None:
    try:
        lines = Path(row.source_raw_path).read_bytes().splitlines(keepends=True)
    except OSError as error:
        raise Phase13ArchiveError("SOURCE_RAW_HASH_MISMATCH") from error
    if hashlib.sha256(b"".join(lines)).hexdigest() != row.source_raw_sha256:
        raise Phase13ArchiveError("SOURCE_RAW_HASH_MISMATCH")
    start, end = row.raw_record_range
    if start < 0 or end < start or end >= len(lines) or end - start + 1 != len(row.events):
        raise Phase13ArchiveError("SOURCE_RAW_RANGE_MISMATCH")
    try:
        selected = tuple(json.loads(line) for line in lines[start : end + 1])
    except json.JSONDecodeError as error:
        raise Phase13ArchiveError("SOURCE_RAW_RANGE_MISMATCH") from error
    if tuple(event.model_dump(mode="json") for event in row.events) != selected:
        raise Phase13ArchiveError("SOURCE_RAW_RANGE_MISMATCH")


def _validate_events(row: SourceAttemptRow) -> None:
    for index, event in enumerate(row.events):
        if not event.source_checkpoint_id.startswith("checkpoint-"):
            raise Phase13ArchiveError("SOURCE_CHECKPOINT_MISMATCH")
        if event.native_state_id != NATIVE_STATE:
            raise Phase13ArchiveError("NATIVE_STATE_MISMATCH")
        if event.call_owner_id != EXECUTION_OWNER:
            raise Phase13ArchiveError("EVENT_CALL_OWNER_MISMATCH")
        expected = () if index == 0 else (row.events[index - 1].event_id,)
        if event.lineage_parent_ids != expected:
            raise Phase13ArchiveError("EVENT_LINEAGE_MISMATCH")
        if index and row.events[index - 1].state_after_sha256 != event.state_before_sha256:
            raise Phase13ArchiveError("EVENT_STATE_CHAIN_MISMATCH")


def _validate_windows(
    archive: Phase13Archive, attempts: dict[str, SourceAttemptRow]
) -> dict[str, tuple[int, ...]]:
    scores: dict[str, tuple[int, ...]] = {}
    for row in archive.derived_windows:
        source = attempts.get(row.source_run_id)
        if source is None:
            raise Phase13ArchiveError("DERIVED_SOURCE_RUN_MISSING")
        end = row.window_length - 1
        selected = tuple(event for event in source.events if event.event_time <= end)
        if row.source_event_range != (0, end) or row.event_ids != tuple(e.event_id for e in selected):
            raise Phase13ArchiveError("DERIVED_EVENT_RANGE_MISMATCH")
        if row.source_raw_sha256 != source.source_raw_sha256:
            raise Phase13ArchiveError("DERIVED_SOURCE_HASH_MISMATCH")
        if row.status != "ESTIMABLE":
            raise Phase13ArchiveError("WINDOW_STATUS_PROMOTION_FORBIDDEN")
        if row.family_id != FAMILY:
            raise Phase13ArchiveError("WINDOW_FAMILY_MISMATCH")
        if row.owner_id != OFFLINE_OWNER:
            raise Phase13ArchiveError("OFFLINE_OWNER_MISMATCH")
        scores[row.window_id] = tuple(event.verified_score for event in selected)
    return scores


def _validate_ledgers(archive: Phase13Archive, attempts: dict[str, SourceAttemptRow]) -> None:
    event_calls = {
        event.semantic_call_id
        for attempt in attempts.values()
        for event in attempt.events
    }
    ledger_calls = {row.semantic_call_id for row in archive.provider_ledger}
    if event_calls != ledger_calls:
        raise Phase13ArchiveError("PROVIDER_LEDGER_MISMATCH")
    if any(row.execution_owner_id != EXECUTION_OWNER for row in archive.provider_ledger):
        raise Phase13ArchiveError("PROVIDER_OWNER_MISMATCH")
    if any(not row.transport_attempt_ids for row in archive.provider_ledger):
        raise Phase13ArchiveError("PROVIDER_LEDGER_MISMATCH")
    operations = ("prefix_derivation", "paired_seed_bootstrap", "report_rendering")
    if tuple(row.operation for row in archive.offline_ledger) != operations:
        raise Phase13ArchiveError("OFFLINE_LEDGER_MISMATCH")
    if any(row.provider_calls != 0 for row in archive.offline_ledger):
        raise Phase13ArchiveError("OFFLINE_PROVIDER_WORK_FORBIDDEN")
    if any(row.owner_id != OFFLINE_OWNER for row in archive.offline_ledger):
        raise Phase13ArchiveError("OFFLINE_OWNER_MISMATCH")


def _validate_aggregates_and_claims(
    archive: Phase13Archive, window_scores: dict[str, tuple[int, ...]]
) -> None:
    windows = {row.window_id: row for row in archive.derived_windows}
    aggregates = {}
    for row in archive.aggregates:
        if row.original_weights != row.weights:
            raise Phase13ArchiveError("WEIGHT_RENORMALIZATION_FORBIDDEN")
        if row.family_id != FAMILY:
            raise Phase13ArchiveError("AGGREGATE_FAMILY_MISMATCH")
        try:
            sources = tuple(windows[source_id] for source_id in row.source_ids)
        except KeyError as error:
            raise Phase13ArchiveError("AGGREGATE_SOURCE_MISSING") from error
        source_run_ids = {source.source_run_id for source in sources}
        if set(row.weights) != source_run_ids:
            raise Phase13ArchiveError("WEIGHT_RENORMALIZATION_FORBIDDEN")
        if row.status != "ESTIMABLE" or any(source.status != "ESTIMABLE" for source in sources):
            raise Phase13ArchiveError("AGGREGATE_STATUS_PROMOTION_FORBIDDEN")
        estimate = mean(mean(window_scores[source.window_id]) for source in sources)
        if not isinstance(row.estimate, float) or not math.isclose(row.estimate, estimate):
            raise Phase13ArchiveError("AGGREGATE_RECONSTRUCTION_MISMATCH")
        aggregates[row.aggregate_id] = row
    for claim in archive.claims:
        aggregate = aggregates.get(claim.aggregate_id)
        if aggregate is None or claim.status != "supported":
            raise Phase13ArchiveError("CLAIM_RECONSTRUCTION_MISMATCH")
        if claim.family_id != aggregate.family_id or claim.estimate != aggregate.estimate:
            raise Phase13ArchiveError("CLAIM_RECONSTRUCTION_MISMATCH")


__all__ = (
    "Phase13ArchiveError",
    "Phase13ArchiveReport",
    "validate_phase13_archive",
)
